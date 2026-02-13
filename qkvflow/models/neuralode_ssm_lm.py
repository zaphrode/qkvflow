"""
Neural ODE Language Model with SSM (State Space Model) replacing FFN.

This module implements a time-continuous transformer where:
- Attention mechanism remains time-varying
- FFN is replaced with a time-varying SSM layer
"""

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from haliax import NamedArray, Axis
from haliax.jax_utils import maybe_rng_split, named_call
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Optional, Dict, Any

from levanter.models.gpt2 import Gpt2Config
from qkvflow.nn.time_embed import SinusoidalPosEmb
from qkvflow.nn.dynamic import TemporalLayerNorm, Attention


class TemporalSSM(eqx.Module):
    """Time-varying Structured State Space Model

    Implements a simplified SSM with parameters A(t), B(t), C(t), D(t), Δ(t)
    generated from a time-dependent hypernetwork.
    """

    # Hypernetwork components
    lin1: hnn.Linear
    lin2: hnn.Linear

    # Parameter generators
    f_A: hnn.Linear
    f_B: hnn.Linear
    f_C: hnn.Linear
    f_D: hnn.Linear
    f_delta: hnn.Linear

    # Axes
    Embed: hax.AxisSpec = eqx.field(static=True)
    StateSize: hax.Axis = eqx.field(static=True)
    TembedDim: hax.AxisSpec = eqx.field(static=True)

    @staticmethod
    def init(SinusodialDim, TembedDim, Embed, state_size=64, *, key):
        k_lin1, k_lin2, k_A, k_B, k_C, k_D, k_delta = jrandom.split(key, 7)

        StateSize = hax.Axis("StateSize", state_size)
        TembedDim_alias = TembedDim.alias("TembedDim_alias")

        lin1 = hnn.Linear.init(SinusodialDim, TembedDim_alias, key=k_lin1)
        lin2 = hnn.Linear.init(TembedDim_alias, TembedDim, key=k_lin2)

        # SSM parameter generators
        f_A = hnn.Linear.init(TembedDim, StateSize, key=k_A, use_bias=True)
        f_B = hnn.Linear.init(
            TembedDim, (StateSize, Embed), key=k_B, use_bias=True)
        f_C = hnn.Linear.init(
            TembedDim, (Embed, StateSize), key=k_C, use_bias=True)
        f_D = hnn.Linear.init(TembedDim, Embed, key=k_D, use_bias=True)
        f_delta = hnn.Linear.init(
            TembedDim, StateSize, key=k_delta, use_bias=True)

        return TemporalSSM(
            lin1, lin2, f_A, f_B, f_C, f_D, f_delta,
            Embed, StateSize, TembedDim
        )

    def _get_params(self, time_embed):
        """Generate SSM parameters from time embedding"""
        t_emb = self.lin1(time_embed)
        t_emb = hnn.silu(t_emb)
        t_emb = self.lin2(t_emb)

        # A: diagonal state transition (negative for stability)
        A_diag = -hnn.softplus(self.f_A(t_emb))
        # B: input-to-state matrix
        B = self.f_B(t_emb)
        # C: state-to-output matrix
        C = self.f_C(t_emb)
        # D: skip connection
        D = self.f_D(t_emb)
        # delta: discretization timestep (positive)
        delta = hnn.softplus(self.f_delta(t_emb)) + 1e-4

        return A_diag, B, C, D, delta

    @named_call
    def __call__(self, time_embed, x, *, key=None):
        """Forward pass through SSM

        Args:
            time_embed: Time embedding from hypernetwork
            x: Input with shape (..., position, embed)
            key: Random key (unused, for compatibility)

        Returns:
            Output with same shape as input
        """
        A_diag, B, C, D, delta = self._get_params(time_embed)

        # Discretize: convert continuous dynamics to discrete
        A_bar = hax.exp(delta * A_diag)
        B_bar = delta * B

        # Selective scan along sequence dimension
        def scan_fn(h, x_t):
            # h: hidden state (StateSize,)
            # x_t: input token (Embed,)
            h_new = A_bar * h + hax.dot("embed", B_bar, x_t)
            y = hax.dot("StateSize", C, h_new)
            return h_new, y

        # Initialize hidden state with batch dimensions
        batch_axes = tuple(ax for ax in x.axes if ax not in [
                           self.Embed] and ax.name != "position")
        h_0 = hax.zeros(batch_axes + (self.StateSize,))

        # Run scan along position axis
        _, outputs = hax.scan(scan_fn, axis="position")(h_0, x)

        # Add skip connection
        outputs = outputs + D * x
        return hax.auto_sharded(outputs)


class SSMBlock(eqx.Module):
    """Transformer block: ẋ(t) = f_attn(x,t) + g_ssm(x,t)"""

    config: Gpt2Config = eqx.field(static=True)
    attn_ln: TemporalLayerNorm
    attn: Attention
    ssm_ln: TemporalLayerNorm
    ssm: TemporalSSM
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config, SinusodialDim, TembedDim, ssm_state_size=64, *, key):
        k_attn, k_ssm, k_ln1, k_ln2 = jrandom.split(key, 4)

        attn_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln1
        )
        attn = Attention.init(config, SinusodialDim, TembedDim, key=k_attn)

        ssm_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln2
        )
        ssm = TemporalSSM.init(SinusodialDim, TembedDim, config.Embed,
                               state_size=ssm_state_size, key=k_ssm)

        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        return SSMBlock(config, attn_ln, attn, ssm_ln, ssm, resid_dropout)

    @named_call
    def __call__(self, time_embed, x, mask, layer_idx, *, key):
        """Forward pass: attention + SSM"""
        k1, k2, k3, k4 = maybe_rng_split(key, 4)

        # f_attn(x, t)
        attn_output = self.attn(time_embed, self.attn_ln(time_embed, x),
                                mask, layer_idx, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2)

        # g_ssm(x, t)
        ssm_output = self.ssm(time_embed, self.ssm_ln(time_embed, x), key=k3)
        ssm_output = self.resid_dropout(ssm_output, key=k4)

        return attn_output + ssm_output


class NeuralOdeSSMTransformer(eqx.Module):
    """Neural ODE Transformer with SSM replacing FFN

    Architecture:
        dx/dt = sum_i [ f_attn_i(x,t) + g_ssm_i(x,t) ]
    where each layer contributes to the derivative.
    """

    config: Gpt2Config = eqx.field(static=True)
    time_emb: SinusoidalPosEmb
    blocks: list[SSMBlock]
    ln_f: TemporalLayerNorm
    dropout: hnn.Dropout

    @staticmethod
    def init(config, SinusodialDim, TembedDim, ssm_state_size=64, *, key):
        k_emb, k_blocks, k_ln = jrandom.split(key, 3)

        # Time embedding
        time_emb = SinusoidalPosEmb.init(SinusodialDim, key=k_emb)
        
        # Resize SinusodialDim to match the output of time embedding (sinusodial_dim * 2 + 1)
        SinusodialDim = SinusodialDim.resize(SinusodialDim.size * 2 + 1)

        # SSM blocks
        block_keys = jrandom.split(k_blocks, config.num_layers)
        blocks = [
            SSMBlock.init(config, SinusodialDim, TembedDim,
                          ssm_state_size=ssm_state_size, key=k)
            for k in block_keys
        ]

        # Final layer norm
        ln_f = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )

        dropout = hnn.Dropout(pdrop=config.embed_pdrop)

        return NeuralOdeSSMTransformer(config, time_emb, blocks, ln_f, dropout)

    @named_call
    def __call__(self, x, attn_mask, t, *, key):
        """
        Args:
            x: Input embeddings with axes (batch, position, embed)
            attn_mask: Attention mask
            t: Time value in [0, 1]
            key: Random key for dropout

        Returns:
            Transformed embeddings
        """
        # Generate time embedding
        time_embed = self.time_emb(t)

        # Split key for dropout and blocks
        k_drop, k_blocks = maybe_rng_split(key, 2)
        x = self.dropout(x, key=k_drop)

        # Forward through all blocks
        block_keys = maybe_rng_split(k_blocks, len(self.blocks))
        for i, (block, k) in enumerate(zip(self.blocks, block_keys)):
            x = x + block(time_embed, x, attn_mask, i, key=k)

        # Final layer norm
        x = self.ln_f(time_embed, x)

        return x


class NeuralOdeSSMLMHeadModel(eqx.Module):
    """Complete Language Model with Neural ODE + SSM"""

    Vocab: hax.Axis = eqx.field(static=True)
    config: Gpt2Config = eqx.field(static=True)
    embeddings: hnn.Embedding
    transformer: NeuralOdeSSMTransformer
    lm_head: hnn.Linear

    # ODE solver config
    time_embedding_dim: int = eqx.field(static=True)
    sinusoidal_dim: int = eqx.field(static=True)
    ssm_state_size: int = eqx.field(static=True)

    @staticmethod
    def init(
        Vocab: hax.Axis,
        config: Gpt2Config,
        time_embedding_dim: int = 64,
        sinusoidal_dim: int = 32,
        ssm_state_size: int = 64,
        *,
        key
    ):
        k_emb, k_trans, k_head = jrandom.split(key, 3)

        # Token embeddings
        embeddings = hnn.Embedding.init(
            Vocab, config.Embed, key=k_emb
        )

        # Time embedding dimensions
        TembedDim = hax.Axis("TembedDim", time_embedding_dim)
        # SinusoidalPosEmb takes sinusoidal_dim as input and outputs sinusoidal_dim * 2 + 1 features
        SinusodialDim = hax.Axis("SinusodialDim", sinusoidal_dim)

        # Transformer with SSM
        transformer = NeuralOdeSSMTransformer.init(
            config, SinusodialDim, TembedDim,
            ssm_state_size=ssm_state_size, key=k_trans
        )

        # Language model head (unembedding)
        lm_head = hnn.Linear.init(
            config.Embed, Vocab, key=k_head, use_bias=False
        )

        return NeuralOdeSSMLMHeadModel(
            Vocab, config, embeddings, transformer, lm_head,
            time_embedding_dim, sinusoidal_dim, ssm_state_size
        )

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        t: float = 1.0,
        attn_mask: Optional[NamedArray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """Forward pass through the model

        Args:
            input_ids: Token IDs with axis (batch, position)
            t: ODE time (0 = initial, 1 = final state)
            attn_mask: Optional attention mask
            key: Random key for dropout

        Returns:
            Logits with axes (batch, position, vocab)
        """
        # Embed tokens
        x = self.embeddings(input_ids)

        # Create default causal mask if not provided
        if attn_mask is None:
            attn_mask = hax.nn.attention.causal_mask(
                self.config.Pos, self.config.KeyPos
            )

        # Time as scalar
        t_scalar = NamedArray(jnp.array(t), ())

        # Forward through transformer
        x = self.transformer(x, attn_mask, t_scalar, key=key)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    def compute_loss(
        self,
        input_ids: NamedArray,
        targets: NamedArray,
        t: float = 1.0,
        loss_mask: Optional[NamedArray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """Compute cross-entropy loss

        Args:
            input_ids: Input token IDs
            targets: Target token IDs  
            t: ODE time
            loss_mask: Optional mask for loss computation
            key: Random key

        Returns:
            Scalar loss value
        """
        logits = self(input_ids, t=t, key=key)

        # Convert targets to one-hot encoding (required by hax.nn.cross_entropy_loss)
        import jax.numpy as jnp
        targets_onehot_array = jax.nn.one_hot(targets.array, self.Vocab.size)
        targets_onehot = hax.named(targets_onehot_array, tuple(targets.axes) + (self.Vocab,))

        # Compute cross-entropy loss
        loss = hax.nn.cross_entropy_loss(
            logits, self.Vocab, targets_onehot,
            reduction=hax.mean,
            where=loss_mask
        )

        # Return scalar value (loss is already scalar after reduction)
        return loss.array if hasattr(loss, 'array') else loss


def count_parameters(model) -> Dict[str, int]:
    """Count parameters in each component"""
    counts = {}

    # Embeddings
    emb_params = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model.embeddings, eqx.is_array)))
    counts['embeddings'] = emb_params

    # Transformer blocks
    trans_params = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model.transformer, eqx.is_array)))
    counts['transformer'] = trans_params

    # LM head
    head_params = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model.lm_head, eqx.is_array)))
    counts['lm_head'] = head_params

    counts['total'] = sum(counts.values())

    return counts
