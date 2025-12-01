"""
Neural ODE Language Model with MLP (Baseline)

This is the baseline version using standard MLP blocks instead of SSM.
Used for comparison with the SSM variant.
"""

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.random as jrandom

from levanter.models.gpt2 import Gpt2Config
from qkvflow.nn.time_embed import SinusoidalPosEmb
from qkvflow.nn.dynamic import TemporalLayerNorm, Attention, MLP, Block
from haliax.jax_utils import maybe_rng_split, named_call


class NeuralOdeLMHeadModel(eqx.Module):
    """Neural ODE Language Model with MLP blocks (baseline)"""
    
    embeddings: hnn.Embedding
    transformer: "NeuralOdeTransformer"
    lm_head: hnn.Linear
    Vocab: hax.Axis = eqx.field(static=True)
    
    @staticmethod
    def init(
        Vocab: hax.Axis,
        config: Gpt2Config,
        time_embedding_dim: int = 64,
        sinusoidal_dim: int = 32,
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
        SinusodialDim = hax.Axis("SinusodialDim", sinusoidal_dim)
        
        # Transformer with MLP
        transformer = NeuralOdeTransformer.init(
            config, SinusodialDim, TembedDim, key=k_trans
        )
        
        # Language model head (unembedding)
        lm_head = hnn.Linear.init(
            config.Embed, Vocab, key=k_head, use_bias=False
        )
        
        return NeuralOdeLMHeadModel(embeddings, transformer, lm_head, Vocab)
    
    @named_call
    def __call__(self, input_ids, *, t=1.0, key=None):
        """
        Args:
            input_ids: Input token IDs with axes (batch, position)
            t: Time value in [0, 1] for ODE integration
            key: Random key for dropout
            
        Returns:
            logits: Output logits with axes (batch, position, vocab)
        """
        # Embed tokens
        x = self.embeddings(input_ids)
        
        # Create causal attention mask
        Pos = input_ids.resolve_axis("position")
        attn_mask = hax.nn.attention.causal_mask(Pos, Pos)
        
        # Create scalar time value for time embedding
        import jax.numpy as jnp
        t_scalar = hax.named(jnp.array(t), ())
        
        # Pass through transformer
        x = self.transformer(x, attn_mask, t_scalar, key=key)
        
        # Output projection
        logits = self.lm_head(x)
        
        return hax.auto_sharded(logits)
    
    def compute_loss(self, input_ids, targets, *, t=1.0, key=None):
        """Compute cross-entropy loss"""
        logits = self(input_ids, t=t, key=key)
        
        # Convert targets to one-hot encoding (required by hax.nn.cross_entropy_loss)
        import jax
        targets_onehot_array = jax.nn.one_hot(targets.array, self.Vocab.size)
        targets_onehot = hax.named(targets_onehot_array, tuple(targets.axes) + (self.Vocab,))
        
        loss = hax.nn.cross_entropy_loss(
            logits, self.Vocab, targets_onehot, reduction=hax.mean
        )
        return loss.array if hasattr(loss, 'array') else loss


class NeuralOdeTransformer(eqx.Module):
    """Transformer with time-varying MLP blocks"""
    
    time_emb: SinusoidalPosEmb
    blocks: list
    ln_f: TemporalLayerNorm
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_emb, k_blocks, k_ln = jrandom.split(key, 3)
        
        # Time embedding
        time_emb = SinusoidalPosEmb.init(SinusodialDim, key=k_emb)
        
        # Resize SinusodialDim to match the output of time embedding (sinusodial_dim * 2 + 1)
        SinusodialDim = SinusodialDim.resize(SinusodialDim.size * 2 + 1)
        
        # MLP blocks
        block_keys = jrandom.split(k_blocks, config.num_layers)
        blocks = [
            Block.init(config, SinusodialDim, TembedDim, key=k)
            for k in block_keys
        ]
        
        # Final layer norm
        ln_f = TemporalLayerNorm.init(
            config.Embed, 
            eps=config.layer_norm_epsilon,
            use_bias=config.use_bias,
            TembedDim=TembedDim,
            SinusodialDim=SinusodialDim,
            key=k_ln
        )
        
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)
        
        return NeuralOdeTransformer(time_emb, blocks, ln_f, dropout)
    
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

