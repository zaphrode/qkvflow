#!/usr/bin/env python3
"""
Test Time-Indexed SSM: Combining parameter sharing with state space models.

This compares three architectures:
1. Standard Transformer (separate MLP weights per layer)
2. Time-Indexed MLP (shared MLP weights with time modulation)
3. Time-Indexed SSM (shared SSM weights with time modulation)

Usage:
    python scripts/test_time_indexed_ssm.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass

from levanter.models.gpt2 import Gpt2Config
from qkvflow.nn.time_embed import SinusoidalPosEmb
from qkvflow.nn.dynamic import TemporalLayerNorm, Attention
from haliax.jax_utils import maybe_rng_split, named_call

# Import time-indexed models
import importlib.util
spec = importlib.util.spec_from_file_location(
    "time_indexed_models", 
    "/home/nahid/Documents/qkvflow/scripts/test_time_indexed_weights.py"
)
time_indexed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_indexed)

TimeIndexedAttention = time_indexed.TimeIndexedAttention
StandardTransformer = time_indexed.StandardTransformer


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    seq_len: int = 128
    vocab_size: int = 10000
    batch_size: int = 4
    num_steps: int = 1000
    learning_rate: float = 3e-4
    ssm_state_size: int = 64


class TimeIndexedSSM(eqx.Module):
    """SSM with time-indexed weight sharing
    
    Shares base SSM weights across layers, modulated by time.
    """
    
    # Hypernetwork for time embedding
    lin1: hnn.Linear
    lin2: hnn.Linear
    
    # Base SSM parameter generators (shared across layers)
    base_f_A: hnn.Linear
    base_f_B: hnn.Linear
    base_f_C: hnn.Linear
    base_f_D: hnn.Linear
    base_f_delta: hnn.Linear
    
    # Time modulation networks
    time_mod_A: hnn.Linear
    time_mod_B: hnn.Linear
    time_mod_C: hnn.Linear
    time_mod_D: hnn.Linear
    time_mod_delta: hnn.Linear
    
    # Axes
    Embed: hax.AxisSpec = eqx.field(static=True)
    StateSize: hax.Axis = eqx.field(static=True)
    TembedDim: hax.AxisSpec = eqx.field(static=True)
    
    @staticmethod
    def init(SinusodialDim, TembedDim, Embed, state_size=64, *, key):
        keys = jrandom.split(key, 12)
        
        StateSize = hax.Axis("StateSize", state_size)
        TembedDim_alias = TembedDim.alias("TembedDim_alias")
        
        # Time embedding hypernetwork
        lin1 = hnn.Linear.init(SinusodialDim, TembedDim_alias, key=keys[0])
        lin2 = hnn.Linear.init(TembedDim_alias, TembedDim, key=keys[1])
        
        # Base SSM parameter generators
        base_f_A = hnn.Linear.init(TembedDim, StateSize, key=keys[2], use_bias=True)
        base_f_B = hnn.Linear.init(TembedDim, (StateSize, Embed), key=keys[3], use_bias=True)
        base_f_C = hnn.Linear.init(TembedDim, (Embed, StateSize), key=keys[4], use_bias=True)
        base_f_D = hnn.Linear.init(TembedDim, Embed, key=keys[5], use_bias=True)
        base_f_delta = hnn.Linear.init(TembedDim, StateSize, key=keys[6], use_bias=True)
        
        # Time modulation (small networks that scale parameters)
        time_mod_A = hnn.Linear.init(SinusodialDim, StateSize, key=keys[7], use_bias=True)
        time_mod_B = hnn.Linear.init(SinusodialDim, (StateSize, Embed), key=keys[8], use_bias=True)
        time_mod_C = hnn.Linear.init(SinusodialDim, (Embed, StateSize), key=keys[9], use_bias=True)
        time_mod_D = hnn.Linear.init(SinusodialDim, Embed, key=keys[10], use_bias=True)
        time_mod_delta = hnn.Linear.init(SinusodialDim, StateSize, key=keys[11], use_bias=True)
        
        return TimeIndexedSSM(
            lin1, lin2,
            base_f_A, base_f_B, base_f_C, base_f_D, base_f_delta,
            time_mod_A, time_mod_B, time_mod_C, time_mod_D, time_mod_delta,
            Embed, StateSize, TembedDim
        )
    
    def _get_params(self, time_embed):
        """Generate time-dependent SSM parameters"""
        # Base time embedding
        t_emb = self.lin1(time_embed)
        t_emb = hnn.silu(t_emb)
        t_emb = self.lin2(t_emb)
        
        # Base parameters
        A_base = -hnn.softplus(self.base_f_A(t_emb))
        B_base = self.base_f_B(t_emb)
        C_base = self.base_f_C(t_emb)
        D_base = self.base_f_D(t_emb)
        delta_base = hnn.softplus(self.base_f_delta(t_emb)) + 1e-4
        
        # Time-dependent modulation
        A_scale = hnn.sigmoid(self.time_mod_A(time_embed))
        B_scale = hnn.sigmoid(self.time_mod_B(time_embed))
        C_scale = hnn.sigmoid(self.time_mod_C(time_embed))
        D_scale = hnn.sigmoid(self.time_mod_D(time_embed))
        delta_scale = hnn.sigmoid(self.time_mod_delta(time_embed))
        
        # Modulated parameters
        A_diag = A_base * A_scale
        B = B_base * B_scale
        C = C_base * C_scale
        D = D_base * D_scale
        delta = delta_base * delta_scale
        
        return A_diag, B, C, D, delta
    
    @named_call
    def __call__(self, time_embed, x, *, key=None):
        """Forward pass with time-modulated SSM"""
        A_diag, B, C, D, delta = self._get_params(time_embed)
        
        # Discretize
        A_bar = hax.exp(delta * A_diag)
        B_bar = delta * B
        
        # Selective scan
        def scan_fn(h, x_t):
            h_new = A_bar * h + hax.dot("embed", B_bar, x_t)
            y = hax.dot("StateSize", C, h_new)
            return h_new, y
        
        # Initialize hidden state
        batch_axes = tuple(ax for ax in x.axes if ax not in [self.Embed] and ax.name != "position")
        h_0 = hax.zeros(batch_axes + (self.StateSize,))
        _, outputs = hax.scan(scan_fn, axis="position")(h_0, x)
        
        outputs = outputs + D * x
        return hax.auto_sharded(outputs)


class TimeIndexedSSMBlock(eqx.Module):
    """Transformer block with time-indexed SSM"""
    
    config: Gpt2Config = eqx.field(static=True)
    attn_ln: TemporalLayerNorm
    attn: TimeIndexedAttention
    ssm_ln: TemporalLayerNorm
    ssm: TimeIndexedSSM
    resid_dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, ssm_state_size=64, *, key):
        k_attn, k_ssm, k_ln = jrandom.split(key, 3)
        
        attn_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )
        
        attn = TimeIndexedAttention.init(config, SinusodialDim, TembedDim, key=k_attn)
        
        ssm_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )
        
        ssm = TimeIndexedSSM.init(
            SinusodialDim, TembedDim, config.Embed,
            state_size=ssm_state_size, key=k_ssm
        )
        
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return TimeIndexedSSMBlock(config, attn_ln, attn, ssm_ln, ssm, resid_dropout)
    
    @named_call
    def __call__(self, time_embed, x, mask, layer_idx, *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)
        
        # Attention path
        attn_out = self.attn(time_embed, self.attn_ln(time_embed, x), 
                            mask, layer_idx, key=k1)
        attn_out = self.resid_dropout(attn_out, key=k2)
        x = x + attn_out
        
        # SSM path
        ssm_out = self.ssm(time_embed, self.ssm_ln(time_embed, x), key=k3)
        ssm_out = self.resid_dropout(ssm_out, key=k4)
        x = x + ssm_out
        
        return hax.auto_sharded(x)


class TimeIndexedSSMTransformer(eqx.Module):
    """Transformer with time-indexed SSM weight sharing"""
    
    config: Gpt2Config = eqx.field(static=True)
    time_emb: SinusoidalPosEmb
    shared_block: TimeIndexedSSMBlock  # Single shared SSM block!
    ln_f: TemporalLayerNorm
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, ssm_state_size=64, *, key):
        k_emb, k_block, k_ln = jrandom.split(key, 3)
        
        time_emb = SinusoidalPosEmb.init(SinusodialDim, key=k_emb)
        SinusodialDim_out = SinusodialDim.resize(SinusodialDim.size * 2 + 1)
        
        # Single shared SSM block for ALL layers
        shared_block = TimeIndexedSSMBlock.init(
            config, SinusodialDim_out, TembedDim,
            ssm_state_size=ssm_state_size, key=k_block
        )
        
        ln_f = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim_out, key=k_ln
        )
        
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)
        
        return TimeIndexedSSMTransformer(config, time_emb, shared_block, ln_f, dropout)
    
    @named_call
    def __call__(self, x, *, key):
        keys = maybe_rng_split(key, self.config.num_layers + 1)
        
        # Create causal mask
        Pos = x.resolve_axis("position")
        attn_mask = hax.nn.attention.causal_mask(Pos, Pos)
        
        # Apply dropout to embeddings
        x = self.dropout(x, key=keys[0])
        
        # Apply the SAME SSM block multiple times with different time indices
        for layer_idx in range(self.config.num_layers):
            t = (layer_idx + 1) / self.config.num_layers
            time_embed = self.time_emb(hax.named(jnp.array(t), ()))
            x = self.shared_block(time_embed, x, attn_mask, layer_idx, key=keys[layer_idx + 1])
        
        # Final layer norm
        time_embed = self.time_emb(hax.named(jnp.array(1.0), ()))
        x = self.ln_f(time_embed, x)
        
        return hax.auto_sharded(x)


def count_parameters(model):
    """Count parameters in a model"""
    return sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_array)
    ))


def create_dummy_batch(key, batch_size, seq_len, vocab_size, Vocab, Pos):
    """Create a dummy training batch"""
    Batch = hax.Axis("batch", batch_size)
    
    input_ids = jrandom.randint(key, (batch_size, seq_len), 0, vocab_size)
    input_ids = hax.named(input_ids, (Batch, Pos))
    
    targets = jnp.roll(input_ids.array, shift=-1, axis=-1)
    targets = hax.named(targets, (Batch, Pos))
    
    return input_ids, targets


@eqx.filter_jit
def train_step(model, embeddings, lm_head, opt_state, optimizer, 
               input_ids, targets, Vocab, key):
    """Single training step"""
    
    def loss_fn(model, embeddings, lm_head):
        x = embeddings(input_ids)
        x = model(x, key=key)
        logits = lm_head(x)
        
        targets_onehot = jax.nn.one_hot(targets.array, Vocab.size)
        targets_onehot = hax.named(targets_onehot, tuple(targets.axes) + (Vocab,))
        
        loss = hax.nn.cross_entropy_loss(
            logits, Vocab, targets_onehot, reduction=hax.mean
        )
        
        return loss.scalar()
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, embeddings, lm_head)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss


def run_experiment(model_type: str, config: ExperimentConfig, seed: int = 42):
    """Run training experiment
    
    Args:
        model_type: "standard", "time_indexed_mlp", or "time_indexed_ssm"
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {model_type.replace('_', ' ').title()}")
    print(f"{'='*70}")
    
    key = jrandom.PRNGKey(seed)
    k_model, k_emb, k_head, k_train = jrandom.split(key, 4)
    
    # Setup
    gpt_config = Gpt2Config(
        seq_len=config.seq_len,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    )
    
    Vocab = hax.Axis("vocab", config.vocab_size)
    TembedDim = hax.Axis("TembedDim", 64)
    SinusodialDim = hax.Axis("SinusodialDim", 32)
    Batch = hax.Axis("batch", config.batch_size)
    Pos = gpt_config.Pos
    
    # Create model
    if model_type == "standard":
        model = StandardTransformer.init(
            gpt_config, SinusodialDim, TembedDim, key=k_model
        )
    elif model_type == "time_indexed_mlp":
        TimeIndexedTransformer = time_indexed.TimeIndexedTransformer
        model = TimeIndexedTransformer.init(
            gpt_config, SinusodialDim, TembedDim, key=k_model
        )
    elif model_type == "time_indexed_ssm":
        model = TimeIndexedSSMTransformer.init(
            gpt_config, SinusodialDim, TembedDim,
            ssm_state_size=config.ssm_state_size, key=k_model
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Embeddings and LM head
    embeddings = hnn.Embedding.init(Vocab, gpt_config.Embed, key=k_emb)
    lm_head = hnn.Linear.init(gpt_config.Embed, Vocab, key=k_head)
    
    # Count parameters
    total_params = (count_parameters(model) + 
                   count_parameters(embeddings) + 
                   count_parameters(lm_head))
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer
    import optax
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=0.01)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training loop
    print(f"\nTraining for {config.num_steps} steps...")
    losses = []
    step_times = []
    
    for step in range(config.num_steps):
        # Get batch
        k_train, k_batch = jrandom.split(k_train)
        input_ids, targets = create_dummy_batch(
            k_batch, config.batch_size, config.seq_len, 
            config.vocab_size, Vocab, Pos
        )
        
        # Training step
        k_train, k_step = jrandom.split(k_train)
        start_time = time.time()
        
        model, opt_state, loss = train_step(
            model, embeddings, lm_head, opt_state, optimizer,
            input_ids, targets, Vocab, k_step
        )
        
        step_time = time.time() - start_time
        losses.append(float(loss))
        step_times.append(step_time)
        
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            avg_time = np.mean(step_times[-100:])
            print(f"  Step {step+1:4d} | Loss: {avg_loss:.4f} | Time: {avg_time*1000:.1f}ms")
    
    final_loss = float(np.mean(losses[-100:]))
    avg_step_time = float(np.mean(step_times))
    
    print(f"\n✓ Training complete")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Avg step time: {avg_step_time*1000:.1f}ms")
    
    return {
        'model_type': model_type,
        'params': total_params,
        'losses': losses,
        'final_loss': final_loss,
        'avg_step_time': avg_step_time
    }


def plot_comparison(results_list):
    """Plot three-way comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. Training loss curves
    ax = axes[0, 0]
    for res, color in zip(results_list, colors):
        label = res['model_type'].replace('_', ' ').title()
        ax.plot(res['losses'], label=label, alpha=0.7, linewidth=1.5, color=color)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter comparison
    ax = axes[0, 1]
    names = [r['model_type'].replace('_', ' ').title() for r in results_list]
    params = [r['params'] / 1e6 for r in results_list]
    bars = ax.bar(names, params, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Parameters (Millions)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.2f}M',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Final loss comparison
    ax = axes[0, 2]
    final_losses = [r['final_loss'] for r in results_list]
    bars = ax.bar(names, final_losses, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Training speed
    ax = axes[1, 0]
    step_times = [r['avg_step_time'] * 1000 for r in results_list]
    bars = ax.bar(names, step_times, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Step Time (ms)', fontsize=12)
    ax.set_title('Training Speed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, st in zip(bars, step_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{st:.1f}ms',
                ha='center', va='bottom', fontsize=10)
    
    # 5. Parameter efficiency (params per loss point)
    ax = axes[1, 1]
    efficiency = [r['params'] / (1e6 * r['final_loss']) for r in results_list]
    bars = ax.bar(names, efficiency, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Params/Loss (M)', fontsize=12)
    ax.set_title('Parameter Efficiency\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Compression ratio
    ax = axes[1, 2]
    baseline_params = results_list[0]['params']
    compression = [baseline_params / r['params'] for r in results_list]
    bars = ax.bar(names, compression, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression vs Standard\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, comp in zip(bars, compression):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{comp:.1f}x',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('time_indexed_ssm_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot to: time_indexed_ssm_comparison.png")
    plt.close()


def main():
    """Main experiment comparing all three architectures"""
    
    print("\n" + "="*70)
    print("TIME-INDEXED SSM vs MLP COMPARISON")
    print("="*70)
    print("\nComparing three architectures:")
    print("1. Standard Transformer (separate MLP per layer)")
    print("2. Time-Indexed MLP (shared MLP with time modulation)")
    print("3. Time-Indexed SSM (shared SSM with time modulation)")
    print()
    
    config = ExperimentConfig()
    results = []
    
    # Run all three experiments
    for model_type in ["standard", "time_indexed_mlp", "time_indexed_ssm"]:
        results.append(run_experiment(model_type, config, seed=42))
    
    # Plot comparison
    plot_comparison(results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - THREE-WAY COMPARISON")
    print("="*70)
    
    standard = results[0]
    mlp = results[1]
    ssm = results[2]
    
    print(f"\n📊 Parameters:")
    print(f"   Standard:           {standard['params']:>12,} (baseline)")
    print(f"   Time-Indexed MLP:   {mlp['params']:>12,} ({100*(1-mlp['params']/standard['params']):.1f}% reduction)")
    print(f"   Time-Indexed SSM:   {ssm['params']:>12,} ({100*(1-ssm['params']/standard['params']):.1f}% reduction)")
    
    print(f"\n📉 Final Loss:")
    print(f"   Standard:           {standard['final_loss']:.4f}")
    print(f"   Time-Indexed MLP:   {mlp['final_loss']:.4f} ({mlp['final_loss']-standard['final_loss']:+.4f})")
    print(f"   Time-Indexed SSM:   {ssm['final_loss']:.4f} ({ssm['final_loss']-standard['final_loss']:+.4f})")
    
    print(f"\n⚡ Training Speed:")
    print(f"   Standard:           {standard['avg_step_time']*1000:.1f}ms/step")
    print(f"   Time-Indexed MLP:   {mlp['avg_step_time']*1000:.1f}ms/step ({standard['avg_step_time']/mlp['avg_step_time']:.2f}x faster)")
    print(f"   Time-Indexed SSM:   {ssm['avg_step_time']*1000:.1f}ms/step ({standard['avg_step_time']/ssm['avg_step_time']:.2f}x faster)")
    
    print(f"\n💡 Key Insights:")
    
    # MLP vs SSM comparison
    if ssm['final_loss'] < mlp['final_loss']:
        improvement = 100 * (mlp['final_loss'] - ssm['final_loss']) / mlp['final_loss']
        print(f"   ✓ SSM outperforms MLP by {improvement:.1f}%")
    else:
        print(f"   ⚠ MLP performs slightly better than SSM")
    
    # Speed comparison
    if ssm['avg_step_time'] < mlp['avg_step_time']:
        speedup = mlp['avg_step_time'] / ssm['avg_step_time']
        print(f"   ✓ SSM is {speedup:.2f}x faster than MLP")
    else:
        slowdown = ssm['avg_step_time'] / mlp['avg_step_time']
        print(f"   ⚠ SSM is {slowdown:.2f}x slower than MLP")
    
    # Overall
    best = min(results, key=lambda r: r['final_loss'])
    print(f"   🏆 Best performer: {best['model_type'].replace('_', ' ').title()}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()


