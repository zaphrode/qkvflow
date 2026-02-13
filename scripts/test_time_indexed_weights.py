#!/usr/bin/env python3
"""
Test time-indexed parameter sharing across transformer depth.

This experiment explores using the same base weights across all layers,
but modulating them with time-dependent functions.

Usage:
    python scripts/test_time_indexed_weights.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import haliax as hax
import haliax.nn as hnn
import matplotlib.pyplot as plt
import time
from typing import Optional
from dataclasses import dataclass

from levanter.models.gpt2 import Gpt2Config
from qkvflow.nn.time_embed import SinusoidalPosEmb
from qkvflow.nn.dynamic import TemporalLayerNorm, Attention, MLP
from haliax.jax_utils import maybe_rng_split, named_call


@dataclass
class ExperimentConfig:
    """Configuration for time-indexed weight experiments"""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    seq_len: int = 128
    vocab_size: int = 10000
    batch_size: int = 4
    num_steps: int = 1000
    learning_rate: float = 3e-4


class TimeIndexedAttention(eqx.Module):
    """Attention with time-indexed weight sharing"""
    
    # Base weights (shared across all time steps/layers)
    base_qkv: hnn.Linear
    base_out: hnn.Linear
    
    # Time-dependent modulation
    time_mod_qkv: hnn.Linear
    time_mod_out: hnn.Linear
    
    config: Gpt2Config = eqx.field(static=True)
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_qkv, k_out, k_mod1, k_mod2 = jrandom.split(key, 4)
        
        # Base weights (3x for Q, K, V)
        base_qkv = hnn.Linear.init(
            config.Embed, 
            (config.Heads, config.HeadSize, hax.Axis("qkv", 3)),
            key=k_qkv, use_bias=False
        )
        
        base_out = hnn.Linear.init(
            (config.Heads, config.HeadSize),
            config.Embed,
            key=k_out, use_bias=False
        )
        
        # Time modulation networks (small MLPs that scale base weights)
        time_mod_qkv = hnn.Linear.init(
            SinusodialDim,
            config.Embed,  # Output scaling factors
            key=k_mod1, use_bias=True
        )
        
        time_mod_out = hnn.Linear.init(
            SinusodialDim,
            config.Embed,
            key=k_mod2, use_bias=True
        )
        
        dropout = hnn.Dropout(pdrop=config.attn_pdrop)
        
        return TimeIndexedAttention(
            base_qkv, base_out, time_mod_qkv, time_mod_out,
            config, dropout
        )
    
    @named_call
    def __call__(self, time_embed, x, mask, layer_idx, *, key):
        k1, k2 = maybe_rng_split(key, 2)
        
        # Get time-dependent modulation factors (shape: Embed)
        qkv_scale = hnn.sigmoid(self.time_mod_qkv(time_embed))  # [0, 1] scaling
        out_scale = hnn.sigmoid(self.time_mod_out(time_embed))
        
        # Apply base weights with time modulation
        # x shape: (batch, seq, embed)
        # Scale input before projection
        x_scaled = x * qkv_scale
        qkv = self.base_qkv(x_scaled)
        
        # Split into Q, K, V
        q = qkv["qkv", 0]
        k = qkv["qkv", 1]
        v = qkv["qkv", 2]
        
        # Standard attention computation
        attn_scores = hax.dot("head_size", q, k) / jnp.sqrt(self.config.HeadSize.size)
        
        if mask is not None:
            attn_scores = hax.where(mask, attn_scores, -1e9)
        
        attn_weights = hnn.softmax(attn_scores, axis=self.config.Pos)
        attn_weights = self.dropout(attn_weights, key=k1)
        
        attn_out = hax.dot(self.config.Pos, attn_weights, v)
        
        # Project back with time-modulated output weights
        out = self.base_out(attn_out)
        out = out * out_scale  # Scale output
        out = self.dropout(out, key=k2)
        
        return hax.auto_sharded(out)


class TimeIndexedMLP(eqx.Module):
    """MLP with time-indexed weight sharing"""
    
    # Base weights
    base_up: hnn.Linear
    base_down: hnn.Linear
    
    # Time modulation
    time_mod_up: hnn.Linear
    time_mod_down: hnn.Linear
    
    config: Gpt2Config = eqx.field(static=True)
    Mlp: hax.Axis = eqx.field(static=True)
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, *, key):
        k_up, k_down, k_mod1, k_mod2 = jrandom.split(key, 4)
        
        # Create a separate Mlp axis with a different name
        Mlp = hax.Axis("mlp", config.hidden_dim)
        
        base_up = hnn.Linear.init(config.Embed, Mlp, key=k_up, use_bias=True)
        base_down = hnn.Linear.init(Mlp, config.Embed, key=k_down, use_bias=True)
        
        time_mod_up = hnn.Linear.init(
            SinusodialDim, config.Embed, key=k_mod1, use_bias=True
        )
        time_mod_down = hnn.Linear.init(
            SinusodialDim, Mlp, key=k_mod2, use_bias=True
        )
        
        dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return TimeIndexedMLP(
            base_up, base_down, time_mod_up, time_mod_down,
            config, Mlp, dropout
        )
    
    @named_call
    def __call__(self, time_embed, x, *, key):
        k1, k2 = maybe_rng_split(key, 2)
        
        # Time-dependent scaling
        up_scale = hnn.sigmoid(self.time_mod_up(time_embed))
        down_scale = hnn.sigmoid(self.time_mod_down(time_embed))
        
        # Modulated forward pass
        x_scaled = x * up_scale
        hidden = self.base_up(x_scaled)
        hidden = hnn.gelu(hidden)
        hidden = self.dropout(hidden, key=k1)
        
        hidden_scaled = hidden * down_scale
        out = self.base_down(hidden_scaled)
        out = self.dropout(out, key=k2)
        
        return hax.auto_sharded(out)


class TimeIndexedBlock(eqx.Module):
    """Transformer block with time-indexed weights"""
    
    config: Gpt2Config = eqx.field(static=True)
    attn_ln: TemporalLayerNorm
    attn: TimeIndexedAttention
    mlp_ln: TemporalLayerNorm
    mlp: TimeIndexedMLP
    resid_dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_attn, k_mlp, k_ln = jrandom.split(key, 3)
        
        attn_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )
        
        attn = TimeIndexedAttention.init(config, SinusodialDim, TembedDim, key=k_attn)
        
        mlp_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )
        
        mlp = TimeIndexedMLP.init(config, SinusodialDim, key=k_mlp)
        
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return TimeIndexedBlock(config, attn_ln, attn, mlp_ln, mlp, resid_dropout)
    
    @named_call
    def __call__(self, time_embed, x, mask, layer_idx, *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)
        
        # Attention path
        attn_out = self.attn(time_embed, self.attn_ln(time_embed, x), 
                            mask, layer_idx, key=k1)
        attn_out = self.resid_dropout(attn_out, key=k2)
        x = x + attn_out
        
        # MLP path
        mlp_out = self.mlp(time_embed, self.mlp_ln(time_embed, x), key=k3)
        mlp_out = self.resid_dropout(mlp_out, key=k4)
        x = x + mlp_out
        
        return hax.auto_sharded(x)


class TimeIndexedTransformer(eqx.Module):
    """Transformer with time-indexed weight sharing across all layers"""
    
    config: Gpt2Config = eqx.field(static=True)
    time_emb: SinusoidalPosEmb
    shared_block: TimeIndexedBlock  # Single shared block for all layers!
    ln_f: TemporalLayerNorm
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_emb, k_block, k_ln = jrandom.split(key, 3)
        
        time_emb = SinusoidalPosEmb.init(SinusodialDim, key=k_emb)
        
        # Resize for time embedding output
        SinusodialDim_out = SinusodialDim.resize(SinusodialDim.size * 2 + 1)
        
        # Single shared block for ALL layers
        shared_block = TimeIndexedBlock.init(
            config, SinusodialDim_out, TembedDim, key=k_block
        )
        
        ln_f = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim_out, key=k_ln
        )
        
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)
        
        return TimeIndexedTransformer(config, time_emb, shared_block, ln_f, dropout)
    
    @named_call
    def __call__(self, x, *, key):
        keys = maybe_rng_split(key, self.config.num_layers + 1)
        
        # Create causal mask
        Pos = x.resolve_axis("position")
        attn_mask = hax.nn.attention.causal_mask(Pos, Pos)
        
        # Apply dropout to embeddings
        x = self.dropout(x, key=keys[0])
        
        # Apply the SAME block multiple times with different time indices
        for layer_idx in range(self.config.num_layers):
            # Time index in [0, 1]
            t = (layer_idx + 1) / self.config.num_layers
            time_embed = self.time_emb(hax.named(jnp.array(t), ()))
            
            # Use the shared block with time-dependent modulation
            x = self.shared_block(time_embed, x, attn_mask, layer_idx, key=keys[layer_idx + 1])
        
        # Final layer norm at t=1.0
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
        # Embed
        x = embeddings(input_ids)
        
        # Forward
        x = model(x, key=key)
        
        # LM head
        logits = lm_head(x)
        
        # Loss - sparse cross entropy (no one-hot materialization)
        logits_flat = logits.array.reshape(-1, Vocab.size)
        targets_flat = targets.array.reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
        
        return jnp.mean(loss)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, embeddings, lm_head)
    
    # Update
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss


class StandardBlock(eqx.Module):
    """Standard transformer block (for comparison)"""
    
    config: Gpt2Config = eqx.field(static=True)
    attn_ln: TemporalLayerNorm
    attn: Attention
    mlp_ln: TemporalLayerNorm
    mlp: MLP
    resid_dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_attn, k_mlp, k_ln = jrandom.split(key, 3)
        
        attn_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )
        
        attn = Attention.init(config, SinusodialDim, TembedDim, key=k_attn)
        
        mlp_ln = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim, key=k_ln
        )
        
        mlp = MLP.init(config, SinusodialDim, TembedDim, key=k_mlp)
        
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return StandardBlock(config, attn_ln, attn, mlp_ln, mlp, resid_dropout)
    
    @named_call
    def __call__(self, time_embed, x, mask, layer_idx, *, key):
        k1, k2, k3, k4 = maybe_rng_split(key, 4)
        
        attn_out = self.attn(time_embed, self.attn_ln(time_embed, x), 
                            mask, layer_idx, key=k1)
        attn_out = self.resid_dropout(attn_out, key=k2)
        x = x + attn_out
        
        mlp_out = self.mlp(time_embed, self.mlp_ln(time_embed, x), key=k3)
        mlp_out = self.resid_dropout(mlp_out, key=k4)
        x = x + mlp_out
        
        return hax.auto_sharded(x)


class StandardTransformer(eqx.Module):
    """Standard transformer with separate weights per layer"""
    
    config: Gpt2Config = eqx.field(static=True)
    time_emb: SinusoidalPosEmb
    blocks: list[StandardBlock]
    ln_f: TemporalLayerNorm
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_emb, k_blocks, k_ln = jrandom.split(key, 3)
        
        time_emb = SinusoidalPosEmb.init(SinusodialDim, key=k_emb)
        
        # Resize for time embedding output
        SinusodialDim_out = SinusodialDim.resize(SinusodialDim.size * 2 + 1)
        
        # Separate blocks for each layer
        block_keys = jrandom.split(k_blocks, config.num_layers)
        blocks = [
            StandardBlock.init(config, SinusodialDim_out, TembedDim, key=k)
            for k in block_keys
        ]
        
        ln_f = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim_out, key=k_ln
        )
        
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)
        
        return StandardTransformer(config, time_emb, blocks, ln_f, dropout)
    
    @named_call
    def __call__(self, x, *, key):
        keys = maybe_rng_split(key, self.config.num_layers + 1)
        
        # Create causal mask
        Pos = x.resolve_axis("position")
        attn_mask = hax.nn.attention.causal_mask(Pos, Pos)
        
        # Apply dropout to embeddings
        x = self.dropout(x, key=keys[0])
        
        # Apply separate blocks
        for layer_idx, block in enumerate(self.blocks):
            t = (layer_idx + 1) / self.config.num_layers
            time_embed = self.time_emb(hax.named(jnp.array(t), ()))
            x = block(time_embed, x, attn_mask, layer_idx, key=keys[layer_idx + 1])
        
        # Final layer norm
        time_embed = self.time_emb(hax.named(jnp.array(1.0), ()))
        x = self.ln_f(time_embed, x)
        
        return hax.auto_sharded(x)


def run_experiment(use_time_indexed: bool, config: ExperimentConfig, seed: int = 42):
    """Run training experiment"""
    
    print(f"\n{'='*70}")
    print(f"Experiment: {'Time-Indexed' if use_time_indexed else 'Standard'} Transformer")
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
    
    # Create model
    if use_time_indexed:
        model = TimeIndexedTransformer.init(
            gpt_config, SinusodialDim, TembedDim, key=k_model
        )
    else:
        model = StandardTransformer.init(
            gpt_config, SinusodialDim, TembedDim, key=k_model
        )
    
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
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training loop
    print(f"\nTraining for {config.num_steps} steps...")
    loss_history = []
    step_times = []
    
    for step in range(config.num_steps):
        k_train, k_batch, k_step = jrandom.split(k_train, 3)
        
        input_ids, targets = create_dummy_batch(
            k_batch, config.batch_size, config.seq_len, 
            config.vocab_size, Vocab, gpt_config.Pos
        )
        
        start_time = time.time()
        model, opt_state, loss = train_step(
            model, embeddings, lm_head, opt_state, optimizer,
            input_ids, targets, Vocab, k_step
        )
        step_time = time.time() - start_time
        
        loss_history.append(float(loss))
        step_times.append(step_time)
        
        if (step + 1) % 100 == 0:
            avg_loss = jnp.mean(jnp.array(loss_history[-100:]))
            avg_time = jnp.mean(jnp.array(step_times[-100:]))
            print(f"  Step {step+1:4d} | Loss: {avg_loss:.4f} | Time: {avg_time*1000:.1f}ms")
    
    final_loss = float(jnp.mean(jnp.array(loss_history[-100:])))
    avg_step_time = float(jnp.mean(jnp.array(step_times)))
    
    print(f"\n✓ Training complete")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Avg step time: {avg_step_time*1000:.1f}ms")
    
    return {
        'model_type': 'time_indexed' if use_time_indexed else 'standard',
        'params': total_params,
        'loss_history': loss_history,
        'final_loss': final_loss,
        'avg_step_time': avg_step_time
    }


def plot_comparison(results_list):
    """Plot comparison of different approaches"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss curves
    ax = axes[0, 0]
    for res in results_list:
        label = f"{res['model_type'].title()} ({res['params']:,} params)"
        ax.plot(res['loss_history'], label=label, alpha=0.8, linewidth=2)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter comparison
    ax = axes[0, 1]
    names = [r['model_type'].title() for r in results_list]
    params = [r['params'] / 1e6 for r in results_list]  # In millions
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(names, params, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Parameters (Millions)', fontsize=12)
    ax.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.2f}M\n({param/params[0]*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # 3. Final loss comparison
    ax = axes[1, 0]
    final_losses = [r['final_loss'] for r in results_list]
    bars = ax.bar(names, final_losses, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Speed comparison
    ax = axes[1, 1]
    step_times = [r['avg_step_time'] * 1000 for r in results_list]  # In ms
    bars = ax.bar(names, step_times, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Step Time (ms)', fontsize=12)
    ax.set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, st in zip(bars, step_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{st:.1f}ms',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('time_indexed_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved comparison plot to: time_indexed_comparison.png")
    plt.show()


def main():
    """Main experiment"""
    
    print("\n" + "="*70)
    print("TIME-INDEXED PARAMETER SHARING EXPERIMENT")
    print("="*70)
    print("\nThis experiment compares:")
    print("1. Standard Transformer (different weights per layer)")
    print("2. Time-Indexed Transformer (shared weights with time modulation)")
    print()
    
    config = ExperimentConfig()
    
    # Run both experiments
    results = []
    
    # Standard transformer
    results.append(run_experiment(use_time_indexed=False, config=config, seed=42))
    
    # Time-indexed transformer
    results.append(run_experiment(use_time_indexed=True, config=config, seed=42))
    
    # Plot comparison
    plot_comparison(results)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    standard = results[0]
    time_indexed = results[1]
    
    param_reduction = (1 - time_indexed['params'] / standard['params']) * 100
    loss_diff = time_indexed['final_loss'] - standard['final_loss']
    speed_ratio = time_indexed['avg_step_time'] / standard['avg_step_time']
    
    print(f"\n📊 Parameter Reduction: {param_reduction:.1f}%")
    print(f"   Standard: {standard['params']:,} params")
    print(f"   Time-Indexed: {time_indexed['params']:,} params")
    
    print(f"\n📉 Final Loss:")
    print(f"   Standard: {standard['final_loss']:.4f}")
    print(f"   Time-Indexed: {time_indexed['final_loss']:.4f}")
    print(f"   Difference: {loss_diff:+.4f} ({'better' if loss_diff < 0 else 'worse'})")
    
    print(f"\n⚡ Training Speed:")
    print(f"   Standard: {standard['avg_step_time']*1000:.1f}ms/step")
    print(f"   Time-Indexed: {time_indexed['avg_step_time']*1000:.1f}ms/step")
    print(f"   Ratio: {speed_ratio:.2f}x")
    
    print(f"\n💡 Key Insights:")
    if param_reduction > 30:
        print(f"   ✓ Significant parameter reduction ({param_reduction:.1f}%)")
    if abs(loss_diff) < 0.1:
        print(f"   ✓ Comparable performance to standard transformer")
    if speed_ratio < 1.2:
        print(f"   ✓ Minimal speed overhead")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

