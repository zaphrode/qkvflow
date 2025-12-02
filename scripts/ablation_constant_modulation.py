#!/usr/bin/env python3
"""
Ablation Study: Constant Modulation Baseline

This script tests the critical question:
    "Is the gain from time-indexing t? Or just from adding MLP adapters?"

We compare:
1. Time-Indexed MLP (original) - modulation depends on layer index t
2. Constant Modulation - modulation is fixed (t=0), but MLP structure remains
3. Standard Transformer - baseline

If Constant Modulation performs as well as Time-Indexed, the "Neural ODE" narrative
weakens (but we still have a cool "Weight-Adapter" architecture).
If Time-Indexed wins, the theory is validated.

This is the critical scientific control.
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
import time
import pickle
from typing import Optional
from dataclasses import dataclass

from levanter.models.gpt2 import Gpt2Config
from qkvflow.nn.time_embed import SinusoidalPosEmb
from qkvflow.nn.dynamic import TemporalLayerNorm, Attention, MLP
from haliax.jax_utils import maybe_rng_split, named_call

# Import the original models
from test_time_indexed_weights import (
    TimeIndexedTransformer,
    StandardTransformer,
    ExperimentConfig
)


class ConstantModulationAttention(eqx.Module):
    """Attention with CONSTANT (non-time-varying) modulation.
    
    This tests whether gains come from:
    - Time-indexing (variable modulation) OR
    - Just having MLP adapters (fixed modulation)
    """
    
    # Base weights (shared across all layers)
    base_qkv: hnn.Linear
    base_out: hnn.Linear
    
    # CONSTANT modulation (learned but not time-dependent)
    constant_scale_qkv: jnp.ndarray  # Fixed scaling factors
    constant_scale_out: jnp.ndarray
    
    config: Gpt2Config = eqx.field(static=True)
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, *, key):
        k_qkv, k_out, k_scale1, k_scale2 = jrandom.split(key, 4)
        
        # Base weights (same as time-indexed)
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
        
        # Constant scaling factors (initialized near 1.0 for stability)
        embed_size = config.Embed.size
        constant_scale_qkv = jax.random.normal(k_scale1, (embed_size,)) * 0.02 + 1.0
        constant_scale_out = jax.random.normal(k_scale2, (embed_size,)) * 0.02 + 1.0
        
        dropout = hnn.Dropout(pdrop=config.attn_pdrop)
        
        return ConstantModulationAttention(
            base_qkv, base_out, constant_scale_qkv, constant_scale_out,
            config, dropout
        )
    
    @named_call
    def __call__(self, x, mask, *, key):
        """Forward pass with CONSTANT modulation (no time dependency)."""
        k1, k2 = maybe_rng_split(key, 2)
        
        # Apply constant scaling to base weights
        # Note: No time_embed parameter - modulation is fixed
        scale_qkv = jax.nn.sigmoid(self.constant_scale_qkv)
        scale_out = jax.nn.sigmoid(self.constant_scale_out)
        
        # Modulate base weights (element-wise)
        qkv = self.base_qkv(x)
        qkv = qkv * hax.named(scale_qkv, self.config.Embed)  # Broadcast scaling
        
        Q = qkv["qkv", 0]
        K = qkv["qkv", 1]
        V = qkv["qkv", 2]
        
        # Standard attention computation
        attn_scores = hax.dot(Q, K, axis=self.config.HeadSize) / jnp.sqrt(self.config.HeadSize.size)
        
        if mask is not None:
            attn_scores = hax.where(mask, attn_scores, -1e10)
        
        attn_weights = hax.nn.softmax(attn_scores, axis=self.config.Pos)
        attn_weights = self.dropout(attn_weights, key=k1)
        
        output = hax.dot(attn_weights, V, axis=self.config.Pos)
        output = self.base_out(output)
        output = output * hax.named(scale_out, self.config.Embed)
        output = self.dropout(output, key=k2)
        
        return output


class ConstantModulationMLP(eqx.Module):
    """MLP with CONSTANT (non-time-varying) modulation."""
    
    base_up: hnn.Linear
    base_down: hnn.Linear
    
    constant_scale_up: jnp.ndarray
    constant_scale_down: jnp.ndarray
    
    config: Gpt2Config = eqx.field(static=True)
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, *, key):
        k_up, k_down, k_scale1, k_scale2 = jrandom.split(key, 4)
        
        base_up = hnn.Linear.init(config.Embed, config.Mlp, key=k_up, use_bias=True)
        base_down = hnn.Linear.init(config.Mlp, config.Embed, key=k_down, use_bias=True)
        
        embed_size = config.Embed.size
        mlp_size = config.Mlp.size
        constant_scale_up = jax.random.normal(k_scale1, (mlp_size,)) * 0.02 + 1.0
        constant_scale_down = jax.random.normal(k_scale2, (embed_size,)) * 0.02 + 1.0
        
        dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        
        return ConstantModulationMLP(
            base_up, base_down, constant_scale_up, constant_scale_down,
            config, dropout
        )
    
    @named_call
    def __call__(self, x, *, key):
        """Forward pass with CONSTANT modulation."""
        k1, k2 = maybe_rng_split(key, 2)
        
        scale_up = jax.nn.sigmoid(self.constant_scale_up)
        scale_down = jax.nn.sigmoid(self.constant_scale_down)
        
        h = self.base_up(x)
        h = h * hax.named(scale_up, self.config.Mlp)
        h = hax.nn.gelu(h)
        h = self.dropout(h, key=k1)
        
        output = self.base_down(h)
        output = output * hax.named(scale_down, self.config.Embed)
        output = self.dropout(output, key=k2)
        
        return output


class ConstantModulationBlock(eqx.Module):
    """Transformer block with constant modulation."""
    
    ln1: TemporalLayerNorm
    attn: ConstantModulationAttention
    ln2: TemporalLayerNorm
    mlp: ConstantModulationMLP
    
    @staticmethod
    def init(config, *, key):
        k_attn, k_mlp = jrandom.split(key, 2)
        
        ln1 = TemporalLayerNorm.init(config.Embed)
        attn = ConstantModulationAttention.init(config, key=k_attn)
        ln2 = TemporalLayerNorm.init(config.Embed)
        mlp = ConstantModulationMLP.init(config, key=k_mlp)
        
        return ConstantModulationBlock(ln1, attn, ln2, mlp)
    
    @named_call
    def __call__(self, x, mask, *, key):
        k1, k2 = maybe_rng_split(key, 2)
        
        # Attention with residual
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, mask, key=k1)
        x = x + attn_out
        
        # MLP with residual
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm, key=k2)
        x = x + mlp_out
        
        return x


class ConstantModulationTransformer(eqx.Module):
    """Full transformer with constant modulation (ablation baseline)."""
    
    embed: hnn.Embedding
    blocks: list
    ln_f: TemporalLayerNorm
    lm_head: hnn.Linear
    config: Gpt2Config = eqx.field(static=True)
    
    @staticmethod
    def init(Vocab, Embed, Heads, HeadSize, Layers, *, key):
        k_embed, k_blocks, k_head = jrandom.split(key, 3)
        
        config = Gpt2Config(
            num_layers=Layers,
            hidden_dim=Embed.size,
            num_heads=Heads.size,
            seq_len=128,
            gradient_checkpointing=False,
            use_bias=False,
        )
        config.Heads = Heads
        config.HeadSize = HeadSize
        config.Embed = Embed
        config.Vocab = Vocab
        config.Pos = hax.Axis("position", 128)
        config.Mlp = hax.Axis("mlp", 4 * Embed.size)
        
        embed = hnn.Embedding.init(Vocab, Embed, key=k_embed)
        
        # Create blocks with constant modulation
        block_keys = jrandom.split(k_blocks, Layers)
        blocks = [ConstantModulationBlock.init(config, key=k) for k in block_keys]
        
        ln_f = TemporalLayerNorm.init(Embed)
        lm_head = hnn.Linear.init(Embed, Vocab, key=k_head, use_bias=False)
        
        return ConstantModulationTransformer(embed, blocks, ln_f, lm_head, config)
    
    @named_call
    def __call__(self, input_ids, Batch, Pos, *, key):
        keys = maybe_rng_split(key, len(self.blocks))
        
        # Embed tokens
        x = self.embed(input_ids)
        
        # Create causal mask
        mask_val = hax.arange(Pos) <= hax.arange(Pos).broadcast_axis(Pos.alias("key_position"))
        
        # Apply blocks (NO time-indexing - same modulation for all layers)
        for block, k in zip(self.blocks, keys):
            x = block(x, mask_val, key=k)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits.array


def run_ablation_experiment(config: ExperimentConfig, seed: int = 42):
    """Run ablation study comparing time-indexed vs constant modulation."""
    
    print("\n" + "="*70)
    print("ABLATION STUDY: Time-Indexed vs Constant Modulation")
    print("="*70)
    print("\nScientific Question:")
    print("  Is the gain from time-indexing t? Or just from MLP adapters?")
    print("\nExperiment:")
    print("  1. Time-Indexed MLP (original) - modulation = f(t)")
    print("  2. Constant Modulation - modulation = constant")
    print("  3. Standard Transformer - baseline")
    print("="*70)
    
    key = jrandom.PRNGKey(seed)
    
    # Setup axes
    Batch = hax.Axis("batch", config.batch_size)
    Pos = hax.Axis("position", config.seq_len)
    Embed = hax.Axis("embed", config.hidden_dim)
    Vocab = hax.Axis("vocab", config.vocab_size)
    Heads = hax.Axis("heads", config.num_heads)
    HeadSize = hax.Axis("head_size", config.hidden_dim // config.num_heads)
    SinusodialDim = hax.Axis("sinusoidal_dim", 64)
    TembedDim = hax.Axis("tembed_dim", 256)
    
    # Initialize models
    print("\n📦 Initializing models...")
    k1, k2, k3, key = jrandom.split(key, 4)
    
    # Create GPT2 config for models
    gpt2_config = Gpt2Config(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        gradient_checkpointing=False,
        use_bias=False,
    )
    gpt2_config.Heads = Heads
    gpt2_config.HeadSize = HeadSize
    gpt2_config.Embed = Embed
    gpt2_config.Vocab = Vocab
    gpt2_config.Pos = Pos
    gpt2_config.Mlp = hax.Axis("mlp", 4 * config.hidden_dim)
    
    time_indexed_model = TimeIndexedTransformer.init(
        gpt2_config, SinusodialDim, TembedDim, key=k1
    )
    
    constant_mod_model = ConstantModulationTransformer.init(
        Vocab, Embed, Heads, HeadSize, config.num_layers, key=k2
    )
    
    standard_model = StandardTransformer(
        Vocab, Embed, Heads, HeadSize, config.num_layers, key=k3
    )
    
    # Count parameters
    def count_params(model):
        params, _ = eqx.partition(model, eqx.is_array)
        leaves = jax.tree_util.tree_leaves(params)
        return sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))
    
    ti_params = count_params(time_indexed_model)
    const_params = count_params(constant_mod_model)
    std_params = count_params(standard_model)
    
    print(f"\n📊 Parameter Counts:")
    print(f"  Time-Indexed MLP:     {ti_params:>12,} params")
    print(f"  Constant Modulation:  {const_params:>12,} params")
    print(f"  Standard:             {std_params:>12,} params")
    
    # Generate synthetic training data
    print(f"\n🎲 Generating synthetic data ({config.num_steps} steps)...")
    data_key, key = jrandom.split(key)
    train_data = jrandom.randint(
        data_key, 
        (config.num_steps, config.batch_size, config.seq_len),
        0, config.vocab_size
    )
    
    # Training loop for each model
    results = {}
    
    for model_name, model in [
        ("Time-Indexed MLP", time_indexed_model),
        ("Constant Modulation", constant_mod_model),
        ("Standard", standard_model)
    ]:
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        
        # Training setup
        import optax
        optimizer = optax.adam(config.learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        @eqx.filter_jit
        def loss_fn(model, x, key):
            inputs = x[:, :-1]
            targets = x[:, 1:]
            logits = model(inputs, Batch, Pos, key=key)
            logits_flat = logits.reshape(-1, config.vocab_size)
            targets_flat = targets.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
            return jnp.mean(loss)
        
        @eqx.filter_jit
        def train_step(model, opt_state, x, key):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, key)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss
        
        # Train
        losses = []
        start_time = time.time()
        
        for step in range(config.num_steps):
            step_key, key = jrandom.split(key)
            model, opt_state, loss = train_step(model, opt_state, train_data[step], step_key)
            losses.append(float(loss))
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{config.num_steps} - Loss: {loss:.4f}")
        
        elapsed = time.time() - start_time
        final_loss = losses[-1]
        
        print(f"\n✅ Completed in {elapsed:.1f}s")
        print(f"   Final Loss: {final_loss:.4f}")
        
        results[model_name] = {
            'losses': losses,
            'final_loss': final_loss,
            'params': count_params(model),
            'time': elapsed
        }
    
    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nFinal Losses:")
    for name in ["Time-Indexed MLP", "Constant Modulation", "Standard"]:
        print(f"  {name:25s}: {results[name]['final_loss']:.4f}")
    
    # Critical comparison
    ti_loss = results["Time-Indexed MLP"]['final_loss']
    const_loss = results["Constant Modulation"]['final_loss']
    std_loss = results["Standard"]['final_loss']
    
    print(f"\n🔬 Critical Comparison:")
    print(f"  Time-Indexed vs Constant: {((const_loss - ti_loss) / const_loss * 100):+.1f}%")
    print(f"  Time-Indexed vs Standard: {((std_loss - ti_loss) / std_loss * 100):+.1f}%")
    print(f"  Constant vs Standard:     {((std_loss - const_loss) / std_loss * 100):+.1f}%")
    
    print(f"\n📝 Interpretation:")
    if ti_loss < const_loss * 0.95:
        print("  ✅ Time-Indexed significantly better than Constant Modulation")
        print("  → Time-dependency (Neural ODE) provides real benefit")
        print("  → The 'trajectory' of weights over depth matters")
    elif ti_loss < const_loss * 1.05:
        print("  ⚠️  Time-Indexed and Constant Modulation similar")
        print("  → Gains mostly from MLP adapters, not time-dependency")
        print("  → 'Neural ODE' narrative weakens")
        print("  → Still valuable as 'Weight Adapter' architecture")
    else:
        print("  ❌ Constant Modulation better than Time-Indexed")
        print("  → Time-dependency may be adding noise")
        print("  → Simpler fixed modulation is sufficient")
    
    # Save results
    output_path = "ablation_constant_modulation_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    config = ExperimentConfig(
        hidden_dim=256,
        num_heads=4,
        num_layers=6,
        seq_len=128,
        vocab_size=256,  # Character-level
        batch_size=4,
        num_steps=500,  # Shorter for quick ablation
        learning_rate=3e-4
    )
    
    results = run_ablation_experiment(config, seed=42)

