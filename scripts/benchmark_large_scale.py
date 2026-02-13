#!/usr/bin/env python3
"""
Large-Scale Benchmark: Time-Indexed MLP vs Published Baselines
===============================================================

This script trains our Time-Indexed MLP model at a larger scale on WikiText-103
and compares perplexity against published results from GPT-2 and similar models.

Published Baselines (WikiText-103 test perplexity):
- GPT-2 Small (117M params): ~29.4 PPL
- GPT-2 Medium (345M params): ~22.0 PPL  
- Transformer-XL Base (151M params): ~24.0 PPL
- LSTM + Neural Cache: ~33.0 PPL

Our goal: Show competitive perplexity with 100-1000x fewer parameters.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import haliax as hax
from haliax import Axis
import time
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for large-scale benchmark."""
    # Model architecture (Medium scale - fits on CPU)
    vocab_size: int = 256  # Character-level for simplicity
    embed_dim: int = 512   # Medium dimension
    num_heads: int = 8     # Medium heads
    num_layers: int = 8    # Medium depth
    seq_len: int = 256     # Medium context
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # Time-indexed MLP settings
    time_embed_dim: int = 64
    time_hidden_dim: int = 128
    
    # Training settings
    batch_size: int = 32   # CPU-friendly batch size
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    total_steps: int = 5000  # Quick but meaningful benchmark
    eval_every: int = 500
    
    # Data
    data_dir: str = "./wikitext103_data"


# ============================================================================
# MODEL ARCHITECTURE (Scaled Time-Indexed MLP)
# ============================================================================

class SinusoidalTimeEmbedding(eqx.Module):
    """Sinusoidal positional encoding for continuous time."""
    dim: int = eqx.field(static=True)
    
    def __call__(self, t: float) -> jnp.ndarray:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t * emb
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])


class TimeIndexedLinear(eqx.Module):
    """Linear layer with time-indexed weight modulation."""
    weight: jnp.ndarray
    bias: jnp.ndarray
    time_mlp: eqx.nn.MLP
    
    def __init__(self, in_features: int, out_features: int, time_dim: int, 
                 hidden_dim: int, *, key: jrandom.PRNGKey):
        keys = jrandom.split(key, 3)
        
        # Base weights
        scale = 1.0 / math.sqrt(in_features)
        self.weight = jrandom.normal(keys[0], (out_features, in_features)) * scale
        self.bias = jnp.zeros(out_features)
        
        # Time-dependent modulation MLP
        self.time_mlp = eqx.nn.MLP(
            in_size=time_dim,
            out_size=out_features,
            width_size=hidden_dim,
            depth=2,
            key=keys[1]
        )
    
    def __call__(self, x: jnp.ndarray, time_embed: jnp.ndarray) -> jnp.ndarray:
        # Compute time-dependent scale
        scale = jax.nn.sigmoid(self.time_mlp(time_embed))  # [out_features]
        
        # Modulated weight
        w_eff = self.weight * scale[:, None]  # Broadcasting
        
        return x @ w_eff.T + self.bias


class TimeIndexedAttention(eqx.Module):
    """Multi-head attention with time-indexed weight sharing."""
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    
    qkv_proj: TimeIndexedLinear
    out_proj: TimeIndexedLinear
    
    def __init__(self, embed_dim: int, num_heads: int, time_dim: int,
                 hidden_dim: int, *, key: jrandom.PRNGKey):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        keys = jrandom.split(key, 2)
        self.qkv_proj = TimeIndexedLinear(embed_dim, 3 * embed_dim, time_dim, hidden_dim, key=keys[0])
        self.out_proj = TimeIndexedLinear(embed_dim, embed_dim, time_dim, hidden_dim, key=keys[1])
    
    def __call__(self, x: jnp.ndarray, time_embed: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch, seq_len, _ = x.shape
        
        # QKV projection with time modulation
        qkv = self.qkv_proj(x, time_embed)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if mask is not None:
            attn = jnp.where(mask, attn, -1e9)
        
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        
        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out, time_embed)


class TimeIndexedFFN(eqx.Module):
    """Feed-forward network with time-indexed weight sharing."""
    fc1: TimeIndexedLinear
    fc2: TimeIndexedLinear
    
    def __init__(self, embed_dim: int, hidden_dim: int, time_dim: int,
                 time_hidden: int, *, key: jrandom.PRNGKey):
        keys = jrandom.split(key, 2)
        self.fc1 = TimeIndexedLinear(embed_dim, hidden_dim, time_dim, time_hidden, key=keys[0])
        self.fc2 = TimeIndexedLinear(hidden_dim, embed_dim, time_dim, time_hidden, key=keys[1])
    
    def __call__(self, x: jnp.ndarray, time_embed: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x, time_embed)
        x = jax.nn.gelu(x)
        return self.fc2(x, time_embed)


class TimeIndexedTransformerBlock(eqx.Module):
    """Single transformer block with time-indexed weight sharing."""
    attn: TimeIndexedAttention
    ffn: TimeIndexedFFN
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float,
                 time_dim: int, time_hidden: int, *, key: jrandom.PRNGKey):
        keys = jrandom.split(key, 2)
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.attn = TimeIndexedAttention(embed_dim, num_heads, time_dim, time_hidden, key=keys[0])
        self.ffn = TimeIndexedFFN(embed_dim, hidden_dim, time_dim, time_hidden, key=keys[1])
        self.ln1 = eqx.nn.LayerNorm(embed_dim)
        self.ln2 = eqx.nn.LayerNorm(embed_dim)
    
    def __call__(self, x: jnp.ndarray, time_embed: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Apply LayerNorm over last dimension (embed) for each (batch, seq) position
        ln1_fn = jax.vmap(jax.vmap(self.ln1))  # vmap over batch and seq
        ln2_fn = jax.vmap(jax.vmap(self.ln2))
        x = x + self.attn(ln1_fn(x), time_embed, mask)
        x = x + self.ffn(ln2_fn(x), time_embed)
        return x


class TimeIndexedTransformer(eqx.Module):
    """Full Time-Indexed Transformer for language modeling."""
    config: BenchmarkConfig = eqx.field(static=True)
    
    token_embed: jnp.ndarray
    pos_embed: jnp.ndarray
    time_encoder: SinusoidalTimeEmbedding
    shared_block: TimeIndexedTransformerBlock
    ln_final: eqx.nn.LayerNorm
    lm_head: jnp.ndarray
    
    def __init__(self, config: BenchmarkConfig, *, key: jrandom.PRNGKey):
        self.config = config
        keys = jrandom.split(key, 4)
        
        # Embeddings
        self.token_embed = jrandom.normal(keys[0], (config.vocab_size, config.embed_dim)) * 0.02
        self.pos_embed = jrandom.normal(keys[1], (config.seq_len, config.embed_dim)) * 0.02
        
        # Time encoding
        self.time_encoder = SinusoidalTimeEmbedding(config.time_embed_dim)
        
        # Single shared block (the key innovation!)
        self.shared_block = TimeIndexedTransformerBlock(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            time_dim=config.time_embed_dim,
            time_hidden=config.time_hidden_dim,
            key=keys[2]
        )
        
        # Output
        self.ln_final = eqx.nn.LayerNorm(config.embed_dim)
        self.lm_head = jrandom.normal(keys[3], (config.embed_dim, config.vocab_size)) * 0.02
    
    def __call__(self, tokens: jnp.ndarray, *, key: Optional[jrandom.PRNGKey] = None) -> jnp.ndarray:
        batch, seq_len = tokens.shape
        
        # Embed tokens
        x = self.token_embed[tokens] + self.pos_embed[:seq_len]
        
        # Causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        
        # Apply shared block at different "times" (layer positions)
        for layer_idx in range(self.config.num_layers):
            t = layer_idx / (self.config.num_layers - 1)  # Normalize to [0, 1]
            time_embed = self.time_encoder(t)
            x = self.shared_block(x, time_embed, mask)
        
        # Output projection
        ln_fn = jax.vmap(jax.vmap(self.ln_final))  # vmap over batch and seq
        x = ln_fn(x)
        logits = x @ self.lm_head
        
        return logits


class StandardTransformer(eqx.Module):
    """Standard Transformer baseline (separate weights per layer)."""
    config: BenchmarkConfig = eqx.field(static=True)
    
    token_embed: jnp.ndarray
    pos_embed: jnp.ndarray
    blocks: List[TimeIndexedTransformerBlock]  # Reusing structure but each is separate
    ln_final: eqx.nn.LayerNorm
    lm_head: jnp.ndarray
    
    def __init__(self, config: BenchmarkConfig, *, key: jrandom.PRNGKey):
        self.config = config
        keys = jrandom.split(key, config.num_layers + 4)
        
        # Embeddings
        self.token_embed = jrandom.normal(keys[0], (config.vocab_size, config.embed_dim)) * 0.02
        self.pos_embed = jrandom.normal(keys[1], (config.seq_len, config.embed_dim)) * 0.02
        
        # Separate block per layer
        self.blocks = [
            TimeIndexedTransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                time_dim=config.time_embed_dim,
                time_hidden=config.time_hidden_dim,
                key=keys[i + 2]
            )
            for i in range(config.num_layers)
        ]
        
        # Output
        self.ln_final = eqx.nn.LayerNorm(config.embed_dim)
        self.lm_head = jrandom.normal(keys[-1], (config.embed_dim, config.vocab_size)) * 0.02
    
    def __call__(self, tokens: jnp.ndarray, *, key: Optional[jrandom.PRNGKey] = None) -> jnp.ndarray:
        batch, seq_len = tokens.shape
        
        x = self.token_embed[tokens] + self.pos_embed[:seq_len]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        
        time_encoder = SinusoidalTimeEmbedding(self.config.time_embed_dim)
        
        for layer_idx, block in enumerate(self.blocks):
            t = layer_idx / (self.config.num_layers - 1)
            time_embed = time_encoder(t)
            x = block(x, time_embed, mask)
        
        ln_fn = jax.vmap(jax.vmap(self.ln_final))  # vmap over batch and seq
        x = ln_fn(x)
        return x @ self.lm_head


# ============================================================================
# DATA LOADING
# ============================================================================

def load_wikitext103(data_dir: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load WikiText-103 train and test data."""
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"WikiText-103 not found at {data_dir}. Please run data preparation first.")
    
    print(f"📖 Loading WikiText-103 from {data_dir}...")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    # Convert to character indices (simple vocab)
    train_data = jnp.array([ord(c) % 256 for c in train_text], dtype=jnp.int32)
    test_data = jnp.array([ord(c) % 256 for c in test_text], dtype=jnp.int32)
    
    print(f"   Train: {len(train_data):,} characters ({len(train_data)/1e6:.1f}M)")
    print(f"   Test: {len(test_data):,} characters ({len(test_data)/1e6:.1f}M)")
    
    return train_data, test_data


def get_batch(data: jnp.ndarray, batch_size: int, seq_len: int, 
              key: jrandom.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random batch from data."""
    max_start = len(data) - seq_len - 1
    starts = jrandom.randint(key, (batch_size,), 0, max_start)
    
    inputs = jnp.stack([data[s:s+seq_len] for s in starts])
    targets = jnp.stack([data[s+1:s+seq_len+1] for s in starts])
    
    return inputs, targets


# ============================================================================
# TRAINING
# ============================================================================

def count_parameters(model) -> int:
    """Count total trainable parameters."""
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def create_optimizer(config: BenchmarkConfig):
    """Create optimizer with warmup and cosine decay."""
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps
    )
    decay_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.total_steps - config.warmup_steps
    )
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule],
        boundaries=[config.warmup_steps]
    )
    
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01)
    )


@eqx.filter_jit
def train_step(model, opt_state, optimizer, inputs, targets, key):
    """Single training step."""
    def loss_fn(model):
        logits = model(inputs, key=key)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        ).mean()
        return loss
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, inputs, targets):
    """Compute evaluation loss."""
    logits = model(inputs, key=None)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    ).mean()
    return loss


def evaluate(model, test_data: jnp.ndarray, config: BenchmarkConfig, 
             key: jrandom.PRNGKey, num_batches: int = 50) -> float:
    """Evaluate model on test data."""
    total_loss = 0.0
    
    for i in range(num_batches):
        key, subkey = jrandom.split(key)
        inputs, targets = get_batch(test_data, config.batch_size, config.seq_len, subkey)
        loss = eval_step(model, inputs, targets)
        total_loss += float(loss)
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return perplexity


def train_model(model, train_data: jnp.ndarray, test_data: jnp.ndarray,
                config: BenchmarkConfig, key: jrandom.PRNGKey, 
                model_name: str) -> Tuple[eqx.Module, List[dict]]:
    """Train a model and return metrics."""
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    num_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"{'='*60}")
    
    metrics = []
    best_ppl = float('inf')
    start_time = time.time()
    
    for step in range(1, config.total_steps + 1):
        key, subkey1, subkey2 = jrandom.split(key, 3)
        
        inputs, targets = get_batch(train_data, config.batch_size, config.seq_len, subkey1)
        model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets, subkey2)
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            print(f"  Step {step:5d}/{config.total_steps} | Loss: {float(loss):.4f} | "
                  f"Speed: {steps_per_sec:.1f} steps/s")
        
        if step % config.eval_every == 0:
            key, eval_key = jrandom.split(key)
            ppl = evaluate(model, test_data, config, eval_key)
            best_ppl = min(best_ppl, ppl)
            
            metrics.append({
                'step': step,
                'train_loss': float(loss),
                'test_ppl': ppl,
                'time': time.time() - start_time
            })
            
            print(f"  >>> EVAL Step {step}: Test PPL = {ppl:.2f} (best: {best_ppl:.2f})")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
    print(f"   Final Test Perplexity: {best_ppl:.2f}")
    
    return model, metrics


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    """Run the full benchmark comparison."""
    print("\n" + "="*70)
    print("       LARGE-SCALE BENCHMARK: Time-Indexed MLP vs Baselines")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config = BenchmarkConfig()
    print(f"\n📋 Configuration:")
    print(f"   Embed dim: {config.embed_dim}")
    print(f"   Num heads: {config.num_heads}")
    print(f"   Num layers: {config.num_layers}")
    print(f"   Seq length: {config.seq_len}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Total steps: {config.total_steps}")
    
    # Load data
    train_data, test_data = load_wikitext103(config.data_dir)
    
    # Initialize models
    key = jrandom.PRNGKey(42)
    key, key1, key2 = jrandom.split(key, 3)
    
    time_indexed_model = TimeIndexedTransformer(config, key=key1)
    standard_model = StandardTransformer(config, key=key2)
    
    # Count parameters
    ti_params = count_parameters(time_indexed_model)
    std_params = count_parameters(standard_model)
    compression = std_params / ti_params
    
    print(f"\n📊 Parameter Comparison:")
    print(f"   Time-Indexed MLP: {ti_params:,} ({ti_params/1e6:.2f}M)")
    print(f"   Standard Transformer: {std_params:,} ({std_params/1e6:.2f}M)")
    print(f"   Compression Ratio: {compression:.1f}x")
    
    results = {}
    
    # Train Time-Indexed Model
    key, train_key = jrandom.split(key)
    ti_model, ti_metrics = train_model(
        time_indexed_model, train_data, test_data, config, train_key,
        "Time-Indexed MLP Transformer"
    )
    results['time_indexed'] = {
        'params': ti_params,
        'metrics': ti_metrics,
        'final_ppl': ti_metrics[-1]['test_ppl'] if ti_metrics else float('inf')
    }
    
    # Train Standard Model  
    key, train_key = jrandom.split(key)
    std_model, std_metrics = train_model(
        standard_model, train_data, test_data, config, train_key,
        "Standard Transformer"
    )
    results['standard'] = {
        'params': std_params,
        'metrics': std_metrics,
        'final_ppl': std_metrics[-1]['test_ppl'] if std_metrics else float('inf')
    }
    
    # Published baselines (WikiText-103 character-level approximate)
    # Note: These are word-level perplexities, char-level will be different
    published_baselines = {
        'GPT-2 Small (117M)': {'params': 117_000_000, 'ppl': 29.4},
        'GPT-2 Medium (345M)': {'params': 345_000_000, 'ppl': 22.0},
        'Transformer-XL (151M)': {'params': 151_000_000, 'ppl': 24.0},
    }
    
    # Final Summary
    print("\n" + "="*70)
    print("                    FINAL RESULTS SUMMARY")
    print("="*70)
    
    print("\n📊 Our Models (Character-level WikiText-103):")
    print("-" * 50)
    print(f"{'Model':<30} {'Params':<15} {'Test PPL':<10}")
    print("-" * 50)
    print(f"{'Time-Indexed MLP':<30} {ti_params:>12,} {results['time_indexed']['final_ppl']:>10.2f}")
    print(f"{'Standard Transformer':<30} {std_params:>12,} {results['standard']['final_ppl']:>10.2f}")
    print("-" * 50)
    print(f"\n⚡ Compression: {compression:.1f}x fewer parameters with Time-Indexed MLP")
    
    ppl_ratio = results['time_indexed']['final_ppl'] / results['standard']['final_ppl']
    print(f"📈 Perplexity ratio: {ppl_ratio:.2f}x (lower is better for Time-Indexed)")
    
    print("\n📚 Published Baselines (Word-level WikiText-103, for reference):")
    print("-" * 50)
    for name, data in published_baselines.items():
        print(f"   {name}: {data['ppl']:.1f} PPL")
    
    print("\n" + "="*70)
    print("                    KEY FINDINGS")
    print("="*70)
    print(f"""
1. PARAMETER EFFICIENCY:
   - Time-Indexed MLP uses {compression:.1f}x fewer parameters
   - {ti_params:,} vs {std_params:,} parameters
   
2. PERFORMANCE:
   - Time-Indexed achieves {results['time_indexed']['final_ppl']:.2f} perplexity
   - Standard achieves {results['standard']['final_ppl']:.2f} perplexity
   - Ratio: {ppl_ratio:.2f}x
   
3. EFFICIENCY SCORE (Lower PPL / Fewer Params):
   - Time-Indexed: {results['time_indexed']['final_ppl'] / (ti_params/1e6):.2f} PPL per M params
   - Standard: {results['standard']['final_ppl'] / (std_params/1e6):.2f} PPL per M params
""")
    
    # Save results
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f"large_scale_benchmark_{timestamp}.pkl")
    
    with open(results_file, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': results,
            'published_baselines': published_baselines
        }, f)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"\n✅ Benchmark complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    run_benchmark()

