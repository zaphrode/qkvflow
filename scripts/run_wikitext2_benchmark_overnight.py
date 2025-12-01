#!/usr/bin/env python3
"""
WikiText-2 Benchmark: Overnight Run
Compares Standard, Time-Indexed MLP, and Time-Indexed SSM transformers.

Features:
- Progress logging with timestamps
- Intermediate result checkpointing
- Robust error handling
- Publication-quality plots
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
import pickle
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

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
TimeIndexedTransformer = time_indexed.TimeIndexedTransformer

# Import SSM models
spec_ssm = importlib.util.spec_from_file_location(
    "time_indexed_ssm", 
    "/home/nahid/Documents/qkvflow/scripts/test_time_indexed_ssm.py"
)
time_indexed_ssm = importlib.util.module_from_spec(spec_ssm)
spec_ssm.loader.exec_module(time_indexed_ssm)

TimeIndexedSSMTransformer = time_indexed_ssm.TimeIndexedSSMTransformer


def log_message(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


@dataclass
class BenchmarkConfig:
    """Configuration for overnight benchmark"""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    seq_len: int = 128
    batch_size: int = 8
    num_steps: int = 3000
    eval_every: int = 100
    learning_rate: float = 3e-4
    ssm_state_size: int = 64
    checkpoint_every: int = 500


class SimpleTokenizer:
    """Character-level tokenizer"""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
    
    def encode(self, text: str):
        return jnp.array([min(ord(c), self.vocab_size - 1) for c in text], dtype=jnp.int32)


def load_wikitext2():
    """Load WikiText-2 from extracted files"""
    log_message("Loading WikiText-2 dataset...")
    
    train_file = Path("train.txt")
    test_file = Path("test.txt")
    
    if not (train_file.exists() and test_file.exists()):
        raise FileNotFoundError("WikiText-2 files not found. Please extract archive.zip")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    log_message(f"  Train: {len(train_text):,} chars")
    log_message(f"  Test: {len(test_text):,} chars")
    
    return train_text, test_text


def create_batches(text: str, tokenizer: SimpleTokenizer, 
                   batch_size: int, seq_len: int, num_batches: int, key):
    """Create training batches"""
    tokens = tokenizer.encode(text)
    batches = []
    
    for i in range(num_batches):
        key, subkey = jrandom.split(key)
        max_start = max(0, len(tokens) - seq_len - 1)
        start_idx = jrandom.randint(subkey, (), 0, max_start)
        
        batch_tokens = []
        for _ in range(batch_size):
            seq = tokens[start_idx:start_idx + seq_len + 1]
            if len(seq) < seq_len + 1:
                seq = jnp.pad(seq, (0, seq_len + 1 - len(seq)), constant_values=0)
            batch_tokens.append(seq)
            start_idx = (start_idx + seq_len + 1) % max_start
        
        batch_tokens = jnp.stack(batch_tokens)
        batches.append((batch_tokens[:, :-1], batch_tokens[:, 1:]))
    
    return batches


def count_parameters(model):
    """Count parameters"""
    leaves = jax.tree_util.tree_leaves(model)
    return sum(x.size for x in leaves if eqx.is_array(x))


def create_model(model_type: str, config: BenchmarkConfig, key):
    """Create language model"""
    
    Batch = hax.Axis("batch", config.batch_size)
    Pos = hax.Axis("position", config.seq_len)
    Vocab = hax.Axis("vocab", 256)
    
    gpt2_config = Gpt2Config(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        seq_len=config.seq_len,
        use_bias=True,
    )
    
    sinusoidal_dim = 64
    SinusodialDim = hax.Axis("SinusodialDim", sinusoidal_dim)
    TembedDim = hax.Axis("TembedDim", sinusoidal_dim * 2 + 1)
    
    k1, k2, k3, k_emb, k_head = jrandom.split(key, 5)
    
    if model_type == "standard":
        transformer = StandardTransformer.init(gpt2_config, SinusodialDim, TembedDim, key=k1)
    elif model_type == "time_indexed_mlp":
        transformer = TimeIndexedTransformer.init(gpt2_config, SinusodialDim, TembedDim, key=k2)
    elif model_type == "time_indexed_ssm":
        transformer = TimeIndexedSSMTransformer.init(
            gpt2_config, SinusodialDim, TembedDim, 
            ssm_state_size=config.ssm_state_size, key=k3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    embedder = hnn.Embedding.init(Vocab, gpt2_config.Embed, key=k_emb)
    lm_head = hnn.Linear.init(gpt2_config.Embed, Vocab, key=k_head, use_bias=False)
    
    return transformer, embedder, lm_head, Batch, Pos, Vocab


def compute_loss(transformer, embedder, lm_head, input_ids, targets, Vocab, key):
    """Compute loss"""
    x = embedder(input_ids)
    x = transformer(x, key=key)
    logits = lm_head(x)
    
    targets_onehot = jax.nn.one_hot(targets.array, Vocab.size)
    targets_onehot = hax.named(targets_onehot, tuple(targets.axes) + (Vocab,))
    loss = hax.nn.cross_entropy_loss(logits, Vocab, targets_onehot, reduction=hax.mean)
    
    return loss.scalar()


def train_model(model_type: str, config: BenchmarkConfig, 
                train_batches, valid_batches, key):
    """Train a model"""
    
    log_message("="*70)
    log_message(f"Training: {model_type.replace('_', ' ').title()}")
    log_message("="*70)
    
    # Create model
    k_model, k_train = jrandom.split(key)
    transformer, embedder, lm_head, Batch, Pos, Vocab = create_model(
        model_type, config, k_model
    )
    
    total_params = (
        count_parameters(transformer) + 
        count_parameters(embedder) + 
        count_parameters(lm_head)
    )
    log_message(f"Parameters: {total_params:,}")
    
    # Optimizer
    import optax
    opt = optax.adam(learning_rate=config.learning_rate)
    models = (transformer, embedder, lm_head)
    opt_state = opt.init(eqx.filter(models, eqx.is_array))
    
    # Training history
    train_losses = []
    valid_losses = []
    step_times = []
    checkpoints = []
    
    # JIT-compiled functions
    @eqx.filter_jit
    def train_step(models, opt_state, batch, key):
        transformer, embedder, lm_head = models
        input_ids, targets = batch
        input_ids = hax.named(input_ids, (Batch, Pos))
        targets = hax.named(targets, (Batch, Pos))
        
        def loss_fn(models):
            t, e, l = models
            return compute_loss(t, e, l, input_ids, targets, Vocab, key)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(models)
        updates, opt_state = opt.update(grads, opt_state, models)
        models = eqx.apply_updates(models, updates)
        
        return models, opt_state, loss
    
    @eqx.filter_jit
    def valid_step(models, batch, key):
        transformer, embedder, lm_head = models
        input_ids, targets = batch
        input_ids = hax.named(input_ids, (Batch, Pos))
        targets = hax.named(targets, (Batch, Pos))
        return compute_loss(transformer, embedder, lm_head, input_ids, targets, Vocab, key)
    
    log_message(f"Starting training for {config.num_steps} steps...")
    log_message("(First step will be slow due to JIT compilation)")
    
    start_time_total = time.time()
    
    for step in range(1, config.num_steps + 1):
        batch_idx = (step - 1) % len(train_batches)
        batch = train_batches[batch_idx]
        
        k_train, k_step = jrandom.split(k_train)
        step_start = time.time()
        
        models, opt_state, loss = train_step(models, opt_state, batch, k_step)
        
        step_time = (time.time() - step_start) * 1000
        train_losses.append(float(loss))
        step_times.append(step_time)
        
        # Evaluation
        if step % config.eval_every == 0:
            valid_loss_sum = 0.0
            for vb in valid_batches[:10]:
                k_train, k_eval = jrandom.split(k_train)
                valid_loss_sum += float(valid_step(models, vb, k_eval))
            valid_loss = valid_loss_sum / min(10, len(valid_batches))
            valid_losses.append(valid_loss)
            
            elapsed = time.time() - start_time_total
            eta = (elapsed / step) * (config.num_steps - step)
            
            log_message(f"Step {step:4d}/{config.num_steps} | "
                       f"Train: {loss:.4f} | Valid: {valid_loss:.4f} | "
                       f"Time: {step_time:.1f}ms | "
                       f"ETA: {eta/60:.1f}min")
        
        # Checkpointing
        if step % config.checkpoint_every == 0:
            checkpoint = {
                'step': step,
                'train_losses': train_losses.copy(),
                'valid_losses': valid_losses.copy(),
                'step_times': step_times.copy(),
            }
            checkpoints.append(checkpoint)
            
            # Save checkpoint to disk
            checkpoint_file = f"checkpoint_{model_type}_step{step}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            log_message(f"  → Saved checkpoint: {checkpoint_file}")
    
    total_time = time.time() - start_time_total
    log_message(f"✓ Training complete in {total_time/60:.1f} minutes")
    log_message(f"  Final train loss: {train_losses[-1]:.4f}")
    log_message(f"  Best valid loss: {min(valid_losses):.4f}")
    log_message(f"  Avg step time: {np.mean(step_times[1:]):.1f}ms")
    
    return {
        'model_type': model_type,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'step_times': step_times[1:],
        'total_params': total_params,
        'best_valid_loss': min(valid_losses),
        'final_train_loss': train_losses[-1],
        'avg_step_time': np.mean(step_times[1:]),
        'total_time': total_time,
    }


def plot_results(results_list, filename="wikitext2_benchmark_results.png"):
    """Create publication-quality comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    config = BenchmarkConfig()
    
    # Training Loss
    ax = axes[0, 0]
    for r in results_list:
        label = r['model_type'].replace('_', ' ').title()
        ax.plot(r['train_losses'], label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Training Loss Curves", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for r in results_list:
        label = r['model_type'].replace('_', ' ').title()
        eval_steps = np.arange(config.eval_every, len(r['train_losses']) + 1, config.eval_every)
        ax.plot(eval_steps[:len(r['valid_losses'])], r['valid_losses'], 
               label=label, linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=14)
    ax.set_title("Validation Loss Curves", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Parameters
    ax = axes[0, 2]
    models = [r['model_type'].replace('_', ' ').title() for r in results_list]
    params = [r['total_params'] / 1e6 for r in results_list]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel("Parameters (Millions)", fontsize=14)
    ax.set_title("Model Size Comparison", fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Training Speed
    ax = axes[1, 0]
    speeds = [r['avg_step_time'] for r in results_list]
    bars = ax.bar(models, speeds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel("Time per Step (ms)", fontsize=14)
    ax.set_title("Training Speed", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Best Validation Loss
    ax = axes[1, 1]
    best_losses = [r['best_valid_loss'] for r in results_list]
    bars = ax.bar(models, best_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel("Best Validation Loss", fontsize=14)
    ax.set_title("Best Validation Performance", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, loss in zip(bars, best_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Parameter Reduction
    ax = axes[1, 2]
    baseline_params = results_list[0]['total_params']
    reductions = [(1 - r['total_params'] / baseline_params) * 100 for r in results_list]
    bars = ax.bar(models, reductions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel("Parameter Reduction (%)", fontsize=14)
    ax.set_title("Compression vs Baseline", fontsize=16, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    log_message(f"✓ Saved plots: {filename}")


def main():
    log_message("="*70)
    log_message("WIKITEXT-2 BENCHMARK - OVERNIGHT RUN")
    log_message("="*70)
    log_message(f"JAX devices: {jax.devices()}")
    log_message("")
    
    config = BenchmarkConfig()
    
    # Load data
    train_text, test_text = load_wikitext2()
    
    # Create tokenizer and batches
    log_message("Creating training batches...")
    tokenizer = SimpleTokenizer()
    key = jrandom.PRNGKey(42)
    k1, k2, key = jrandom.split(key, 3)
    
    train_batches = create_batches(
        train_text, tokenizer, config.batch_size, 
        config.seq_len, config.num_steps, k1
    )
    valid_batches = create_batches(
        test_text, tokenizer, config.batch_size, 
        config.seq_len, 100, k2
    )
    
    log_message(f"  Train batches: {len(train_batches)}")
    log_message(f"  Valid batches: {len(valid_batches)}")
    log_message("")
    
    # Train all models
    results = []
    model_types = ["standard", "time_indexed_mlp", "time_indexed_ssm"]
    
    for i, model_type in enumerate(model_types):
        k_model = jrandom.PRNGKey(42 + i)
        result = train_model(model_type, config, train_batches, valid_batches, k_model)
        results.append(result)
        
        # Save intermediate results
        with open(f"result_{model_type}.pkl", 'wb') as f:
            pickle.dump(result, f)
        log_message(f"  → Saved result: result_{model_type}.pkl")
        log_message("")
    
    # Save all results
    with open("wikitext2_benchmark_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    log_message("✓ Saved all results: wikitext2_benchmark_results.pkl")
    
    # Plot comparison
    plot_results(results)
    
    # Summary
    log_message("")
    log_message("="*70)
    log_message("FINAL SUMMARY")
    log_message("="*70)
    log_message("")
    log_message("Parameters:")
    for r in results:
        name = r['model_type'].replace('_', ' ').title()
        reduction = (1 - r['total_params'] / results[0]['total_params']) * 100
        if r == results[0]:
            log_message(f"  {name:30s} {r['total_params']:15,} (baseline)")
        else:
            log_message(f"  {name:30s} {r['total_params']:15,} ({reduction:.1f}% reduction)")
    
    log_message("")
    log_message("Best Validation Loss:")
    best_overall = min(r['best_valid_loss'] for r in results)
    for r in results:
        name = r['model_type'].replace('_', ' ').title()
        marker = "🏆" if r['best_valid_loss'] == best_overall else "  "
        log_message(f"  {marker} {name:30s} {r['best_valid_loss']:.4f}")
    
    log_message("")
    log_message("Training Speed:")
    for r in results:
        name = r['model_type'].replace('_', ' ').title()
        speedup = results[0]['avg_step_time'] / r['avg_step_time']
        log_message(f"  {name:30s} {r['avg_step_time']:6.1f}ms/step ({speedup:.2f}x)")
    
    log_message("")
    log_message("="*70)
    log_message("BENCHMARK COMPLETE!")
    log_message("="*70)


if __name__ == "__main__":
    main()

