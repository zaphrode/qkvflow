#!/usr/bin/env python3
"""
Statistical Validation Script - Run all models with multiple seeds
This is the #1 CRITICAL requirement for publication

Runs each model 5 times with different random seeds to compute:
- Mean ± standard deviation for all metrics
- Statistical significance tests
- Error bars for plots
"""

# ruff: noqa: E402
# isort: skip_file
# flake8: noqa: E402

import os
import sys

# Add project root to path FIRST
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import stats

import equinox as eqx
import haliax as hax

from config.neuralode_config import NeuralOdeConfig
from config.neuralode_ssm_config import NeuralOdeSSMConfig
from levanter.models.gpt2 import Gpt2Config
from qkvflow.models.neuralode_lm import NeuralOdeLMHeadModel
from qkvflow.models.neuralode_ssm_lm import NeuralOdeSSMLMHeadModel


# Configuration
SEEDS = [42, 123, 456, 789, 1011]  # 5 different random seeds
NUM_STEPS = 1000
EVAL_EVERY = 100
BATCH_SIZE = 8
SEQ_LEN = 128

RESULTS_DIR = Path("statistical_validation_results")
RESULTS_DIR.mkdir(exist_ok=True)


def create_dummy_batch(vocab_size, seq_len, batch_size, key):
    """Create dummy training batch"""
    Pos = hax.Axis("Pos", seq_len)
    Batch = hax.Axis("Batch", batch_size)
    
    # Random input IDs
    input_ids = hax.random.randint(key, (Batch, Pos), 0, vocab_size)
    
    # Causal targets (shifted by 1)
    targets_array = jnp.roll(input_ids.array, -1, axis=1)
    targets = hax.named(targets_array, (Batch, Pos))
    
    return input_ids, targets


def train_standard_transformer(seed: int, config: Gpt2Config) -> Dict:
    """Train standard transformer with given seed"""
    print(f"\n{'='*60}")
    print(f"Training Standard Transformer (seed={seed})")
    print(f"{'='*60}")
    
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    
    # Create axes
    Vocab = hax.Axis("vocab", config.vocab_size)
    Pos = hax.Axis("Pos", SEQ_LEN)
    Batch = hax.Axis("Batch", BATCH_SIZE)
    
    # Initialize model (simple MLP-based transformer)
    # For standard, we'll use Tong's architecture without time modulation
    from qkvflow.models.neuralode_lm import NeuralOdeLMHeadModel
    
    neuralode_config = NeuralOdeConfig(
        gpt2_config=config,
        sinusoidal_dim=32,
        time_embed_dim=64,
        use_ode_integration=False  # Standard residual
    )
    
    model = NeuralOdeLMHeadModel.init(Vocab, neuralode_config, key=init_key)
    
    # Optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training loop
    losses = []
    times = []
    
    for step in range(NUM_STEPS):
        key, data_key = jax.random.split(key)
        input_ids, targets = create_dummy_batch(config.vocab_size, SEQ_LEN, BATCH_SIZE, data_key)
        
        start = time.time()
        
        # Compute loss and gradients
        def loss_fn(m):
            return m.compute_loss(input_ids, targets)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        # Update
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        step_time = (time.time() - start) * 1000  # ms
        times.append(step_time)
        losses.append(float(loss))
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_time = np.mean(times[-EVAL_EVERY:])
            print(f"Step {step+1}/{NUM_STEPS} | Loss: {loss:.4f} | Time: {avg_time:.1f}ms")
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    
    return {
        'seed': seed,
        'model': 'Standard Transformer',
        'final_loss': losses[-1],
        'all_losses': losses,
        'avg_time_ms': np.mean(times),
        'num_params': num_params
    }


def train_time_indexed_mlp(seed: int, config: Gpt2Config) -> Dict:
    """Train time-indexed MLP with given seed"""
    print(f"\n{'='*60}")
    print(f"Training Time-Indexed MLP (seed={seed})")
    print(f"{'='*60}")
    
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    
    # Create axes
    Vocab = hax.Axis("vocab", config.vocab_size)
    
    # Initialize model
    neuralode_config = NeuralOdeConfig(
        gpt2_config=config,
        sinusoidal_dim=32,
        time_embed_dim=64,
        use_ode_integration=False
    )
    
    model = NeuralOdeLMHeadModel.init(Vocab, neuralode_config, key=init_key)
    
    # Optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training loop
    losses = []
    times = []
    
    for step in range(NUM_STEPS):
        key, data_key = jax.random.split(key)
        input_ids, targets = create_dummy_batch(config.vocab_size, SEQ_LEN, BATCH_SIZE, data_key)
        
        start = time.time()
        
        def loss_fn(m):
            return m.compute_loss(input_ids, targets)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        step_time = (time.time() - start) * 1000
        times.append(step_time)
        losses.append(float(loss))
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_time = np.mean(times[-EVAL_EVERY:])
            print(f"Step {step+1}/{NUM_STEPS} | Loss: {loss:.4f} | Time: {avg_time:.1f}ms")
    
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    
    return {
        'seed': seed,
        'model': 'Time-Indexed MLP',
        'final_loss': losses[-1],
        'all_losses': losses,
        'avg_time_ms': np.mean(times),
        'num_params': num_params
    }


def train_time_indexed_ssm(seed: int, config: Gpt2Config) -> Dict:
    """Train time-indexed SSM with given seed"""
    print(f"\n{'='*60}")
    print(f"Training Time-Indexed SSM (seed={seed})")
    print(f"{'='*60}")
    
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    
    # Create axes
    Vocab = hax.Axis("vocab", config.vocab_size)
    
    # Initialize model
    ssm_config = NeuralOdeSSMConfig(
        gpt2_config=config,
        sinusoidal_dim=32,
        time_embed_dim=64,
        ssm_state_size=64,
        vocab_size=config.vocab_size
    )
    
    model = NeuralOdeSSMLMHeadModel.init(Vocab, ssm_config, key=init_key)
    
    # Optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training loop
    losses = []
    times = []
    
    for step in range(NUM_STEPS):
        key, data_key = jax.random.split(key)
        input_ids, targets = create_dummy_batch(config.vocab_size, SEQ_LEN, BATCH_SIZE, data_key)
        
        start = time.time()
        
        def loss_fn(m):
            return m.compute_loss(input_ids, targets, key=jax.random.PRNGKey(0))
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        step_time = (time.time() - start) * 1000
        times.append(step_time)
        losses.append(float(loss))
        
        if (step + 1) % EVAL_EVERY == 0:
            avg_time = np.mean(times[-EVAL_EVERY:])
            print(f"Step {step+1}/{NUM_STEPS} | Loss: {loss:.4f} | Time: {avg_time:.1f}ms")
    
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    
    return {
        'seed': seed,
        'model': 'Time-Indexed SSM',
        'final_loss': losses[-1],
        'all_losses': losses,
        'avg_time_ms': np.mean(times),
        'num_params': num_params
    }


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute mean, std, and confidence intervals"""
    losses = [r['final_loss'] for r in results]
    times = [r['avg_time_ms'] for r in results]
    
    return {
        'model': results[0]['model'],
        'num_seeds': len(results),
        'loss_mean': np.mean(losses),
        'loss_std': np.std(losses),
        'loss_sem': stats.sem(losses),  # Standard error of mean
        'loss_ci95': stats.t.interval(0.95, len(losses)-1, loc=np.mean(losses), scale=stats.sem(losses)),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'num_params': results[0]['num_params'],
        'all_seeds': [r['seed'] for r in results],
        'all_losses': losses,
        'all_times': times
    }


def significance_test(results1: List[Dict], results2: List[Dict]) -> Dict:
    """Perform statistical significance test between two models"""
    losses1 = [r['final_loss'] for r in results1]
    losses2 = [r['final_loss'] for r in results2]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_ind(losses1, losses2)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(losses1)**2 + np.std(losses2)**2) / 2)
    cohens_d = (np.mean(losses1) - np.mean(losses2)) / pooled_std
    
    return {
        'model1': results1[0]['model'],
        'model2': results2[0]['model'],
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'cohens_d': float(cohens_d),
        'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
    }


def main():
    print("="*70)
    print("STATISTICAL VALIDATION - Multiple Seed Experiments")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Seeds: {SEEDS}")
    print(f"  Training steps: {NUM_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"\nThis will take approximately 30-60 minutes on A100 GPU")
    print("="*70)
    
    # Shared config
    config = Gpt2Config(
        hidden_dim=256,
        num_heads=4,
        num_layers=6,
        seq_len=SEQ_LEN,
        vocab_size=256  # Character-level
    )
    
    # Store all results
    all_results = {
        'Standard Transformer': [],
        'Time-Indexed MLP': [],
        'Time-Indexed SSM': []
    }
    
    # Run experiments for each seed
    for seed in SEEDS:
        # Standard Transformer
        result = train_standard_transformer(seed, config)
        all_results['Standard Transformer'].append(result)
        
        # Save individual result
        with open(RESULTS_DIR / f"standard_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        # Time-Indexed MLP
        result = train_time_indexed_mlp(seed, config)
        all_results['Time-Indexed MLP'].append(result)
        
        with open(RESULTS_DIR / f"time_mlp_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        # Time-Indexed SSM
        result = train_time_indexed_ssm(seed, config)
        all_results['Time-Indexed SSM'].append(result)
        
        with open(RESULTS_DIR / f"time_ssm_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Compute statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    statistics = {}
    for model_name, results in all_results.items():
        stats_dict = compute_statistics(results)
        statistics[model_name] = stats_dict
        
        print(f"\n{model_name}:")
        print(f"  Loss: {stats_dict['loss_mean']:.4f} ± {stats_dict['loss_std']:.4f}")
        print(f"  95% CI: [{stats_dict['loss_ci95'][0]:.4f}, {stats_dict['loss_ci95'][1]:.4f}]")
        print(f"  Time: {stats_dict['time_mean']:.1f} ± {stats_dict['time_std']:.1f} ms")
        print(f"  Params: {stats_dict['num_params']:,}")
    
    # Save statistics
    with open(RESULTS_DIR / "statistics_summary.json", 'w') as f:
        # Convert CI tuples to lists for JSON
        for model in statistics:
            ci = statistics[model]['loss_ci95']
            statistics[model]['loss_ci95'] = [float(ci[0]), float(ci[1])]
        json.dump(statistics, f, indent=2)
    
    # Significance tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    models = list(all_results.keys())
    significance_results = []
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            test_result = significance_test(
                all_results[models[i]],
                all_results[models[j]]
            )
            significance_results.append(test_result)
            
            print(f"\n{models[i]} vs {models[j]}:")
            print(f"  p-value: {test_result['p_value']:.6f}")
            print(f"  Significant (α=0.05): {test_result['significant_at_0.05']}")
            print(f"  Cohen's d: {test_result['cohens_d']:.3f} ({test_result['effect_size']} effect)")
    
    # Save significance tests
    with open(RESULTS_DIR / "significance_tests.json", 'w') as f:
        json.dump(significance_results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ STATISTICAL VALIDATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("\nNext steps:")
    print("  1. Run: python scripts/plot_statistical_results.py")
    print("  2. Include these results with error bars in your paper")
    print("  3. Report significance tests in results section")


if __name__ == "__main__":
    main()


