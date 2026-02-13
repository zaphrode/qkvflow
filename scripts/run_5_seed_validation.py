#!/usr/bin/env python3
"""
Simplified Multi-Seed Validation
Runs comparison with 5 different seeds and aggregates results
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats
import jax
import jax.random as jrandom

# Import the comparison script directly
from scripts.compare_vs_tong_neuralode import (
    ComparisonConfig, load_wikitext2, SimpleTokenizer,
    create_batches, train_model
)

SEEDS = [42, 123, 456, 789, 1011]
RESULTS_DIR = Path("statistical_validation_results")
RESULTS_DIR.mkdir(exist_ok=True)


def log_message(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_comparison_with_seed(seed, config, train_text, test_text):
    """Run comparison with specific seed"""
    log_message(f"Running comparison with seed {seed}...")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    key = jrandom.PRNGKey(seed)
    k1, k2, key = jrandom.split(key, 3)
    
    # Create batches
    train_batches = create_batches(
        train_text, tokenizer, config.batch_size,
        config.seq_len, config.num_steps, k1
    )
    valid_batches = create_batches(
        test_text, tokenizer, config.batch_size,
        config.seq_len, 100, k2
    )
    
    # Train all models
    results = []
    model_types = ["standard", "tong_neuralode", "time_indexed_mlp", "time_indexed_ssm"]
    
    for i, model_type in enumerate(model_types):
        k_model = jrandom.PRNGKey(seed + i * 100)  # Different seed per model
        try:
            log_message(f"  Training {model_type}...")
            result = train_model(model_type, config, train_batches, valid_batches, k_model)
            results.append(result)
            log_message(f"    ✓ {model_type}: loss={result['best_valid_loss']:.4f}")
        except Exception as e:
            log_message(f"    ✗ {model_type} failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def aggregate_results(all_seeds_results):
    """Aggregate results across seeds"""
    # Organize by model type
    by_model = {}
    
    for seed_results in all_seeds_results:
        for result in seed_results:
            model_type = result['model_type']
            if model_type not in by_model:
                by_model[model_type] = {
                    'losses': [],
                    'speeds': [],
                    'params': result['total_params']
                }
            by_model[model_type]['losses'].append(result['best_valid_loss'])
            by_model[model_type]['speeds'].append(result['avg_step_time'])
    
    # Compute statistics
    statistics = {}
    for model_type, data in by_model.items():
        losses = np.array(data['losses'])
        speeds = np.array(data['speeds'])
        
        statistics[model_type] = {
            'loss_mean': float(np.mean(losses)),
            'loss_std': float(np.std(losses)),
            'loss_sem': float(stats.sem(losses)),
            'loss_ci95': [
                float(stats.t.interval(0.95, len(losses)-1, 
                                      loc=np.mean(losses), 
                                      scale=stats.sem(losses))[0]),
                float(stats.t.interval(0.95, len(losses)-1, 
                                      loc=np.mean(losses), 
                                      scale=stats.sem(losses))[1])
            ],
            'speed_mean': float(np.mean(speeds)),
            'speed_std': float(np.std(speeds)),
            'params': int(data['params']),
            'all_losses': [float(x) for x in losses],
            'all_speeds': [float(x) for x in speeds],
            'num_seeds': len(losses)
        }
    
    return statistics


def significance_tests(statistics):
    """Perform pairwise significance tests"""
    models = list(statistics.keys())
    tests = []
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            losses1 = np.array(statistics[model1]['all_losses'])
            losses2 = np.array(statistics[model2]['all_losses'])
            
            if len(losses1) < 2 or len(losses2) < 2:
                continue
            
            t_stat, p_value = stats.ttest_ind(losses1, losses2)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(losses1) + np.var(losses2)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(losses1) - np.mean(losses2)) / pooled_std
            else:
                cohens_d = 0.0
            
            tests.append({
                'model1': model1,
                'model2': model2,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': bool(p_value < 0.05),
                'significant_at_0.01': bool(p_value < 0.01),
                'cohens_d': float(cohens_d),
                'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
            })
    
    return tests


def print_summary(statistics, tests):
    """Print formatted summary"""
    log_message("\n" + "="*70)
    log_message("STATISTICAL VALIDATION RESULTS")
    log_message("="*70)
    
    log_message("\nModel Performance (Mean ± Std over {} seeds):".format(
        statistics[list(statistics.keys())[0]]['num_seeds']
    ))
    log_message("-"*70)
    
    for model_type, stats_dict in statistics.items():
        name = model_type.replace('_', ' ').title()
        log_message(f"\n{name}:")
        log_message(f"  Loss: {stats_dict['loss_mean']:.4f} ± {stats_dict['loss_std']:.4f}")
        log_message(f"  95% CI: [{stats_dict['loss_ci95'][0]:.4f}, {stats_dict['loss_ci95'][1]:.4f}]")
        log_message(f"  Speed: {stats_dict['speed_mean']:.1f} ± {stats_dict['speed_std']:.1f} ms/step")
        log_message(f"  Params: {stats_dict['params']:,}")
    
    log_message("\n" + "-"*70)
    log_message("Statistical Significance Tests:")
    log_message("-"*70)
    
    for test in tests:
        if test['significant_at_0.05']:
            name1 = test['model1'].replace('_', ' ').title()
            name2 = test['model2'].replace('_', ' ').title()
            sig_marker = "**" if test['significant_at_0.01'] else "*"
            
            log_message(f"\n{name1} vs {name2}:")
            log_message(f"  p-value: {test['p_value']:.6f} {sig_marker}")
            log_message(f"  Cohen's d: {test['cohens_d']:.3f} ({test['effect_size']} effect)")
    
    log_message("\n" + "="*70)
    log_message("Legend: * p<0.05, ** p<0.01")
    log_message("="*70)


def main():
    log_message("="*70)
    log_message("5-SEED STATISTICAL VALIDATION")
    log_message("="*70)
    log_message(f"Seeds: {SEEDS}")
    log_message(f"Output directory: {RESULTS_DIR}/")
    log_message(f"JAX devices: {jax.devices()}")
    log_message("")
    log_message("This will take approximately 30-60 minutes on A100 GPU")
    log_message("="*70)
    
    # Configuration
    config = ComparisonConfig()
    
    # Load data once
    log_message("\nLoading WikiText-2 data...")
    train_text, test_text = load_wikitext2()
    log_message(f"  Train: {len(train_text)} chars")
    log_message(f"  Test: {len(test_text)} chars")
    
    # Run for each seed
    all_seeds_results = []
    
    for idx, seed in enumerate(SEEDS):
        log_message(f"\n{'='*70}")
        log_message(f"SEED {idx+1}/{len(SEEDS)}: {seed}")
        log_message(f"{'='*70}")
        
        try:
            results = run_comparison_with_seed(seed, config, train_text, test_text)
            if results:
                all_seeds_results.append(results)
                
                # Save individual seed results
                seed_file = RESULTS_DIR / f"seed_{seed}_results.pkl"
                with open(seed_file, 'wb') as f:
                    pickle.dump(results, f)
                log_message(f"✓ Saved: {seed_file}")
            else:
                log_message(f"✗ No results for seed {seed}")
        
        except Exception as e:
            log_message(f"✗ ERROR with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_seeds_results) < 2:
        log_message("\n✗ ERROR: Not enough successful runs (need at least 2)")
        return
    
    # Aggregate results
    log_message("\n" + "="*70)
    log_message("AGGREGATING RESULTS")
    log_message("="*70)
    
    statistics = aggregate_results(all_seeds_results)
    tests = significance_tests(statistics)
    
    # Save aggregated results
    with open(RESULTS_DIR / "statistics_summary.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    log_message(f"✓ Saved: {RESULTS_DIR}/statistics_summary.json")
    
    with open(RESULTS_DIR / "significance_tests.json", 'w') as f:
        json.dump(tests, f, indent=2)
    log_message(f"✓ Saved: {RESULTS_DIR}/significance_tests.json")
    
    # Print summary
    print_summary(statistics, tests)
    
    log_message("\n" + "="*70)
    log_message("✅ STATISTICAL VALIDATION COMPLETE!")
    log_message("="*70)
    log_message(f"\nResults saved to: {RESULTS_DIR}/")
    log_message("\nNext steps:")
    log_message("  1. python scripts/plot_statistical_results.py")
    log_message("  2. Include these results in your paper")
    log_message("  3. Convert RESEARCH_SUMMARY.md to LaTeX")


if __name__ == "__main__":
    main()


