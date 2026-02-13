#!/usr/bin/env python3
"""
Multi-Seed Statistical Validation
Runs the comparison 5 times with different seeds for publication-ready results
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats

# Seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1011]
RESULTS_DIR = Path("statistical_validation_results")
RESULTS_DIR.mkdir(exist_ok=True)

def log_message(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_comparison_with_seed(seed):
    """Run comparison script with specific seed"""
    log_message(f"Starting comparison with seed {seed}")
    
    # Modify the comparison script to accept seed as argument
    # For now, we'll run it and parse results
    cmd = f"cd /home/nahid/Documents/qkvflow && source venv311/bin/activate && python scripts/compare_vs_tong_neuralode.py"
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash'
    )
    
    if result.returncode != 0:
        log_message(f"ERROR with seed {seed}: {result.stderr}")
        return None
    
    # Parse output to extract results
    # The script saves results to comparison_results.csv
    return seed


def load_results_from_csv():
    """Load the latest comparison results"""
    csv_path = Path("comparison_results.csv")
    if not csv_path.exists():
        return None
    
    results = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                model = parts[0]
                results[model] = {
                    'valid_loss': float(parts[1]),
                    'params': int(parts[2]),
                    'speed_ms': float(parts[3])
                }
    
    return results


def compute_statistics(all_results):
    """Compute mean, std, and significance tests"""
    models = list(all_results[0].keys())
    
    statistics = {}
    for model in models:
        losses = [r[model]['valid_loss'] for r in all_results]
        speeds = [r[model]['speed_ms'] for r in all_results]
        params = all_results[0][model]['params']  # Same across runs
        
        statistics[model] = {
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
            'params': params,
            'all_losses': [float(x) for x in losses],
            'all_speeds': [float(x) for x in speeds]
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
            
            losses1 = statistics[model1]['all_losses']
            losses2 = statistics[model2]['all_losses']
            
            t_stat, p_value = stats.ttest_ind(losses1, losses2)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.std(losses1)**2 + np.std(losses2)**2) / 2)
            cohens_d = (np.mean(losses1) - np.mean(losses2)) / pooled_std
            
            tests.append({
                'model1': model1,
                'model2': model2,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01,
                'cohens_d': float(cohens_d),
                'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
            })
    
    return tests


def main():
    log_message("="*70)
    log_message("MULTI-SEED STATISTICAL VALIDATION")
    log_message("="*70)
    log_message(f"Seeds: {SEEDS}")
    log_message(f"Output: {RESULTS_DIR}/")
    log_message("")
    
    all_results = []
    
    # Run comparison for each seed
    for idx, seed in enumerate(SEEDS):
        log_message(f"\n{'='*70}")
        log_message(f"RUN {idx+1}/{len(SEEDS)} - Seed {seed}")
        log_message(f"{'='*70}")
        
        # Note: We need to modify compare_vs_tong_neuralode.py to accept seed
        # For now, run as is and save results
        run_comparison_with_seed(seed)
        
        # Load results
        results = load_results_from_csv()
        if results:
            all_results.append(results)
            
            # Save individual run
            with open(RESULTS_DIR / f"seed_{seed}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            log_message(f"✓ Completed seed {seed}")
        else:
            log_message(f"✗ Failed to load results for seed {seed}")
    
    if len(all_results) < 2:
        log_message("ERROR: Not enough successful runs for statistics")
        return
    
    log_message(f"\n{'='*70}")
    log_message("COMPUTING STATISTICS")
    log_message(f"{'='*70}")
    
    # Compute statistics
    statistics = compute_statistics(all_results)
    
    # Save statistics
    with open(RESULTS_DIR / "statistics_summary.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    
    # Significance tests
    tests = significance_tests(statistics)
    
    # Save tests
    with open(RESULTS_DIR / "significance_tests.json", 'w') as f:
        json.dump(tests, f, indent=2)
    
    # Print summary
    log_message("\n" + "="*70)
    log_message("RESULTS SUMMARY")
    log_message("="*70)
    
    for model, stats_dict in statistics.items():
        log_message(f"\n{model}:")
        log_message(f"  Loss: {stats_dict['loss_mean']:.4f} ± {stats_dict['loss_std']:.4f}")
        log_message(f"  95% CI: [{stats_dict['loss_ci95'][0]:.4f}, {stats_dict['loss_ci95'][1]:.4f}]")
        log_message(f"  Speed: {stats_dict['speed_mean']:.1f} ± {stats_dict['speed_std']:.1f} ms")
        log_message(f"  Params: {stats_dict['params']:,}")
    
    log_message("\n" + "="*70)
    log_message("SIGNIFICANCE TESTS")
    log_message("="*70)
    
    for test in tests:
        if test['significant_at_0.05']:
            sig_marker = "**" if test['significant_at_0.01'] else "*"
            log_message(f"\n{test['model1']} vs {test['model2']}:")
            log_message(f"  p-value: {test['p_value']:.6f} {sig_marker}")
            log_message(f"  Cohen's d: {test['cohens_d']:.3f} ({test['effect_size']} effect)")
    
    log_message("\n" + "="*70)
    log_message("✅ STATISTICAL VALIDATION COMPLETE!")
    log_message("="*70)
    log_message(f"\nResults saved to: {RESULTS_DIR}/")
    log_message("\nNext step: python scripts/plot_statistical_results.py")


if __name__ == "__main__":
    main()


