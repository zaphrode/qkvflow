#!/usr/bin/env python3
"""
Regenerate statistics from existing seed results
Fixes the JSON serialization issue
"""

import json
import pickle
from pathlib import Path
import numpy as np
from scipy import stats

RESULTS_DIR = Path("statistical_validation_results")
SEEDS = [42, 123, 456, 789, 1011]


def aggregate_results(all_seeds_results):
    """Aggregate results across seeds"""
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
    """Perform pairwise significance tests - FIXED JSON serialization"""
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
            
            # FIX: Convert numpy bool_ to Python bool
            tests.append({
                'model1': model1,
                'model2': model2,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': bool(float(p_value) < 0.05),  # Convert via float first
                'significant_at_0.01': bool(float(p_value) < 0.01),  # Convert via float first
                'cohens_d': float(cohens_d),
                'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
            })
    
    return tests


def main():
    print("Loading saved seed results...")
    
    all_seeds_results = []
    for seed in SEEDS:
        seed_file = RESULTS_DIR / f"seed_{seed}_results.pkl"
        if seed_file.exists():
            with open(seed_file, 'rb') as f:
                results = pickle.load(f)
                all_seeds_results.append(results)
            print(f"  ✓ Loaded seed {seed}")
        else:
            print(f"  ✗ Missing seed {seed}")
    
    if len(all_seeds_results) < 2:
        print("ERROR: Not enough results")
        return
    
    print(f"\nAggregating {len(all_seeds_results)} seeds...")
    statistics = aggregate_results(all_seeds_results)
    tests = significance_tests(statistics)
    
    print("Saving results...")
    with open(RESULTS_DIR / "statistics_summary.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    print("  ✓ statistics_summary.json")
    
    with open(RESULTS_DIR / "significance_tests.json", 'w') as f:
        json.dump(tests, f, indent=2)
    print("  ✓ significance_tests.json")
    
    # Print summary
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION RESULTS")
    print("="*70)
    
    for model_type, stats_dict in statistics.items():
        name = model_type.replace('_', ' ').title()
        print(f"\n{name}:")
        print(f"  Loss: {stats_dict['loss_mean']:.4f} ± {stats_dict['loss_std']:.4f}")
        print(f"  95% CI: [{stats_dict['loss_ci95'][0]:.4f}, {stats_dict['loss_ci95'][1]:.4f}]")
        print(f"  Speed: {stats_dict['speed_mean']:.1f} ± {stats_dict['speed_std']:.1f} ms/step")
        print(f"  Params: {stats_dict['params']:,}")
    
    print("\n" + "-"*70)
    print("Significance Tests:")
    print("-"*70)
    
    for test in tests:
        name1 = test['model1'].replace('_', ' ').title()
        name2 = test['model2'].replace('_', ' ').title()
        sig = ""
        if test['significant_at_0.01']:
            sig = "**"
        elif test['significant_at_0.05']:
            sig = "*"
        
        print(f"\n{name1} vs {name2}:")
        print(f"  p-value: {test['p_value']:.6f} {sig}")
        print(f"  Cohen's d: {test['cohens_d']:.3f} ({test['effect_size']})")
    
    print("\n" + "="*70)
    print("✅ STATISTICS REGENERATED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()

