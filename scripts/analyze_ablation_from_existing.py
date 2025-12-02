#!/usr/bin/env python3
"""
Analyze Ablation Study from Existing Results

This uses the already-computed statistical validation results to analyze
the ablation question without needing to retrain models.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

def load_existing_results():
    """Load results from statistical validation."""
    results_dir = Path("statistical_validation_results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Load all seed results
    seed_files = list(results_dir.glob("seed_*_results.pkl"))
    
    if not seed_files:
        print(f"❌ No results files found in {results_dir}")
        return None
    
    print(f"📂 Found {len(seed_files)} result files")
    
    # Load and aggregate
    all_results = {}
    for seed_file in seed_files:
        with open(seed_file, 'rb') as f:
            data = pickle.load(f)
            # Data is a list of dicts with 'model_type' keys
            if isinstance(data, list):
                for model_data in data:
                    model_name = model_data.get('model_type', 'unknown')
                    if model_name not in all_results:
                        all_results[model_name] = []
                    all_results[model_name].append(model_data)
            elif isinstance(data, dict):
                for model_name, model_data in data.items():
                    if model_name not in all_results:
                        all_results[model_name] = []
                    all_results[model_name].append(model_data)
    
    return all_results


def analyze_ablation(results):
    """Analyze whether time-indexing provides benefit."""
    
    print("\n" + "="*80)
    print("ABLATION ANALYSIS: Time-Indexed vs Standard")
    print("="*80)
    print("\nScientific Question:")
    print("  Is the gain from time-indexing t? Or just from MLP adapter structure?")
    print("\nData Source: Statistical validation results (5 seeds)")
    print("="*80)
    
    # Extract losses for each model
    model_losses = {}
    for model_name, runs in results.items():
        losses = [run['best_valid_loss'] for run in runs if 'best_valid_loss' in run]
        if losses:
            model_losses[model_name] = {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'all': losses
            }
    
    print("\n📊 Validation Loss (mean ± std over 5 seeds):")
    print("-" * 60)
    for model_name in sorted(model_losses.keys()):
        stats = model_losses[model_name]
        print(f"  {model_name:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Critical comparison
    if 'time_indexed_mlp' in model_losses and 'standard' in model_losses:
        ti_mean = model_losses['time_indexed_mlp']['mean']
        ti_std = model_losses['time_indexed_mlp']['std']
        std_mean = model_losses['standard']['mean']
        std_std = model_losses['standard']['std']
        
        improvement = ((std_mean - ti_mean) / std_mean) * 100
        
        print("\n" + "="*80)
        print("🔍 CRITICAL COMPARISON: Time-Indexed MLP vs Standard")
        print("="*80)
        print(f"\n  Time-Indexed MLP: {ti_mean:.4f} ± {ti_std:.4f}")
        print(f"  Standard:         {std_mean:.4f} ± {std_std:.4f}")
        print(f"\n  Improvement: {improvement:.2f}%")
        
        # Statistical significance (simple t-test)
        from scipy import stats as scipy_stats
        ti_losses = model_losses['time_indexed_mlp']['all']
        std_losses = model_losses['standard']['all']
        t_stat, p_value = scipy_stats.ttest_ind(ti_losses, std_losses)
        
        print(f"  p-value: {p_value:.4f}")
        
        print("\n📝 Interpretation:")
        if p_value < 0.05 and improvement > 5:
            print("  ✅ Time-Indexed provides STATISTICALLY SIGNIFICANT benefit")
            print("  ✅ Improvement > 5% and p < 0.05")
            print("  ✅ This suggests time-dependency (Neural ODE) helps")
            print("")
            print("  Conclusion: The gain appears to come from time-indexing,")
            print("              not just from adding MLP adapter structure.")
            print("              Neural ODE narrative is supported.")
        elif p_value < 0.05 and improvement > 0:
            print("  ⚠️  Time-Indexed provides modest but significant benefit")
            print("  ⚠️  Improvement is small (<5%) but p < 0.05")
            print("")
            print("  Conclusion: Time-indexing helps, but the effect is modest.")
            print("              To confirm, would need constant t=0 baseline.")
        elif improvement > 5:
            print("  ⚠️  Time-Indexed appears better but NOT statistically significant")
            print("  ⚠️  Improvement > 5% but p >= 0.05")
            print("")
            print("  Conclusion: Difference may be within noise.")
            print("              Need more seeds or constant t=0 baseline.")
        else:
            print("  ❌ Time-Indexed does NOT provide meaningful benefit")
            print("")
            print("  Conclusion: The gain (if any) is small and not significant.")
            print("              Time-indexing may not be the key factor.")
    
    # SSM comparison
    if 'time_indexed_ssm' in model_losses and 'standard' in model_losses:
        ssm_mean = model_losses['time_indexed_ssm']['mean']
        ssm_std = model_losses['time_indexed_ssm']['std']
        std_mean = model_losses['standard']['mean']
        
        improvement = ((std_mean - ssm_mean) / std_mean) * 100
        
        print("\n" + "="*80)
        print("🔍 BONUS: Time-Indexed SSM vs Standard")
        print("="*80)
        print(f"\n  Time-Indexed SSM: {ssm_mean:.4f} ± {ssm_std:.4f}")
        print(f"  Standard:         {std_mean:.4f} ± {std_std:.4f}")
        print(f"\n  Improvement: {improvement:.2f}%")
        
        if improvement > 5:
            print("\n  ✅ SSM variant shows even stronger benefit!")
        else:
            print("\n  ⚠️  SSM variant benefit is modest")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\nTo FULLY complete the ablation study:")
    print("  1. Modify TimeIndexedTransformer to use fixed t=0:")
    print("     time_embed = self.time_emb(0.0)  # Constant for all layers")
    print("")
    print("  2. Train this 'constant modulation' variant")
    print("")
    print("  3. Compare:")
    print("     - Time-Indexed (variable t) - current results")
    print("     - Time-Indexed (constant t=0) - new experiment")
    print("     - Standard - current results")
    print("")
    print("  If variable t ≫ constant t=0:")
    print("    ✅ Proves time-dependency matters (Neural ODE validated)")
    print("")
    print("  If variable t ≈ constant t=0:")
    print("    ⚠️  Gain is from MLP adapters, not time-indexing")
    print("")
    print("="*80)


def main():
    print("\n" + "="*80)
    print("ABLATION STUDY ANALYSIS")
    print("Using Existing Statistical Validation Results")
    print("="*80)
    
    results = load_existing_results()
    
    if results is None:
        print("\n❌ Could not load results")
        return
    
    analyze_ablation(results)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Used data from: statistical_validation_results/")


if __name__ == "__main__":
    main()

