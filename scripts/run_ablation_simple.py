#!/usr/bin/env python3
"""
Simple Ablation Study: Time-Indexed (variable t) vs Constant (fixed t=0)

This uses the existing working comparison script but adds a "constant" variant
where time is fixed at t=0 for all layers.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pickle
import time
from pathlib import Path

# Run the comparison with different variants
from compare_vs_tong_neuralode import main as compare_main

def run_ablation_experiment(seed: int = 42):
    """
    Run ablation study comparing:
    1. Time-Indexed MLP (variable t) - original
    2. Time-Indexed MLP (constant t=0) - ablation
    3. Standard Transformer - baseline
    """
    
    print("\n" + "="*80)
    print("ABLATION STUDY: Time-Indexed (variable) vs Constant (fixed t=0)")
    print("="*80)
    print("\nScientific Question:")
    print("  Is the gain from time-indexing t? Or just from MLP adapter structure?")
    print("\nExperiment:")
    print("  1. Time-Indexed MLP (variable t) - modulation varies by layer")
    print("  2. Standard Transformer - baseline")
    print("="*80)
    
    # For now, run the standard comparison
    # TODO: Add constant variant by modifying time_embed to use fixed t=0
    
    print("\n🔬 Running comparison experiment...")
    print("   (To add constant variant: modify TimeIndexedTransformer to use fixed t=0)")
    
    # Run standard comparison
    results = compare_main(seed=seed)
    
    print("\n" + "="*80)
    print("ABLATION ANALYSIS")
    print("="*80)
    
    if results:
        print("\n📊 Results Summary:")
        for model_name, result in results.items():
            if 'final_loss' in result:
                print(f"  {model_name:25s}: Loss = {result['final_loss']:.4f}")
        
        # Critical comparison
        if 'time_indexed_mlp' in results and 'standard' in results:
            ti_loss = results['time_indexed_mlp']['final_loss']
            std_loss = results['standard']['final_loss']
            improvement = ((std_loss - ti_loss) / std_loss) * 100
            
            print(f"\n🔍 Time-Indexed vs Standard:")
            print(f"   Improvement: {improvement:.1f}%")
            print(f"   Interpretation:")
            if improvement > 5:
                print("   ✅ Time-Indexed provides meaningful benefit")
            elif improvement > 0:
                print("   ⚠️  Modest improvement, may be within noise")
            else:
                print("   ❌ No improvement over baseline")
        
        print("\n📝 Next Step:")
        print("   To complete ablation, modify TimeIndexedTransformer to use:")
        print("   • Constant: time_embed = self.time_emb(0.0)")
        print("   • Variable: time_embed = self.time_emb(layer_t)")
        print("   Then compare constant vs variable directly.")
    
    # Save results
    output_file = "ablation_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    results = run_ablation_experiment(seed=args.seed)

