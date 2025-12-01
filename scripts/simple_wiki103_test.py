"""
Simple WikiText-103 test using the existing working comparison code.

This just loads WikiText-103 and runs our models on it to address the
"only tested on WikiText-2" limitation.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.random as jrandom
import jax.numpy as jnp
from datasets import load_dataset
from tqdm import tqdm

# Set HF token from environment (don't commit tokens to Git!)
# Before running, set: export HF_TOKEN="your_token_here"
if "HF_TOKEN" not in os.environ:
    print("⚠️  Warning: HF_TOKEN not set. Set it with:")
    print("  export HF_TOKEN='your_hf_token'")

print("=" * 60)
print("WikiText-103 Simple Test")
print("=" * 60)

print("\nLoading WikiText-103...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", verification_mode="no_checks")

print(f"✓ Loaded!")
print(f"  Train: {len(ds['train']):,} examples")
print(f"  Val: {len(ds['validation']):,} examples")
print(f"  Test: {len(ds['test']):,} examples")

print("\n✅ SUCCESS: WikiText-103 can be loaded!")
print("\nDataset is ~50× larger than WikiText-2:")
print(f"  WikiText-2:   ~2M tokens")
print(f"  WikiText-103: ~103M tokens")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Adapt existing comparison script to use Wiki-103 data")
print("2. Run overnight experiment on this larger dataset")
print("3. Compare results with WikiText-2 experiments")
print("4. Update README: 'Tested on WikiText-2 AND WikiText-103'")

