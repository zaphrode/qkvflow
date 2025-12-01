#!/usr/bin/env python3
"""
Convert WikiText-103 from HuggingFace to text files.
This allows us to use the WORKING comparison script without modifications.
"""

import os
from datasets import load_dataset
from pathlib import Path

# Set token from environment
if "HF_TOKEN" not in os.environ:
    raise ValueError("HF_TOKEN environment variable not set. Export it before running: export HF_TOKEN='your_token'")

print("Loading WikiText-103...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", verification_mode="no_checks")

print(f"✓ Loaded")
print(f"  Train: {len(ds['train']):,} examples")
print(f"  Validation: {len(ds['validation']):,} examples")

# Create output directory
output_dir = Path("wikitext103_data")
output_dir.mkdir(exist_ok=True)

# Save train (subsample to reasonable size)
print("\nSaving train.txt (subsampling 50k examples)...")
with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
    for i, example in enumerate(ds['train']):
        if i >= 50000:  # Subsample for reasonable size
            break
        f.write(example['text'] + "\n")

print("✓ train.txt saved")

# Save validation
print("\nSaving test.txt (using validation split)...")
with open(output_dir / "test.txt", "w", encoding="utf-8") as f:
    for example in ds['validation']:
        f.write(example['text'] + "\n")

print("✓ test.txt saved")

# Check sizes
train_size = (output_dir / "train.txt").stat().st_size / 1024 / 1024
test_size = (output_dir / "test.txt").stat().st_size / 1024 / 1024

print(f"\n✅ WikiText-103 data prepared!")
print(f"  train.txt: {train_size:.1f} MB")
print(f"  test.txt: {test_size:.1f} MB")
print(f"\nTo use with comparison script:")
print(f"  cd {output_dir}")
print(f"  python ../scripts/compare_vs_tong_neuralode.py")

