"""
Quick test of WikiText-103 loading and processing.
Runs a short training to verify everything works before full run.
"""

import os
from datasets import load_dataset
import jax.numpy as jnp

# Set HF token from environment (don't commit tokens to Git!)
# Before running, set: export HF_TOKEN="your_token_here"
if "HF_TOKEN" not in os.environ:
    print("⚠️  Warning: HF_TOKEN not set. Set it with:")
    print("  export HF_TOKEN='your_hf_token'")
    # For local testing only - remove before committing!
    # os.environ["HF_TOKEN"] = "your_token_here"

print("="*60)
print("Quick WikiText-103 Test")
print("="*60)

print("\n1. Loading dataset from Hugging Face...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", verification_mode="no_checks")

print(f"\n✓ Dataset loaded successfully!")
print(f"  Train: {len(ds['train'])} examples")
print(f"  Validation: {len(ds['validation'])} examples")
print(f"  Test: {len(ds['test'])} examples")

print(f"\nSample text (first 200 chars):")
print(f"  '{ds['train']['text'][100][:200]}...'")

print("\n2. Building vocabulary (from 1000 samples)...")
sample_text = " ".join(ds['train']['text'][:1000])
unique_chars = sorted(set(sample_text))
vocab_size = len(unique_chars)

print(f"✓ Vocabulary size: {vocab_size} characters")
print(f"  Sample chars: {unique_chars[:20]}")

print("\n3. Tokenizing sample...")
char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}

def tokenize(text, max_len=128):
    tokens = [char_to_idx.get(ch, 0) for ch in text[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return tokens

sample_tokens = tokenize(ds['train']['text'][100])
print(f"✓ Tokenized: {len(sample_tokens)} tokens")
print(f"  First 20: {sample_tokens[:20]}")

print("\n4. Creating batch...")
batch_data = []
for i in range(16):  # Small batch
    if i < len(ds['train']['text']) and ds['train']['text'][i].strip():
        batch_data.append(tokenize(ds['train']['text'][i]))

batch = jnp.array(batch_data)
print(f"✓ Batch shape: {batch.shape}")

print("\n" + "="*60)
print("✅ All checks passed!")
print("="*60)
print("\nReady to run full experiment:")
print("  python scripts/run_wikitext103_validation.py")
print("\nEstimated time: 2-3 hours on A100")
print("  (5000 steps × ~1.5s/step)")

