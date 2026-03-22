#!/usr/bin/env python3
"""
Diagnostic script to understand why the model has impossibly low perplexity.
"""

import sys
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
import optax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from qkvflow.config import Gpt2Config, Gpt2Embeddings, ACT2FN

# Import model classes for pickle loading
from train_publication_model import (
    TimeIndexedTransformer, TimeIndexedBlock, TimeIndexedAttention,
    TimeIndexedMLP, TimeModulatedLinear, TimeEmbedding
)

print("=" * 70)
print("MODEL DIAGNOSTIC")
print("=" * 70)

# Load the checkpoint
checkpoint_path = "/data1/fypnahid/qkvflow/checkpoints_v2/best_model.pkl"
print(f"\nLoading checkpoint from {checkpoint_path}...")

with open(checkpoint_path, "rb") as f:
    checkpoint = pickle.load(f)

model = checkpoint["model"]
config = checkpoint["config"]

print(f"Model config: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}")

# Setup
Batch = hax.Axis("batch", 2)
Pos = hax.Axis("position", config['sequence_length'])
Vocab = hax.Axis("vocab", config['vocab_size'])

def compute_loss(model, input_ids, labels):
    """Compute loss"""
    input_named = hax.named(input_ids, (Batch, Pos))
    logits = model(input_named, key=jrandom.PRNGKey(0), inference=True)
    logits_flat = logits.array.reshape(-1, Vocab.size)
    labels_flat = labels.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
    return jnp.mean(loss)

print("\n" + "=" * 70)
print("TEST 1: Random tokens (should have loss ~10.8, PPL ~50000)")
print("=" * 70)

key = jrandom.PRNGKey(42)
random_input = jrandom.randint(key, (2, config['sequence_length']), 0, config['vocab_size'])
random_labels = jrandom.randint(jrandom.split(key)[0], (2, config['sequence_length']), 0, config['vocab_size'])

loss = compute_loss(model, random_input, random_labels)
print(f"Random input → Loss: {float(loss):.4f}, PPL: {np.exp(float(loss)):.2f}")

print("\n" + "=" * 70)
print("TEST 2: Constant tokens (input=100, label=100)")
print("=" * 70)

const_input = jnp.full((2, config['sequence_length']), 100, dtype=jnp.int32)
const_labels = jnp.full((2, config['sequence_length']), 100, dtype=jnp.int32)

loss = compute_loss(model, const_input, const_labels)
print(f"Constant tokens → Loss: {float(loss):.4f}, PPL: {np.exp(float(loss)):.2f}")

print("\n" + "=" * 70)
print("TEST 3: Sequential tokens (label = input + 1)")
print("=" * 70)

seq_input = jnp.arange(config['sequence_length'], dtype=jnp.int32) % 1000
seq_input = jnp.stack([seq_input, seq_input])  # batch of 2
seq_labels = (seq_input + 1) % config['vocab_size']

loss = compute_loss(model, seq_input, seq_labels)
print(f"Sequential tokens → Loss: {float(loss):.4f}, PPL: {np.exp(float(loss)):.2f}")

print("\n" + "=" * 70)
print("TEST 4: Check model output distribution")
print("=" * 70)

input_named = hax.named(random_input, (Batch, Pos))
logits = model(input_named, key=jrandom.PRNGKey(0), inference=True)
logits_np = np.array(logits.array)

print(f"Logits shape: {logits_np.shape}")
print(f"Logits range: [{logits_np.min():.2f}, {logits_np.max():.2f}]")
print(f"Logits mean: {logits_np.mean():.4f}, std: {logits_np.std():.4f}")

# Check softmax distribution
probs = jax.nn.softmax(logits_np, axis=-1)
print(f"Max prob per position (mean): {probs.max(axis=-1).mean():.4f}")
print(f"Entropy per position (mean): {-(probs * np.log(probs + 1e-10)).sum(axis=-1).mean():.4f}")

# Check if model predicts same token regardless of input
print("\n" + "=" * 70)
print("TEST 5: Does output depend on input?")
print("=" * 70)

input1 = jnp.zeros((2, config['sequence_length']), dtype=jnp.int32)
input2 = jnp.ones((2, config['sequence_length']), dtype=jnp.int32) * 500

logits1 = model(hax.named(input1, (Batch, Pos)), key=jrandom.PRNGKey(0), inference=True)
logits2 = model(hax.named(input2, (Batch, Pos)), key=jrandom.PRNGKey(0), inference=True)

diff = np.abs(logits1.array - logits2.array).mean()
print(f"Mean absolute difference in logits for different inputs: {diff:.6f}")

if diff < 0.01:
    print("⚠️  WARNING: Model outputs are nearly identical regardless of input!")
    print("   This suggests the model has collapsed to a degenerate solution.")
else:
    print("✅ Model outputs differ based on input (good)")

print("\n" + "=" * 70)
print("TEST 6: Check actual predictions on real text")
print("=" * 70)

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for the language model."
tokens = tokenizer.encode(test_text)
print(f"Test text: {test_text[:50]}...")
print(f"Tokens: {tokens[:20]}...")

# Pad/truncate to sequence length
if len(tokens) < config['sequence_length'] + 1:
    tokens = tokens + [tokenizer.eos_token_id] * (config['sequence_length'] + 1 - len(tokens))
tokens = tokens[:config['sequence_length'] + 1]

input_ids = jnp.array([tokens[:-1], tokens[:-1]], dtype=jnp.int32)
labels = jnp.array([tokens[1:], tokens[1:]], dtype=jnp.int32)

loss = compute_loss(model, input_ids, labels)
print(f"Real text → Loss: {float(loss):.4f}, PPL: {np.exp(float(loss)):.2f}")

# Check what tokens the model predicts
input_named = hax.named(input_ids, (Batch, Pos))
logits = model(input_named, key=jrandom.PRNGKey(0), inference=True)
predicted_tokens = jnp.argmax(logits.array, axis=-1)[0]

print(f"\nFirst 10 actual tokens:    {tokens[1:11]}")
print(f"First 10 predicted tokens: {list(predicted_tokens[:10])}")

matches = (predicted_tokens[:len(tokens)-1] == jnp.array(tokens[1:])).sum()
print(f"Token match rate: {matches}/{len(tokens)-1} = {matches/(len(tokens)-1)*100:.1f}%")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
