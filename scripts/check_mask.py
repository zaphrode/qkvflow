#!/usr/bin/env python3
"""Check if causal mask is applied correctly"""

import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn

Pos = hax.Axis("position", 8)
KeyPos = Pos.alias("key_position")

print("Testing different ways to create causal mask:")
print("=" * 60)

# Method 1: Same axis twice
print("\n1. hnn.attention.causal_mask(Pos, Pos):")
mask1 = hnn.attention.causal_mask(Pos, Pos)
print(f"   Shape: {mask1.array.shape}")
print(f"   Axes: {mask1.axes}")

# Method 2: Different axes (query_pos, key_pos)
print("\n2. hnn.attention.causal_mask(Pos, KeyPos):")
mask2 = hnn.attention.causal_mask(Pos, KeyPos)
print(f"   Shape: {mask2.array.shape}")
print(f"   Axes: {mask2.axes}")
print(f"   Values:\n{mask2.array.astype(int)}")

# Check what the mask means
print("\n" + "=" * 60)
print("Interpreting the mask:")
print("=" * 60)
print("For causal attention, position i should only see positions j <= i")
print("The mask should be a lower triangular matrix (including diagonal)")

if len(mask2.array.shape) == 2:
    print(f"\nmask2[0,0] (pos 0 sees pos 0): {mask2.array[0,0]}")
    print(f"mask2[0,7] (pos 0 sees pos 7): {mask2.array[0,7]}")
    print(f"mask2[7,0] (pos 7 sees pos 0): {mask2.array[7,0]}")
    print(f"mask2[7,7] (pos 7 sees pos 7): {mask2.array[7,7]}")
    
    # Check if it's lower triangular
    is_lower_tri = jnp.allclose(mask2.array, jnp.tril(jnp.ones((8,8))))
    is_upper_tri = jnp.allclose(mask2.array, jnp.triu(jnp.ones((8,8))))
    print(f"\nIs lower triangular (correct): {is_lower_tri}")
    print(f"Is upper triangular (inverted): {is_upper_tri}")

print("\n" + "=" * 60)
print("Testing hax.where behavior:")
print("=" * 60)

# Simulate attention scores
attn_scores = hax.ones((Pos, KeyPos))
print(f"\nOriginal attention scores (all 1s)")

# Apply mask using hax.where
# hax.where(condition, x, y) returns x where condition is True, y otherwise
masked = hax.where(mask2, attn_scores, hax.full((Pos, KeyPos), -1e9))
print(f"\nhax.where(mask, attn, -1e9):")
print(masked.array)

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
if len(mask2.array.shape) == 2 and mask2.array[0,7] == 0:
    print("✅ Mask is correct: True where attention is allowed")
    print("   Using hax.where(mask, attn, -inf) is CORRECT")
elif len(mask2.array.shape) == 2 and mask2.array[0,7] == 1:
    print("⚠️  Mask is inverted: True where attention should be BLOCKED")
    print("   Need to use hax.where(mask, -inf, attn) instead")
else:
    print("❌ Mask has unexpected shape or values")
