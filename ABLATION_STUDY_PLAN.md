# Ablation Study Plan

## Scientific Question

**Is the gain from time-indexing t? Or just from adding MLP adapter structure?**

## Hypothesis Test

Compare three configurations:
1. **Time-Indexed MLP** (original) - modulation = f(t) where t varies by layer
2. **Constant Modulation** - modulation = f(0) fixed for all layers  
3. **Standard Transformer** - baseline

## Expected Outcomes

### If Time-Indexed ≫ Constant Modulation:
- ✅ Time-dependency (Neural ODE) provides real benefit
- ✅ The "trajectory" of weights over depth matters
- ✅ Validates Neural ODE narrative

### If Time-Indexed ≈ Constant Modulation:
- ⚠️  Gains mostly from MLP adapters, not time-dependency
- ⚠️  "Neural ODE" narrative weakens
- ✅ Still valuable as "Weight Adapter" architecture

### If Constant > Time-Indexed:
- ❌ Time-dependency may be adding noise
- ❌ Simpler fixed modulation is sufficient

## Implementation Strategy

Instead of creating a completely new model class, **modify the existing TimeIndexedTransformer**:

### Option 1: Fixed Time Embedding (Simplest)
```python
# In TimeIndexedAttention.__call__():
# Instead of: time_embed = self.time_emb(layer_idx)
# Use: time_embed = self.time_emb(0.0)  # Always use t=0
```

This tests whether the time-varying nature matters at all.

### Option 2: Learned Constant Scales
Create a variant where scaling factors are learned but not time-dependent.

## Next Steps

1. **Quick Test**: Modify time-indexed model to use fixed t=0
2. **Run Comparison**: Same training setup as existing experiments
3. **Analyze Results**: Compare final losses
4. **Document**: Update README with findings

## Why This Matters

This ablation is the difference between:
- ❌ "Interesting toy experiment"
- ✅ **"Rigorous scientific contribution"**

It addresses the core scientific validity of the approach.

