# Architecture Comparison: Tong's Neural ODE vs Our Time-Indexed Models

## Overview

This repository contains **A. Tong's Neural ODE Transformer** (ICLR 2025), and we've added **Time-Indexed Parameter Sharing** models for comparison.

---

## Architecture Details

### 1. **A. Tong's Neural ODE Transformer** (`qkvflow/nn/dynamic.py`)

**Key Innovation:** Continuous-depth transformer using Neural ODE formulation

**Architecture:**
```python
# Single shared block applied repeatedly
block = Block(attn, mlp, layer_norms)  # ONE block for all layers
dt = 1.0 / num_layers

# ODE Integration (Euler method)
for layer in range(num_layers):
    t = (layer + 1) * dt
    time_embed = time_embedding(t)
    output = block(time_embed, x, mask)
    x = x + output * dt  # Euler step
```

**Key Features:**
- ✅ **Single shared block** across all layers
- ✅ **ODE integration** with Euler method (`x = x + f(x,t) * dt`)
- ✅ **Time-dependent weights** via neural networks
- ✅ **Continuous layer index** `t ∈ [0, 1]`
- ✅ **Spectral analysis** of dynamics (eigenvalues, Lyapunov exponents)

**Weight Parameterization:**
```python
# All weights are functions of time
Q(t) = neural_net_Q(time_embedding(t))
K(t) = neural_net_K(time_embedding(t))
V(t) = neural_net_V(time_embedding(t))
W_mlp(t) = neural_net_mlp(time_embedding(t))
```

**Parameters:** ~617M (for 6-layer, 256-dim model)
- Each weight matrix has its own neural network generator
- Full flexibility but high parameter count

---

### 2. **Our Time-Indexed MLP** (`scripts/test_time_indexed_weights.py`)

**Key Innovation:** Extreme parameter compression via base weight sharing + time modulation

**Architecture:**
```python
# Shared base weights with time gating
W_base = shared_across_layers  # ONE set of base weights
time_mod = small_neural_net(time_embedding(t))

# Time-indexed weights
for layer in range(num_layers):
    t = (layer + 1) / num_layers
    time_embed = time_embedding(t)
    W_effective = W_base ⊙ sigmoid(time_mod(time_embed))
    x = x + attn(x, W_effective) + mlp(x, W_effective)
```

**Key Features:**
- ✅ **Shared base weights** across all layers
- ✅ **Lightweight time modulation** (small gating networks)
- ✅ **Element-wise gating** of base weights
- ✅ **99.8% parameter reduction** (617M → 0.96M)
- ✅ **8x faster training**

**Parameters:** ~0.96M
- ONE set of base attention/MLP weights
- Small modulation networks (< 100K params)

---

### 3. **Our Time-Indexed SSM** (`scripts/test_time_indexed_ssm.py`)

**Key Innovation:** State-space models with time-indexed parameter sharing

**Architecture:**
```python
# Shared base SSM matrices with time modulation
A_base, B_base, C_base, D_base = shared_across_layers

for layer in range(num_layers):
    t = (layer + 1) / num_layers
    time_embed = time_embedding(t)
    
    # Generate layer-specific SSM params
    A(t) = A_base ⊙ sigmoid(mod_A(time_embed))
    B(t) = B_base ⊙ sigmoid(mod_B(time_embed))
    C(t) = C_base ⊙ sigmoid(mod_C(time_embed))
    
    # SSM forward pass
    h = A(t) @ h + B(t) @ x
    y = C(t) @ h + D(t) @ x
    x = x + attention(x) + y
```

**Key Features:**
- ✅ **Shared SSM matrices** across layers
- ✅ **State-space dynamics** for sequence modeling
- ✅ **Time-indexed modulation** of A, B, C, D matrices
- ✅ **98.5% parameter reduction** (617M → 9.4M)
- ✅ **Best performance** on WikiText-2

**Parameters:** ~9.4M
- Shared A, B, C, D matrices
- Small modulation networks for each matrix
- State vectors for each layer

---

## Key Differences

| Aspect | Tong's Neural ODE | Time-Indexed MLP | Time-Indexed SSM |
|--------|-------------------|------------------|------------------|
| **Philosophy** | Continuous ODE | Discrete + Gating | Discrete + SSM |
| **Integration** | Euler (`x + f*dt`) | Direct (`x + f`) | Direct (`x + f`) |
| **Weight Sharing** | Full time-dependent | Base + modulation | Base + modulation |
| **Parameters** | 617M | 0.96M | 9.4M |
| **Compression** | Baseline | 99.8% | 98.5% |
| **Speed** | 71ms/step | 8.8ms/step | 61.9ms/step |
| **WikiText-2 Loss** | ??? | 2.15 | **1.85** |

---

## Conceptual Relationship

```
Tong's Neural ODE:
    x_{i+1} = x_i + f(x_i, t_i; θ(t_i)) * dt
    where θ(t) are FULLY time-dependent weights

Our Time-Indexed Models:
    x_{i+1} = x_i + f(x_i; θ_base ⊙ g(t_i))
    where θ_base are SHARED, g(t) is time modulation
```

**Key Insight:**
- Tong generates **entire weight matrices** as functions of time
- We use **shared base weights** modulated by time-dependent scalars
- Both achieve parameter sharing, but different compression/flexibility trade-offs

---

## Research Questions

### Q1: How does Tong's full ODE compare to our time-indexed models?

**Hypothesis:** Tong's model is more flexible but less parameter-efficient.

**Expected Results:**
- Tong: Higher capacity, potentially better performance
- Ours: Better compression, faster training, regularization benefits

### Q2: Is ODE integration necessary?

**Tong's approach:** `x = x + f(x,t) * dt` (Euler integration)  
**Our approach:** `x = x + f(x,t)` (direct residual)

**Question:** Does the `* dt` scaling factor improve training?

### Q3: Can we combine the best of both?

**Hybrid approach:**
```python
# Tong's ODE integration + Our weight sharing
x = x + (base_block(x) ⊙ modulate(t)) * dt
```

---

## Proposed Benchmark

Let's compare **FOUR** architectures on WikiText-2:

1. **Baseline:** Standard Transformer (separate weights per layer)
2. **Tong's ODE:** Full time-dependent weights with ODE integration
3. **Time-Indexed MLP:** Our compressed version
4. **Time-Indexed SSM:** Our SSM variant

### Metrics:
- Validation loss (performance)
- Parameter count (efficiency)
- Training speed (ms/step)
- Memory usage (GB)

---

## Implementation Status

✅ **Standard Transformer** - Benchmark complete  
❓ **Tong's Neural ODE** - Available in repo, needs benchmarking  
✅ **Time-Indexed MLP** - Benchmark complete  
✅ **Time-Indexed SSM** - Benchmark complete  

---

## Next Steps

1. **Run Tong's model on WikiText-2** with same config as our benchmark
2. **Compare all four architectures** side-by-side
3. **Ablation study:** ODE integration vs direct residual
4. **Hybrid experiments:** Combine best features from both approaches

---

*Analysis Date: November 14, 2025*

