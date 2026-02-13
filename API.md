# API Documentation

This document describes the core Python API for using the Time-Indexed Parameter Sharing models.

---

## Quick Start

```python
from qkvflow.config.neuralode_ssm_config import Gpt2Config
from qkvflow.models.neuralode_lm import NeuralODELM
from qkvflow.models.neuralode_ssm_lm import NeuralODESSMLM

# See example_usage.py for complete working examples
```

---

## Configuration

### `Gpt2Config`

Configuration class for all models.

**Location:** `qkvflow/config/neuralode_ssm_config.py`

**Key Parameters:**

```python
config = Gpt2Config(
    num_layers=6,              # Number of transformer layers
    hidden_dim=256,            # Hidden dimension (embedding size)
    num_heads=4,               # Number of attention heads
    vocab_size=1000,           # Vocabulary size
    max_position_embeddings=512,  # Maximum sequence length
    use_bias=True,             # Whether to use bias in linear layers
    activation_function="gelu", # Activation function
    attn_pdrop=0.1,            # Attention dropout probability
    resid_pdrop=0.1,           # Residual dropout probability
    embd_pdrop=0.1,            # Embedding dropout probability
)
```

**Properties (auto-computed, don't pass to constructor):**
- `Embed`: Axis for embedding dimension
- `Heads`: Axis for attention heads  
- `HeadSize`: Axis for per-head dimension

---

## Models

### `NeuralODELM` (Time-Indexed MLP)

Language model with time-indexed parameter sharing using MLP blocks.

**Location:** `qkvflow/models/neuralode_lm.py`

**Initialization:**

```python
import jax.random as jrandom

key = jrandom.PRNGKey(0)
model = NeuralODELM.init(config, key=key)
```

**Forward Pass:**

```python
import haliax as hax

# input_ids: NamedArray with axes (Batch, Position)
# Returns: NamedArray with axes (Batch, Position, Vocab)
logits = model(input_ids, t=None, key=key)
```

**Parameters:**
- `input_ids`: Token IDs as `hax.NamedArray` with shape `(Batch, Position)`
- `t`: Optional time parameter (for Neural ODE integration, default: None)
- `key`: JAX random key for dropout

**Compute Loss:**

```python
# target_ids: NamedArray with axes (Batch, Position)
# Returns: Scalar loss value
loss = model.compute_loss(input_ids, target_ids, t=None, key=key)
```

**Architecture:**
- Shared base weight matrices across all layers
- Lightweight time-modulation network (64 hidden units)
- Standard multi-head attention with time-indexed weights
- MLP feed-forward with time-indexed weights

**Typical Size:** ~0.7M parameters (for hidden_dim=256, 6 layers)

---

### `NeuralODESSMLM` (Time-Indexed SSM)

Language model with time-indexed parameter sharing using Mamba-style SSM blocks.

**Location:** `qkvflow/models/neuralode_ssm_lm.py`

**Initialization:**

```python
key = jrandom.PRNGKey(0)
model = NeuralODESSMLM.init(config, key=key)
```

**Usage:** Same API as `NeuralODELM`

```python
logits = model(input_ids, t=None, key=key)
loss = model.compute_loss(input_ids, target_ids, t=None, key=key)
```

**Architecture:**
- Shared base SSM matrices across all layers
- Selective state space model (Mamba-style)
- Time-indexed parameter modulation
- More parameters than MLP variant due to SSM state matrices

**Typical Size:** ~4.9M parameters (for hidden_dim=256, 6 layers)

---

## Data Format

All models expect inputs as **Haliax NamedArrays**.

### Input IDs

```python
import haliax as hax
import jax.numpy as jnp

# Define axes
Batch = hax.Axis("batch", batch_size)
Position = hax.Axis("position", seq_len)

# Create named array
tokens = jnp.array([[1, 2, 3, ...], ...])  # Shape: (batch_size, seq_len)
input_ids = hax.named(tokens, (Batch, Position))
```

### Target IDs

Same format as input IDs. Typically, targets are input IDs shifted by one position:

```python
targets = jnp.roll(tokens, -1, axis=1)
target_ids = hax.named(targets, (Batch, Position))
```

---

## Training

### Basic Training Loop

```python
import optax
import equinox as eqx

# Initialize
model = NeuralODELM.init(config, key=model_key)
optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Training step
@eqx.filter_jit
def train_step(model, opt_state, batch, key):
    input_ids, target_ids = batch
    
    def loss_fn(model):
        return model.compute_loss(input_ids, target_ids, t=None, key=key)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss

# Run training
for step in range(num_steps):
    batch = next(data_iterator)
    model, opt_state, loss = train_step(model, opt_state, batch, key)
    print(f"Step {step}, Loss: {loss:.4f}")
```

See `example_usage.py` for complete working example.

---

## Utilities

### Parameter Counting

```python
import jax
import equinox as eqx

num_params = sum(
    x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
)
print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
```

### Model Inspection

```python
# Print model structure
print(model)

# Access specific components
embedding = model.transformer.wte  # Word token embeddings
position_embedding = model.transformer.wpe  # Position embeddings
blocks = model.transformer.blocks  # Time-indexed blocks
lm_head = model.lm_head  # Output projection
```

---

## Comparison Scripts

### Compare All Models

```python
# Run from command line
python scripts/compare_vs_tong_neuralode.py
```

This compares:
- Standard Transformer
- Tong's Neural ODE
- Time-Indexed MLP
- Time-Indexed SSM

**Output:** Prints loss values and saves results to `tong_comparison_results.pkl`

### Statistical Validation

```python
# Run from command line
python scripts/run_5_seed_validation.py
```

Runs all models with 5 different random seeds.

**Output:**
- `statistical_validation_results/statistics_summary.json`
- `statistical_validation_results/significance_tests.json`
- Individual `.pkl` files for each seed

### Generate Figures

```python
# Run from command line
python scripts/plot_statistical_results.py
```

**Output:** Publication-quality figures in `publication_figures/`

---

## File Organization

```
qkvflow/
├── qkvflow/
│   ├── models/
│   │   ├── neuralode_lm.py         # Time-Indexed MLP
│   │   └── neuralode_ssm_lm.py     # Time-Indexed SSM
│   ├── nn/                          # Core neural network modules
│   │   ├── dynamic.py               # Time-indexed layers
│   │   └── ...
│   └── config/
│       └── neuralode_ssm_config.py  # Configuration
├── scripts/
│   ├── compare_vs_tong_neuralode.py  # Main comparison
│   ├── run_5_seed_validation.py      # Statistical validation
│   └── plot_statistical_results.py   # Figure generation
└── example_usage.py                  # Simple Python examples
```

---

## Dependencies

**Core:**
- JAX 0.4.28+ (with CUDA for GPU)
- Equinox 0.11.4+ (for PyTree models)
- Haliax 1.3+ (for named arrays)

**Training:**
- Optax (for optimizers)

**Visualization:**
- Matplotlib, Seaborn

**Full list:** See `requirements.txt`

---

## Known Issues & Limitations

1. **Haliax dependency:** Named arrays require understanding Haliax API
2. **JAX-specific:** Models use JAX functional programming style
3. **No checkpointing API:** Manual save/load required
4. **Limited documentation:** Some internal modules lack docstrings
5. **No type hints:** Many functions missing type annotations

---

## Examples

### Complete Working Examples

See `example_usage.py` for:
- Creating toy datasets
- Initializing models
- Running forward passes
- Computing losses
- Training steps

Run with:
```bash
python example_usage.py
```

---

## Support

**Issues:** Please open a GitHub issue for:
- Bugs in the code
- API questions
- Feature requests

**Not supported:**
- Production deployments (research code only)
- Custom architectures (exploratory work)
- Large-scale training (not tested beyond small models)

---

*Last updated: December 2025*

