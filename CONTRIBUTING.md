# Contributing Guidelines

Thank you for your interest in this project!

---

## ⚠️ Important Context

This is a **personal research repository** for exploratory work extending Tong et al.'s Neural ODE Transformers.

**Current status:**
- Work in progress
- Small-scale experiments only
- No formal release or versioning
- Code quality varies (research-grade, not production)

**What this means:**
- Contributions welcome, but may not be immediately reviewed
- Breaking changes may occur without notice
- No stability guarantees

---

## 🐛 Reporting Issues

### Before Opening an Issue

1. **Check existing issues** to avoid duplicates
2. **Verify the issue** with the latest code from `main`
3. **Provide context** about your setup (Python version, JAX version, GPU/CPU)

### Bug Reports Should Include

- **Description:** Clear description of the bug
- **Reproduction:** Minimal code to reproduce the issue
- **Expected vs Actual:** What you expected vs what happened
- **Environment:**
  - Python version
  - JAX version
  - GPU/CPU
  - OS

**Example:**

```
### Bug Description
`NeuralODELM.compute_loss()` crashes with `KeyError: 'Vocab'` when batch_size=1

### Reproduction
```python
config = Gpt2Config(vocab_size=1000, hidden_dim=256, num_layers=6)
model = NeuralODELM.init(config, key=jrandom.PRNGKey(0))
# ... (minimal code)
```

### Environment
- Python 3.11
- JAX 0.4.28 (CUDA 12.1)
- Ubuntu 22.04
```

---

## 🔧 Code Contributions

### Small Fixes (Typos, Docs, Minor Bugs)

1. Fork the repository
2. Create a branch: `git checkout -b fix/typo-in-readme`
3. Make your changes
4. Push and create a Pull Request
5. Describe what you fixed

**These are always welcome!**

### Larger Changes (Features, Refactoring)

**Please open an issue first** to discuss:
- What you want to change
- Why it's needed
- How you plan to implement it

This avoids wasted effort if the direction doesn't fit the project goals.

---

## 📝 Code Style

### Current State (Not Ideal)

The codebase currently:
- ❌ Has minimal type hints
- ❌ Has sparse docstrings
- ❌ No automated linting (black, ruff, etc.)
- ❌ No CI/CD
- ❌ No unit tests

This is research code prioritizing **experimentation speed** over **engineering rigor**.

### If Contributing Code

**Please:**
- Follow existing code patterns
- Add docstrings for public functions
- Comment non-obvious logic
- Keep changes focused and minimal

**Don't worry about:**
- Perfect type coverage
- 100% documentation
- Setting up CI/CD
- Writing tests (though appreciated if you do!)

### Example of Good Docstring

```python
def compute_loss(self, input_ids, target_ids, t=None, key=None):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        input_ids: Token IDs, NamedArray with shape (Batch, Position)
        target_ids: Target token IDs, same shape as input_ids
        t: Optional time parameter for Neural ODE (default: None)
        key: JAX random key for dropout
        
    Returns:
        Scalar loss value (float)
    """
    # ... implementation
```

---

## 🧪 Testing

### Current State

- No formal test suite
- Manual testing via `example_usage.py` and comparison scripts
- Integration tests in `scripts/` that run full experiments

### If Adding Tests

Appreciated but not required! If you want to add tests:

```python
# tests/test_models.py (example structure)
import pytest
import jax.random as jrandom
from qkvflow.models.neuralode_lm import NeuralODELM
from qkvflow.config.neuralode_ssm_config import Gpt2Config

def test_model_forward_pass():
    """Test that model forward pass runs without errors."""
    config = Gpt2Config(vocab_size=100, hidden_dim=64, num_layers=2)
    model = NeuralODELM.init(config, key=jrandom.PRNGKey(0))
    # ... assertions
```

---

## 📚 Documentation

### What Needs Documentation

- [x] README (updated)
- [x] API.md (basic coverage)
- [x] example_usage.py (working examples)
- [ ] Internal module docstrings (sparse)
- [ ] Architecture diagrams (would be nice!)
- [ ] Video walkthrough (future)

### How to Contribute Documentation

- **Clarifications:** Open a PR directly
- **New guides:** Open an issue first to discuss scope
- **Typos/fixes:** Just open a PR

---

## 🔬 Research Contributions

### Reproducibility Issues

If you can't reproduce the reported results:
1. Open an issue with your setup and results
2. Share your exact command/script
3. Include random seeds you tried

### New Experiments

If you run new experiments (e.g., different datasets, architectures):
1. **Share your findings** in an issue or discussion
2. **Include details:** config, dataset, random seeds, results
3. **Consider a PR** if you have clean, reproducible code

---

## 🚫 What NOT to Contribute

**Please don't:**
- Add heavy dependencies (keep it lightweight)
- Refactor large portions without discussion
- Add unrelated features
- Change core architecture without strong justification
- Add production-focused tooling (Docker, k8s, etc.) - this is research code

---

## 📧 Questions?

- **General questions:** Open a GitHub issue
- **Research discussions:** Also GitHub issues (use "Discussion" label)
- **Private matters:** Contact repository owner directly

---

## 🙏 Acknowledgment

All contributors will be acknowledged in:
- The README
- Future paper acknowledgments (if applicable)
- Release notes (if we ever do formal releases)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License (same as the rest of the project).

---

*This is a living document - suggestions for improvement welcome!*

