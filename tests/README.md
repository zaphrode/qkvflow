# Tests

This directory contains tests to validate model implementations and ensure reported metrics match actual code behavior.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test class
pytest tests/test_models.py::TestTimeIndexedMLPTransformer -v

# Run specific test
pytest tests/test_models.py::TestTimeIndexedMLPTransformer::test_parameter_count -v
```

## Test Coverage

### test_models.py

**Purpose:** Validate model architectures, shapes, and parameter counts

**Tests:**
- Model initialization (no crashes)
- Forward pass shape validation
- Parameter count validation (matches README claims)
- Compression ratio validation

**Models tested:**
- Standard Transformer baseline (~308M params)
- Time-Indexed MLP (~0.7M params, 430× compression)
- Time-Indexed SSM (~4.9M params, 63× compression)

## Why These Tests Matter

1. **Reproducibility:** Ensures parameter counts in papers/README match code
2. **Regression detection:** Catches accidental architectural changes
3. **Scientific integrity:** Validates compression ratio claims
4. **Professional polish:** Shows engineering rigor to reviewers

## Future Tests (TODO)

- [ ] Ablation: constant modulation baseline (scientific control)
- [ ] Training convergence tests
- [ ] Numerical stability tests
- [ ] Memory usage profiling

