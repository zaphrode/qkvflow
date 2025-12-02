"""
Test model architectures: shape validation and parameter counts.

This ensures that reported parameter counts in the README match actual implementations.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from haliax import Axis
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.test_time_indexed_weights import (
    TimeIndexedTransformer,
    StandardTransformer,
)
from scripts.test_time_indexed_ssm import TimeIndexedSSMTransformer


# Test configuration matching the paper experiments
class TestConfig:
    """Configuration matching WikiText-2/103 experiments."""
    hidden_size = 256
    num_layers = 6
    num_heads = 4
    seq_len = 128
    vocab_size = 256  # Character-level
    batch_size = 8
    head_dim = hidden_size // num_heads


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture
def key():
    """Provide random key for model initialization."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def batch_axis(test_config):
    """Create batch axis."""
    return Axis(name="batch", size=test_config.batch_size)


@pytest.fixture
def position_axis(test_config):
    """Create position axis."""
    return Axis(name="position", size=test_config.seq_len)


@pytest.fixture
def embed_axis(test_config):
    """Create embedding axis."""
    return Axis(name="embed", size=test_config.hidden_size)


@pytest.fixture
def vocab_axis(test_config):
    """Create vocabulary axis."""
    return Axis(name="vocab", size=test_config.vocab_size)


@pytest.fixture
def sinusoidal_dim_axis():
    """Create sinusoidal dimension axis."""
    return Axis(name="sinusoidal_dim", size=64)


@pytest.fixture
def tembed_dim_axis():
    """Create time embedding dimension axis."""
    return Axis(name="tembed_dim", size=256)


def count_parameters(model):
    """Count total trainable parameters in a model."""
    params, _ = eqx.partition(model, eqx.is_array)
    
    def count_leaves(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))
    
    return count_leaves(params)


class TestStandardTransformer:
    """Test Standard Transformer baseline."""
    
    def test_initialization(self, test_config, key, batch_axis, position_axis, 
                           embed_axis, vocab_axis):
        """Test that Standard Transformer initializes without errors."""
        model = StandardTransformer(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            key=key,
        )
        assert model is not None
    
    def test_forward_pass_shape(self, test_config, key, batch_axis, position_axis,
                                embed_axis, vocab_axis):
        """Test that forward pass produces correct output shape."""
        model = StandardTransformer(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            key=key,
        )
        
        # Create dummy input
        x = jnp.zeros((test_config.batch_size, test_config.seq_len), dtype=jnp.int32)
        
        # Forward pass
        output = model(x, batch_axis, position_axis)
        
        # Check output shape: should be (batch, position, vocab)
        assert output.shape == (test_config.batch_size, test_config.seq_len, 
                               test_config.vocab_size)
    
    def test_parameter_count(self, test_config, key, batch_axis, position_axis,
                            embed_axis, vocab_axis):
        """Test that parameter count is reasonable for standard transformer."""
        model = StandardTransformer(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            key=key,
        )
        
        param_count = count_parameters(model)
        
        # Standard transformer should have ~300M params with this config
        # (from paper: 308.5M)
        # Allow 20% tolerance for implementation details
        expected = 308.5e6
        tolerance = 0.20
        
        assert param_count > expected * (1 - tolerance), \
            f"Expected ~{expected:.1e} params, got {param_count:.1e}"
        assert param_count < expected * (1 + tolerance), \
            f"Expected ~{expected:.1e} params, got {param_count:.1e}"


class TestTimeIndexedMLPTransformer:
    """Test Time-Indexed MLP variant."""
    
    def test_initialization(self, test_config, key, batch_axis, position_axis,
                           embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that Time-Indexed MLP initializes without errors."""
        model = TimeIndexedTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        assert model is not None
    
    def test_forward_pass_shape(self, test_config, key, batch_axis, position_axis,
                                embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that forward pass produces correct output shape."""
        model = TimeIndexedTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        
        # Create dummy input
        x = jnp.zeros((test_config.batch_size, test_config.seq_len), dtype=jnp.int32)
        
        # Forward pass
        output = model(x, batch_axis, position_axis)
        
        # Check output shape
        assert output.shape == (test_config.batch_size, test_config.seq_len,
                               test_config.vocab_size)
    
    def test_parameter_count(self, test_config, key, batch_axis, position_axis,
                            embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that parameter count matches reported value (0.7M)."""
        model = TimeIndexedTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        
        param_count = count_parameters(model)
        
        # Paper reports 0.7M params for Time-Indexed MLP
        expected = 0.7e6
        tolerance = 0.20  # 20% tolerance
        
        assert param_count > expected * (1 - tolerance), \
            f"Expected ~{expected:.1e} params, got {param_count:.1e}"
        assert param_count < expected * (1 + tolerance), \
            f"Expected ~{expected:.1e} params, got {param_count:.1e}"
    
    def test_compression_ratio(self, test_config, key, batch_axis, position_axis,
                               embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that compression ratio vs standard is approximately 430×."""
        standard_model = StandardTransformer(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            key=key,
        )
        
        time_indexed_model = TimeIndexedTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        
        standard_params = count_parameters(standard_model)
        time_indexed_params = count_parameters(time_indexed_model)
        
        compression_ratio = standard_params / time_indexed_params
        
        # Paper reports ~430× compression
        expected_ratio = 430
        tolerance = 0.25  # 25% tolerance
        
        assert compression_ratio > expected_ratio * (1 - tolerance), \
            f"Expected ~{expected_ratio}× compression, got {compression_ratio:.1f}×"
        assert compression_ratio < expected_ratio * (1 + tolerance), \
            f"Expected ~{expected_ratio}× compression, got {compression_ratio:.1f}×"


class TestTimeIndexedSSMTransformer:
    """Test Time-Indexed SSM variant."""
    
    def test_initialization(self, test_config, key, batch_axis, position_axis,
                           embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that Time-Indexed SSM initializes without errors."""
        model = TimeIndexedSSMTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        assert model is not None
    
    def test_forward_pass_shape(self, test_config, key, batch_axis, position_axis,
                                embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that forward pass produces correct output shape."""
        model = TimeIndexedSSMTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        
        # Create dummy input
        x = jnp.zeros((test_config.batch_size, test_config.seq_len), dtype=jnp.int32)
        
        # Forward pass
        output = model(x, batch_axis, position_axis)
        
        # Check output shape
        assert output.shape == (test_config.batch_size, test_config.seq_len,
                               test_config.vocab_size)
    
    def test_parameter_count(self, test_config, key, batch_axis, position_axis,
                            embed_axis, vocab_axis, sinusoidal_dim_axis, tembed_dim_axis):
        """Test that parameter count matches reported value (4.9M)."""
        model = TimeIndexedSSMTransformer.init(
            Vocab=vocab_axis,
            Embed=embed_axis,
            Heads=Axis(name="heads", size=test_config.num_heads),
            HeadSize=Axis(name="head_size", size=test_config.head_dim),
            Layers=test_config.num_layers,
            SinusodialDim=sinusoidal_dim_axis,
            TembedDim=tembed_dim_axis,
            key=key,
        )
        
        param_count = count_parameters(model)
        
        # Paper reports 4.9M params for Time-Indexed SSM
        expected = 4.9e6
        tolerance = 0.20  # 20% tolerance
        
        assert param_count > expected * (1 - tolerance), \
            f"Expected ~{expected:.1e} params, got {param_count:.1e}"
        assert param_count < expected * (1 + tolerance), \
            f"Expected ~{expected:.1e} params, got {param_count:.1e}"


if __name__ == "__main__":
    # Allow running directly with python for quick testing
    print("Run with: pytest tests/test_models.py -v")
    print("\nTo run specific test:")
    print("  pytest tests/test_models.py::TestTimeIndexedMLPTransformer::test_parameter_count -v")

