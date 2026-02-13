"""
Simple example of using the Time-Indexed Parameter Sharing models.

This demonstrates how to instantiate and train the models without notebooks.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
from jaxtyping import PRNGKeyArray

from qkvflow.config.neuralode_ssm_config import Gpt2Config
from qkvflow.models.neuralode_lm import NeuralODELM
from qkvflow.models.neuralode_ssm_lm import NeuralODESSMLM


def create_toy_dataset(
    vocab_size: int = 1000,
    seq_len: int = 128,
    batch_size: int = 4,
    key: PRNGKeyArray = None
) -> tuple[hax.NamedArray, hax.NamedArray]:
    """Create a simple toy dataset for testing."""
    if key is None:
        key = jrandom.PRNGKey(42)
    
    # Define axes
    Batch = hax.Axis("batch", batch_size)
    Pos = hax.Axis("position", seq_len)
    
    # Generate random tokens
    tokens = jrandom.randint(key, (batch_size, seq_len), 0, vocab_size)
    input_ids = hax.named(tokens, (Batch, Pos))
    
    # Targets are inputs shifted by one
    targets = jnp.roll(tokens, -1, axis=1)
    target_ids = hax.named(targets, (Batch, Pos))
    
    return input_ids, target_ids


def example_time_indexed_mlp():
    """Example: Time-Indexed MLP model."""
    print("=" * 60)
    print("Example 1: Time-Indexed MLP (0.7M parameters)")
    print("=" * 60)
    
    # Configuration
    config = Gpt2Config(
        num_layers=6,
        hidden_dim=256,
        num_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
        use_bias=True,
        activation_function="gelu",
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
    )
    
    # Initialize model
    key = jrandom.PRNGKey(0)
    model_key, data_key = jrandom.split(key)
    
    print(f"\nConfig: {config.num_layers} layers, {config.hidden_dim} hidden dim")
    print("Initializing Time-Indexed MLP model...")
    
    model = NeuralODELM.init(config, key=model_key)
    
    # Create toy data
    input_ids, target_ids = create_toy_dataset(
        vocab_size=config.vocab_size,
        seq_len=128,
        batch_size=4,
        key=data_key
    )
    
    # Forward pass
    print("\nRunning forward pass...")
    logits_key = jrandom.PRNGKey(1)
    logits = model(input_ids, t=None, key=logits_key)
    
    print(f"Input shape: {input_ids.axes}")
    print(f"Output shape: {logits.axes}")
    
    # Compute loss
    print("\nComputing loss...")
    loss_key = jrandom.PRNGKey(2)
    loss = model.compute_loss(input_ids, target_ids, t=None, key=loss_key)
    
    print(f"Loss: {loss:.4f}")
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"\nTotal parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return model, input_ids, target_ids


def example_time_indexed_ssm():
    """Example: Time-Indexed SSM model."""
    print("\n" + "=" * 60)
    print("Example 2: Time-Indexed SSM (4.9M parameters)")
    print("=" * 60)
    
    # Configuration
    config = Gpt2Config(
        num_layers=6,
        hidden_dim=256,
        num_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
        use_bias=True,
        activation_function="gelu",
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
    )
    
    # Initialize model
    key = jrandom.PRNGKey(0)
    model_key, data_key = jrandom.split(key)
    
    print(f"\nConfig: {config.num_layers} layers, {config.hidden_dim} hidden dim")
    print("Initializing Time-Indexed SSM model...")
    
    model = NeuralODESSMLM.init(config, key=model_key)
    
    # Create toy data
    input_ids, target_ids = create_toy_dataset(
        vocab_size=config.vocab_size,
        seq_len=128,
        batch_size=4,
        key=data_key
    )
    
    # Forward pass
    print("\nRunning forward pass...")
    logits_key = jrandom.PRNGKey(1)
    logits = model(input_ids, t=None, key=logits_key)
    
    print(f"Input shape: {input_ids.axes}")
    print(f"Output shape: {logits.axes}")
    
    # Compute loss
    print("\nComputing loss...")
    loss_key = jrandom.PRNGKey(2)
    loss = model.compute_loss(input_ids, target_ids, t=None, key=loss_key)
    
    print(f"Loss: {loss:.4f}")
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"\nTotal parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return model, input_ids, target_ids


def example_training_step():
    """Example: Single training step with optimizer."""
    print("\n" + "=" * 60)
    print("Example 3: Training Step")
    print("=" * 60)
    
    import optax
    
    # Create model and data
    config = Gpt2Config(
        num_layers=6,
        hidden_dim=256,
        num_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
    )
    
    key = jrandom.PRNGKey(42)
    model_key, data_key, train_key = jrandom.split(key, 3)
    
    model = NeuralODELM.init(config, key=model_key)
    input_ids, target_ids = create_toy_dataset(key=data_key)
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training step
    @eqx.filter_jit
    def train_step(model, opt_state, input_ids, target_ids, key):
        def loss_fn(model):
            return model.compute_loss(input_ids, target_ids, t=None, key=key)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    print("Running training step...")
    model, opt_state, loss = train_step(model, opt_state, input_ids, target_ids, train_key)
    
    print(f"Initial loss: {loss:.4f}")
    print("✓ Training step completed successfully!")
    
    return model, opt_state


if __name__ == "__main__":
    print("\n" + "🚀 Time-Indexed Parameter Sharing Examples")
    print("=" * 60)
    
    # Example 1: MLP variant
    mlp_model, inputs, targets = example_time_indexed_mlp()
    
    # Example 2: SSM variant
    ssm_model, inputs, targets = example_time_indexed_ssm()
    
    # Example 3: Training
    trained_model, opt_state = example_training_step()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Modify config for your use case")
    print("  2. Use your own dataset")
    print("  3. Run full training with scripts/compare_vs_tong_neuralode.py")
    print()

