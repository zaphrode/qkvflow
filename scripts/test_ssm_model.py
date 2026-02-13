#!/usr/bin/env python3
"""
Quick test script to verify Neural ODE + SSM model works correctly

Usage:
    python scripts/test_ssm_model.py
"""

# Add parent directory to path BEFORE any local imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import everything else
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
import equinox as eqx
from levanter.models.gpt2 import Gpt2Config
from qkvflow.models.neuralode_ssm_lm import NeuralOdeSSMLMHeadModel, count_parameters
from config.neuralode_ssm_config import NeuralOdeSSMConfig


def test_forward_pass():
    """Test basic forward pass"""
    print("\n" + "="*60)
    print("Test 1: Forward Pass")
    print("="*60)

    # Create small config
    config = NeuralOdeSSMConfig.small_ssm()

    # Initialize model
    key = jrandom.PRNGKey(42)
    k_model, k_forward = jrandom.split(key)

    print("\nInitializing model...")
    # Create Vocab axis
    Vocab = hax.Axis("vocab", config.vocab_size)

    model = NeuralOdeSSMLMHeadModel.init(
        Vocab,
        config.gpt2_config,
        time_embedding_dim=config.time_embedding_dim,
        sinusoidal_dim=config.sinusoidal_dim,
        ssm_state_size=config.ssm_state_size,
        key=k_model
    )

    print("✓ Model initialized successfully")

    # Print parameter counts
    param_counts = count_parameters(model)
    print("\nParameter counts:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # Create dummy input
    batch_size = 4
    seq_len = config.gpt2_config.seq_len

    Batch = hax.Axis("batch", batch_size)
    input_ids = jrandom.randint(
        k_forward, (batch_size, seq_len),
        0, config.vocab_size
    )
    input_ids = hax.named(input_ids, (Batch, config.gpt2_config.Pos))

    print(f"\nInput shape: batch={batch_size}, seq_len={seq_len}")

    # Forward pass
    print("Running forward pass...")
    logits = model(input_ids, t=1.0, key=k_forward)

    print(f"✓ Forward pass successful")
    print(f"  Output axes: {logits.axes}")
    print(f"  Output shape: {logits.array.shape}")
    print(
        f"  Logits range: [{jnp.min(logits.array):.2f}, {jnp.max(logits.array):.2f}]")

    return True


def test_loss_computation():
    """Test loss computation"""
    print("\n" + "="*60)
    print("Test 2: Loss Computation")
    print("="*60)

    config = NeuralOdeSSMConfig.small_ssm()

    key = jrandom.PRNGKey(123)
    k_model, k_loss = jrandom.split(key)

    print("\nInitializing model...")
    Vocab = hax.Axis("vocab", config.vocab_size)
    model = NeuralOdeSSMLMHeadModel.init(
        Vocab,
        config.gpt2_config,
        time_embedding_dim=config.time_embedding_dim,
        sinusoidal_dim=config.sinusoidal_dim,
        ssm_state_size=config.ssm_state_size,
        key=k_model
    )

    # Create dummy data
    batch_size = 2
    Batch = hax.Axis("batch", batch_size)

    input_ids = jrandom.randint(
        k_loss, (batch_size, config.gpt2_config.seq_len),
        0, config.vocab_size
    )
    input_ids = hax.named(input_ids, (Batch, config.gpt2_config.Pos))

    # Targets are shifted by 1
    targets = jnp.roll(input_ids.array, shift=-1, axis=-1)
    targets = hax.named(targets, (Batch, config.gpt2_config.Pos))

    print("Computing loss...")
    loss = model.compute_loss(input_ids, targets, t=1.0, key=k_loss)

    print(f"✓ Loss computation successful")
    print(f"  Loss value: {loss:.4f}")
    print(f"  Loss is finite: {jnp.isfinite(loss)}")
    print(f"  Loss is positive: {loss > 0}")

    return jnp.isfinite(loss) and loss > 0


def test_gradient_computation():
    """Test gradient computation"""
    print("\n" + "="*60)
    print("Test 3: Gradient Computation")
    print("="*60)

    config = NeuralOdeSSMConfig.small_ssm()

    key = jrandom.PRNGKey(456)
    k_model, k_grad = jrandom.split(key)

    print("\nInitializing model...")
    Vocab = hax.Axis("vocab", config.vocab_size)
    model = NeuralOdeSSMLMHeadModel.init(
        Vocab,
        config.gpt2_config,
        time_embedding_dim=config.time_embedding_dim,
        sinusoidal_dim=config.sinusoidal_dim,
        ssm_state_size=config.ssm_state_size,
        key=k_model
    )

    # Create dummy data
    batch_size = 2
    Batch = hax.Axis("batch", batch_size)

    input_ids = jrandom.randint(
        k_grad, (batch_size, config.gpt2_config.seq_len),
        0, config.vocab_size
    )
    input_ids = hax.named(input_ids, (Batch, config.gpt2_config.Pos))

    targets = jnp.roll(input_ids.array, shift=-1, axis=-1)
    targets = hax.named(targets, (Batch, config.gpt2_config.Pos))

    # Compute gradients
    print("Computing gradients...")

    def loss_fn(model):
        return model.compute_loss(input_ids, targets, t=1.0, key=k_grad)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # Check gradient statistics
    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    grad_norms = [jnp.linalg.norm(g.flatten())
                  for g in grad_leaves if g is not None]

    print(f"✓ Gradient computation successful")
    print(f"  Number of gradient tensors: {len(grad_norms)}")
    print(f"  Mean gradient norm: {jnp.mean(jnp.array(grad_norms)):.6f}")
    print(f"  Max gradient norm: {jnp.max(jnp.array(grad_norms)):.6f}")
    print(
        f"  All gradients finite: {all(jnp.isfinite(g).all() for g in grad_leaves)}")

    return all(jnp.isfinite(g).all() for g in grad_leaves)


def test_time_variation():
    """Test that SSM parameters vary with time"""
    print("\n" + "="*60)
    print("Test 4: Time-Varying SSM Behavior")
    print("="*60)

    config = NeuralOdeSSMConfig.small_ssm()

    key = jrandom.PRNGKey(789)
    k_model, k_test = jrandom.split(key)

    print("\nInitializing model...")
    Vocab = hax.Axis("vocab", config.vocab_size)
    model = NeuralOdeSSMLMHeadModel.init(
        Vocab,
        config.gpt2_config,
        time_embedding_dim=config.time_embedding_dim,
        sinusoidal_dim=config.sinusoidal_dim,
        ssm_state_size=config.ssm_state_size,
        key=k_model
    )

    # Create dummy input
    batch_size = 2
    Batch = hax.Axis("batch", batch_size)
    input_ids = jrandom.randint(
        k_test, (batch_size, config.gpt2_config.seq_len),
        0, config.vocab_size
    )
    input_ids = hax.named(input_ids, (Batch, config.gpt2_config.Pos))

    # Test at different time points
    times = [0.0, 0.5, 1.0]
    outputs = []

    print("\nTesting forward pass at different time points...")
    for t in times:
        logits = model(input_ids, t=t, key=k_test)
        outputs.append(logits.array)
        print(
            f"  t={t:.1f}: output range [{jnp.min(logits.array):.2f}, {jnp.max(logits.array):.2f}]")

    # Check if outputs are different at different times
    diff_0_1 = jnp.abs(outputs[0] - outputs[2]).mean()
    print(f"\n✓ Time variation test complete")
    print(f"  Mean absolute difference (t=0 vs t=1): {diff_0_1:.4f}")
    print(f"  Outputs vary with time: {diff_0_1 > 0.01}")

    return diff_0_1 > 0.01


def test_ssm_vs_mlp_comparison():
    """Compare SSM and MLP model sizes"""
    print("\n" + "="*60)
    print("Test 5: SSM vs MLP Parameter Comparison")
    print("="*60)

    try:
        from qkvflow.models.neuralode_lm import NeuralOdeLMHeadModel
    except ImportError:
        print("⚠️  Skipping: neuralode_lm module not found (baseline model not implemented yet)")
        return True

    config = NeuralOdeSSMConfig.small_ssm()

    key = jrandom.PRNGKey(999)
    k_ssm, k_mlp = jrandom.split(key)

    # Initialize SSM model
    print("\nInitializing SSM model...")
    Vocab = hax.Axis("vocab", config.vocab_size)
    ssm_model = NeuralOdeSSMLMHeadModel.init(
        Vocab,
        config.gpt2_config,
        time_embedding_dim=config.time_embedding_dim,
        sinusoidal_dim=config.sinusoidal_dim,
        ssm_state_size=config.ssm_state_size,
        key=k_ssm
    )
    ssm_params = count_parameters(ssm_model)

    # Initialize MLP model
    print("Initializing MLP model (baseline)...")
    mlp_model = NeuralOdeLMHeadModel.init(
        Vocab,
        config.gpt2_config,
        time_embedding_dim=config.time_embedding_dim,
        sinusoidal_dim=config.sinusoidal_dim,
        key=k_mlp
    )
    mlp_total = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(mlp_model, eqx.is_array)))

    print("\n" + "-"*60)
    print("Parameter Comparison:")
    print("-"*60)
    print(f"SSM model:  {ssm_params['total']:>12,} parameters")
    print(f"MLP model:  {mlp_total:>12,} parameters")
    print(f"Reduction:  {mlp_total - ssm_params['total']:>12,} parameters")
    print(f"Ratio:      {ssm_params['total'] / mlp_total:>12.2%}")
    print("-"*60)

    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("NEURAL ODE + SSM MODEL VALIDATION")
    print("="*60)

    tests = [
        ("Forward Pass", test_forward_pass),
        ("Loss Computation", test_loss_computation),
        ("Gradient Computation", test_gradient_computation),
        ("Time Variation", test_time_variation),
        ("SSM vs MLP Comparison", test_ssm_vs_mlp_comparison),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {name}")

    all_passed = all(r for _, r in results)

    print("="*60)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Model is ready for training.")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review errors above.")
    print()

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
