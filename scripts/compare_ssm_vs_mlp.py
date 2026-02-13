#!/usr/bin/env python3
"""
Quick comparison script: SSM vs MLP baseline

This script runs a short training comparison (1000 steps) to verify
SSM performance before committing to full training.

Usage:
    python scripts/compare_ssm_vs_mlp.py
"""

# Add parent directory to path BEFORE any local imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import everything else
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
import equinox as eqx
import optax
from levanter.models.gpt2 import Gpt2Config
from qkvflow.models.neuralode_ssm_lm import NeuralOdeSSMLMHeadModel, count_parameters
from qkvflow.models.neuralode_lm import NeuralOdeLMHeadModel
from config.neuralode_ssm_config import NeuralOdeSSMConfig


@dataclass
class ComparisonResult:
    """Results from a single model run"""
    model_type: str
    total_params: int
    final_loss: float
    avg_step_time: float
    memory_usage_gb: float
    loss_history: List[float]


def create_dummy_batch(key, batch_size, seq_len, vocab_size):
    """Create a dummy training batch"""
    Batch = hax.Axis("batch", batch_size)
    Pos = hax.Axis("position", seq_len)

    input_ids = jrandom.randint(key, (batch_size, seq_len), 0, vocab_size)
    input_ids = hax.named(input_ids, (Batch, Pos))

    # Targets are shifted by 1
    targets = jnp.roll(input_ids.array, shift=-1, axis=-1)
    targets = hax.named(targets, (Batch, Pos))

    return input_ids, targets


@eqx.filter_jit
def train_step(model, opt_state, optimizer, input_ids, targets, key):
    """Single training step"""
    def loss_fn(model):
        return model.compute_loss(input_ids, targets, t=1.0, key=key)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def train_model(
    model_type: str,
    config: NeuralOdeSSMConfig,
    num_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seed: int = 42
) -> ComparisonResult:
    """Train a model for comparison"""

    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*60}")

    key = jrandom.PRNGKey(seed)
    k_model, k_train = jrandom.split(key)

    # Create model
    print("Initializing model...")
    
    # Create Vocab axis
    Vocab = hax.Axis("vocab", config.vocab_size)
    
    if model_type == "ssm":
        model = NeuralOdeSSMLMHeadModel.init(
            Vocab,
            config.gpt2_config,
            time_embedding_dim=config.time_embedding_dim,
            sinusoidal_dim=config.sinusoidal_dim,
            ssm_state_size=config.ssm_state_size,
            key=k_model
        )
        param_counts = count_parameters(model)
        total_params = param_counts['total']
        print(f"  Total parameters: {total_params:,}")
        print(f"  Embeddings: {param_counts['embeddings']:,}")
        print(f"  Transformer: {param_counts['transformer']:,}")

    else:  # mlp
        model = NeuralOdeLMHeadModel.init(
            Vocab,
            config.gpt2_config,
            time_embedding_dim=config.time_embedding_dim,
            sinusoidal_dim=config.sinusoidal_dim,
            key=k_model
        )
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(model, eqx.is_array)))
        print(f"  Total parameters: {total_params:,}")

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=0.01)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    loss_history = []
    step_times = []

    for step in range(num_steps):
        # Generate batch
        k_train, k_batch, k_step = jrandom.split(k_train, 3)
        input_ids, targets = create_dummy_batch(
            k_batch, batch_size, config.gpt2_config.seq_len,
            config.vocab_size
        )

        # Training step
        start_time = time.time()
        model, opt_state, loss = train_step(
            model, opt_state, optimizer, input_ids, targets, k_step
        )
        step_time = time.time() - start_time

        loss_history.append(float(loss))
        step_times.append(step_time)

        # Log progress
        if (step + 1) % 100 == 0:
            avg_loss = jnp.mean(jnp.array(loss_history[-100:]))
            avg_time = jnp.mean(jnp.array(step_times[-100:]))
            tokens_per_sec = (
                batch_size * config.gpt2_config.seq_len) / avg_time
            print(f"  Step {step+1:4d} | Loss: {avg_loss:.4f} | "
                  f"Time: {avg_time*1000:.1f}ms | Tokens/s: {tokens_per_sec:.0f}")

    # Estimate memory usage (rough approximation)
    memory_gb = (total_params * 4) / (1024**3)  # 4 bytes per param (fp32)

    result = ComparisonResult(
        model_type=model_type,
        total_params=total_params,
        final_loss=float(jnp.mean(jnp.array(loss_history[-100:]))),
        avg_step_time=float(jnp.mean(jnp.array(step_times))),
        memory_usage_gb=memory_gb,
        loss_history=loss_history
    )

    print(f"\n✓ Training complete")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Avg step time: {result.avg_step_time*1000:.1f}ms")

    return result


def generate_comparison_report(ssm_result: ComparisonResult, mlp_result: ComparisonResult):
    """Generate detailed comparison report"""

    print("\n" + "="*70)
    print(" "*20 + "COMPARISON REPORT")
    print("="*70)

    # Parameters
    print("\n📊 Model Size:")
    print("-"*70)
    print(f"{'Model':<15} {'Parameters':<20} {'Memory (GB)':<20}")
    print("-"*70)
    print(f"{'SSM':<15} {ssm_result.total_params:>12,}      {ssm_result.memory_usage_gb:>8.2f}")
    print(f"{'MLP':<15} {mlp_result.total_params:>12,}      {mlp_result.memory_usage_gb:>8.2f}")
    print("-"*70)
    param_reduction = mlp_result.total_params - ssm_result.total_params
    param_ratio = ssm_result.total_params / mlp_result.total_params
    print(f"{'Reduction':<15} {param_reduction:>12,}      ({param_ratio:.1%})")
    print()

    # Performance
    print("⚡ Training Speed:")
    print("-"*70)
    print(f"{'Model':<15} {'Avg Step Time':<20} {'Tokens/sec':<20}")
    print("-"*70)

    batch_size = 4
    seq_len = 512  # From config

    ssm_tokens_per_sec = (batch_size * seq_len) / ssm_result.avg_step_time
    mlp_tokens_per_sec = (batch_size * seq_len) / mlp_result.avg_step_time

    print(f"{'SSM':<15} {ssm_result.avg_step_time*1000:>8.1f} ms        {ssm_tokens_per_sec:>10.0f}")
    print(f"{'MLP':<15} {mlp_result.avg_step_time*1000:>8.1f} ms        {mlp_tokens_per_sec:>10.0f}")
    print("-"*70)
    speedup = mlp_result.avg_step_time / ssm_result.avg_step_time
    if speedup > 1:
        print(f"SSM is {speedup:.2f}x faster")
    else:
        print(f"MLP is {1/speedup:.2f}x faster")
    print()

    # Loss
    print("📈 Training Loss:")
    print("-"*70)
    print(f"{'Model':<15} {'Final Loss':<20} {'Min Loss':<20}")
    print("-"*70)
    print(f"{'SSM':<15} {ssm_result.final_loss:>8.4f}           "
          f"{min(ssm_result.loss_history):>8.4f}")
    print(f"{'MLP':<15} {mlp_result.final_loss:>8.4f}           "
          f"{min(mlp_result.loss_history):>8.4f}")
    print("-"*70)
    loss_diff = ssm_result.final_loss - mlp_result.final_loss
    if abs(loss_diff) < 0.01:
        print(f"Loss is comparable (Δ={loss_diff:+.4f})")
    elif ssm_result.final_loss < mlp_result.final_loss:
        print(f"SSM achieves lower loss (Δ={loss_diff:+.4f})")
    else:
        print(f"MLP achieves lower loss (Δ={loss_diff:+.4f})")
    print()

    # Verdict
    print("✅ Verdict:")
    print("-"*70)

    verdicts = []

    # Parameter efficiency
    verdicts.append(f"✓ SSM uses {param_ratio:.1%} of MLP parameters")

    # Speed
    if speedup > 1.1:
        verdicts.append(f"✓ SSM is {speedup:.2f}x faster")
    elif speedup < 0.9:
        verdicts.append(f"⚠ SSM is {1/speedup:.2f}x slower")
    else:
        verdicts.append(f"≈ Similar speed ({speedup:.2f}x)")

    # Loss
    if abs(loss_diff) < 0.05:
        verdicts.append(f"✓ Comparable loss (Δ={loss_diff:+.4f})")
    elif ssm_result.final_loss < mlp_result.final_loss - 0.05:
        verdicts.append(f"✓ SSM achieves better loss")
    else:
        verdicts.append(f"⚠ SSM has higher loss (Δ={loss_diff:+.4f})")

    for v in verdicts:
        print(v)

    print()

    # Recommendation
    print("🎯 Recommendation:")
    print("-"*70)

    if ssm_result.final_loss < mlp_result.final_loss + 0.1 and param_ratio < 0.7:
        print("✅ SSM looks promising! Proceed with full training:")
        print("   - Competitive loss with fewer parameters")
        print("   - Potential for better long-context performance")
        print("   - Faster inference on long sequences")
        print()
        print("   Next steps:")
        print("   1. Run full training (1B tokens)")
        print("   2. Evaluate on long sequences (2K+ tokens)")
        print("   3. Ablate SSM state size (32/64/128/256)")

    elif ssm_result.final_loss > mlp_result.final_loss + 0.2:
        print("⚠️  SSM performance needs improvement:")
        print("   - Significantly higher loss than baseline")
        print("   - Debug suggestions:")
        print("     • Increase SSM state size (--ssm_state_size 128)")
        print("     • Adjust learning rate (--learning_rate 1e-4)")
        print("     • Check gradient norms for instability")
        print("     • Verify SSM parameter initialization")

    else:
        print("➡️  Mixed results - more investigation needed:")
        print("   - Run longer comparison (10K+ steps)")
        print("   - Try different SSM configurations")
        print("   - Evaluate on validation set")

    print("="*70)


def main():
    """Run comparison"""

    print("\n" + "="*70)
    print(" "*15 + "SSM vs MLP COMPARISON")
    print("="*70)
    print("\nThis will train both models for 1000 steps to compare:")
    print("  • Parameter counts")
    print("  • Training speed")
    print("  • Loss convergence")
    print()
    print("⏱️  Estimated time: 5-10 minutes")
    print()

    # Configuration
    config = NeuralOdeSSMConfig.small_ssm()

    print("Configuration:")
    print(f"  Hidden dim: {config.gpt2_config.hidden_dim}")
    print(f"  Num layers: {config.gpt2_config.num_layers}")
    print(f"  Seq length: {config.gpt2_config.seq_len}")
    print(f"  SSM state size: {config.ssm_state_size}")

    # Train both models
    ssm_result = train_model("ssm", config, num_steps=1000, seed=42)
    mlp_result = train_model("mlp", config, num_steps=1000, seed=42)

    # Generate report
    generate_comparison_report(ssm_result, mlp_result)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"comparison_{timestamp}.json"

    results = {
        "ssm": {
            "params": ssm_result.total_params,
            "final_loss": ssm_result.final_loss,
            "avg_step_time": ssm_result.avg_step_time,
        },
        "mlp": {
            "params": mlp_result.total_params,
            "final_loss": mlp_result.final_loss,
            "avg_step_time": mlp_result.avg_step_time,
        },
        "config": {
            "hidden_dim": config.gpt2_config.hidden_dim,
            "num_layers": config.gpt2_config.num_layers,
            "ssm_state_size": config.ssm_state_size,
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print()


if __name__ == "__main__":
    main()
