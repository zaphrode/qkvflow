#!/usr/bin/env python3
"""
Training script for Neural ODE + SSM Language Model

Usage:
    # Train SSM model
    python scripts/train_neuralode_ssm.py --model ssm --ssm_state_size 64
    
    # Train baseline MLP for comparison  
    python scripts/train_neuralode_ssm.py --model mlp
"""

# Add parent directory to path BEFORE any local imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import everything else
import argparse
import time
from pathlib import Path
from typing import Optional
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jax.experimental import multihost_utils
from tqdm import tqdm
from levanter.data.text import LMDatasetConfig, TokenSeqDataset
from levanter.models.gpt2 import Gpt2Config
from levanter.tracker import CompositeTracker, LoggerConfig, log_configuration
from qkvflow.models.neuralode_ssm_lm import NeuralOdeSSMLMHeadModel, count_parameters
from qkvflow.models.neuralode_lm import NeuralOdeLMHeadModel
from config.neuralode_ssm_config import NeuralOdeSSMConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Neural ODE + SSM LM")

    # Model selection
    parser.add_argument("--model", type=str, default="ssm", choices=["ssm", "mlp"],
                        help="Model type: 'ssm' or 'mlp' (baseline)")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size configuration")

    # SSM-specific
    parser.add_argument("--ssm_state_size", type=int, default=64,
                        help="SSM hidden state dimension")
    parser.add_argument("--time_embedding_dim", type=int, default=64,
                        help="Time embedding dimension")
    parser.add_argument("--sinusoidal_dim", type=int, default=32,
                        help="Sinusoidal encoding dimension")

    # Data
    parser.add_argument("--dataset", type=str, default="wikitext-103",
                        help="Dataset name")
    parser.add_argument("--cache_dir", type=str, default="./data_cache",
                        help="Cache directory for dataset")
    parser.add_argument("--max_tokens", type=int, default=10_000_000,
                        help="Maximum training tokens")

    # Training
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_batches", type=int, default=100,
                        help="Number of eval batches")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N steps")

    # Logging
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qkvflow-ssm",
                        help="W&B project name")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def create_model(args, config, key):
    """Create model based on args"""
    if args.model == "ssm":
        model = NeuralOdeSSMLMHeadModel.init(
            config.gpt2_config,
            time_embedding_dim=args.time_embedding_dim,
            sinusoidal_dim=args.sinusoidal_dim,
            ssm_state_size=args.ssm_state_size,
            key=key
        )
        print(f"\n✓ Created Neural ODE + SSM model")
        param_counts = count_parameters(model)
        for k, v in param_counts.items():
            print(f"  {k}: {v:,} parameters")

    else:  # mlp
        model = NeuralOdeLMHeadModel.init(
            config.gpt2_config,
            time_embedding_dim=args.time_embedding_dim,
            sinusoidal_dim=args.sinusoidal_dim,
            key=key
        )
        print(f"\n✓ Created Neural ODE + MLP model (baseline)")
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(model, eqx.is_array)))
        print(f"  total: {total_params:,} parameters")

    return model


def create_optimizer(args, total_steps):
    """Create optimizer with learning rate schedule"""
    # Cosine decay with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=total_steps,
        end_value=args.learning_rate * 0.1
    )

    # AdamW optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.scale_by_adam(),
        optax.add_decayed_weights(args.weight_decay),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0)
    )

    return optimizer, schedule


def prepare_batch(batch, config):
    """Prepare batch for training"""
    # Assume batch is a dict with 'input_ids'
    input_ids = batch['input_ids']

    # Create targets (shift by 1)
    targets = jnp.roll(input_ids, shift=-1, axis=-1)

    # Convert to named arrays
    Batch = hax.Axis("batch", input_ids.shape[0])
    Pos = config.gpt2_config.Pos

    input_ids_named = hax.named(input_ids, (Batch, Pos))
    targets_named = hax.named(targets, (Batch, Pos))

    return input_ids_named, targets_named


@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, key):
    """Single training step"""
    input_ids, targets = batch

    def loss_fn(model):
        return model.compute_loss(input_ids, targets, t=1.0, key=key)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, batch, key):
    """Single evaluation step"""
    input_ids, targets = batch
    loss = model.compute_loss(input_ids, targets, t=1.0, key=key)
    return loss


def train(args):
    """Main training loop"""
    print(f"\n{'='*60}")
    print(
        f"Training Neural ODE {'SSM' if args.model == 'ssm' else 'MLP'} Language Model")
    print(f"{'='*60}\n")

    # Setup
    key = jrandom.PRNGKey(args.seed)
    k_model, k_train, k_eval = jrandom.split(key, 3)

    # Configuration
    if args.config == "small":
        config = NeuralOdeSSMConfig.small_ssm()
    elif args.config == "medium":
        config = NeuralOdeSSMConfig.medium_ssm()
    else:
        config = NeuralOdeSSMConfig.large_ssm()

    # Override with CLI args
    config.ssm_state_size = args.ssm_state_size
    config.time_embedding_dim = args.time_embedding_dim
    config.sinusoidal_dim = args.sinusoidal_dim

    print(f"Config: {args.config}")
    print(f"  Hidden dim: {config.gpt2_config.hidden_dim}")
    print(f"  Num layers: {config.gpt2_config.num_layers}")
    print(f"  Num heads: {config.gpt2_config.num_heads}")
    print(f"  Seq length: {config.gpt2_config.seq_len}")
    if args.model == "ssm":
        print(f"  SSM state size: {config.ssm_state_size}")

    # Create model
    model = create_model(args, config, k_model)

    # Calculate training steps
    total_steps = args.max_tokens // (args.batch_size *
                                      config.gpt2_config.seq_len)
    print(f"\nTraining for {total_steps:,} steps ({args.max_tokens:,} tokens)")

    # Create optimizer
    optimizer, lr_schedule = create_optimizer(args, total_steps)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / f"{args.model}_{args.config}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*60}\n")

    step = 0
    total_loss = 0.0
    start_time = time.time()

    # Note: This is a simplified training loop
    # In practice, you'd integrate with your data loader
    print("⚠️  WARNING: This script needs integration with your data pipeline!")
    print("    Please connect to your dataset loading logic.")
    print(f"\n{'='*60}\n")

    # Placeholder training loop structure
    try:
        for step in tqdm(range(total_steps), desc="Training"):
            # TODO: Get actual batch from data loader
            # batch = next(train_loader)
            # input_ids, targets = prepare_batch(batch, config)

            # Generate dummy batch for now (REMOVE THIS)
            k_train, k_batch = jrandom.split(k_train)
            dummy_input = jrandom.randint(
                k_batch,
                (args.batch_size, config.gpt2_config.seq_len),
                0, config.gpt2_config.vocab_size
            )
            Batch = hax.Axis("batch", args.batch_size)
            dummy_input = hax.named(
                dummy_input, (Batch, config.gpt2_config.Pos))
            batch = (dummy_input, dummy_input)  # Dummy targets

            # Training step
            k_train, k_step = jrandom.split(k_train)
            model, opt_state, loss = train_step(
                model, opt_state, optimizer, batch, k_step
            )

            total_loss += loss

            # Logging
            if (step + 1) % args.log_every == 0:
                avg_loss = total_loss / args.log_every
                lr = lr_schedule(step)
                elapsed = time.time() - start_time
                tokens_per_sec = (args.log_every * args.batch_size *
                                  config.gpt2_config.seq_len) / elapsed

                print(f"Step {step+1:6d} | Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f}")

                total_loss = 0.0
                start_time = time.time()

            # Evaluation
            if (step + 1) % args.eval_every == 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {step+1}")
                print(f"{'='*60}")

                # TODO: Run evaluation on validation set
                print("⚠️  Evaluation not yet implemented")
                print(f"{'='*60}\n")

            # Checkpointing
            if (step + 1) % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step+1}.eqx"
                print(f"\n💾 Saving checkpoint to {checkpoint_path}")
                eqx.tree_serialise_leaves(checkpoint_path, model)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    # Final checkpoint
    final_path = checkpoint_dir / "checkpoint_final.eqx"
    print(f"\n💾 Saving final checkpoint to {final_path}")
    eqx.tree_serialise_leaves(final_path, model)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
