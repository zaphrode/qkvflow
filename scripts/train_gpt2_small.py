#!/usr/bin/env python3
"""
Train GPT-2 Small scale models on OpenWebText.

This script trains:
- Standard Transformer (117M params)
- Time-Indexed MLP (0.27M params)

With full checkpointing, monitoring, and evaluation.

Usage:
    python scripts/train_gpt2_small.py \
        --config config/gpt2_small/time_indexed_mlp.yaml \
        --output_dir checkpoints/time_indexed_mlp_gpt2
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import haliax as hax
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse
import pickle
from tqdm import tqdm

from transformers import GPT2TokenizerFast
from datasets import load_dataset

# Import your model implementations
from scripts.test_time_indexed_weights import TimeIndexedTransformer, StandardTransformer
from levanter.models.gpt2 import Gpt2Config


def load_config(config_path: str) -> Dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict, key: jax.Array):
    """
    Initialize model based on config.
    
    Returns:
        model: Initialized model
        num_params: Total parameter count
    """
    model_config = config['model']
    model_type = model_config['type']
    
    print(f"\n🏗️  Initializing {model_type} model...")
    
    # Create Gpt2Config
    gpt_config = Gpt2Config(
        hidden_dim=model_config['hidden_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        seq_len=model_config['seq_length'],
        use_bias=model_config.get('use_bias', True),
    )
    
    # Initialize model
    if model_type == "time_indexed_mlp":
        SinusodialDim = hax.Axis("SinusodialDim", model_config['sinusoidal_dim'])
        TembedDim = hax.Axis("TembedDim", model_config['tembed_dim'])
        model = TimeIndexedTransformer.init(gpt_config, SinusodialDim, TembedDim, key=key)
    
    elif model_type == "standard":
        model = StandardTransformer.init(gpt_config, key=key)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    
    print(f"   ✓ Model initialized")
    print(f"   ✓ Parameters: {num_params:,}")
    print(f"   ✓ Expected: ~{config['expected']['total_params']:,}")
    
    return model, num_params


def create_optimizer(config: Dict, num_steps: int):
    """Create optimizer with learning rate schedule"""
    train_config = config['training']
    
    # Learning rate schedule
    warmup_steps = train_config['warmup_steps']
    max_lr = train_config['learning_rate']
    min_lr = max_lr * train_config.get('min_lr_ratio', 0.1)
    
    # Warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=max_lr,
        warmup_steps=warmup_steps,
        decay_steps=num_steps - warmup_steps,
        end_value=min_lr
    )
    
    # Optimizer: AdamW with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config['max_grad_norm']),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config['model']['weight_decay']
        )
    )
    
    print(f"\n📈 Optimizer configured:")
    print(f"   Learning rate: {max_lr:.2e} (warmup: {warmup_steps} steps)")
    print(f"   Weight decay: {config['model']['weight_decay']}")
    print(f"   Gradient clip: {train_config['max_grad_norm']}")
    
    return optimizer, schedule


def create_data_loader(config: Dict, split: str = "train"):
    """
    Create streaming data loader for OpenWebText.
    
    Returns iterator that yields (input_ids, targets)
    """
    data_config = config['data']
    train_config = config['training']
    
    print(f"\n📚 Setting up {split} data loader...")
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # GPT-2 doesn't have a padding token by default, use EOS token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset (streaming)
    dataset = load_dataset(
        data_config['dataset'],
        split="train",
        streaming=True,
        cache_dir=data_config.get('cache_dir', None)
    )
    
    # Create train/val split (simple modulo split)
    is_val = (split == "validation")
    
    batch_size = train_config['batch_size']
    seq_len = config['model']['seq_length']
    
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Sequence length: {seq_len}")
    print(f"   ✓ Tokenizer vocab: {tokenizer.vocab_size}")
    
    def data_iterator():
        """Generator that yields batches"""
        batch_inputs = []
        
        for idx, example in enumerate(dataset):
            # Simple split: val = every 100th example
            is_val_sample = (idx % 100 == 0)
            if is_val != is_val_sample:
                continue
            
            # Tokenize
            text = example['text']
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=seq_len + 1,  # +1 for target
                padding="max_length"
            )
            
            input_ids = tokens['input_ids'][:seq_len + 1]
            
            # Skip if too short
            if len(input_ids) < seq_len + 1:
                continue
            
            batch_inputs.append(input_ids)
            
            # Yield when batch is full
            if len(batch_inputs) == batch_size:
                # Convert to JAX arrays
                batch_array = jnp.array(batch_inputs, dtype=jnp.int32)
                
                # Split into inputs and targets (shifted by 1)
                inputs = batch_array[:, :-1]  # [batch, seq_len]
                targets = batch_array[:, 1:]  # [batch, seq_len]
                
                yield inputs, targets
                batch_inputs = []
    
    return data_iterator()


@eqx.filter_jit
def train_step(model, optimizer_state, inputs, targets, is_time_indexed: bool):
    """
    Single training step.
    
    Returns:
        loss, model, optimizer_state, grads
    """
    Pos = hax.Axis("position", inputs.shape[1])
    Batch = hax.Axis("batch", inputs.shape[0])
    
    # Create named arrays
    inputs_named = hax.named(inputs, (Batch, Pos))
    targets_named = hax.named(targets, (Batch, Pos))
    
    def loss_fn(model):
        """Compute loss"""
        # Forward pass
        if is_time_indexed:
            key = jrandom.PRNGKey(0)  # Fixed key for training
            logits = model(inputs_named, key=key)
        else:
            logits = model(inputs_named)
        
        # Compute cross-entropy loss (sparse)
        logits_flat = logits.array.reshape(-1, 256)  # Vocab size
        targets_flat = targets_named.array.reshape(-1)
        
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits_flat, targets_flat
        ))
        
        return loss
    
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    
    # Apply gradients
    updates, optimizer_state = optimizer_state.update(grads, optimizer_state)
    model = eqx.apply_updates(model, updates)
    
    return loss, model, optimizer_state, grads


def train(
    model,
    optimizer,
    config: Dict,
    output_dir: Path,
    resume_step: int = 0
):
    """
    Main training loop.
    """
    train_config = config['training']
    model_type = config['model']['type']
    is_time_indexed = (model_type == "time_indexed_mlp")
    
    max_steps = train_config['max_steps']
    log_every = train_config['log_every']
    eval_every = train_config['eval_every']
    save_every = train_config['save_every']
    
    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Create data loaders
    train_loader = create_data_loader(config, split="train")
    val_loader = create_data_loader(config, split="validation")
    
    # Training log
    log_file = output_dir / "train.log"
    metrics_file = output_dir / "metrics.jsonl"
    
    print(f"\n{'='*70}")
    print(f"🚀 STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Model: {model_type}")
    print(f"Steps: {max_steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Training loop
    step = resume_step
    train_losses = []
    start_time = time.time()
    
    with tqdm(total=max_steps, initial=step, desc="Training") as pbar:
        for inputs, targets in train_loader:
            if step >= max_steps:
                break
            
            # Training step
            step_start = time.time()
            loss, model, opt_state, grads = train_step(
                model, opt_state, inputs, targets, is_time_indexed
            )
            step_time = time.time() - step_start
            
            train_losses.append(float(loss))
            step += 1
            
            # Logging
            if step % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / len(train_losses[-log_every:])
                
                metrics = {
                    "step": step,
                    "loss": float(loss),
                    "avg_loss": avg_loss,
                    "step_time": step_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Log to file
                with open(metrics_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "step_time": f"{step_time*1000:.1f}ms"
                })
            
            pbar.update(1)
            
            # Evaluation
            if step % eval_every == 0:
                print(f"\n📊 Evaluation at step {step}...")
                val_loss = evaluate(model, val_loader, is_time_indexed, num_steps=100)
                print(f"   Validation loss: {val_loss:.4f}")
                
                # Log validation
                with open(metrics_file, 'a') as f:
                    f.write(json.dumps({
                        "step": step,
                        "val_loss": val_loss,
                        "timestamp": datetime.now().isoformat()
                    }) + '\n')
            
            # Checkpointing
            if step % save_every == 0:
                print(f"\n💾 Saving checkpoint at step {step}...")
                save_checkpoint(model, opt_state, step, output_dir)
    
    # Final save
    print(f"\n💾 Saving final checkpoint...")
    save_checkpoint(model, opt_state, step, output_dir, is_final=True)
    
    # Training summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")


def evaluate(model, val_loader, is_time_indexed: bool, num_steps: int = 100):
    """Run evaluation"""
    losses = []
    
    for i, (inputs, targets) in enumerate(val_loader):
        if i >= num_steps:
            break
        
        Pos = hax.Axis("position", inputs.shape[1])
        Batch = hax.Axis("batch", inputs.shape[0])
        
        inputs_named = hax.named(inputs, (Batch, Pos))
        targets_named = hax.named(targets, (Batch, Pos))
        
        # Forward pass
        if is_time_indexed:
            key = jrandom.PRNGKey(0)
            logits = model(inputs_named, key=key)
        else:
            logits = model(inputs_named)
        
        # Compute loss
        logits_flat = logits.array.reshape(-1, 256)
        targets_flat = targets_named.array.reshape(-1)
        
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits_flat, targets_flat
        ))
        
        losses.append(float(loss))
    
    return sum(losses) / len(losses)


def save_checkpoint(model, opt_state, step: int, output_dir: Path, is_final: bool = False):
    """Save model checkpoint"""
    if is_final:
        ckpt_dir = output_dir / "final"
    else:
        ckpt_dir = output_dir / f"step_{step}"
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(ckpt_dir / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save optimizer state
    with open(ckpt_dir / "optimizer.pkl", 'wb') as f:
        pickle.dump(opt_state, f)
    
    # Save metadata
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat()
    }
    with open(ckpt_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 Small models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Initialize model
    key = jrandom.PRNGKey(args.seed)
    model, num_params = setup_model(config, key)
    
    # Create optimizer
    optimizer, schedule = create_optimizer(config, config['training']['max_steps'])
    
    # Train
    train(model, optimizer, config, output_dir)


if __name__ == "__main__":
    main()

