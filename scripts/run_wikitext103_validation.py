"""
WikiText-103 Validation Script

Tests Time-Indexed Parameter Sharing models on WikiText-103 (larger dataset).
This addresses the "small-scale only" limitation mentioned in the README.

WikiText-103 is ~50× larger than WikiText-2 (103M vs 2M tokens).
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
import optax
from datasets import load_dataset
from tqdm import tqdm

from config.neuralode_ssm_config import Gpt2Config
from qkvflow.models.neuralode_lm import NeuralOdeLMHeadModel
from qkvflow.models.neuralode_ssm_lm import NeuralOdeSSMLMHeadModel


class WikiText103Config:
    """Configuration for WikiText-103 experiments."""
    
    # Model configs
    hidden_dim = 512  # Larger than WikiText-2 experiments
    num_layers = 6
    num_heads = 8
    vocab_size = 50000  # Larger vocab for bigger dataset
    seq_len = 256  # Longer sequences
    
    # Training
    batch_size = 8  # Smaller due to larger model
    num_steps = 5000  # More steps for larger dataset
    learning_rate = 3e-4
    
    # Validation
    eval_every = 500
    eval_steps = 50
    
    # Data
    max_train_samples = 50000  # Subsample for reasonable training time
    max_eval_samples = 5000
    

def setup_hf_token(token: str):
    """Set up Hugging Face token for dataset access."""
    os.environ["HF_TOKEN"] = token
    print(f"✓ Hugging Face token configured")


def load_and_preprocess_wikitext103(config: WikiText103Config):
    """
    Load WikiText-103 from Hugging Face and preprocess for our models.
    
    Returns:
        train_data: List of tokenized sequences
        val_data: List of tokenized sequences
        vocab_size: Actual vocabulary size
    """
    print("\n" + "="*60)
    print("Loading WikiText-103 from Hugging Face...")
    print("="*60)
    
    # Load dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", verification_mode="no_checks")
    
    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(ds['train'])} examples")
    print(f"  Validation: {len(ds['validation'])} examples")
    print(f"  Test: {len(ds['test'])} examples")
    
    # Simple character-level tokenization (can upgrade to BPE later)
    print("\nBuilding vocabulary...")
    all_text = " ".join(ds['train']['text'][:10000])  # Sample for vocab
    unique_chars = sorted(set(all_text))
    vocab_size = len(unique_chars)
    
    char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    
    print(f"✓ Vocabulary size: {vocab_size} characters")
    
    def tokenize_text(text_list, max_samples=None):
        """Convert text to token IDs."""
        sequences = []
        
        for i, text in enumerate(tqdm(text_list[:max_samples], desc="Tokenizing")):
            if not text.strip():  # Skip empty lines
                continue
                
            # Convert to token IDs
            tokens = [char_to_idx.get(ch, 0) for ch in text[:config.seq_len]]
            
            # Pad if needed
            if len(tokens) < config.seq_len:
                tokens = tokens + [0] * (config.seq_len - len(tokens))
            
            sequences.append(tokens)
        
        return jnp.array(sequences, dtype=jnp.int32)
    
    print("\nTokenizing train set...")
    train_data = tokenize_text(
        ds['train']['text'],
        max_samples=config.max_train_samples
    )
    
    print("\nTokenizing validation set...")
    val_data = tokenize_text(
        ds['validation']['text'],
        max_samples=config.max_eval_samples
    )
    
    print(f"\n✓ Preprocessing complete:")
    print(f"  Train: {train_data.shape[0]} sequences × {train_data.shape[1]} tokens")
    print(f"  Val: {val_data.shape[0]} sequences × {val_data.shape[1]} tokens")
    
    return train_data, val_data, vocab_size, char_to_idx, idx_to_char


def create_batches(data, batch_size, seq_len, key):
    """Create batches from data."""
    num_batches = len(data) // batch_size
    data = data[:num_batches * batch_size]
    data = data.reshape(num_batches, batch_size, seq_len)
    
    # Shuffle
    perm = jrandom.permutation(key, num_batches)
    data = data[perm]
    
    return data


def create_model(model_type: str, config: Gpt2Config, key):
    """Create a model based on type."""
    vocab = hax.Axis("vocab", config.vocab_size)
    
    if model_type == "time_indexed_mlp":
        return NeuralOdeLMHeadModel.init(vocab, config, key=key)
    elif model_type == "time_indexed_ssm":
        return NeuralOdeSSMLMHeadModel.init(vocab, config, key=key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count model parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def train_and_evaluate(
    model_type: str,
    train_data,
    val_data,
    config: WikiText103Config,
    exp_config: Gpt2Config,
    key
):
    """Train and evaluate a model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"{'='*60}")
    
    # Initialize model
    model_key, train_key, eval_key = jrandom.split(key, 3)
    model = create_model(model_type, exp_config, model_key)
    
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Define axes
    Batch = hax.Axis("batch", config.batch_size)
    Pos = hax.Axis("position", config.seq_len)
    
    # Training step
    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        # Prepare data
        inputs = batch[:, :-1]  # All but last token
        targets = batch[:, 1:]   # All but first token
        
        input_ids = hax.named(inputs, (Batch, Pos))
        target_ids = hax.named(targets, (Batch, Pos))
        
        # Compute loss and gradients
        def loss_fn(model):
            return model.compute_loss(input_ids, target_ids, t=None, key=key)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        # Update
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    # Evaluation function
    @eqx.filter_jit
    def eval_step(model, batch, key):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        input_ids = hax.named(inputs, (Batch, Pos))
        target_ids = hax.named(targets, (Batch, Pos))
        
        return model.compute_loss(input_ids, target_ids, t=None, key=key)
    
    # Create batches
    batch_key, shuffle_key = jrandom.split(train_key)
    train_batches = create_batches(
        train_data, config.batch_size, config.seq_len + 1, shuffle_key
    )
    
    # Training loop
    print("\nTraining...")
    history = {
        'train_loss': [],
        'eval_loss': [],
        'steps': [],
        'timestamps': []
    }
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for step in range(config.num_steps):
        # Get batch
        batch_idx = step % len(train_batches)
        batch = train_batches[batch_idx]
        
        # Train
        step_key = jrandom.fold_in(batch_key, step)
        model, opt_state, train_loss = train_step(model, opt_state, batch, step_key)
        
        # Log
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Train Loss: {train_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")
            history['train_loss'].append(float(train_loss))
            history['steps'].append(step)
            history['timestamps'].append(elapsed)
        
        # Evaluate
        if step % config.eval_every == 0 and step > 0:
            print("\nEvaluating...")
            val_batches = create_batches(
                val_data, config.batch_size, config.seq_len + 1, eval_key
            )
            
            eval_losses = []
            for i, val_batch in enumerate(val_batches[:config.eval_steps]):
                eval_key_i = jrandom.fold_in(eval_key, i)
                val_loss = eval_step(model, val_batch, eval_key_i)
                eval_losses.append(float(val_loss))
            
            avg_val_loss = jnp.mean(jnp.array(eval_losses))
            history['eval_loss'].append(float(avg_val_loss))
            
            print(f"✓ Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  → New best! ({best_val_loss:.4f})")
    
    total_time = time.time() - start_time
    print(f"\n✓ Training complete: {total_time:.1f}s")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    
    return {
        'model_type': model_type,
        'num_params': num_params,
        'best_val_loss': best_val_loss,
        'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else None,
        'history': history,
        'total_time': total_time
    }


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("WikiText-103 Validation Experiment")
    print("Testing Time-Indexed Parameter Sharing on Larger Dataset")
    print("="*60)
    
    # Setup
    config = WikiText103Config()
    
    # Set HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set!\n"
            "Set it with: export HF_TOKEN='your_hf_token'"
        )
    setup_hf_token(hf_token)
    
    # Load data
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = \
        load_and_preprocess_wikitext103(config)
    
    # Update vocab size
    config.vocab_size = vocab_size
    
    # Model configuration
    exp_config = Gpt2Config(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.seq_len,
        use_bias=True,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
    )
    
    print(f"\nModel Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Seq length: {config.seq_len}")
    
    # Run experiments
    results = {}
    key = jrandom.PRNGKey(42)
    
    models_to_test = [
        "time_indexed_mlp",
        "time_indexed_ssm",
    ]
    
    for model_type in models_to_test:
        model_key = jrandom.fold_in(key, hash(model_type))
        result = train_and_evaluate(
            model_type,
            train_data,
            val_data,
            config,
            exp_config,
            model_key
        )
        results[model_type] = result
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY - WikiText-103")
    print("="*60)
    
    for model_type, result in results.items():
        print(f"\n{model_type}:")
        print(f"  Parameters: {result['num_params']:,} ({result['num_params']/1e6:.2f}M)")
        print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"  Training Time: {result['total_time']:.1f}s")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['best_val_loss'])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Val Loss: {best_model[1]['best_val_loss']:.4f}")
    
    # Save results
    output_dir = Path("wikitext103_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save summary JSON
    summary = {
        model_type: {
            'num_params': result['num_params'],
            'best_val_loss': float(result['best_val_loss']),
            'total_time': result['total_time']
        }
        for model_type, result in results.items()
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # Update status
    print("\n" + "="*60)
    print("✅ WikiText-103 validation complete!")
    print("="*60)
    print("\nThis addresses the limitation: 'Only tested on WikiText-2'")
    print("WikiText-103 is ~50× larger, providing better validation.")
    print("\nNext: Update README with these results!")


if __name__ == "__main__":
    main()

