#!/usr/bin/env python3
"""
CRITICAL ABLATION: Time-Indexed (variable t) vs Constant (t=0)

This is the scientific control that tests:
    "Is the gain from time-indexing t? Or just from MLP adapters?"

Compares THREE models:
1. Standard Transformer - baseline
2. Time-Indexed MLP (variable t) - original approach
3. Constant t=0 MLP - ablation control (same architecture, fixed t=0)

If variable t ≫ constant t=0:
  ✅ Time-dependency (Neural ODE) is crucial
  ✅ Neural ODE narrative validated

If variable t ≈ constant t=0:
  ⚠️  Gain is from MLP adapters, not time-indexing
  ⚠️  "Weight Adapter" architecture (not "Neural ODE")
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
import haliax.nn as hnn
import time
import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from levanter.models.gpt2 import Gpt2Config

# Import models
from pathlib import Path
scripts_dir = Path(__file__).parent

import importlib.util
spec = importlib.util.spec_from_file_location(
    "time_indexed_models", 
    scripts_dir / "test_time_indexed_weights.py"
)
time_indexed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_indexed)

StandardTransformer = time_indexed.StandardTransformer
TimeIndexedTransformer = time_indexed.TimeIndexedTransformer

# Import constant t=0 variant
spec_t0 = importlib.util.spec_from_file_location(
    "constant_t0", 
    scripts_dir / "test_constant_t0.py"
)
constant_t0 = importlib.util.module_from_spec(spec_t0)
spec_t0.loader.exec_module(constant_t0)

ConstantT0Transformer = constant_t0.ConstantT0Transformer


def log_message(msg):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


@dataclass
class AblationConfig:
    """Configuration for ablation comparison"""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    seq_len: int = 128
    batch_size: int = 8
    num_steps: int = 1000
    eval_every: int = 100
    learning_rate: float = 3e-4


class SimpleTokenizer:
    """Character-level tokenizer"""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
    
    def encode(self, text: str):
        return jnp.array([min(ord(c), self.vocab_size - 1) for c in text], dtype=jnp.int32)


def load_wikitext2():
    """Load WikiText-2"""
    log_message("Loading WikiText-2...")
    
    train_file = Path("train.txt")
    test_file = Path("test.txt")
    
    # Try wikitext103_data directory if not in root
    if not train_file.exists():
        train_file = Path("wikitext103_data/train.txt")
        test_file = Path("wikitext103_data/test.txt")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    log_message(f"  Train: {len(train_text):,} chars")
    log_message(f"  Test: {len(test_text):,} chars")
    
    return train_text, test_text


def create_batches(text: str, tokenizer: SimpleTokenizer, 
                   batch_size: int, seq_len: int, num_batches: int, key):
    """Create batches"""
    tokens = tokenizer.encode(text)
    batches = []
    
    for i in range(num_batches):
        key, subkey = jrandom.split(key)
        max_start = max(0, len(tokens) - seq_len - 1)
        start_idx = jrandom.randint(subkey, (), 0, max_start)
        
        batch_tokens = []
        for _ in range(batch_size):
            seq = tokens[start_idx:start_idx + seq_len + 1]
            if len(seq) < seq_len + 1:
                seq = jnp.pad(seq, (0, seq_len + 1 - len(seq)), constant_values=0)
            batch_tokens.append(seq)
            start_idx = (start_idx + seq_len + 1) % max_start
        
        batch_tokens = jnp.stack(batch_tokens)
        batches.append((batch_tokens[:, :-1], batch_tokens[:, 1:]))
    
    return batches


def count_parameters(model):
    """Count parameters"""
    leaves = jax.tree_util.tree_leaves(model)
    return sum(x.size for x in leaves if eqx.is_array(x))


def train_model(model_name: str, transformer, embedder, lm_head, config, 
                train_batches, valid_batches, Batch, Pos, Vocab, key):
    """Train a model"""
    
    log_message("="*70)
    log_message(f"Model: {model_name}")
    log_message("="*70)
    
    total_params = (
        count_parameters(transformer) + 
        count_parameters(embedder) + 
        count_parameters(lm_head)
    )
    log_message(f"Parameters: {total_params:,}")
    
    # Optimizer
    opt = optax.adam(learning_rate=config.learning_rate)
    models = (transformer, embedder, lm_head)
    opt_state = opt.init(eqx.filter(models, eqx.is_array))
    
    # History
    train_losses = []
    valid_losses = []
    step_times = []
    
    # Create causal mask
    attn_mask = hax.nn.attention.causal_mask(Pos, Pos)
    
    # JIT functions
    @eqx.filter_jit
    def train_step(models, opt_state, batch, key):
        transformer, embedder, lm_head = models
        input_ids, targets = batch
        input_ids = hax.named(input_ids, (Batch, Pos))
        targets = hax.named(targets, (Batch, Pos))
        
        def loss_fn(models):
            t, e, l = models
            x = e(input_ids)
            x = t(x, key=key)
            logits = l(x)
            
            # Sparse cross-entropy (no one-hot materialization)
            logits_flat = logits.array.reshape(-1, Vocab.size)
            targets_flat = targets.array.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
            return jnp.mean(loss)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(models)
        updates, opt_state = opt.update(grads, opt_state, models)
        models = eqx.apply_updates(models, updates)
        
        return models, opt_state, loss
    
    @eqx.filter_jit
    def valid_step(models, batch, key):
        transformer, embedder, lm_head = models
        input_ids, targets = batch
        input_ids = hax.named(input_ids, (Batch, Pos))
        targets = hax.named(targets, (Batch, Pos))
        
        x = embedder(input_ids)
        x = transformer(x, key=key)
        logits = lm_head(x)
        
        # Sparse cross-entropy
        logits_flat = logits.array.reshape(-1, Vocab.size)
        targets_flat = targets.array.reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
        return jnp.mean(loss)
    
    log_message(f"Training for {config.num_steps} steps...")
    log_message("(First step will be slow - JIT compilation)")
    
    for step in range(1, config.num_steps + 1):
        batch_idx = (step - 1) % len(train_batches)
        batch = train_batches[batch_idx]
        
        k_step, key = jrandom.split(key)
        step_start = time.time()
        
        models, opt_state, loss = train_step(models, opt_state, batch, k_step)
        
        step_time = (time.time() - step_start) * 1000
        train_losses.append(float(loss))
        step_times.append(step_time)
        
        # Evaluation
        if step % config.eval_every == 0:
            valid_loss_sum = 0.0
            for vb in valid_batches[:10]:
                k_eval, key = jrandom.split(key)
                valid_loss_sum += float(valid_step(models, vb, k_eval))
            valid_loss = valid_loss_sum / min(10, len(valid_batches))
            valid_losses.append(valid_loss)
            
            log_message(f"Step {step:4d}/{config.num_steps} | "
                       f"Train: {loss:.4f} | Valid: {valid_loss:.4f} | "
                       f"Time: {step_time:.1f}ms")
    
    log_message(f"✓ Complete")
    log_message(f"  Final train loss: {train_losses[-1]:.4f}")
    log_message(f"  Best valid loss: {min(valid_losses):.4f}")
    log_message(f"  Avg step time: {np.mean(step_times[1:]):.1f}ms")
    
    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'step_times': step_times[1:],
        'total_params': total_params,
        'best_valid_loss': min(valid_losses),
        'final_train_loss': train_losses[-1],
        'avg_step_time': np.mean(step_times[1:]),
    }


def main():
    log_message("="*70)
    log_message("ABLATION STUDY: Time-Indexed (variable t) vs Constant (t=0)")
    log_message("="*70)
    log_message("")
    log_message("Scientific Question:")
    log_message("  Is the gain from time-indexing t? Or just from MLP adapters?")
    log_message("")
    log_message("Models:")
    log_message("  1. Standard Transformer - baseline")
    log_message("  2. Time-Indexed MLP (variable t) - t varies by layer")
    log_message("  3. Constant t=0 MLP - t=0 for ALL layers (control)")
    log_message("="*70)
    log_message("")
    
    config = AblationConfig()
    
    # Load data
    train_text, test_text = load_wikitext2()
    
    # Create batches
    log_message("Creating batches...")
    tokenizer = SimpleTokenizer()
    key = jrandom.PRNGKey(42)
    k1, k2, key = jrandom.split(key, 3)
    
    train_batches = create_batches(
        train_text, tokenizer, config.batch_size, 
        config.seq_len, config.num_steps, k1
    )
    valid_batches = create_batches(
        test_text, tokenizer, config.batch_size, 
        config.seq_len, 100, k2
    )
    
    log_message(f"  Train: {len(train_batches)} batches")
    log_message(f"  Valid: {len(valid_batches)} batches")
    log_message("")
    
    # Setup axes
    Batch = hax.Axis("batch", config.batch_size)
    Pos = hax.Axis("position", config.seq_len)
    Vocab = hax.Axis("vocab", 256)
    
    gpt2_config = Gpt2Config(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        seq_len=config.seq_len,
        use_bias=True,
    )
    
    SinusodialDim = hax.Axis("SinusodialDim", 32)
    TembedDim = hax.Axis("TembedDim", 64)
    
    # Initialize all three models
    results = []
    
    for model_idx, (model_name, model_class) in enumerate([
        ("Standard Transformer", StandardTransformer),
        ("Time-Indexed (variable t)", TimeIndexedTransformer),
        ("Constant t=0", ConstantT0Transformer),
    ]):
        k_model = jrandom.PRNGKey(42 + model_idx)
        k_trans, k_emb, k_head, k_train = jrandom.split(k_model, 4)
        
        try:
            # Create model
            transformer = model_class.init(gpt2_config, SinusodialDim, TembedDim, key=k_trans)
            embedder = hnn.Embedding.init(Vocab, gpt2_config.Embed, key=k_emb)
            lm_head = hnn.Linear.init(gpt2_config.Embed, Vocab, key=k_head, use_bias=False)
            
            # Train
            result = train_model(
                model_name, transformer, embedder, lm_head, config,
                train_batches, valid_batches, Batch, Pos, Vocab, k_train
            )
            results.append(result)
            
        except Exception as e:
            log_message(f"✗ Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        log_message("")
    
    # Save results
    with open("ablation_t0_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    log_message("✓ Saved results: ablation_t0_results.pkl")
    
    # Analysis
    log_message("")
    log_message("="*70)
    log_message("ABLATION RESULTS")
    log_message("="*70)
    log_message("")
    
    # Find each model
    standard = next((r for r in results if "Standard" in r['model_name']), None)
    variable_t = next((r for r in results if "variable" in r['model_name']), None)
    constant_t0 = next((r for r in results if "Constant" in r['model_name']), None)
    
    if all([standard, variable_t, constant_t0]):
        log_message("Best Validation Loss:")
        log_message(f"  Standard:           {standard['best_valid_loss']:.4f}")
        log_message(f"  Variable t:         {variable_t['best_valid_loss']:.4f}")
        log_message(f"  Constant t=0:       {constant_t0['best_valid_loss']:.4f}")
        log_message("")
        
        # Critical comparisons
        var_vs_std = ((standard['best_valid_loss'] - variable_t['best_valid_loss']) / 
                      standard['best_valid_loss']) * 100
        const_vs_std = ((standard['best_valid_loss'] - constant_t0['best_valid_loss']) / 
                       standard['best_valid_loss']) * 100
        var_vs_const = ((constant_t0['best_valid_loss'] - variable_t['best_valid_loss']) / 
                       constant_t0['best_valid_loss']) * 100
        
        log_message("Improvements vs Standard:")
        log_message(f"  Variable t:    {var_vs_std:+.2f}%")
        log_message(f"  Constant t=0:  {const_vs_std:+.2f}%")
        log_message("")
        log_message("🔬 CRITICAL COMPARISON: Variable t vs Constant t=0:")
        log_message(f"   Variable t improvement: {var_vs_const:+.2f}%")
        log_message("")
        
        # Interpretation
        log_message("="*70)
        log_message("SCIENTIFIC INTERPRETATION")
        log_message("="*70)
        log_message("")
        
        if var_vs_const > 3.0:
            log_message("✅ CONCLUSION: Time-dependency (Neural ODE) is CRUCIAL")
            log_message("")
            log_message("   Variable t significantly outperforms Constant t=0")
            log_message("   → Gain comes from time-indexing (varying t by layer)")
            log_message("   → NOT just from MLP adapter structure")
            log_message("   → Neural ODE narrative is VALIDATED")
            log_message("")
            log_message("   Implication: The continuous-depth formulation matters.")
            log_message("                Weight trajectory over depth is meaningful.")
            
        elif var_vs_const > -3.0:
            log_message("⚠️  CONCLUSION: Time-dependency provides MODEST benefit")
            log_message("")
            log_message("   Variable t ≈ Constant t=0 (within 3%)")
            log_message("   → Gain is primarily from MLP adapter structure")
            log_message("   → Time-indexing helps, but not dramatically")
            log_message("   → 'Weight Adapter' framing more accurate than 'Neural ODE'")
            log_message("")
            log_message("   Implication: The architectural innovation is the adapter,")
            log_message("                not the time-dependency per se.")
            
        else:
            log_message("❌ CONCLUSION: Time-dependency may be HARMFUL")
            log_message("")
            log_message("   Constant t=0 outperforms Variable t")
            log_message("   → Time-indexing adds noise or instability")
            log_message("   → Simpler fixed modulation is better")
            log_message("   → Neural ODE narrative is NOT supported")
            log_message("")
            log_message("   Implication: Revisit the time-dependency assumption.")
        
        log_message("")
        log_message("="*70)
    
    return results


if __name__ == "__main__":
    main()

