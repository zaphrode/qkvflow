#!/usr/bin/env python3
"""
Publication-Ready Training: Time-Indexed Transformer

Model size comparable to GPT-2 Small for fair comparison:
- Hidden dim: 768
- Layers: 12 (but shared via time-indexing)
- Heads: 12
- Sequence length: 512

Uses gradient accumulation and proper validation with dropout.
"""

import os
import sys
import json
import pickle
import time
import argparse
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import haliax as hax
import haliax.nn as hnn
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from qkvflow.config import Gpt2Config, Gpt2Embeddings, ACT2FN

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


@dataclass
class TrainingConfig:
    """Publication-ready training configuration"""
    # Model - GPT-2 Small equivalent
    vocab_size: int = 50257
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Time-indexed specific
    time_embed_dim: int = 256
    sinusoidal_dim: int = 128
    
    # Regularization - IMPORTANT to prevent overfitting
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training - memory efficient
    micro_batch_size: int = 2        # Small batch that fits in GPU
    gradient_accumulation: int = 32  # Effective batch = 2 * 32 = 64
    sequence_length: int = 512       # Reduced for memory
    
    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Early stopping
    patience: int = 5  # Stop if val loss doesn't improve for this many evals
    min_delta: float = 0.01  # Minimum improvement to count as improvement
    
    # Checkpointing
    eval_every: int = 500  # Evaluate more frequently
    save_every: int = 2500
    
    # Paths
    data_path: str = "/data1/fypnahid/qkvflow/openwebtext"
    output_dir: str = "/data1/fypnahid/qkvflow/checkpoints_v2"
    
    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation


# =============================================================================
# TIME-INDEXED TRANSFORMER BLOCKS
# =============================================================================

class TimeEmbedding(eqx.Module):
    """Sinusoidal time embedding for layer depth"""
    embed_dim: int = eqx.field(static=True)
    
    @staticmethod
    def init(embed_dim: int):
        return TimeEmbedding(embed_dim=embed_dim)
    
    def __call__(self, t: float) -> jnp.ndarray:
        """Generate embedding for time t in [0, 1]"""
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
        args = t * freqs
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
        return embedding


class TimeModulatedLinear(eqx.Module):
    """Linear layer with time-dependent modulation"""
    weight: jnp.ndarray
    bias: jnp.ndarray
    time_scale: jnp.ndarray  # Modulates weight based on time
    time_shift: jnp.ndarray  # Shifts output based on time
    
    in_axis: hax.Axis = eqx.field(static=True)
    out_axis: hax.Axis = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)
    
    @staticmethod
    def init(in_axis: hax.Axis, out_axis: hax.Axis, time_dim: int, *, key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        
        # Standard weight initialization
        std = 0.02
        weight = jrandom.normal(k1, (out_axis.size, in_axis.size)) * std
        bias = jnp.zeros(out_axis.size)
        
        # Time modulation parameters
        time_scale = jrandom.normal(k3, (time_dim, out_axis.size)) * 0.01
        time_shift = jnp.zeros((time_dim, out_axis.size))
        
        return TimeModulatedLinear(
            weight=weight,
            bias=bias,
            time_scale=time_scale,
            time_shift=time_shift,
            in_axis=in_axis,
            out_axis=out_axis,
            time_dim=time_dim,
        )
    
    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray) -> hax.NamedArray:
        # Compute time-dependent scale and shift
        scale = 1.0 + jnp.tanh(time_embed @ self.time_scale)  # [out_dim]
        shift = time_embed @ self.time_shift  # [out_dim]
        
        # Apply linear transformation
        x_array = x.array
        out = jnp.einsum('...i,oi->...o', x_array, self.weight) + self.bias
        
        # Apply time modulation
        out = out * scale + shift
        
        # Reconstruct NamedArray - replace input axis with output axis at same position
        out_axes = tuple(
            self.out_axis if ax.name == self.in_axis.name else ax
            for ax in x.axes
        )
        return hax.named(out, out_axes)


class TimeIndexedAttention(eqx.Module):
    """Multi-head attention with time-indexed parameters and dropout"""
    config: Gpt2Config = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    
    q_proj: TimeModulatedLinear
    k_proj: TimeModulatedLinear
    v_proj: TimeModulatedLinear
    out_proj: TimeModulatedLinear
    
    @staticmethod
    def init(config: Gpt2Config, time_dim: int, dropout_rate: float = 0.1, *, key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        
        HeadDim = hax.Axis("head_dim", config.HeadSize.size * config.Heads.size)
        
        return TimeIndexedAttention(
            config=config,
            time_dim=time_dim,
            dropout_rate=dropout_rate,
            q_proj=TimeModulatedLinear.init(config.Embed, HeadDim, time_dim, key=k1),
            k_proj=TimeModulatedLinear.init(config.Embed, HeadDim, time_dim, key=k2),
            v_proj=TimeModulatedLinear.init(config.Embed, HeadDim, time_dim, key=k3),
            out_proj=TimeModulatedLinear.init(HeadDim, config.Embed, time_dim, key=k4),
        )
    
    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray, mask, *, key=None, inference: bool = False):
        # Project Q, K, V with time modulation
        HeadDim = hax.Axis("head_dim", self.config.HeadSize.size * self.config.Heads.size)
        
        q = self.q_proj(x, time_embed)
        k = self.k_proj(x, time_embed)
        v = self.v_proj(x, time_embed)
        
        # Reshape to multi-head
        q = q.unflatten_axis(HeadDim, (self.config.Heads, self.config.HeadSize))
        k = k.unflatten_axis(HeadDim, (self.config.Heads, self.config.HeadSize))
        v = v.unflatten_axis(HeadDim, (self.config.Heads, self.config.HeadSize))
        
        # Rename for attention - key and value get "key_position" axis
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        # Get actual KeyPos axis from tensor (not config) to handle variable sequence lengths
        KeyPos = k.resolve_axis("key_position")
        
        # Scaled dot-product attention
        scale = self.config.HeadSize.size ** -0.5
        attn = hax.dot(self.config.HeadSize, q, k) * scale
        
        if mask is not None:
            attn = hax.where(mask, attn, -1e9)
        
        attn = hnn.softmax(attn, axis=KeyPos)
        
        # Attention dropout (only during training)
        if not inference and self.dropout_rate > 0 and key is not None:
            k1, key = jrandom.split(key)
            dropout_mask = jrandom.bernoulli(k1, 1.0 - self.dropout_rate, attn.array.shape)
            attn = hax.named(attn.array * dropout_mask / (1.0 - self.dropout_rate), attn.axes)
        
        # Apply attention to values
        out = hax.dot(KeyPos, attn, v)
        
        # Reshape back and project
        out = out.flatten_axes((self.config.Heads, self.config.HeadSize), HeadDim)
        out = self.out_proj(out, time_embed)
        
        # Output dropout
        if not inference and self.dropout_rate > 0 and key is not None:
            dropout_mask = jrandom.bernoulli(key, 1.0 - self.dropout_rate, out.array.shape)
            out = hax.named(out.array * dropout_mask / (1.0 - self.dropout_rate), out.axes)
        
        return out


class TimeIndexedMLP(eqx.Module):
    """MLP with time-indexed parameters and dropout"""
    fc1: TimeModulatedLinear
    fc2: TimeModulatedLinear
    act: callable = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    
    @staticmethod
    def init(config: Gpt2Config, time_dim: int, dropout_rate: float = 0.1, *, key):
        k1, k2 = jrandom.split(key)
        return TimeIndexedMLP(
            fc1=TimeModulatedLinear.init(config.Embed, config.Mlp, time_dim, key=k1),
            fc2=TimeModulatedLinear.init(config.Mlp, config.Embed, time_dim, key=k2),
            act=ACT2FN[config.activation_function],
            dropout_rate=dropout_rate,
        )
    
    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray, *, key=None, inference: bool = False):
        x = self.fc1(x, time_embed)
        x = self.act(x)
        
        # Dropout after activation (only during training)
        if not inference and self.dropout_rate > 0 and key is not None:
            dropout_mask = jrandom.bernoulli(key, 1.0 - self.dropout_rate, x.array.shape)
            x = hax.named(x.array * dropout_mask / (1.0 - self.dropout_rate), x.axes)
        
        x = self.fc2(x, time_embed)
        return x


class TimeIndexedBlock(eqx.Module):
    """Single transformer block with time-indexed parameters (shared across layers)"""
    ln1: hnn.LayerNorm
    attn: TimeIndexedAttention
    ln2: hnn.LayerNorm
    mlp: TimeIndexedMLP
    dropout_rate: float = eqx.field(static=True)
    
    @staticmethod
    def init(config: Gpt2Config, time_dim: int, dropout_rate: float = 0.1, *, key):
        k1, k2 = jrandom.split(key)
        return TimeIndexedBlock(
            ln1=hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon),
            attn=TimeIndexedAttention.init(config, time_dim, dropout_rate=dropout_rate, key=k1),
            ln2=hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon),
            mlp=TimeIndexedMLP.init(config, time_dim, dropout_rate=dropout_rate, key=k2),
            dropout_rate=dropout_rate,
        )
    
    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray, mask, *, key=None, inference: bool = False):
        # Pre-norm architecture with dropout
        # Split into 4 keys to avoid PRNG reuse (each dropout needs its own key)
        if key is not None:
            k1, k2, k3, k4 = jrandom.split(key, 4)
        else:
            k1 = k2 = k3 = k4 = None
        
        # Attention with residual dropout
        attn_out = self.attn(self.ln1(x), time_embed, mask, key=k1, inference=inference)
        if not inference and self.dropout_rate > 0 and k2 is not None:
            dropout_mask = jrandom.bernoulli(k2, 1.0 - self.dropout_rate, attn_out.array.shape)
            attn_out = hax.named(attn_out.array * dropout_mask / (1.0 - self.dropout_rate), attn_out.axes)
        x = x + attn_out
        
        # MLP with residual dropout (k3 for MLP internal, k4 for residual - avoid key reuse)
        mlp_out = self.mlp(self.ln2(x), time_embed, key=k3, inference=inference)
        if not inference and self.dropout_rate > 0 and k4 is not None:
            dropout_mask = jrandom.bernoulli(k4, 1.0 - self.dropout_rate, mlp_out.array.shape)
            mlp_out = hax.named(mlp_out.array * dropout_mask / (1.0 - self.dropout_rate), mlp_out.axes)
        x = x + mlp_out
        
        return x


class TimeIndexedTransformer(eqx.Module):
    """Full Time-Indexed Transformer for language modeling"""
    config: Gpt2Config = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    
    embeddings: Gpt2Embeddings
    time_embed: TimeEmbedding
    block: TimeIndexedBlock  # Single block, applied multiple times
    ln_f: hnn.LayerNorm
    
    @staticmethod
    def init(Vocab: hax.Axis, config: Gpt2Config, num_layers: int, time_dim: int, dropout_rate: float = 0.1, *, key):
        k1, k2, k3 = jrandom.split(key, 3)
        
        return TimeIndexedTransformer(
            config=config,
            num_layers=num_layers,
            time_dim=time_dim,
            dropout_rate=dropout_rate,
            embeddings=Gpt2Embeddings.init(Vocab, config, key=k1),
            time_embed=TimeEmbedding.init(time_dim),
            block=TimeIndexedBlock.init(config, time_dim, dropout_rate=dropout_rate, key=k2),
            ln_f=hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon),
        )
    
    def __call__(self, input_ids: hax.NamedArray, *, key=None, inference: bool = False):
        # For inference, use a dummy key if none provided (dropout will be skipped anyway)
        if key is None and inference:
            key = jrandom.PRNGKey(0)
        
        # Embed tokens
        k_emb, key = jrandom.split(key) if key is not None else (None, None)
        x = self.embeddings.embed(input_ids, key=k_emb)
        
        # Embedding dropout (only during training)
        if not inference and self.dropout_rate > 0 and key is not None:
            k_drop, key = jrandom.split(key)
            dropout_mask = jrandom.bernoulli(k_drop, 1.0 - self.dropout_rate, x.array.shape)
            x = hax.named(x.array * dropout_mask / (1.0 - self.dropout_rate), x.axes)
        
        # Causal mask - CRITICAL: Must use different axes for query and key positions
        # Using same axis twice returns 1D mask (all True) which breaks causality!
        Pos = input_ids.resolve_axis("position")
        KeyPos = Pos.alias("key_position")
        mask = hnn.attention.causal_mask(Pos, KeyPos)
        
        # Apply the SAME block multiple times with different time embeddings
        for layer_idx in range(self.num_layers):
            # Time = layer_idx / (num_layers - 1), normalized to [0, 1]
            t = layer_idx / max(1, self.num_layers - 1)
            time_emb = self.time_embed(t)
            
            layer_key = jrandom.fold_in(key, layer_idx) if key is not None else None
            x = self.block(x, time_emb, mask, key=layer_key, inference=inference)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.embeddings.unembed(x)
        
        return logits


# =============================================================================
# DATA LOADING
# =============================================================================

class OpenWebTextLoader:
    """Efficient data loader for OpenWebText"""
    
    def __init__(self, data_path: str, tokenizer, batch_size: int, seq_len: int):
        from datasets import load_from_disk
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        log(f"Loading dataset from {data_path}")
        self.dataset = load_from_disk(data_path)
        
        if hasattr(self.dataset, 'keys'):
            self.dataset = self.dataset['train']
        
        self.num_examples = len(self.dataset)
        log(f"Dataset has {self.num_examples:,} examples")
    
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Yield batches indefinitely"""
        token_buffer = []
        example_idx = 0
        
        while True:
            # Fill buffer
            while len(token_buffer) < self.batch_size * (self.seq_len + 1) * 2:
                if example_idx >= self.num_examples:
                    example_idx = 0
                    np.random.shuffle(np.arange(self.num_examples))
                
                text = self.dataset[example_idx]['text']
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                example_idx += 1
            
            # Create batch
            batch_input = []
            batch_labels = []
            
            for _ in range(self.batch_size):
                seq = token_buffer[:self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len:]
                
                batch_input.append(seq[:-1])
                batch_labels.append(seq[1:])
            
            yield {
                "input_ids": jnp.array(batch_input, dtype=jnp.int32),
                "labels": jnp.array(batch_labels, dtype=jnp.int32),
            }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def count_parameters(model) -> int:
    """Count trainable parameters"""
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def compute_loss(model, batch, Batch, Pos, Vocab, key, inference: bool = False):
    """Compute cross-entropy loss"""
    input_ids = hax.named(batch["input_ids"], (Batch, Pos))
    labels = batch["labels"]
    
    logits = model(input_ids, key=key, inference=inference)
    
    logits_flat = logits.array.reshape(-1, Vocab.size)
    labels_flat = labels.reshape(-1)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
    return jnp.mean(loss)


def create_optimizer(config: TrainingConfig, num_params: int):
    """Create optimizer with warmup and cosine decay"""
    
    # Ensure warmup doesn't exceed max steps
    warmup = min(config.warmup_steps, config.max_steps // 2)
    decay_steps = max(config.max_steps - warmup, 1)
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )
    
    return optimizer


def train(config: TrainingConfig):
    """Main training loop with gradient accumulation"""
    
    log("=" * 70)
    log("PUBLICATION-READY TRAINING: Time-Indexed Transformer")
    log("=" * 70)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(f"{config.output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Setup
    log(f"\nDevices: {jax.devices()}")
    log(f"\nModel config:")
    log(f"  Hidden dim: {config.hidden_dim}")
    log(f"  Layers: {config.num_layers}")
    log(f"  Heads: {config.num_heads}")
    log(f"  Sequence length: {config.sequence_length}")
    log(f"\nTraining config:")
    log(f"  Micro batch: {config.micro_batch_size}")
    log(f"  Gradient accumulation: {config.gradient_accumulation}")
    log(f"  Effective batch: {config.effective_batch_size}")
    log(f"  Max steps: {config.max_steps}")
    
    # Define axes
    Batch = hax.Axis("batch", config.micro_batch_size)
    Pos = hax.Axis("position", config.sequence_length)
    Vocab = hax.Axis("vocab", config.vocab_size)
    
    # Create model config
    gpt_config = Gpt2Config(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        seq_len=config.sequence_length,
    )
    
    # Initialize model
    log("\nInitializing model...")
    key = jrandom.PRNGKey(42)
    k_model, key = jrandom.split(key)
    
    model = TimeIndexedTransformer.init(
        Vocab=Vocab,
        config=gpt_config,
        num_layers=config.num_layers,
        time_dim=config.time_embed_dim,
        dropout_rate=config.dropout,
        key=k_model,
    )
    
    num_params = count_parameters(model)
    log(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create optimizer
    log("\nCreating optimizer...")
    optimizer = create_optimizer(config, num_params)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Load tokenizer
    log("\nLoading tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loader
    log("\nLoading data...")
    train_loader = OpenWebTextLoader(
        config.data_path, tokenizer, config.micro_batch_size, config.sequence_length
    )
    train_iter = iter(train_loader)
    
    # JIT compile training step (training mode with dropout)
    @eqx.filter_jit
    def compute_grads(model, batch, key):
        def loss_fn(model):
            return compute_loss(model, batch, Batch, Pos, Vocab, key, inference=False)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        return loss, grads
    
    @eqx.filter_jit
    def apply_grads(model, opt_state, grads):
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state
    
    # JIT compile validation step (inference mode, no dropout)
    # KEY FIX: Pass a dummy key for validation to avoid the dropout key error
    @eqx.filter_jit
    def compute_val_loss(model, batch):
        # Use a fixed key for deterministic validation (dropout is disabled via inference=True)
        dummy_key = jrandom.PRNGKey(0)
        return compute_loss(model, batch, Batch, Pos, Vocab, dummy_key, inference=True)
    
    # Load WikiText-103 for validation
    log("\nLoading WikiText-103 validation set...")
    from datasets import load_dataset
    try:
        wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        
        # Concatenate all validation text and tokenize
        all_val_text = "\n".join([t for t in wikitext['text'] if len(t.strip()) > 0])
        log(f"Total validation text length: {len(all_val_text):,} characters")
        
        all_val_tokens = tokenizer.encode(all_val_text)
        log(f"Total validation tokens: {len(all_val_tokens):,}")
        
        # Create validation sequences
        val_tokens = []
        seq_len = config.sequence_length + 1
        for i in range(0, len(all_val_tokens) - seq_len, seq_len):
            val_tokens.append(all_val_tokens[i:i + seq_len])
            if len(val_tokens) >= 500:  # Limit to 500 sequences for efficiency
                break
        
        log(f"Validation samples: {len(val_tokens)} (each {config.sequence_length} tokens)")
    except Exception as e:
        log(f"WARNING: Could not load WikiText-103: {e}")
        import traceback
        traceback.print_exc()
        log("Training without proper validation!")
        val_tokens = []
    
    def compute_validation_loss(model):
        """Compute loss on WikiText-103 validation set"""
        if not val_tokens:
            return None
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, min(len(val_tokens), 100), config.micro_batch_size):
            batch_seqs = val_tokens[i:i + config.micro_batch_size]
            if len(batch_seqs) < config.micro_batch_size:
                continue
            
            batch = {
                "input_ids": jnp.array([s[:-1] for s in batch_seqs], dtype=jnp.int32),
                "labels": jnp.array([s[1:] for s in batch_seqs], dtype=jnp.int32),
            }
            
            loss = compute_val_loss(model, batch)
            total_loss += float(loss)
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    # Training loop
    log("\n" + "=" * 70)
    log("Starting training with PROPER VALIDATION...")
    log(f"Dropout rate: {config.dropout}")
    log(f"Early stopping patience: {config.patience}")
    log("=" * 70 + "\n")
    
    metrics_history = []
    step_times = []
    accumulated_loss = 0.0
    accumulated_grads = None
    
    # Early stopping tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None
    
    global_step = 0
    micro_step = 0
    start_time = time.time()
    
    try:
        while global_step < config.max_steps:
            step_start = time.time()
            
            # Accumulate gradients
            for accum_step in range(config.gradient_accumulation):
                batch = next(train_iter)
                key, subkey = jrandom.split(key)
                
                loss, grads = compute_grads(model, batch, subkey)
                accumulated_loss += float(loss)
                
                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda a, b: a + b, accumulated_grads, grads
                    )
                
                micro_step += 1
            
            # Average gradients
            accumulated_grads = jax.tree_util.tree_map(
                lambda g: g / config.gradient_accumulation, accumulated_grads
            )
            avg_loss = accumulated_loss / config.gradient_accumulation
            
            # Apply gradients
            model, opt_state = apply_grads(model, opt_state, accumulated_grads)
            
            # Reset accumulators
            accumulated_loss = 0.0
            accumulated_grads = None
            
            global_step += 1
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Logging
            if global_step % 10 == 0:
                tokens_per_sec = config.effective_batch_size * config.sequence_length / step_time
                log(f"Step {global_step:6d}/{config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {step_time*1000:.0f}ms | "
                    f"Tok/s: {tokens_per_sec:.0f}")
            
            # Evaluation with proper validation
            if global_step % config.eval_every == 0:
                # Compute training perplexity
                train_ppl = np.exp(avg_loss)
                
                # Compute VALIDATION loss (key for detecting overfitting!)
                val_loss = compute_validation_loss(model)
                val_ppl = np.exp(val_loss) if val_loss is not None else None
                
                metrics = {
                    "step": global_step,
                    "train_loss": avg_loss,
                    "train_ppl": float(train_ppl),
                    "val_loss": float(val_loss) if val_loss else None,
                    "val_ppl": float(val_ppl) if val_ppl else None,
                    "step_time_ms": np.mean(step_times[-100:]) * 1000,
                    "tokens_seen": global_step * config.effective_batch_size * config.sequence_length,
                }
                metrics_history.append(metrics)
                
                if val_loss is not None:
                    log(f"\n📊 Step {global_step}:")
                    log(f"   Train Loss: {avg_loss:.4f} (PPL: {train_ppl:.2f})")
                    log(f"   Val Loss:   {val_loss:.4f} (PPL: {val_ppl:.2f})")
                    
                    # Early stopping check
                    if val_loss < best_val_loss - config.min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        log(f"   ✅ New best validation loss!")
                        
                        # Save best model
                        best_model_path = f"{config.output_dir}/best_model.pkl"
                        with open(best_model_path, "wb") as f:
                            pickle.dump({"model": model, "config": config_dict, "step": global_step, "val_loss": val_loss}, f)
                    else:
                        patience_counter += 1
                        log(f"   ⚠️  No improvement ({patience_counter}/{config.patience})")
                        
                        if patience_counter >= config.patience:
                            log(f"\n🛑 Early stopping triggered! Best val loss: {best_val_loss:.4f}")
                            break
                else:
                    log(f"\n📊 Step {global_step}: Train Loss={avg_loss:.4f}, Train PPL={train_ppl:.2f}")
                
                log("")
                
                # Save metrics
                with open(f"{config.output_dir}/metrics.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)
            
            # Checkpointing
            if global_step % config.save_every == 0:
                checkpoint_path = f"{config.output_dir}/checkpoint_{global_step:06d}.pkl"
                log(f"💾 Saving checkpoint to {checkpoint_path}")
                
                checkpoint = {
                    "model": model,
                    "opt_state": opt_state,
                    "step": global_step,
                    "config": config_dict,
                }
                
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint, f)
                
                # Keep only last 3 checkpoints
                checkpoints = sorted(Path(config.output_dir).glob("checkpoint_*.pkl"))
                for old_ckpt in checkpoints[:-3]:
                    old_ckpt.unlink()
                    log(f"🗑️  Removed old checkpoint: {old_ckpt}")
    
    except KeyboardInterrupt:
        log("\n⚠️  Training interrupted!")
    
    # Final save
    total_time = time.time() - start_time
    log(f"\n{'='*70}")
    log(f"Training complete!")
    log(f"  Total steps: {global_step}")
    log(f"  Total time: {total_time/3600:.2f} hours")
    log(f"  Final loss: {avg_loss:.4f}")
    log(f"{'='*70}")
    
    # Save final model
    final_path = f"{config.output_dir}/final_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"model": model, "config": config_dict}, f)
    log(f"Saved final model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train publication-ready Time-Indexed Transformer")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--time_embed_dim", type=int, default=256)
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for regularization")
    
    # Training
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    # Checkpointing
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2500)
    
    # Paths
    parser.add_argument("--data_path", type=str, default="/data1/fypnahid/qkvflow/openwebtext")
    parser.add_argument("--output_dir", type=str, default="/data1/fypnahid/qkvflow/checkpoints_v2")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        patience=args.patience,
        eval_every=args.eval_every,
        save_every=args.save_every,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    
    train(config)


if __name__ == "__main__":
    main()
