#!/usr/bin/env python3
"""
V2 Training: Modern Transformer Architecture with Optional Time-Indexing

Architecture upgrades over v1:
- RoPE (Rotary Position Embeddings) instead of learned absolute positions
- RMSNorm instead of LayerNorm
- SwiGLU activation instead of GELU
- Proper weight initialization (scaled by 1/sqrt(2*n_layers))
- Label smoothing
- Trapezoidal LR schedule (warmup → stable → decay)
- Larger effective batch size

Supports two modes:
  --mode baseline    → Standard Transformer (Model A)
  --mode time_index  → Time-Indexed Transformer (Model B)
"""

import os
import sys
import json
import pickle
import time
import argparse
import math
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax
import haliax as hax
import haliax.nn as hnn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    seq_len: int = 512
    vocab_size: int = 50257
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # Time-indexed specific
    time_embed_dim: int = 256

    @property
    def Embed(self) -> hax.Axis:
        return hax.Axis("embed", self.hidden_dim)

    @property
    def Heads(self) -> hax.Axis:
        return hax.Axis("heads", self.num_heads)

    @property
    def HeadSize(self) -> hax.Axis:
        return hax.Axis("head_size", self.hidden_dim // self.num_heads)

    @property
    def Pos(self) -> hax.Axis:
        return hax.Axis("position", self.seq_len)

    @property
    def Mlp(self) -> hax.Axis:
        return hax.Axis("mlp", int(self.hidden_dim * self.mlp_ratio))

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass
class TrainingConfig:
    micro_batch_size: int = 4
    gradient_accumulation: int = 64
    sequence_length: int = 512
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    stable_steps: int = 20000
    max_steps: int = 150000
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    patience: int = 15
    min_delta: float = 0.005
    eval_every: int = 500
    save_every: int = 5000
    data_path: str = "/data1/fypnahid/qkvflow/openwebtext"
    output_dir: str = "/data1/fypnahid/qkvflow/checkpoints_v5"
    mode: str = "time_index"  # "baseline" or "time_index"

    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(eqx.Module):
    axis: hax.Axis = eqx.field(static=True)
    weight: hax.NamedArray
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(axis: hax.Axis, eps: float = 1e-6):
        weight = hax.ones(axis)
        return RMSNorm(axis=axis, weight=weight, eps=eps)

    def __call__(self, x: hax.NamedArray) -> hax.NamedArray:
        variance = (x * x).mean(self.axis)
        inv = hax.rsqrt(variance + self.eps)
        return self.weight * (x * inv)


# =============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================

def precompute_rope_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute sin/cos for RoPE."""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half_dim).astype(jnp.float32) / half_dim))
    positions = jnp.arange(seq_len).astype(jnp.float32)
    angles = jnp.outer(positions, freqs)
    cos_vals = jnp.cos(angles)
    sin_vals = jnp.sin(angles)
    return cos_vals, sin_vals


def apply_rope(x_array: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embeddings to tensor. x shape: (..., seq_len, head_dim)"""
    half = x_array.shape[-1] // 2
    x1 = x_array[..., :half]
    x2 = x_array[..., half:]
    seq_len = x_array.shape[-2]
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return jnp.concatenate([out1, out2], axis=-1)


# =============================================================================
# DROPOUT UTILITY
# =============================================================================

def apply_dropout(x: hax.NamedArray, rate: float, key, inference: bool) -> hax.NamedArray:
    if inference or rate <= 0 or key is None:
        return x
    mask = jrandom.bernoulli(key, 1.0 - rate, x.array.shape)
    return hax.named(x.array * mask / (1.0 - rate), x.axes)


# =============================================================================
# SwiGLU MLP
# =============================================================================

class SwiGLUMLP(eqx.Module):
    """SwiGLU MLP: gate(x) * up(x) → down"""
    w_gate: hnn.Linear
    w_up: hnn.Linear
    w_down: hnn.Linear
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, *, key):
        k1, k2, k3 = jrandom.split(key, 3)
        Embed = config.Embed
        Mlp = config.Mlp
        return SwiGLUMLP(
            w_gate=hnn.Linear.init(Embed, Mlp, key=k1, use_bias=False),
            w_up=hnn.Linear.init(Embed, Mlp, key=k2, use_bias=False),
            w_down=hnn.Linear.init(Mlp, Embed, key=k3, use_bias=False),
            dropout_rate=config.dropout,
        )

    def __call__(self, x: hax.NamedArray, *, key=None, inference: bool = False) -> hax.NamedArray:
        gate = hnn.silu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        x = apply_dropout(x, self.dropout_rate, key, inference)
        x = self.w_down(x)
        return x


# =============================================================================
# ATTENTION (shared between baseline and time-indexed)
# =============================================================================

class MultiHeadAttention(eqx.Module):
    """Multi-head attention with RoPE."""
    config: ModelConfig = eqx.field(static=True)
    w_q: hnn.Linear
    w_k: hnn.Linear
    w_v: hnn.Linear
    w_o: hnn.Linear
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, *, key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        Embed = config.Embed
        HeadDim = hax.Axis("head_dim", config.num_heads * config.head_dim)
        return MultiHeadAttention(
            config=config,
            w_q=hnn.Linear.init(Embed, HeadDim, key=k1, use_bias=False),
            w_k=hnn.Linear.init(Embed, HeadDim, key=k2, use_bias=False),
            w_v=hnn.Linear.init(Embed, HeadDim, key=k3, use_bias=False),
            w_o=hnn.Linear.init(HeadDim, Embed, key=k4, use_bias=False),
            dropout_rate=config.dropout,
        )

    def __call__(self, x: hax.NamedArray, mask, rope_cos, rope_sin,
                 *, key=None, inference: bool = False) -> hax.NamedArray:
        cfg = self.config
        HeadDim = hax.Axis("head_dim", cfg.num_heads * cfg.head_dim)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.unflatten_axis(HeadDim, (cfg.Heads, cfg.HeadSize))
        k = k.unflatten_axis(HeadDim, (cfg.Heads, cfg.HeadSize))
        v = v.unflatten_axis(HeadDim, (cfg.Heads, cfg.HeadSize))

        # Apply RoPE to Q and K
        q_arr = q.array  # (batch?, heads, seq, head_size)
        k_arr = k.array

        # RoPE needs to operate on the last two dims: (seq, head_size)
        # Array shape depends on axis ordering; apply via reshape
        q_shape = q_arr.shape
        k_shape = k_arr.shape
        q_2d = q_arr.reshape(-1, q_shape[-2], q_shape[-1])
        k_2d = k_arr.reshape(-1, k_shape[-2], k_shape[-1])
        q_2d = apply_rope(q_2d, rope_cos, rope_sin)
        k_2d = apply_rope(k_2d, rope_cos, rope_sin)
        q = hax.named(q_2d.reshape(q_shape), q.axes)
        k = hax.named(k_2d.reshape(k_shape), k.axes)

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        KeyPos = k.resolve_axis("key_position")

        scale = cfg.head_dim ** -0.5
        attn = hax.dot(cfg.HeadSize, q, k) * scale

        if mask is not None:
            attn = hax.where(mask, attn, -1e9)

        attn = hnn.softmax(attn, axis=KeyPos)

        if not inference and self.dropout_rate > 0 and key is not None:
            k1, key = jrandom.split(key)
            attn = apply_dropout(attn, self.dropout_rate, k1, inference)

        out = hax.dot(KeyPos, attn, v)
        out = out.flatten_axes((cfg.Heads, cfg.HeadSize), HeadDim)
        out = self.w_o(out)
        return out


# =============================================================================
# TIME-MODULATED VARIANTS
# =============================================================================

class TimeEmbedding(eqx.Module):
    embed_dim: int = eqx.field(static=True)

    @staticmethod
    def init(embed_dim: int):
        return TimeEmbedding(embed_dim=embed_dim)

    def __call__(self, t: float) -> jnp.ndarray:
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
        args = t * freqs
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)])


class TimeModulation(eqx.Module):
    """Applies time-dependent scale and shift to a hidden representation."""
    scale_proj: jnp.ndarray
    shift_proj: jnp.ndarray
    time_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    @staticmethod
    def init(time_dim: int, hidden_dim: int, *, key):
        k1, k2 = jrandom.split(key)
        return TimeModulation(
            scale_proj=jrandom.normal(k1, (time_dim, hidden_dim)) * 0.01,
            shift_proj=jnp.zeros((time_dim, hidden_dim)),
            time_dim=time_dim,
            hidden_dim=hidden_dim,
        )

    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray) -> hax.NamedArray:
        scale = 1.0 + jnp.tanh(time_embed @ self.scale_proj)
        shift = time_embed @ self.shift_proj
        return hax.named(x.array * scale + shift, x.axes)


class TimeModulatedAttention(eqx.Module):
    """MHA with RoPE + time modulation on Q/K projections."""
    config: ModelConfig = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)
    w_q: hnn.Linear
    w_k: hnn.Linear
    w_v: hnn.Linear
    w_o: hnn.Linear
    time_mod_q: TimeModulation
    time_mod_k: TimeModulation
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, time_dim: int, *, key):
        k1, k2, k3, k4, k5, k6 = jrandom.split(key, 6)
        Embed = config.Embed
        HeadDim = hax.Axis("head_dim", config.num_heads * config.head_dim)
        return TimeModulatedAttention(
            config=config,
            time_dim=time_dim,
            w_q=hnn.Linear.init(Embed, HeadDim, key=k1, use_bias=False),
            w_k=hnn.Linear.init(Embed, HeadDim, key=k2, use_bias=False),
            w_v=hnn.Linear.init(Embed, HeadDim, key=k3, use_bias=False),
            w_o=hnn.Linear.init(HeadDim, Embed, key=k4, use_bias=False),
            time_mod_q=TimeModulation.init(time_dim, HeadDim.size, key=k5),
            time_mod_k=TimeModulation.init(time_dim, HeadDim.size, key=k6),
            dropout_rate=config.dropout,
        )

    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray, mask,
                 rope_cos, rope_sin, *, key=None, inference: bool = False) -> hax.NamedArray:
        cfg = self.config
        HeadDim = hax.Axis("head_dim", cfg.num_heads * cfg.head_dim)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Time modulation on Q and K
        q = self.time_mod_q(q, time_embed)
        k = self.time_mod_k(k, time_embed)

        q = q.unflatten_axis(HeadDim, (cfg.Heads, cfg.HeadSize))
        k = k.unflatten_axis(HeadDim, (cfg.Heads, cfg.HeadSize))
        v = v.unflatten_axis(HeadDim, (cfg.Heads, cfg.HeadSize))

        # RoPE
        q_shape = q.array.shape
        k_shape = k.array.shape
        q_2d = q.array.reshape(-1, q_shape[-2], q_shape[-1])
        k_2d = k.array.reshape(-1, k_shape[-2], k_shape[-1])
        q_2d = apply_rope(q_2d, rope_cos, rope_sin)
        k_2d = apply_rope(k_2d, rope_cos, rope_sin)
        q = hax.named(q_2d.reshape(q_shape), q.axes)
        k = hax.named(k_2d.reshape(k_shape), k.axes)

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        KeyPos = k.resolve_axis("key_position")

        scale = cfg.head_dim ** -0.5
        attn = hax.dot(cfg.HeadSize, q, k) * scale

        if mask is not None:
            attn = hax.where(mask, attn, -1e9)
        attn = hnn.softmax(attn, axis=KeyPos)

        if not inference and self.dropout_rate > 0 and key is not None:
            k1, key = jrandom.split(key)
            attn = apply_dropout(attn, self.dropout_rate, k1, inference)

        out = hax.dot(KeyPos, attn, v)
        out = out.flatten_axes((cfg.Heads, cfg.HeadSize), HeadDim)
        out = self.w_o(out)
        return out


class TimeModulatedSwiGLU(eqx.Module):
    """SwiGLU with time-modulated gating."""
    w_gate: hnn.Linear
    w_up: hnn.Linear
    w_down: hnn.Linear
    time_mod_gate: TimeModulation
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, time_dim: int, *, key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        Embed = config.Embed
        Mlp = config.Mlp
        return TimeModulatedSwiGLU(
            w_gate=hnn.Linear.init(Embed, Mlp, key=k1, use_bias=False),
            w_up=hnn.Linear.init(Embed, Mlp, key=k2, use_bias=False),
            w_down=hnn.Linear.init(Mlp, Embed, key=k3, use_bias=False),
            time_mod_gate=TimeModulation.init(time_dim, Mlp.size, key=k4),
            dropout_rate=config.dropout,
        )

    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray,
                 *, key=None, inference: bool = False) -> hax.NamedArray:
        gate = self.w_gate(x)
        gate = self.time_mod_gate(gate, time_embed)
        gate = hnn.silu(gate)
        up = self.w_up(x)
        x = gate * up
        x = apply_dropout(x, self.dropout_rate, key, inference)
        x = self.w_down(x)
        return x


# =============================================================================
# TRANSFORMER BLOCKS
# =============================================================================

class BaselineBlock(eqx.Module):
    """Standard transformer block: RMSNorm + MHA(RoPE) + SwiGLU"""
    norm1: RMSNorm
    attn: MultiHeadAttention
    norm2: RMSNorm
    mlp: SwiGLUMLP
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, *, key):
        k1, k2 = jrandom.split(key)
        return BaselineBlock(
            norm1=RMSNorm.init(config.Embed, eps=config.rms_norm_eps),
            attn=MultiHeadAttention.init(config, key=k1),
            norm2=RMSNorm.init(config.Embed, eps=config.rms_norm_eps),
            mlp=SwiGLUMLP.init(config, key=k2),
            dropout_rate=config.dropout,
        )

    def __call__(self, x: hax.NamedArray, mask, rope_cos, rope_sin,
                 *, key=None, inference: bool = False) -> hax.NamedArray:
        if key is not None:
            k1, k2, k3 = jrandom.split(key, 3)
        else:
            k1 = k2 = k3 = None

        h = self.attn(self.norm1(x), mask, rope_cos, rope_sin, key=k1, inference=inference)
        h = apply_dropout(h, self.dropout_rate, k2, inference)
        x = x + h

        h = self.mlp(self.norm2(x), key=k3, inference=inference)
        h = apply_dropout(h, self.dropout_rate, k3, inference)
        x = x + h
        return x


class TimeIndexedBlock(eqx.Module):
    """Time-indexed block: RMSNorm + TimeModulatedMHA(RoPE) + TimeModulatedSwiGLU"""
    norm1: RMSNorm
    attn: TimeModulatedAttention
    norm2: RMSNorm
    mlp: TimeModulatedSwiGLU
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(config: ModelConfig, time_dim: int, *, key):
        k1, k2 = jrandom.split(key)
        return TimeIndexedBlock(
            norm1=RMSNorm.init(config.Embed, eps=config.rms_norm_eps),
            attn=TimeModulatedAttention.init(config, time_dim, key=k1),
            norm2=RMSNorm.init(config.Embed, eps=config.rms_norm_eps),
            mlp=TimeModulatedSwiGLU.init(config, time_dim, key=k2),
            dropout_rate=config.dropout,
        )

    def __call__(self, x: hax.NamedArray, time_embed: jnp.ndarray, mask,
                 rope_cos, rope_sin, *, key=None, inference: bool = False) -> hax.NamedArray:
        if key is not None:
            k1, k2, k3, k4 = jrandom.split(key, 4)
        else:
            k1 = k2 = k3 = k4 = None

        h = self.attn(self.norm1(x), time_embed, mask, rope_cos, rope_sin,
                      key=k1, inference=inference)
        h = apply_dropout(h, self.dropout_rate, k2, inference)
        x = x + h

        h = self.mlp(self.norm2(x), time_embed, key=k3, inference=inference)
        h = apply_dropout(h, self.dropout_rate, k4, inference)
        x = x + h
        return x


# =============================================================================
# FULL MODELS
# =============================================================================

class BaselineTransformer(eqx.Module):
    """Standard Transformer with modern defaults (Model A)."""
    config: ModelConfig = eqx.field(static=True)
    token_embeddings: hnn.Embedding
    blocks: list
    norm_f: RMSNorm
    rope_cos: jnp.ndarray = eqx.field(static=True)
    rope_sin: jnp.ndarray = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(Vocab: hax.Axis, config: ModelConfig, *, key):
        k_emb, key = jrandom.split(key)
        blocks = []
        for i in range(config.num_layers):
            k_block, key = jrandom.split(key)
            blocks.append(BaselineBlock.init(config, key=k_block))

        cos, sin = precompute_rope_frequencies(config.head_dim, config.seq_len, config.rope_theta)
        return BaselineTransformer(
            config=config,
            token_embeddings=hnn.Embedding.init(Vocab, config.Embed, key=k_emb),
            blocks=blocks,
            norm_f=RMSNorm.init(config.Embed, eps=config.rms_norm_eps),
            rope_cos=cos,
            rope_sin=sin,
            dropout_rate=config.dropout,
        )

    def __call__(self, input_ids: hax.NamedArray, *, key=None, inference: bool = False):
        if key is None and inference:
            key = jrandom.PRNGKey(0)

        x = self.token_embeddings(input_ids)

        k_drop, key = jrandom.split(key) if key is not None else (None, None)
        x = apply_dropout(x, self.dropout_rate, k_drop, inference)

        Pos = input_ids.resolve_axis("position")
        KeyPos = Pos.alias("key_position")
        mask = hnn.attention.causal_mask(Pos, KeyPos)

        for i, block in enumerate(self.blocks):
            layer_key = jrandom.fold_in(key, i) if key is not None else None
            x = block(x, mask, self.rope_cos, self.rope_sin, key=layer_key, inference=inference)

        x = self.norm_f(x)
        logits = self.token_embeddings.unembed(x)
        return logits


class TimeIndexedTransformerV2(eqx.Module):
    """Time-Indexed Transformer with modern defaults (Model B).
    Single block applied N times with continuous time embeddings."""
    config: ModelConfig = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)
    token_embeddings: hnn.Embedding
    time_embed: TimeEmbedding
    block: TimeIndexedBlock
    norm_f: RMSNorm
    rope_cos: jnp.ndarray = eqx.field(static=True)
    rope_sin: jnp.ndarray = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    @staticmethod
    def init(Vocab: hax.Axis, config: ModelConfig, *, key):
        k_emb, k_block, key = jrandom.split(key, 3)
        time_dim = config.time_embed_dim
        cos, sin = precompute_rope_frequencies(config.head_dim, config.seq_len, config.rope_theta)
        return TimeIndexedTransformerV2(
            config=config,
            num_layers=config.num_layers,
            time_dim=time_dim,
            token_embeddings=hnn.Embedding.init(Vocab, config.Embed, key=k_emb),
            time_embed=TimeEmbedding.init(time_dim),
            block=TimeIndexedBlock.init(config, time_dim, key=k_block),
            norm_f=RMSNorm.init(config.Embed, eps=config.rms_norm_eps),
            rope_cos=cos,
            rope_sin=sin,
            dropout_rate=config.dropout,
        )

    def __call__(self, input_ids: hax.NamedArray, *, key=None, inference: bool = False):
        if key is None and inference:
            key = jrandom.PRNGKey(0)

        x = self.token_embeddings(input_ids)

        k_drop, key = jrandom.split(key) if key is not None else (None, None)
        x = apply_dropout(x, self.dropout_rate, k_drop, inference)

        Pos = input_ids.resolve_axis("position")
        KeyPos = Pos.alias("key_position")
        mask = hnn.attention.causal_mask(Pos, KeyPos)

        for layer_idx in range(self.num_layers):
            t = layer_idx / max(1, self.num_layers - 1)
            time_emb = self.time_embed(t)
            layer_key = jrandom.fold_in(key, layer_idx) if key is not None else None
            x = self.block(x, time_emb, mask, self.rope_cos, self.rope_sin,
                          key=layer_key, inference=inference)

        x = self.norm_f(x)
        logits = self.token_embeddings.unembed(x)
        return logits


# =============================================================================
# DATA LOADING
# =============================================================================

class OpenWebTextLoader:
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
        token_buffer = []
        example_idx = 0
        while True:
            while len(token_buffer) < self.batch_size * (self.seq_len + 1) * 2:
                if example_idx >= self.num_examples:
                    example_idx = 0
                text = self.dataset[example_idx]['text']
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                example_idx += 1
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
# TRAINING
# =============================================================================

def count_parameters(model) -> int:
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def compute_loss(model, batch, Batch, Pos, Vocab, key, label_smoothing=0.0, inference=False):
    input_ids = hax.named(batch["input_ids"], (Batch, Pos))
    labels = batch["labels"]
    logits = model(input_ids, key=key, inference=inference)

    logits_flat = logits.array.reshape(-1, Vocab.size)
    labels_flat = labels.reshape(-1)

    if label_smoothing > 0 and not inference:
        num_classes = Vocab.size
        one_hot = jax.nn.one_hot(labels_flat, num_classes)
        smooth = one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.sum(smooth * log_probs, axis=-1)
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
    return jnp.mean(loss)


def create_optimizer(config: TrainingConfig):
    """Trapezoidal LR: warmup → stable at peak → cosine decay"""
    warmup = config.warmup_steps
    stable = config.stable_steps
    total = config.max_steps
    decay_steps = max(total - warmup - stable, 1)
    peak_lr = config.learning_rate
    end_lr = peak_lr * config.min_lr_ratio

    def schedule(step):
        # Warmup phase
        warmup_lr = peak_lr * jnp.minimum(step / jnp.maximum(warmup, 1), 1.0)
        # Stable phase
        # Cosine decay phase
        decay_step = jnp.maximum(step - warmup - stable, 0)
        decay_frac = jnp.minimum(decay_step / decay_steps, 1.0)
        cosine_lr = end_lr + 0.5 * (peak_lr - end_lr) * (1.0 + jnp.cos(jnp.pi * decay_frac))
        # Select phase
        lr = jnp.where(step < warmup, warmup_lr,
                        jnp.where(step < warmup + stable, peak_lr, cosine_lr))
        return lr

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay, b1=0.9, b2=0.95),
    )
    return optimizer


def train(config: TrainingConfig, resume_from: Optional[str] = None):
    hidden_dim = getattr(config, '_hidden_dim', 768)
    num_heads = getattr(config, '_num_heads', 12)
    num_layers = getattr(config, '_num_layers', 12)
    model_config = ModelConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=config.sequence_length,
        dropout=config.dropout,
        time_embed_dim=256,
    )

    mode_label = "BASELINE (Model A)" if config.mode == "baseline" else "TIME-INDEXED (Model B)"
    log("=" * 70)
    log(f"V2 TRAINING: {mode_label}")
    log("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)
    config_dict = {k: v for k, v in config.__dict__.items()}
    config_dict["model_config"] = {k: v for k, v in model_config.__dict__.items()
                                    if not callable(v) and not k.startswith("_")}
    with open(f"{config.output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    log(f"\nDevices: {jax.devices()}")
    log(f"\nArchitecture: RoPE + RMSNorm + SwiGLU")
    log(f"Mode: {config.mode}")
    log(f"Hidden: {model_config.hidden_dim}, Layers: {model_config.num_layers}, Heads: {model_config.num_heads}")
    log(f"Sequence length: {config.sequence_length}")
    log(f"\nTraining:")
    log(f"  Micro batch: {config.micro_batch_size}")
    log(f"  Gradient accumulation: {config.gradient_accumulation}")
    log(f"  Effective batch: {config.effective_batch_size}")
    log(f"  Peak LR: {config.learning_rate}")
    log(f"  Schedule: warmup({config.warmup_steps}) → stable({config.stable_steps}) → cosine decay")
    log(f"  Label smoothing: {config.label_smoothing}")
    log(f"  Max steps: {config.max_steps}")

    Batch = hax.Axis("batch", config.micro_batch_size)
    Pos = hax.Axis("position", config.sequence_length)
    Vocab = hax.Axis("vocab", model_config.vocab_size)

    log("\nInitializing model...")
    key = jrandom.PRNGKey(42)
    k_model, key = jrandom.split(key)

    if config.mode == "baseline":
        model = BaselineTransformer.init(Vocab=Vocab, config=model_config, key=k_model)
    else:
        model = TimeIndexedTransformerV2.init(Vocab=Vocab, config=model_config, key=k_model)

    num_params = count_parameters(model)
    log(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    log("\nCreating optimizer...")
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Resume from checkpoint
    resume_step = 0
    if resume_from is not None:
        log(f"\n📂 Resuming from checkpoint: {resume_from}")
        with open(resume_from, "rb") as f:
            ckpt = pickle.load(f)
        model = ckpt["model"]
        opt_state = ckpt["opt_state"]
        resume_step = ckpt["step"]
        log(f"   Restored model and optimizer at step {resume_step}")

        # Load existing metrics
        metrics_path = f"{config.output_dir}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
            log(f"   Loaded {len(existing_metrics)} existing metric entries")
        else:
            existing_metrics = []

    log("\nLoading tokenizer...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    log("\nLoading data...")
    train_loader = OpenWebTextLoader(config.data_path, tokenizer, config.micro_batch_size, config.sequence_length)
    train_iter = iter(train_loader)

    @eqx.filter_jit
    def compute_grads(model, batch, key):
        def loss_fn(model):
            return compute_loss(model, batch, Batch, Pos, Vocab, key,
                              label_smoothing=config.label_smoothing, inference=False)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        return loss, grads

    @eqx.filter_jit
    def apply_grads(model, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state

    @eqx.filter_jit
    def compute_val_loss(model, batch):
        dummy_key = jrandom.PRNGKey(0)
        return compute_loss(model, batch, Batch, Pos, Vocab, dummy_key,
                          label_smoothing=0.0, inference=True)

    # Load WikiText-103 validation
    log("\nLoading WikiText-103 validation set...")
    from datasets import load_dataset
    try:
        wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        all_val_text = "\n".join([t for t in wikitext['text'] if len(t.strip()) > 0])
        log(f"Validation text: {len(all_val_text):,} chars")
        all_val_tokens = tokenizer.encode(all_val_text)
        log(f"Validation tokens: {len(all_val_tokens):,}")

        val_tokens = []
        seq_len = config.sequence_length + 1
        for i in range(0, len(all_val_tokens) - seq_len, seq_len):
            val_tokens.append(all_val_tokens[i:i + seq_len])
            if len(val_tokens) >= 500:
                break
        log(f"Validation samples: {len(val_tokens)}")
    except Exception as e:
        log(f"WARNING: Could not load WikiText-103: {e}")
        val_tokens = []

    def compute_validation_loss(model):
        if not val_tokens:
            return None
        total_loss = 0.0
        num_batches = 0
        for i in range(0, min(len(val_tokens), 200), config.micro_batch_size):
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
    log(f"\n{'='*70}")
    log("Starting training...")
    log(f"{'='*70}\n")

    metrics_history = []
    accumulated_loss = 0.0
    accumulated_grads = None
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    start_time = time.time()

    if resume_from is not None:
        global_step = resume_step
        metrics_history = existing_metrics
        # Restore best_val_loss from existing metrics
        val_losses = [m["val_loss"] for m in metrics_history if m.get("val_loss") is not None]
        if val_losses:
            best_val_loss = min(val_losses)
            log(f"   Restored best val loss: {best_val_loss:.4f}")
        log(f"   Resuming training from step {global_step}")

    try:
        while global_step < config.max_steps:
            step_start = time.time()

            for accum_step in range(config.gradient_accumulation):
                batch = next(train_iter)
                key, subkey = jrandom.split(key)
                loss, grads = compute_grads(model, batch, subkey)
                accumulated_loss += float(loss)
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(lambda a, b: a + b, accumulated_grads, grads)

            accumulated_grads = jax.tree_util.tree_map(
                lambda g: g / config.gradient_accumulation, accumulated_grads)
            avg_loss = accumulated_loss / config.gradient_accumulation

            model, opt_state = apply_grads(model, opt_state, accumulated_grads)
            accumulated_loss = 0.0
            accumulated_grads = None
            global_step += 1

            step_time = time.time() - step_start

            if global_step % 10 == 0:
                tokens_per_sec = config.effective_batch_size * config.sequence_length / step_time
                log(f"Step {global_step:6d}/{config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {step_time*1000:.0f}ms | "
                    f"Tok/s: {tokens_per_sec:.0f}")

            if global_step % config.eval_every == 0:
                train_ppl = np.exp(min(avg_loss, 20))
                val_loss = compute_validation_loss(model)
                val_ppl = np.exp(min(val_loss, 20)) if val_loss is not None else None

                metrics = {
                    "step": global_step,
                    "train_loss": avg_loss,
                    "train_ppl": float(train_ppl),
                    "val_loss": float(val_loss) if val_loss else None,
                    "val_ppl": float(val_ppl) if val_ppl else None,
                    "tokens_seen": global_step * config.effective_batch_size * config.sequence_length,
                }
                metrics_history.append(metrics)

                if val_loss is not None:
                    log(f"\n📊 Step {global_step}:")
                    log(f"   Train Loss: {avg_loss:.4f} (PPL: {train_ppl:.2f})")
                    log(f"   Val Loss:   {val_loss:.4f} (PPL: {val_ppl:.2f})")

                    if val_loss < best_val_loss - config.min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        log(f"   ✅ New best val loss!")
                        best_path = f"{config.output_dir}/best_model.pkl"
                        with open(best_path, "wb") as f:
                            pickle.dump({"model": model, "config": config_dict, "step": global_step}, f)
                    else:
                        patience_counter += 1
                        log(f"   ⚠️  No improvement ({patience_counter}/{config.patience})")
                        if patience_counter >= config.patience:
                            log(f"\n🛑 Early stopping! Best val loss: {best_val_loss:.4f}")
                            break
                    log("")

                with open(f"{config.output_dir}/metrics.json", "w") as f:
                    json.dump(metrics_history, f, indent=2)

            if global_step % config.save_every == 0:
                ckpt_path = f"{config.output_dir}/checkpoint_{global_step:06d}.pkl"
                log(f"💾 Saving {ckpt_path}")
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"model": model, "opt_state": opt_state, "step": global_step, "config": config_dict}, f)
                checkpoints = sorted(Path(config.output_dir).glob("checkpoint_*.pkl"))
                for old_ckpt in checkpoints[:-3]:
                    old_ckpt.unlink()

    except KeyboardInterrupt:
        log("\n⚠️ Training interrupted!")

    total_time = time.time() - start_time
    log(f"\n{'='*70}")
    log(f"Training complete!")
    log(f"  Mode: {config.mode}")
    log(f"  Steps: {global_step}")
    log(f"  Time: {total_time/3600:.2f} hours")
    log(f"  Best val loss: {best_val_loss:.4f} (PPL: {np.exp(min(best_val_loss, 20)):.2f})")
    log(f"{'='*70}")

    final_path = f"{config.output_dir}/final_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"model": model, "config": config_dict}, f)
    log(f"Saved final model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="V2 Training: Modern Transformer")
    parser.add_argument("--mode", type=str, default="time_index", choices=["baseline", "time_index"])
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--stable_steps", type=int, default=20000)
    parser.add_argument("--max_steps", type=int, default=150000)
    parser.add_argument("--patience", type=int, default=999)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--data_path", type=str, default="/data1/fypnahid/qkvflow/openwebtext")
    parser.add_argument("--output_dir", type=str, default="/data1/fypnahid/qkvflow/checkpoints_v5")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pkl to resume from")
    args = parser.parse_args()

    config = TrainingConfig(
        mode=args.mode,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        max_steps=args.max_steps,
        patience=args.patience,
        eval_every=args.eval_every,
        save_every=args.save_every,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    config._hidden_dim = args.hidden_dim
    config._num_heads = args.num_heads
    config._num_layers = args.num_layers
    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
