#!/usr/bin/env python3
"""
Measure FLOPs and wall-clock time for different models.

This script provides rigorous compute budget comparisons beyond just parameter counts.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import time
import pickle
from dataclasses import dataclass
from typing import Dict, Any

from levanter.models.gpt2 import Gpt2Config
from test_time_indexed_weights import TimeIndexedTransformer, StandardTransformer
from test_time_indexed_ssm import TimeIndexedSSMTransformer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    seq_len: int = 128
    vocab_size: int = 256
    batch_size: int = 8
    num_warmup: int = 10
    num_runs: int = 100


def count_parameters(*models) -> int:
    """Count total trainable parameters."""
    total = 0
    for model in models:
        params, _ = eqx.partition(model, eqx.is_array)
        leaves = jax.tree_util.tree_leaves(params)
        total += sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))
    return total


def estimate_flops(config: BenchmarkConfig, model_type: str) -> Dict[str, float]:
    """
    Estimate FLOPs for a forward pass.
    
    Based on standard transformer FLOP counting:
    - Attention: 2 * batch * seq_len^2 * hidden_dim (Q@K^T and attn@V)
    - MLP: 2 * batch * seq_len * hidden_dim * mlp_dim (up and down projections)
    - Per layer: attention + MLP FLOPs
    """
    B = config.batch_size
    L = config.seq_len
    D = config.hidden_dim
    H = config.num_heads
    d_head = D // H
    N = config.num_layers
    M = 4 * D  # MLP hidden dim (standard 4x expansion)
    
    # Attention FLOPs per layer
    # QKV projection: 3 * B * L * D * D
    qkv_proj = 3 * B * L * D * D
    # Attention scores: B * H * L * L * d_head
    attn_scores = B * H * L * L * d_head
    # Attention output: B * H * L * L * d_head
    attn_out = B * H * L * L * d_head
    # Output projection: B * L * D * D
    out_proj = B * L * D * D
    attn_flops_per_layer = qkv_proj + attn_scores + attn_out + out_proj
    
    # MLP FLOPs per layer
    # Up projection: B * L * D * M
    up_proj = B * L * D * M
    # Down projection: B * L * M * D
    down_proj = B * L * M * D
    mlp_flops_per_layer = up_proj + down_proj
    
    # Total per layer
    flops_per_layer = attn_flops_per_layer + mlp_flops_per_layer
    
    if model_type == "standard":
        # Standard transformer: full FLOPs for each layer
        total_flops = N * flops_per_layer
    elif model_type == "time_indexed_mlp":
        # Time-indexed: shared weights, but still do the same compute
        # Plus: time embedding MLP overhead (small)
        time_embed_flops = N * (64 * 256 + 256 * D)  # Sinusoidal -> scaling factors
        total_flops = N * flops_per_layer + time_embed_flops
    elif model_type == "time_indexed_ssm":
        # SSM: attention same, MLP replaced with SSM
        # SSM FLOPs: B * L * D * state_dim (recurrent computation)
        ssm_state_dim = 64
        ssm_flops_per_layer = B * L * D * ssm_state_dim * 2  # input and state update
        total_flops = N * (attn_flops_per_layer + ssm_flops_per_layer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Embedding and output head
    embed_flops = B * L * config.vocab_size * D
    head_flops = B * L * D * config.vocab_size
    total_flops += embed_flops + head_flops
    
    return {
        'total_flops': total_flops,
        'flops_per_layer': flops_per_layer,
        'attn_flops': attn_flops_per_layer,
        'mlp_flops': mlp_flops_per_layer,
    }


def measure_wallclock_time(forward_fn, num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
    """
    Measure wall-clock time with warmup.
    
    Returns:
        - mean_time_ms: Average time per forward pass (milliseconds)
        - std_time_ms: Standard deviation
        - throughput_samples_per_sec: Samples per second
    """
    # Warmup
    for _ in range(num_warmup):
        forward_fn()
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        forward_fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = jnp.array(times)
    
    return {
        'mean_time_ms': float(jnp.mean(times)),
        'std_time_ms': float(jnp.std(times)),
        'min_time_ms': float(jnp.min(times)),
        'max_time_ms': float(jnp.max(times)),
        'throughput_samples_per_sec': 1000.0 / float(jnp.mean(times)),
    }


def benchmark_model(model_type: str, config: BenchmarkConfig, seed: int = 42) -> Dict[str, Any]:
    """Benchmark a single model: FLOPs and wall-clock time."""
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_type.upper()}")
    print(f"{'='*70}")
    
    key = jrandom.PRNGKey(seed)
    
    # Setup axes
    Batch = hax.Axis("batch", config.batch_size)
    Pos = hax.Axis("position", config.seq_len)
    Embed = hax.Axis("embed", config.hidden_dim)
    Vocab = hax.Axis("vocab", config.vocab_size)
    Heads = hax.Axis("heads", config.num_heads)
    HeadSize = hax.Axis("head_size", config.hidden_dim // config.num_heads)
    SinusodialDim = hax.Axis("sinusoidal_dim", 64)
    TembedDim = hax.Axis("tembed_dim", 256)
    
    # Create config
    gpt2_config = Gpt2Config(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        gradient_checkpointing=False,
        use_bias=False,
    )
    gpt2_config.Heads = Heads
    gpt2_config.HeadSize = HeadSize
    gpt2_config.Embed = Embed
    gpt2_config.Vocab = Vocab
    gpt2_config.Pos = Pos
    gpt2_config.Mlp = hax.Axis("mlp", 4 * config.hidden_dim)
    
    # Initialize model
    k_model, k_emb, k_head, key = jrandom.split(key, 4)
    
    if model_type == "standard":
        transformer = StandardTransformer(Vocab, Embed, Heads, HeadSize, config.num_layers, key=k_model)
        embedder = hnn.Embedding.init(Vocab, Embed, key=k_emb)
        lm_head = hnn.Linear.init(Embed, Vocab, key=k_head, use_bias=False)
    elif model_type == "time_indexed_mlp":
        transformer = TimeIndexedTransformer.init(gpt2_config, SinusodialDim, TembedDim, key=k_model)
        embedder = hnn.Embedding.init(Vocab, Embed, key=k_emb)
        lm_head = hnn.Linear.init(Embed, Vocab, key=k_head, use_bias=False)
    elif model_type == "time_indexed_ssm":
        transformer = TimeIndexedSSMTransformer.init(gpt2_config, SinusodialDim, TembedDim, ssm_state_size=64, key=k_model)
        embedder = hnn.Embedding.init(Vocab, Embed, key=k_emb)
        lm_head = hnn.Linear.init(Embed, Vocab, key=k_head, use_bias=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    num_params = count_parameters(transformer, embedder, lm_head)
    print(f"📊 Parameters: {num_params:,}")
    
    # Estimate FLOPs
    flop_estimates = estimate_flops(config, model_type)
    gflops = flop_estimates['total_flops'] / 1e9
    print(f"💾 Estimated FLOPs per forward: {gflops:.2f} GFLOP")
    
    # Create dummy input
    dummy_input = jrandom.randint(key, (config.batch_size, config.seq_len), 0, config.vocab_size)
    
    # Define forward function
    if model_type == "standard":
        @jax.jit
        def forward():
            return transformer(dummy_input, Batch, Pos, key=key)
    else:
        mask_val = hax.arange(Pos) <= hax.arange(Pos).broadcast_axis(Pos.alias("key_position"))
        @jax.jit
        def forward():
            inputs = hax.named(dummy_input, (Batch, Pos))
            embedded = embedder(inputs)
            hidden = transformer(embedded, mask_val, key=key)
            logits = lm_head(hidden)
            return logits.array
    
    # Measure wall-clock time
    print("⏱️  Measuring wall-clock time...")
    timing_results = measure_wallclock_time(forward, num_warmup=config.num_warmup, num_runs=config.num_runs)
    
    print(f"   Mean time: {timing_results['mean_time_ms']:.2f} ± {timing_results['std_time_ms']:.2f} ms")
    print(f"   Throughput: {timing_results['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Compute FLOPs/sec
    flops_per_sec = flop_estimates['total_flops'] * timing_results['throughput_samples_per_sec']
    tflops_per_sec = flops_per_sec / 1e12
    print(f"   TFLOP/s: {tflops_per_sec:.2f}")
    
    # Compute efficiency metrics
    flops_per_param = flop_estimates['total_flops'] / num_params
    print(f"   FLOPs per parameter: {flops_per_param:.1f}")
    
    return {
        'model_type': model_type,
        'num_params': num_params,
        'flop_estimates': flop_estimates,
        'timing': timing_results,
        'flops_per_sec': flops_per_sec,
        'tflops_per_sec': tflops_per_sec,
        'flops_per_param': flops_per_param,
    }


def run_all_benchmarks(config: BenchmarkConfig, seed: int = 42):
    """Run benchmarks for all models and compare."""
    
    print("\n" + "="*70)
    print("FLOPS AND WALL-CLOCK TIME BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Warmup runs: {config.num_warmup}")
    print(f"  Benchmark runs: {config.num_runs}")
    
    results = {}
    for model_type in ["standard", "time_indexed_mlp", "time_indexed_ssm"]:
        try:
            results[model_type] = benchmark_model(model_type, config, seed)
        except Exception as e:
            print(f"\n❌ Error benchmarking {model_type}: {e}")
            continue
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<25} {'Params':<15} {'Time (ms)':<15} {'GFLOP':<12} {'TFLOP/s':<12}")
    print("-"*79)
    
    for model_type, result in results.items():
        model_name = model_type.replace("_", " ").title()
        params_m = result['num_params'] / 1e6
        time_ms = result['timing']['mean_time_ms']
        gflops = result['flop_estimates']['total_flops'] / 1e9
        tflops_s = result['tflops_per_sec']
        
        print(f"{model_name:<25} {params_m:>7.1f}M       {time_ms:>7.2f}        {gflops:>7.2f}     {tflops_s:>7.2f}")
    
    # Relative comparisons
    if "standard" in results and "time_indexed_mlp" in results:
        print(f"\n📊 Time-Indexed MLP vs Standard:")
        std_time = results["standard"]['timing']['mean_time_ms']
        ti_time = results["time_indexed_mlp"]['timing']['mean_time_ms']
        speedup = std_time / ti_time
        print(f"   Speed: {speedup:.2f}× {'faster' if speedup > 1 else 'slower'}")
        
        std_params = results["standard"]['num_params']
        ti_params = results["time_indexed_mlp"]['num_params']
        compression = std_params / ti_params
        print(f"   Compression: {compression:.1f}×")
        
        std_flops = results["standard"]['flop_estimates']['total_flops']
        ti_flops = results["time_indexed_mlp"]['flop_estimates']['total_flops']
        flop_ratio = ti_flops / std_flops
        print(f"   Relative FLOPs: {flop_ratio:.2f}× ({'more' if flop_ratio > 1 else 'fewer'} compute)")
    
    # Save results
    output_file = "flops_benchmark_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    config = BenchmarkConfig(
        hidden_dim=256,
        num_heads=4,
        num_layers=6,
        seq_len=128,
        vocab_size=256,
        batch_size=8,
        num_warmup=10,
        num_runs=100,
    )
    
    results = run_all_benchmarks(config, seed=42)

