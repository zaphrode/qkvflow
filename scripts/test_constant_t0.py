#!/usr/bin/env python3
"""
Constant t=0 Transformer for Ablation Study

This is a CRITICAL scientific control. It tests whether gains come from:
- Time-indexing (variable t) OR
- Just having MLP adapter structure (constant t=0)

This file is identical to test_time_indexed_weights.py except:
- Line 294: t = 0.0 (CONSTANT) instead of (layer_idx + 1) / num_layers (VARIABLE)
- Line 301: time_embed at 0.0 (CONSTANT) instead of 1.0
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
from typing import Optional
from dataclasses import dataclass

from levanter.models.gpt2 import Gpt2Config
from qkvflow.nn.time_embed import SinusoidalPosEmb
from qkvflow.nn.dynamic import TemporalLayerNorm
from haliax.jax_utils import maybe_rng_split, named_call

# Import the time-indexed components from the original file
from test_time_indexed_weights import TimeIndexedAttention, TimeIndexedMLP, TimeIndexedBlock


class ConstantT0Transformer(eqx.Module):
    """Transformer with CONSTANT t=0 modulation (ablation baseline).
    
    This uses the SAME architecture as TimeIndexedTransformer, but:
    - All layers use t=0.0 (constant)
    - NOT t=(layer_idx+1)/num_layers (variable)
    
    If this performs as well as TimeIndexedTransformer:
      → Gain is from MLP adapters, not time-dependency
    
    If TimeIndexedTransformer performs better:
      → Time-dependency (Neural ODE) is crucial
    """
    
    config: Gpt2Config = eqx.field(static=True)
    time_emb: SinusoidalPosEmb
    shared_block: TimeIndexedBlock  # Same block as TimeIndexedTransformer
    ln_f: TemporalLayerNorm
    dropout: hnn.Dropout
    
    @staticmethod
    def init(config, SinusodialDim, TembedDim, *, key):
        k_emb, k_block, k_ln = jrandom.split(key, 3)
        
        time_emb = SinusoidalPosEmb.init(SinusodialDim, key=k_emb)
        
        # Resize for time embedding output
        SinusodialDim_out = SinusodialDim.resize(SinusodialDim.size * 2 + 1)
        
        # Single shared block (same as TimeIndexedTransformer)
        shared_block = TimeIndexedBlock.init(
            config, SinusodialDim_out, TembedDim, key=k_block
        )
        
        ln_f = TemporalLayerNorm.init(
            config.Embed, eps=config.layer_norm_epsilon,
            use_bias=config.use_bias, TembedDim=TembedDim,
            SinusodialDim=SinusodialDim_out, key=k_ln
        )
        
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)
        
        return ConstantT0Transformer(config, time_emb, shared_block, ln_f, dropout)
    
    @named_call
    def __call__(self, x, *, key):
        keys = maybe_rng_split(key, self.config.num_layers + 1)
        
        # Create causal mask
        Pos = x.resolve_axis("position")
        attn_mask = hax.nn.attention.causal_mask(Pos, Pos)
        
        # Apply dropout to embeddings
        x = self.dropout(x, key=keys[0])
        
        # CRITICAL DIFFERENCE: Use t=0.0 for ALL layers (constant)
        # This is the ablation control
        t = 0.0  # ← CONSTANT (vs variable in TimeIndexedTransformer)
        time_embed = self.time_emb(hax.named(jnp.array(t), ()))
        
        # Apply the SAME block multiple times with SAME time embedding
        for layer_idx in range(self.config.num_layers):
            # Note: time_embed is the SAME for all layers
            x = self.shared_block(time_embed, x, attn_mask, layer_idx, key=keys[layer_idx + 1])
        
        # Final layer norm at t=0.0 (constant)
        x = self.ln_f(time_embed, x)
        
        return hax.auto_sharded(x)


if __name__ == "__main__":
    print("This module defines ConstantT0Transformer for ablation studies.")
    print("Use in compare_ablation_t0.py for experiments.")

