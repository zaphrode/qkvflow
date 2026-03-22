"""
Standalone configuration for GPT-2 style models.
Replaces levanter.models.gpt2 to avoid dependency conflicts.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Callable
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import haliax as hax
import haliax.nn as hnn


@dataclass
class Gpt2Config:
    """GPT-2 model configuration"""
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    seq_len: int = 1024
    vocab_size: int = 50257
    mlp_ratio: float = 4.0
    activation_function: str = "gelu"
    layer_norm_epsilon: float = 1e-5
    dropout: float = 0.0
    
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


# Activation functions
def _gelu(x):
    return hnn.gelu(x)

def _relu(x):
    return hnn.relu(x)

def _silu(x):
    return hnn.silu(x)

ACT2FN: Dict[str, Callable] = {
    "gelu": _gelu,
    "relu": _relu,
    "silu": _silu,
    "swish": _silu,
}


def dot_product_attention(
    query: hax.NamedArray,
    key: hax.NamedArray,
    value: hax.NamedArray,
    mask: Optional[hax.NamedArray] = None,
    dropout: float = 0.0,
    key_rng: Optional[jax.random.PRNGKey] = None,
) -> hax.NamedArray:
    """Compute scaled dot-product attention."""
    # Get head dimension for scaling
    head_dim = query.resolve_axis("head_size").size
    scale = head_dim ** -0.5
    
    # Compute attention scores
    attn_weights = hax.dot("head_size", query, key) * scale
    
    # Apply mask if provided
    if mask is not None:
        attn_weights = hax.where(mask, attn_weights, -1e9)
    
    # Softmax over key positions
    key_pos = key.resolve_axis("key_position")
    attn_weights = hnn.softmax(attn_weights, axis=key_pos)
    
    # Apply dropout
    if dropout > 0.0 and key_rng is not None:
        keep_prob = 1.0 - dropout
        mask = jrandom.bernoulli(key_rng, keep_prob, attn_weights.array.shape)
        attn_weights = hax.named(attn_weights.array * mask / keep_prob, attn_weights.axes)
    
    # Apply attention to values
    output = hax.dot(key_pos, attn_weights, value)
    
    return output


class Gpt2Embeddings(eqx.Module):
    """Token and position embeddings for GPT-2"""
    Vocab: hax.Axis = eqx.field(static=True)
    config: Gpt2Config = eqx.field(static=True)
    
    token_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding
    dropout: hnn.Dropout
    
    @staticmethod
    def init(Vocab: hax.Axis, config: Gpt2Config, *, key):
        k1, k2 = jrandom.split(key)
        
        return Gpt2Embeddings(
            Vocab=Vocab,
            config=config,
            token_embeddings=hnn.Embedding.init(Vocab, config.Embed, key=k1),
            position_embeddings=hnn.Embedding.init(config.Pos, config.Embed, key=k2),
            dropout=hnn.Dropout(config.dropout),
        )
    
    def embed(self, input_ids: hax.NamedArray, *, key=None) -> hax.NamedArray:
        """Embed tokens and add position embeddings"""
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Position embeddings - use input's actual position axis
        input_pos = input_ids.resolve_axis("position")
        seq_len = input_pos.size
        
        # Create position indices with the input's position axis (not config.Pos)
        # This ensures axis consistency when sequence length differs from config
        positions = hax.named(jnp.arange(seq_len), input_pos)
        pos_emb = self.position_embeddings(positions)
        
        x = x + pos_emb
        
        # Apply dropout if key is provided
        if key is not None:
            x = self.dropout(x, key=key)
        
        return x
    
    def unembed(self, x: hax.NamedArray) -> hax.NamedArray:
        """Project hidden states to vocabulary logits"""
        return self.token_embeddings.unembed(x)
