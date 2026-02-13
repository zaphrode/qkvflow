#!/usr/bin/env python3
"""
Efficient data loader for OpenWebText.

Streams data from HuggingFace datasets and tokenizes on-the-fly.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import numpy as np
from typing import Iterator, Tuple
from pathlib import Path


class OpenWebTextLoader:
    """
    Efficient data loader for OpenWebText.
    
    Features:
    - Streams from HuggingFace datasets (no full load into memory)
    - Tokenizes on-the-fly with GPT-2 tokenizer
    - Returns JAX arrays ready for training
    - Handles batching and sequence packing
    """
    
    def __init__(
        self,
        split: str = "train",
        batch_size: int = 8,
        seq_length: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        cache_dir: str = "./data/openwebtext_cache"
    ):
        """
        Initialize data loader.
        
        Args:
            split: 'train' or 'validation'
            batch_size: Number of sequences per batch
            seq_length: Length of each sequence (e.g., 1024 for GPT-2)
            shuffle: Whether to shuffle data
            seed: Random seed
            cache_dir: Where to cache downloaded data
        """
        self.split = split
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.seed = seed
        
        print(f"📦 Initializing OpenWebText loader...")
        print(f"   Split: {split}")
        print(f"   Batch size: {batch_size}")
        print(f"   Sequence length: {seq_length}")
        
        # Load tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load dataset (streaming mode for efficiency)
        print(f"   Loading dataset (streaming mode)...")
        self.dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=True,  # Don't load all data into memory
            cache_dir=cache_dir
        )
        
        # Create train/val split
        # For streaming, we'll use a simple modulo-based split
        # Train: indices % 100 != 0
        # Val: indices % 100 == 0 (1% of data)
        self.is_validation = (split == "validation")
        
        print(f"   ✓ Loader initialized")
    
    def _should_include_sample(self, idx: int) -> bool:
        """Determine if sample should be in this split"""
        is_val_sample = (idx % 100 == 0)
        return is_val_sample if self.is_validation else (not is_val_sample)
    
    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Iterate over batches.
        
        Yields:
            (input_ids, attention_mask) as JAX arrays
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        """
        batch_inputs = []
        batch_masks = []
        
        for idx, example in enumerate(self.dataset):
            # Skip if not in our split
            if not self._should_include_sample(idx):
                continue
            
            # Tokenize
            text = example['text']
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_length,
                padding="max_length",
                return_attention_mask=True
            )
            
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            
            batch_inputs.append(input_ids)
            batch_masks.append(attention_mask)
            
            # Yield batch when full
            if len(batch_inputs) == self.batch_size:
                yield (
                    jnp.array(batch_inputs, dtype=jnp.int32),
                    jnp.array(batch_masks, dtype=jnp.int32)
                )
                batch_inputs = []
                batch_masks = []
    
    def get_batch(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a single random batch (for validation).
        
        Returns:
            (input_ids, attention_mask)
        """
        # For validation, just return next batch from iterator
        batch_iter = iter(self)
        try:
            return next(batch_iter)
        except StopIteration:
            # Reset iterator if exhausted
            batch_iter = iter(self)
            return next(batch_iter)


def test_loader():
    """Test the data loader"""
    print("\n" + "="*70)
    print("TESTING OPENWEBTEXT LOADER")
    print("="*70)
    
    print("\n1. Creating training loader...")
    train_loader = OpenWebTextLoader(
        split="train",
        batch_size=4,
        seq_length=128,  # Shorter for testing
    )
    
    print("\n2. Fetching first batch...")
    batch_iter = iter(train_loader)
    input_ids, attention_mask = next(batch_iter)
    
    print(f"\n3. Batch shapes:")
    print(f"   input_ids: {input_ids.shape}")
    print(f"   attention_mask: {attention_mask.shape}")
    
    print(f"\n4. First sequence (first 50 tokens):")
    print(f"   Token IDs: {input_ids[0, :50]}")
    
    print(f"\n5. Decoded text:")
    decoded = train_loader.tokenizer.decode(input_ids[0])
    print(f"   {decoded[:500]}...")
    
    print("\n✅ Loader test complete!")


if __name__ == "__main__":
    test_loader()





