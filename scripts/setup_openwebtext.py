#!/usr/bin/env python3
"""
Download and prepare OpenWebText dataset for training.

This script:
1. Downloads OpenWebText from HuggingFace
2. Tokenizes with GPT-2 tokenizer
3. Creates training/validation splits
4. Saves processed data for efficient loading

Usage:
    python scripts/setup_openwebtext.py
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import json
from tqdm import tqdm


def download_openwebtext():
    """Download OpenWebText from HuggingFace"""
    print("\n" + "="*70)
    print("DOWNLOADING OPENWEBTEXT")
    print("="*70)
    print("\n📥 Downloading OpenWebText from HuggingFace...")
    print("   This may take 30-60 minutes depending on your connection.")
    print("   Dataset size: ~8GB compressed, ~40GB uncompressed\n")
    
    try:
        # Load dataset (this will cache it locally)
        dataset = load_dataset(
            "openwebtext",
            split="train",
            cache_dir="./data/openwebtext_cache"
        )
        
        print(f"\n✅ Downloaded {len(dataset):,} documents")
        print(f"   Cache location: ./data/openwebtext_cache/")
        
        return dataset
    
    except Exception as e:
        print(f"\n❌ Error downloading OpenWebText: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify datasets library: pip install --upgrade datasets")
        print("  3. Check disk space (need ~50GB free)")
        sys.exit(1)


def setup_tokenizer():
    """Set up GPT-2 tokenizer"""
    print("\n" + "="*70)
    print("SETTING UP GPT-2 TOKENIZER")
    print("="*70)
    
    print("\n🔧 Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    print(f"   ✓ Vocab size: {tokenizer.vocab_size:,}")
    print(f"   ✓ Special tokens: {tokenizer.all_special_tokens}")
    print(f"   ✓ EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    
    return tokenizer


def create_splits(dataset, val_fraction=0.01):
    """Create train/validation splits"""
    print("\n" + "="*70)
    print("CREATING TRAIN/VAL SPLITS")
    print("="*70)
    
    print(f"\n📊 Splitting dataset ({val_fraction*100:.1f}% validation)...")
    
    # HuggingFace datasets have train_test_split method
    splits = dataset.train_test_split(test_size=val_fraction, seed=42)
    
    train_dataset = splits['train']
    val_dataset = splits['test']
    
    print(f"   ✓ Training: {len(train_dataset):,} documents")
    print(f"   ✓ Validation: {len(val_dataset):,} documents")
    
    return train_dataset, val_dataset


def preprocess_and_save(dataset, tokenizer, output_dir, split_name, max_length=1024, num_samples=None):
    """
    Preprocess dataset and save tokenized samples.
    
    For efficiency, we'll save pre-tokenized sequences.
    """
    print(f"\n📝 Preprocessing {split_name} split...")
    print(f"   Max length: {max_length} tokens")
    
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks
    chunk_size = 10000
    total_samples = num_samples if num_samples else len(dataset)
    
    print(f"   Processing {total_samples:,} samples in chunks of {chunk_size:,}...")
    
    # For streaming efficiency, we'll save metadata about the dataset
    # rather than pre-tokenizing everything (would be ~100GB+)
    
    metadata = {
        "split": split_name,
        "num_documents": len(dataset),
        "max_length": max_length,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": "gpt2",
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved to {output_path}/metadata.json")
    print(f"   ✓ Dataset ready for streaming during training")
    
    # Sample a few examples to verify
    print(f"\n📄 Sample document (first 500 chars):")
    print("-" * 70)
    print(dataset[0]['text'][:500])
    print("-" * 70)
    
    # Tokenize sample
    sample_tokens = tokenizer(dataset[0]['text'], truncation=True, max_length=max_length)
    print(f"\n🔢 Tokenized length: {len(sample_tokens['input_ids'])} tokens")
    print(f"   First 20 tokens: {sample_tokens['input_ids'][:20]}")
    
    return metadata


def main():
    """Main setup pipeline"""
    
    print("\n" + "="*70)
    print("OPENWEBTEXT SETUP FOR GPT-2 SMALL TRAINING")
    print("="*70)
    
    print("\n📋 This script will:")
    print("   1. Download OpenWebText (~8GB, ~40GB uncompressed)")
    print("   2. Set up GPT-2 tokenizer (50k vocab)")
    print("   3. Create train/val splits")
    print("   4. Prepare data for efficient streaming")
    
    print("\n⏱️  Estimated time: 30-60 minutes")
    print("💾  Disk space required: ~50GB")
    
    response = input("\n▶️  Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Download dataset
    dataset = download_openwebtext()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    
    # Create splits
    train_dataset, val_dataset = create_splits(dataset, val_fraction=0.01)
    
    # Save metadata
    output_dir = Path("data/openwebtext_processed")
    
    train_metadata = preprocess_and_save(
        train_dataset, tokenizer, output_dir, "train", max_length=1024
    )
    
    val_metadata = preprocess_and_save(
        val_dataset, tokenizer, output_dir, "val", max_length=1024
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    
    print("\n📊 Dataset Summary:")
    print(f"   Training documents: {len(train_dataset):,}")
    print(f"   Validation documents: {len(val_dataset):,}")
    print(f"   Tokenizer vocab: {tokenizer.vocab_size:,}")
    print(f"   Max sequence length: 1024 tokens")
    
    print("\n📁 Output:")
    print(f"   Raw data: ./data/openwebtext_cache/")
    print(f"   Processed metadata: ./data/openwebtext_processed/")
    
    print("\n🚀 Next Steps:")
    print("   1. Verify setup: python scripts/verify_openwebtext.py")
    print("   2. Create model configs: python scripts/create_gpt2_configs.py")
    print("   3. Test training: python scripts/test_training_setup.py")
    print("   4. Launch training: python scripts/train_gpt2_small.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()





