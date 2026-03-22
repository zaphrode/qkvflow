#!/usr/bin/env python3
"""
Evaluate HuggingFace pretrained GPT-2 Small (124M) on WikiText-103 validation.

Uses the EXACT same evaluation protocol as train_v2.py:
- WikiText-103 raw v1, validation split
- GPT-2 tokenizer, seq_len=512
- Up to 500 sequences, evaluate first 200
- Cross-entropy loss (no label smoothing)
"""

import json
import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

SEQ_LEN = 512
MAX_SEQUENCES = 500
EVAL_SEQUENCES = 200
BATCH_SIZE = 4

def main():
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("\nLoading GPT-2 Small (124M)...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading WikiText-103 validation set...")
    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    all_val_text = "\n".join([t for t in wikitext['text'] if len(t.strip()) > 0])
    print(f"Validation text: {len(all_val_text):,} characters")

    all_val_tokens = tokenizer.encode(all_val_text)
    print(f"Validation tokens: {len(all_val_tokens):,}")

    val_sequences = []
    seq_len_plus_one = SEQ_LEN + 1
    for i in range(0, len(all_val_tokens) - seq_len_plus_one, seq_len_plus_one):
        val_sequences.append(all_val_tokens[i:i + seq_len_plus_one])
        if len(val_sequences) >= MAX_SEQUENCES:
            break
    print(f"Validation sequences: {len(val_sequences)} (each {SEQ_LEN} tokens)")

    print(f"\nEvaluating on first {EVAL_SEQUENCES} sequences (batch_size={BATCH_SIZE})...")
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, min(len(val_sequences), EVAL_SEQUENCES), BATCH_SIZE):
            batch_seqs = val_sequences[i:i + BATCH_SIZE]
            if len(batch_seqs) < BATCH_SIZE:
                continue

            input_ids = torch.tensor([s[:-1] for s in batch_seqs], dtype=torch.long).to(device)
            labels = torch.tensor([s[1:] for s in batch_seqs], dtype=torch.long).to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat)

            total_loss += loss.item()
            num_batches += 1

            if (i // BATCH_SIZE) % 10 == 0:
                running_loss = total_loss / num_batches
                running_ppl = math.exp(min(running_loss, 20))
                print(f"  Batch {num_batches}: running loss={running_loss:.4f}, ppl={running_ppl:.2f}")

    avg_loss = total_loss / num_batches
    ppl = math.exp(min(avg_loss, 20))

    print(f"\n{'='*60}")
    print(f"GPT-2 Small (124M) -- WikiText-103 Validation")
    print(f"{'='*60}")
    print(f"  Val Loss:       {avg_loss:.4f}")
    print(f"  Val Perplexity: {ppl:.2f}")
    print(f"  Parameters:     {num_params:,}")
    print(f"  Sequences:      {num_batches * BATCH_SIZE}")
    print(f"  Seq Length:     {SEQ_LEN}")
    print(f"{'='*60}")

    results = {
        "model": "gpt2-small-pretrained",
        "params": num_params,
        "val_loss": avg_loss,
        "val_ppl": ppl,
        "eval_sequences": num_batches * BATCH_SIZE,
        "seq_len": SEQ_LEN,
    }
    with open("gpt2_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to gpt2_baseline_results.json")


if __name__ == "__main__":
    main()
