#!/usr/bin/env python3
"""Plot model comparison including pretrained GPT-2 Small as external baseline."""

import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("checkpoints_v5_baseline/metrics.json") as f:
    baseline = json.load(f)
with open("checkpoints_v5_timeindex/metrics.json") as f:
    small = json.load(f)
with open("checkpoints_v5_timeindex_large/metrics.json") as f:
    large = json.load(f)
with open("gpt2_baseline_results.json") as f:
    gpt2 = json.load(f)

def best_entry(data):
    return min(data, key=lambda x: x["val_loss"] if x.get("val_loss") is not None else 999)

baseline_best = best_entry(baseline)
small_best = best_entry(small)
large_best = best_entry(large)

models = [
    ("GPT-2 Small\n(pretrained, 124M)", gpt2["val_loss"], gpt2["val_ppl"], 124.4, "#4CAF50"),
    ("Baseline\n(LLaMA-style, 152M)", baseline_best["val_loss"], math.exp(min(baseline_best["val_loss"], 20)), 152, "#2196F3"),
    ("Time-Indexed\nLarge (95M)\n[still training]", large_best["val_loss"], math.exp(min(large_best["val_loss"], 20)), 95, "#F44336"),
    ("Time-Indexed\nSmall (50M)", small_best["val_loss"], math.exp(min(small_best["val_loss"], 20)), 50, "#FF9800"),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Bar chart: Val Loss ---
ax = axes[0]
names = [m[0] for m in models]
losses = [m[1] for m in models]
colors = [m[4] for m in models]
x = np.arange(len(models))
bars = ax.bar(x, losses, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

for bar, loss in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{loss:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Validation Loss (WikiText-103)", fontsize=13)
ax.set_title("Validation Loss Comparison", fontsize=14, fontweight='bold')
ax.set_ylim(0, max(losses) * 1.15)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=gpt2["val_loss"], color='#4CAF50', linestyle='--', alpha=0.5, linewidth=1)

# --- Bar chart: Val PPL ---
ax = axes[1]
ppls = [m[2] for m in models]
bars = ax.bar(x, ppls, color=colors, width=0.6, edgecolor='white', linewidth=1.5)

for bar, ppl in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{ppl:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Validation Perplexity (WikiText-103)", fontsize=13)
ax.set_title("Validation Perplexity Comparison", fontsize=14, fontweight='bold')
ax.set_ylim(0, max(ppls) * 1.15)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=gpt2["val_ppl"], color='#4CAF50', linestyle='--', alpha=0.5, linewidth=1)

fig.suptitle("Model Comparison — WikiText-103 Validation\n"
             "GPT-2 pretrained on ~40B tokens  |  Our models trained on ~2-4B tokens (OpenWebText)",
             fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig("model_comparison_with_gpt2.png", dpi=150, bbox_inches='tight')
print("Saved to model_comparison_with_gpt2.png")

# --- Also plot training curves with GPT-2 horizontal line ---
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

def extract(data, key):
    steps = [m["step"] for m in data]
    vals = [m[key] for m in data]
    return steps, vals

ax = axes2[0]
s, v = extract(baseline, "val_loss")
ax.plot(s, v, label="Baseline (LLaMA-style, 152M)", color="#2196F3", linewidth=2)
s, v = extract(large, "val_loss")
ax.plot(s, v, label="Time-Indexed Large (95M)", color="#F44336", linewidth=2)
s, v = extract(small, "val_loss")
ax.plot(s, v, label="Time-Indexed Small (50M)", color="#FF9800", linewidth=2)
ax.axhline(y=gpt2["val_loss"], color='#4CAF50', linestyle='--', linewidth=2,
           label=f'GPT-2 Small pretrained ({gpt2["val_loss"]:.2f})')
ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Validation Loss", fontsize=13)
ax.set_title("Training Progress vs GPT-2 Baseline", fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes2[1]
s, v = extract(baseline, "val_ppl")
ax.plot(s, v, label="Baseline (LLaMA-style, 152M)", color="#2196F3", linewidth=2)
s, v = extract(large, "val_ppl")
ax.plot(s, v, label="Time-Indexed Large (95M)", color="#F44336", linewidth=2)
s, v = extract(small, "val_ppl")
ax.plot(s, v, label="Time-Indexed Small (50M)", color="#FF9800", linewidth=2)
ax.axhline(y=gpt2["val_ppl"], color='#4CAF50', linestyle='--', linewidth=2,
           label=f'GPT-2 Small pretrained ({gpt2["val_ppl"]:.1f})')
ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Validation Perplexity", fontsize=13)
ax.set_title("Training Progress vs GPT-2 Baseline", fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

fig2.suptitle("Training Curves with Pretrained GPT-2 Reference\n"
              "(GPT-2 trained on ~40B tokens; our models on ~2-4B tokens of OpenWebText)",
              fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig("training_curves_with_gpt2.png", dpi=150, bbox_inches='tight')
print("Saved to training_curves_with_gpt2.png")
