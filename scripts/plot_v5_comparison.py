#!/usr/bin/env python3
"""Plot v5 training comparison: Baseline vs Time-Indexed Large vs Time-Indexed Small"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

with open("checkpoints_v5_baseline/metrics.json") as f:
    baseline = json.load(f)
with open("checkpoints_v5_timeindex/metrics.json") as f:
    small = json.load(f)
with open("checkpoints_v5_timeindex_large/metrics.json") as f:
    large = json.load(f)

max_step = max(m["step"] for m in large)

def extract(data, key, max_step):
    steps = [m["step"] for m in data if m["step"] <= max_step]
    vals = [m[key] for m in data if m["step"] <= max_step]
    return steps, vals

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Val Loss ---
ax = axes[0]
s, v = extract(baseline, "val_loss", max_step)
ax.plot(s, v, label="Baseline (152M, 12 ind. layers)", color="#2196F3", linewidth=2)
s, v = extract(large, "val_loss", max_step)
ax.plot(s, v, label="Time-Indexed Large (95M, shared)", color="#F44336", linewidth=2)
s, v = extract(small, "val_loss", max_step)
ax.plot(s, v, label="Time-Indexed Small (50M, shared)", color="#FF9800", linewidth=2)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Validation Loss", fontsize=13)
ax.set_title("Validation Loss vs Steps", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_step + 500)

# --- Val PPL ---
ax = axes[1]
s, v = extract(baseline, "val_ppl", max_step)
ax.plot(s, v, label="Baseline (152M, 12 ind. layers)", color="#2196F3", linewidth=2)
s, v = extract(large, "val_ppl", max_step)
ax.plot(s, v, label="Time-Indexed Large (95M, shared)", color="#F44336", linewidth=2)
s, v = extract(small, "val_ppl", max_step)
ax.plot(s, v, label="Time-Indexed Small (50M, shared)", color="#FF9800", linewidth=2)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Validation Perplexity", fontsize=13)
ax.set_title("Validation Perplexity vs Steps", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_step + 500)

fig.suptitle("V5 Training Comparison (OpenWebText, seq_len=512, WikiText-103 val)",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("v5_training_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved to v5_training_comparison.png")

# Also plot full range with all data
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# --- Val Loss (full) ---
ax = axes2[0]
s, v = extract(baseline, "val_loss", 999999)
ax.plot(s, v, label=f"Baseline (152M) — final {v[-1]:.3f}", color="#2196F3", linewidth=2)
s, v = extract(large, "val_loss", 999999)
ax.plot(s, v, label=f"Time-Idx Large (95M) — latest {v[-1]:.3f}", color="#F44336", linewidth=2)
s, v = extract(small, "val_loss", 999999)
ax.plot(s, v, label=f"Time-Idx Small (50M) — final {v[-1]:.3f}", color="#FF9800", linewidth=2)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Validation Loss", fontsize=13)
ax.set_title("Validation Loss — Full Training", fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# --- Val PPL (full) ---
ax = axes2[1]
s, v = extract(baseline, "val_ppl", 999999)
ax.plot(s, v, label=f"Baseline (152M) — final {v[-1]:.1f}", color="#2196F3", linewidth=2)
s, v = extract(large, "val_ppl", 999999)
ax.plot(s, v, label=f"Time-Idx Large (95M) — latest {v[-1]:.1f}", color="#F44336", linewidth=2)
s, v = extract(small, "val_ppl", 999999)
ax.plot(s, v, label=f"Time-Idx Small (50M) — final {v[-1]:.1f}", color="#FF9800", linewidth=2)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Validation Perplexity", fontsize=13)
ax.set_title("Validation Perplexity — Full Training", fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

fig2.suptitle("V5 Training Comparison — All Available Data",
              fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("v5_training_comparison_full.png", dpi=150, bbox_inches='tight')
print(f"Saved to v5_training_comparison_full.png")
