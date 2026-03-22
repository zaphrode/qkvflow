#!/usr/bin/env python3
"""Comprehensive benchmark plots: all models vs GPT-2 Small."""

import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})

with open("checkpoints_v5_baseline/metrics.json") as f:
    baseline_152 = json.load(f)
with open("checkpoints_v5_baseline_95m/metrics.json") as f:
    baseline_95 = json.load(f)
with open("checkpoints_v5_timeindex/metrics.json") as f:
    ti_small = json.load(f)
with open("checkpoints_v5_timeindex_large/metrics.json") as f:
    ti_large = json.load(f)
with open("gpt2_baseline_results.json") as f:
    gpt2 = json.load(f)

MODELS = {
    "Baseline 152M": {"data": baseline_152, "color": "#2196F3", "params": 151.9, "ls": "-"},
    "Baseline 95M":  {"data": baseline_95,  "color": "#03A9F4", "params": 93.8,  "ls": "--"},
    "TI-Large 95M":  {"data": ti_large,     "color": "#F44336", "params": 94.5,  "ls": "-"},
    "TI-Small 50M":  {"data": ti_small,     "color": "#FF9800", "params": 50.4,  "ls": "-"},
}

def extract(data, key):
    steps = [m["step"] for m in data]
    vals = [m[key] for m in data]
    return np.array(steps), np.array(vals)

def best(data):
    return min(data, key=lambda x: x["val_loss"] if x.get("val_loss") is not None else 999)

# =========================================================================
# FIGURE 1: Training curves (Val Loss) with GPT-2 reference
# =========================================================================
fig1, ax = plt.subplots(figsize=(10, 6))
for name, m in MODELS.items():
    s, v = extract(m["data"], "val_loss")
    ax.plot(s, v, label=f'{name} ({m["params"]:.0f}M)', color=m["color"],
            linewidth=2, linestyle=m["ls"])
ax.axhline(y=gpt2["val_loss"], color='#4CAF50', linestyle='--', linewidth=2, alpha=0.7,
           label=f'GPT-2 Small pretrained (124M) — {gpt2["val_loss"]:.2f}')
ax.set_xlabel("Training Steps")
ax.set_ylabel("Validation Loss (WikiText-103)")
ax.set_title("Training Progress — All Models vs GPT-2 Small", fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(3.0, 8.5)
fig1.tight_layout()
fig1.savefig("fig1_val_loss_curves.png", dpi=150, bbox_inches='tight')
print("Saved fig1_val_loss_curves.png")

# =========================================================================
# FIGURE 2: Training curves (Val PPL) — zoomed to useful range
# =========================================================================
fig2, ax = plt.subplots(figsize=(10, 6))
for name, m in MODELS.items():
    s, v = extract(m["data"], "val_ppl")
    ax.plot(s, v, label=f'{name}', color=m["color"], linewidth=2, linestyle=m["ls"])
ax.axhline(y=gpt2["val_ppl"], color='#4CAF50', linestyle='--', linewidth=2, alpha=0.7,
           label=f'GPT-2 Small (124M) — PPL {gpt2["val_ppl"]:.1f}')
ax.set_xlabel("Training Steps")
ax.set_ylabel("Validation Perplexity")
ax.set_title("Validation Perplexity — All Models vs GPT-2 Small", fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(20, 300)
fig2.tight_layout()
fig2.savefig("fig2_val_ppl_curves.png", dpi=150, bbox_inches='tight')
print("Saved fig2_val_ppl_curves.png")

# =========================================================================
# FIGURE 3: Apples-to-apples — Baseline 95M vs TI-Large 95M
# =========================================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

max_shared_step = min(
    max(m["step"] for m in baseline_95),
    max(m["step"] for m in ti_large),
)

s1, v1 = extract(baseline_95, "val_loss")
s2, v2 = extract(ti_large, "val_loss")
mask1 = s1 <= max_shared_step
mask2 = s2 <= max_shared_step
ax1.plot(s1[mask1], v1[mask1], label="Baseline 95M (independent layers)", color="#03A9F4", linewidth=2.5)
ax1.plot(s2[mask2], v2[mask2], label="TI-Large 95M (shared + modulated)", color="#F44336", linewidth=2.5)
ax1.axhline(y=gpt2["val_loss"], color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'GPT-2 Small ({gpt2["val_loss"]:.2f})')
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Validation Loss")
ax1.set_title("Equal Params (~95M): Val Loss", fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

s1, v1 = extract(baseline_95, "val_ppl")
s2, v2 = extract(ti_large, "val_ppl")
ax2.plot(s1[mask1], v1[mask1], label="Baseline 95M", color="#03A9F4", linewidth=2.5)
ax2.plot(s2[mask2], v2[mask2], label="TI-Large 95M", color="#F44336", linewidth=2.5)
ax2.axhline(y=gpt2["val_ppl"], color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'GPT-2 Small (PPL {gpt2["val_ppl"]:.1f})')
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Validation Perplexity")
ax2.set_title("Equal Params (~95M): Val Perplexity", fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(60, 300)

fig3.suptitle("Apples-to-Apples: Same Parameter Budget (~95M), Same Training Steps",
              fontsize=14, fontweight='bold', y=1.02)
fig3.tight_layout()
fig3.savefig("fig3_apples_to_apples.png", dpi=150, bbox_inches='tight')
print("Saved fig3_apples_to_apples.png")

# =========================================================================
# FIGURE 4: Bar chart — Best PPL per model + GPT-2
# =========================================================================
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

model_names = [
    "GPT-2 Small\n(pretrained)\n124M",
    "Baseline\n(152M)",
    "Baseline\n(95M)\n[still training]",
    "TI-Large\n(95M)\n[stopped @44k]",
    "TI-Small\n(50M)",
]
best_losses = [
    gpt2["val_loss"],
    best(baseline_152)["val_loss"],
    best(baseline_95)["val_loss"],
    best(ti_large)["val_loss"],
    best(ti_small)["val_loss"],
]
best_ppls = [
    gpt2["val_ppl"],
    best(baseline_152)["val_ppl"],
    math.exp(min(best(baseline_95)["val_loss"], 20)),
    best(ti_large)["val_ppl"],
    best(ti_small)["val_ppl"],
]
colors = ["#4CAF50", "#2196F3", "#03A9F4", "#F44336", "#FF9800"]

x = np.arange(len(model_names))
bars = ax1.bar(x, best_losses, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, best_losses):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=9)
ax1.set_ylabel("Best Validation Loss")
ax1.set_title("Best Validation Loss — All Models", fontweight='bold')
ax1.set_ylim(0, max(best_losses) * 1.15)
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(y=gpt2["val_loss"], color='#4CAF50', linestyle='--', alpha=0.4, linewidth=1)

bars = ax2.bar(x, best_ppls, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, best_ppls):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=9)
ax2.set_ylabel("Best Validation Perplexity")
ax2.set_title("Best Validation Perplexity — All Models", fontweight='bold')
ax2.set_ylim(0, max(best_ppls) * 1.15)
ax2.grid(True, axis='y', alpha=0.3)
ax2.axhline(y=gpt2["val_ppl"], color='#4CAF50', linestyle='--', alpha=0.4, linewidth=1)

fig4.suptitle("Model Comparison — WikiText-103 Validation\n"
              "GPT-2 pretrained on ~40B tokens | Our models on 3-6B tokens (OpenWebText)",
              fontsize=13, fontweight='bold', y=1.04)
fig4.tight_layout()
fig4.savefig("fig4_bar_comparison.png", dpi=150, bbox_inches='tight')
print("Saved fig4_bar_comparison.png")

# =========================================================================
# FIGURE 5: Parameter efficiency — PPL vs Param Count
# =========================================================================
fig5, ax = plt.subplots(figsize=(9, 6))

param_counts = [gpt2["params"]/1e6, 151.9, 93.8, 94.5, 50.4]
ppls = best_ppls
labels = ["GPT-2 Small\n(pretrained, ~40B tok)", "Baseline 152M\n(3.3B tok)",
          "Baseline 95M\n(3.2B tok)", "TI-Large 95M\n(5.4B tok)", "TI-Small 50M\n(3.6B tok)"]
colors_scatter = ["#4CAF50", "#2196F3", "#03A9F4", "#F44336", "#FF9800"]
markers = ["*", "s", "s", "o", "o"]
sizes = [300, 150, 150, 150, 150]

for i, (px, py, lab, col, mk, sz) in enumerate(zip(param_counts, ppls, labels, colors_scatter, markers, sizes)):
    ax.scatter(px, py, c=col, s=sz, marker=mk, zorder=5, edgecolors='white', linewidth=1.5)
    offset_x = 3 if i != 3 else -3
    ha = 'left' if i != 3 else 'right'
    ax.annotate(lab, (px, py), textcoords="offset points",
                xytext=(offset_x, 10), ha=ha, fontsize=8.5, color=col, fontweight='bold')

ax.set_xlabel("Parameters (Millions)")
ax.set_ylabel("Best Validation Perplexity")
ax.set_title("Parameter Efficiency: PPL vs Model Size", fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(30, 170)
ax.set_ylim(20, 130)

ax.annotate('', xy=(94.5, 77.6), xytext=(93.8, 81.8),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(80, 79, 'Same params\ndifferent arch', fontsize=8, color='gray', ha='center', style='italic')

fig5.tight_layout()
fig5.savefig("fig5_param_efficiency.png", dpi=150, bbox_inches='tight')
print("Saved fig5_param_efficiency.png")

# =========================================================================
# FIGURE 6: Val loss vs tokens seen (fairer compute comparison)
# =========================================================================
fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name, m in MODELS.items():
    tokens = [d.get("tokens_seen", d["step"] * 131072) for d in m["data"]]
    tokens_b = [t / 1e9 for t in tokens]
    losses = [d["val_loss"] for d in m["data"]]
    ppls_curve = [d["val_ppl"] for d in m["data"]]
    ax1.plot(tokens_b, losses, label=name, color=m["color"], linewidth=2, linestyle=m["ls"])
    ax2.plot(tokens_b, ppls_curve, label=name, color=m["color"], linewidth=2, linestyle=m["ls"])

ax1.axhline(y=gpt2["val_loss"], color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'GPT-2 Small (~40B tok)')
ax1.set_xlabel("Tokens Seen (Billions)")
ax1.set_ylabel("Validation Loss")
ax1.set_title("Val Loss vs Training Tokens", fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.axhline(y=gpt2["val_ppl"], color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'GPT-2 Small (PPL {gpt2["val_ppl"]:.1f})')
ax2.set_xlabel("Tokens Seen (Billions)")
ax2.set_ylabel("Validation Perplexity")
ax2.set_title("Val PPL vs Training Tokens", fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(20, 300)

fig6.suptitle("Performance vs Training Compute (Tokens Seen)",
              fontsize=14, fontweight='bold', y=1.02)
fig6.tight_layout()
fig6.savefig("fig6_tokens_comparison.png", dpi=150, bbox_inches='tight')
print("Saved fig6_tokens_comparison.png")

print("\nAll figures saved!")
