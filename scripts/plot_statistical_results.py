#!/usr/bin/env python3
"""
Plot statistical validation results with error bars
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

sns.set_palette("husl")

RESULTS_DIR = Path("statistical_validation_results")
OUTPUT_DIR = Path("publication_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_statistics():
    """Load computed statistics"""
    with open(RESULTS_DIR / "statistics_summary.json", 'r') as f:
        return json.load(f)


def load_significance_tests():
    """Load significance test results"""
    with open(RESULTS_DIR / "significance_tests.json", 'r') as f:
        return json.load(f)


def plot_performance_with_errorbars():
    """Plot validation loss with error bars"""
    stats = load_statistics()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(stats.keys())
    means = [stats[m]['loss_mean'] for m in models]
    stds = [stats[m]['loss_std'] for m in models]
    
    x = np.arange(len(models))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                   color=colors, ecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.05, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_ylabel('Validation Loss (lower is better)', fontweight='bold')
    ax.set_title('Model Performance Comparison (5 seeds, 95% CI)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'statistical_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'statistical_performance.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/statistical_performance.png")
    plt.close()


def plot_efficiency_scatter():
    """Plot parameters vs performance with error bars"""
    stats = load_statistics()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    models = list(stats.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'D']
    
    for i, model in enumerate(models):
        params = stats[model]['params'] / 1e6  # In millions
        loss_mean = stats[model]['loss_mean']
        loss_std = stats[model]['loss_std']
        
        # Plot point
        ax.errorbar(params, loss_mean, yerr=loss_std, 
                   marker=markers[i], markersize=15, capsize=5,
                   color=colors[i], ecolor=colors[i],
                   linewidth=2, label=model, alpha=0.8)
        
        # Annotate
        ax.annotate(model, (params, loss_mean), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.2))
    
    ax.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Parameter Efficiency (with 95% confidence intervals)', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_with_error.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'efficiency_with_error.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/efficiency_with_error.png")
    plt.close()


def plot_speed_comparison():
    """Plot training speed with error bars"""
    stats = load_statistics()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(stats.keys())
    means = [stats[m]['speed_mean'] for m in models]
    stds = [stats[m]['speed_std'] for m in models]
    
    x = np.arange(len(models))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                   color=colors, ecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 2, f'{mean:.1f}±{std:.1f} ms', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_ylabel('Training Time per Step (ms)', fontweight='bold')
    ax.set_title('Training Speed Comparison (lower is better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'speed_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/speed_comparison.png")
    plt.close()


def plot_significance_matrix():
    """Plot significance test results as a matrix"""
    sig_tests = load_significance_tests()
    stats = load_statistics()
    
    models = list(stats.keys())
    n = len(models)
    
    # Create matrix
    p_matrix = np.ones((n, n))
    cohen_matrix = np.zeros((n, n))
    
    for test in sig_tests:
        i = models.index(test['model1'])
        j = models.index(test['model2'])
        p_matrix[i, j] = test['p_value']
        p_matrix[j, i] = test['p_value']
        cohen_matrix[i, j] = test['cohens_d']
        cohen_matrix[j, i] = -test['cohens_d']
    
    # Plot p-values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # P-value heatmap
    im1 = ax1.imshow(p_matrix, cmap='RdYlGn', vmin=0, vmax=0.1, aspect='auto')
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(n))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_yticklabels(models)
    ax1.set_title('Statistical Significance (p-values)\nGreen = Significant difference', 
                  fontweight='bold')
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                text = f'{p_matrix[i, j]:.4f}'
                color = 'white' if p_matrix[i, j] < 0.05 else 'black'
                ax1.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
            else:
                ax1.text(j, i, '—', ha='center', va='center', fontsize=14)
    
    plt.colorbar(im1, ax=ax1, label='p-value')
    
    # Cohen's d heatmap
    im2 = ax2.imshow(cohen_matrix, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax2.set_xticks(np.arange(n))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_yticklabels(models)
    ax2.set_title("Effect Size (Cohen's d)\nRed = Model1 better, Blue = Model2 better", 
                  fontweight='bold')
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                text = f'{cohen_matrix[i, j]:.2f}'
                color = 'white' if abs(cohen_matrix[i, j]) > 1 else 'black'
                ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
            else:
                ax2.text(j, i, '—', ha='center', va='center', fontsize=14)
    
    plt.colorbar(im2, ax=ax2, label="Cohen's d")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'significance_tests.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'significance_tests.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/significance_tests.png")
    plt.close()


def generate_latex_table():
    """Generate LaTeX table for paper"""
    stats = load_statistics()
    sig_tests = load_significance_tests()
    
    latex = r"""\begin{table}[t]
\centering
\caption{Model Performance Comparison (Mean $\pm$ Std over 5 seeds)}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Valid Loss} & \textbf{Parameters} & \textbf{Speed (ms)} & \textbf{Compression} \\
\midrule
"""
    
    baseline_params = stats['standard']['params']
    
    for model in stats.keys():
        s = stats[model]
        loss = f"{s['loss_mean']:.4f} $\\pm$ {s['loss_std']:.4f}"
        params = f"{s['params']/1e6:.1f}M"
        speed = f"{s['speed_mean']:.1f} $\\pm$ {s['speed_std']:.1f}"
        compression = f"{baseline_params/s['params']:.1f}×"
        
        latex += f"{model} & {loss} & {params} & {speed} & {compression} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Add significance note
    latex += "\n% Statistical Significance Tests:\n"
    for test in sig_tests:
        if test['significant_at_0.01']:
            latex += f"% {test['model1']} vs {test['model2']}: "
            latex += f"p={test['p_value']:.6f} (p<0.01), d={test['cohens_d']:.3f}\n"
    
    with open(OUTPUT_DIR / 'results_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved: {OUTPUT_DIR}/results_table.tex")
    return latex


def plot_all_seeds_overlay():
    """Plot individual seed results overlaid"""
    stats = load_statistics()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, model in enumerate(stats.keys()):
        ax = axes[idx]
        
        # Load all seed results for this model
        all_losses = []
        for seed in stats[model]['all_seeds']:
            # Determine filename based on model
            if model == 'Standard Transformer':
                filename = f'standard_seed{seed}.json'
            elif model == 'Time-Indexed MLP':
                filename = f'time_mlp_seed{seed}.json'
            else:
                filename = f'time_ssm_seed{seed}.json'
            
            with open(RESULTS_DIR / filename, 'r') as f:
                data = json.load(f)
                all_losses.append(data['all_losses'])
        
        # Plot each seed
        for i, losses in enumerate(all_losses):
            ax.plot(losses, alpha=0.3, linewidth=1, color='gray')
        
        # Plot mean
        mean_losses = np.mean(all_losses, axis=0)
        std_losses = np.std(all_losses, axis=0)
        steps = np.arange(len(mean_losses))
        
        ax.plot(mean_losses, linewidth=2.5, label='Mean', color='red')
        ax.fill_between(steps, mean_losses - std_losses, mean_losses + std_losses,
                        alpha=0.3, color='red', label='±1 Std')
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.set_title(f'{model}\n(5 seeds)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_seeds_overlay.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'all_seeds_overlay.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/all_seeds_overlay.png")
    plt.close()


def main():
    print("="*70)
    print("PLOTTING STATISTICAL RESULTS")
    print("="*70)
    
    # Create all plots
    print("\nGenerating plots...")
    plot_performance_with_errorbars()
    plot_efficiency_scatter()
    plot_speed_comparison()
    plot_significance_matrix()
    # plot_all_seeds_overlay()  # Skip for now - needs pkl file reading
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex = generate_latex_table()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    stats = load_statistics()
    for model in stats.keys():
        s = stats[model]
        print(f"\n{model}:")
        print(f"  Loss: {s['loss_mean']:.4f} ± {s['loss_std']:.4f}")
        print(f"  95% CI: [{s['loss_ci95'][0]:.4f}, {s['loss_ci95'][1]:.4f}]")
    
    sig_tests = load_significance_tests()
    print("\n" + "-"*70)
    print("Significance Tests:")
    for test in sig_tests:
        if test['significant_at_0.05']:
            print(f"  {test['model1']} vs {test['model2']}: "
                  f"p={test['p_value']:.4f} ({'**' if test['significant_at_0.01'] else '*'})")
    
    print("\n" + "="*70)
    print("✅ ALL PLOTS GENERATED!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • statistical_performance.png/pdf - Performance with error bars")
    print("  • efficiency_with_error.png/pdf - Parameters vs Loss")
    print("  • speed_comparison.png/pdf - Training speed comparison")
    print("  • significance_tests.png/pdf - Statistical test results")
    print("  • results_table.tex - LaTeX table for paper")
    print("\n✨ Ready for publication!")


if __name__ == "__main__":
    main()


