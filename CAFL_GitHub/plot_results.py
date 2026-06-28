"""
Graph Generator for Context-Aware Adaptive Federated Learning Paper
Generates all figures needed for Section 4.5 and ablation analysis
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# STYLE SETTINGS
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 12,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 12,
    'legend.fontsize'  : 10,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'figure.dpi'       : 150,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

COLORS = {
    'CAFL'    : '#2196F3',   # blue
    'FedAvg'  : '#F44336',   # red
    'FedProx' : '#4CAF50',   # green
    'NoPhase2': '#FF9800',   # orange
    'NoPhase3': '#9C27B0',   # purple
    'NoPhase4': '#795548',   # brown
}

RESULTS_DIR = 'results/'
FIGURES_DIR = 'results/figures/'
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_results():
    exp_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    abl_path = os.path.join(RESULTS_DIR, 'ablation_results.json')

    with open(exp_path) as f:
        exp_data = json.load(f)
    with open(abl_path) as f:
        abl_data = json.load(f)

    return exp_data, abl_data


# ─────────────────────────────────────────────
# FIGURE 1: Accuracy over Rounds (Convergence)
# Shows CAFL vs FedAvg vs FedProx per round
# ─────────────────────────────────────────────
def plot_convergence(exp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Convergence: Accuracy over Communication Rounds',
                 fontsize=14, fontweight='bold', y=1.02)

    datasets = [
        ('FLY-SMOTE_alpha0.1', 'FLY-SMOTE (α=0.1)', axes[0]),
        ('FD-IDS_alpha0.1',    'FD-IDS (α=0.1)',     axes[1]),
    ]

    for exp_key, title, ax in datasets:
        if exp_key not in exp_data:
            continue

        results = exp_data[exp_key]
        for method, color in [('CAFL', COLORS['CAFL']),
                               ('FedAvg', COLORS['FedAvg']),
                               ('FedProx', COLORS['FedProx'])]:
            if method not in results:
                continue
            rounds   = results[method]['round']
            accuracy = results[method]['accuracy']
            ax.plot(rounds, accuracy, color=color,
                    linewidth=2.5, label=method,
                    marker='o', markersize=4)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_convergence.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 2: Bar Chart — Final Metrics Comparison
# Accuracy, Precision, Recall, F1 side by side
# ─────────────────────────────────────────────
def plot_metrics_comparison(exp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Comparison: CAFL vs Baselines',
                 fontsize=14, fontweight='bold', y=1.02)

    datasets = [
        ('FLY-SMOTE_alpha0.1', 'FLY-SMOTE (α=0.1)', axes[0]),
        ('FD-IDS_alpha0.1',    'FD-IDS (α=0.1)',     axes[1]),
    ]

    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_keys   = ['accuracy', 'precision', 'recall', 'f1']
    methods        = ['CAFL', 'FedAvg', 'FedProx']
    bar_colors     = [COLORS['CAFL'], COLORS['FedAvg'], COLORS['FedProx']]

    x      = np.arange(len(metrics_labels))
    width  = 0.25

    for exp_key, title, ax in datasets:
        if exp_key not in exp_data:
            continue

        results = exp_data[exp_key]
        for idx, (method, color) in enumerate(zip(methods, bar_colors)):
            if method not in results:
                continue
            values = [results[method][k][-1] for k in metrics_keys]
            bars   = ax.bar(x + idx * width, values, width,
                            label=method, color=color, alpha=0.85,
                            edgecolor='white', linewidth=0.5)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.12])
        ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_metrics_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 3: Accuracy vs Alpha (Non-IID Level)
# Shows how methods behave under heterogeneity
# ─────────────────────────────────────────────
def plot_alpha_comparison(exp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Accuracy vs Non-IID Level (α)',
                 fontsize=14, fontweight='bold', y=1.02)

    alpha_values = [0.1, 0.3, 0.5]

    dataset_pairs = [
        ('FLY-SMOTE', axes[0]),
        ('FD-IDS',    axes[1]),
    ]

    for dataset_name, ax in dataset_pairs:
        for method, color in [('CAFL', COLORS['CAFL']),
                               ('FedAvg', COLORS['FedAvg']),
                               ('FedProx', COLORS['FedProx'])]:
            accs = []
            for alpha in alpha_values:
                key = f"{dataset_name}_alpha{alpha}"
                if key in exp_data and method in exp_data[key]:
                    accs.append(exp_data[key][method]['accuracy'][-1])
                else:
                    accs.append(None)

            valid = [(a, acc) for a, acc in zip(alpha_values, accs)
                     if acc is not None]
            if valid:
                alphas, accuracies = zip(*valid)
                ax.plot(alphas, accuracies, color=color,
                        linewidth=2.5, label=method,
                        marker='s', markersize=8)

        ax.set_title(f'{dataset_name}', fontweight='bold')
        ax.set_xlabel('Dirichlet α (higher = more IID)')
        ax.set_ylabel('Final Accuracy')
        ax.set_xticks(alpha_values)
        ax.legend()
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_alpha_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 4: Ablation — Phase Removal
# ─────────────────────────────────────────────
def plot_phase_ablation(abl_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study: Impact of Each Phase',
                 fontsize=14, fontweight='bold', y=1.02)

    datasets = [
        ('phase_flysmote', 'FLY-SMOTE (α=0.1)', axes[0]),
        ('phase_fdids',    'FD-IDS (α=0.1)',     axes[1]),
    ]

    variant_labels = {
        'CAFL_Full'    : 'CAFL (Full)',
        'CAFL_NoPhase2': 'w/o Phase 2\n(No Clustering)',
        'CAFL_NoPhase3': 'w/o Phase 3\n(No Drift Det.)',
        'CAFL_NoPhase4': 'w/o Phase 4\n(No Lightweight Agg.)',
    }
    variant_colors = [
        COLORS['CAFL'], COLORS['NoPhase2'],
        COLORS['NoPhase3'], COLORS['NoPhase4']
    ]

    for exp_key, title, ax in datasets:
        if exp_key not in abl_data:
            continue

        results  = abl_data[exp_key]
        variants = list(variant_labels.keys())
        accs     = [results[v]['accuracy'][-1] if v in results else 0
                    for v in variants]
        f1s      = [results[v]['f1'][-1] if v in results else 0
                    for v in variants]

        x     = np.arange(len(variants))
        width = 0.35

        bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',
                       color=variant_colors, alpha=0.85,
                       edgecolor='white')
        bars2 = ax.bar(x + width/2, f1s, width, label='F1-Score',
                       color=variant_colors, alpha=0.5,
                       edgecolor='white', hatch='//')

        # Value labels
        for bar, val in zip(bars1, accs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, f1s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([variant_labels[v] for v in variants],
                           fontsize=9)
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.12])

        # Custom legend
        acc_patch = mpatches.Patch(color='gray', alpha=0.85, label='Accuracy')
        f1_patch  = mpatches.Patch(color='gray', alpha=0.5,
                                   hatch='//', label='F1-Score')
        ax.legend(handles=[acc_patch, f1_patch])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_phase_ablation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 5: Ablation — Cluster Count
# ─────────────────────────────────────────────
def plot_cluster_ablation(abl_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Study: Impact of Number of Clusters (K)',
                 fontsize=14, fontweight='bold', y=1.02)

    datasets = [
        ('clusters_flysmote', 'FLY-SMOTE (α=0.1)', axes[0]),
        ('clusters_fdids',    'FD-IDS (α=0.1)',     axes[1]),
    ]

    for exp_key, title, ax in datasets:
        if exp_key not in abl_data:
            continue

        results = abl_data[exp_key]
        ks      = sorted([int(k.split('=')[1]) for k in results.keys()])
        accs    = [results[f'K={k}']['accuracy'][-1] for k in ks]
        f1s     = [results[f'K={k}']['f1'][-1] for k in ks]

        ax.plot(ks, accs, color=COLORS['CAFL'], linewidth=2.5,
                marker='o', markersize=9, label='Accuracy')
        ax.plot(ks, f1s,  color=COLORS['FedAvg'], linewidth=2.5,
                marker='s', markersize=9, label='F1-Score',
                linestyle='--')

        # Annotate best K
        best_k   = ks[np.argmax(accs)]
        best_acc = max(accs)
        ax.annotate(f'Best K={best_k}',
                    xy=(best_k, best_acc),
                    xytext=(best_k + 0.3, best_acc - 0.01),
                    fontsize=9, color=COLORS['CAFL'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['CAFL']))

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Score')
        ax.set_xticks(ks)
        ax.legend()
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_cluster_ablation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIGURE 6: F1 Score over Rounds (Stability)
# ─────────────────────────────────────────────
def plot_f1_stability(exp_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('F1-Score Stability over Communication Rounds',
                 fontsize=14, fontweight='bold', y=1.02)

    datasets = [
        ('FLY-SMOTE_alpha0.1', 'FLY-SMOTE (α=0.1)', axes[0]),
        ('FD-IDS_alpha0.1',    'FD-IDS (α=0.1)',     axes[1]),
    ]

    for exp_key, title, ax in datasets:
        if exp_key not in exp_data:
            continue

        results = exp_data[exp_key]
        for method, color in [('CAFL', COLORS['CAFL']),
                               ('FedAvg', COLORS['FedAvg']),
                               ('FedProx', COLORS['FedProx'])]:
            if method not in results:
                continue
            rounds = results[method]['round']
            f1s    = results[method]['f1']
            ax.plot(rounds, f1s, color=color, linewidth=2.5,
                    label=method, marker='o', markersize=4)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('F1-Score')
        ax.legend()
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_f1_stability.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)

    exp_data, abl_data = load_results()

    print("\n  Generating figures...")
    plot_convergence(exp_data)
    plot_metrics_comparison(exp_data)
    plot_alpha_comparison(exp_data)
    plot_phase_ablation(abl_data)
    plot_cluster_ablation(abl_data)
    plot_f1_stability(exp_data)

    print(f"\n  All figures saved to: {FIGURES_DIR}")
    print("\n  Figures generated:")
    print("   fig1_convergence.png        — Accuracy over rounds")
    print("   fig2_metrics_comparison.png — Bar chart: Acc/Prec/Rec/F1")
    print("   fig3_alpha_comparison.png   — Accuracy vs alpha")
    print("   fig4_phase_ablation.png     — Phase removal ablation")
    print("   fig5_cluster_ablation.png   — Cluster count ablation")
    print("   fig6_f1_stability.png       — F1 stability over rounds")
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
