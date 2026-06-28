"""
Main Experiment Runner
Compares CAFL vs FedAvg vs FedProx on FD-IDS and FLY-SMOTE datasets
Produces results for Section 4.5 of the paper
"""

import numpy as np
import torch
import json
import os
import time
from data_loader import load_fd_ids, load_fly_smote, split_non_iid
from federated import CAFLFramework
from baselines import FedAvg, FedProx


# ─────────────────────────────────────────────
# EXPERIMENT SETTINGS
# Matches paper Section 4.3 parameter table
# ─────────────────────────────────────────────
CONFIG = {
    # FL settings
    'num_clients'   : 10,
    'num_clusters'  : 3,
    'num_rounds'    : 20,
    'local_epochs'  : 5,
    'lr'            : 0.001,
    'fedprox_mu'    : 0.01,

    # Non-IID settings (Dirichlet alpha values from paper)
    'alpha_values'  : [0.1, 0.3, 0.5],

    # Drift detection
    'drift_threshold': 0.05,

    # Max samples per client for speed (set None to use all data)
    'max_samples'   : 50000,

    # Results output
    'results_dir'   : 'results/'
}


# ─────────────────────────────────────────────
# HELPER: Subsample client data for speed
# ─────────────────────────────────────────────
def subsample_clients(client_data, max_samples_total):
    """Limit total samples across clients for faster experiments"""
    num_clients  = len(client_data)
    per_client   = max_samples_total // num_clients
    subsampled   = {}

    for i in range(num_clients):
        X = client_data[i]["X"]
        y = client_data[i]["y"]

        if len(y) > per_client:
            idx = np.random.choice(len(y), per_client, replace=False)
            subsampled[i] = {"X": X[idx], "y": y[idx]}
        else:
            subsampled[i] = {"X": X, "y": y}

    total = sum(len(subsampled[i]["y"]) for i in range(num_clients))
    print(f"  Subsampled to {total:,} total samples "
          f"({per_client:,} per client)")
    return subsampled


# ─────────────────────────────────────────────
# HELPER: Print comparison table
# ─────────────────────────────────────────────
def print_results_table(results, dataset_name):
    print(f"\n{'='*70}")
    print(f"  RESULTS — {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*52}")

    for method, history in results.items():
        acc  = history['accuracy'][-1]
        prec = history['precision'][-1]
        rec  = history['recall'][-1]
        f1   = history['f1'][-1]
        print(f"  {method:<12} {acc:>10.4f} {prec:>10.4f} "
              f"{rec:>10.4f} {f1:>10.4f}")

    # Calculate improvement of CAFL over FedAvg
    if 'CAFL' in results and 'FedAvg' in results:
        cafl_acc  = results['CAFL']['accuracy'][-1]
        favg_acc  = results['FedAvg']['accuracy'][-1]
        improve   = ((cafl_acc - favg_acc) / (favg_acc + 1e-8)) * 100
        print(f"\n  CAFL improvement over FedAvg: {improve:+.1f}%")

    print(f"{'='*70}")


# ─────────────────────────────────────────────
# RUN ONE EXPERIMENT
# Runs all 3 methods on one dataset + alpha
# ─────────────────────────────────────────────
def run_experiment(X, y, input_dim, num_classes, alpha,
                   dataset_name, config, device):

    print(f"\n{'─'*60}")
    print(f"  Dataset: {dataset_name} | Alpha: {alpha}")
    print(f"{'─'*60}")

    # Split data across clients (Non-IID)
    client_data = split_non_iid(
        X, y,
        num_clients=config['num_clients'],
        alpha=alpha
    )

    # Subsample for manageable runtime
    if config['max_samples']:
        client_data = subsample_clients(
            client_data, config['max_samples']
        )

    results = {}

    # ── Run CAFL (proposed method) ──
    print(f"\n  Running CAFL (Proposed Method)...")
    t0 = time.time()
    cafl = CAFLFramework(
        input_dim=input_dim,
        num_classes=num_classes,
        num_clients=config['num_clients'],
        num_clusters=config['num_clusters'],
        num_rounds=config['num_rounds'],
        local_epochs=config['local_epochs'],
        lr=config['lr'],
        device=device
    )
    results['CAFL'] = cafl.run(client_data, verbose=True)
    print(f"  CAFL time: {time.time()-t0:.1f}s")

    # ── Run FedAvg baseline ──
    print(f"\n  Running FedAvg (Baseline)...")
    t0 = time.time()
    fedavg = FedAvg(
        input_dim=input_dim,
        num_classes=num_classes,
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds'],
        local_epochs=config['local_epochs'],
        lr=config['lr'],
        device=device
    )
    results['FedAvg'] = fedavg.run(client_data, verbose=True)
    print(f"  FedAvg time: {time.time()-t0:.1f}s")

    # ── Run FedProx baseline ──
    print(f"\n  Running FedProx (Baseline)...")
    t0 = time.time()
    fedprox = FedProx(
        input_dim=input_dim,
        num_classes=num_classes,
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds'],
        local_epochs=config['local_epochs'],
        lr=config['lr'],
        mu=config['fedprox_mu'],
        device=device
    )
    results['FedProx'] = fedprox.run(client_data, verbose=True)
    print(f"  FedProx time: {time.time()-t0:.1f}s")

    # Print comparison table
    print_results_table(results, f"{dataset_name} (alpha={alpha})")

    return results


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(all_results, config):
    os.makedirs(config['results_dir'], exist_ok=True)
    path = os.path.join(config['results_dir'], 'experiment_results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Clean results for saving
    clean = {}
    for key, results in all_results.items():
        clean[key] = {}
        for method, history in results.items():
            clean[key][method] = {
                k: [convert(v) for v in vals]
                for k, vals in history.items()
            }

    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)

    print(f"\n  Results saved to: {path}")
    return path


# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
def print_final_summary(all_results):
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — All Experiments")
    print(f"{'='*70}")
    print(f"  {'Experiment':<30} {'CAFL':>8} {'FedAvg':>8} "
          f"{'FedProx':>8} {'Improve':>8}")
    print(f"  {'-'*62}")

    for exp_name, results in all_results.items():
        cafl_acc  = results['CAFL']['accuracy'][-1]
        favg_acc  = results['FedAvg']['accuracy'][-1]
        fprox_acc = results['FedProx']['accuracy'][-1]
        improve   = ((cafl_acc - favg_acc) / (favg_acc + 1e-8)) * 100

        print(f"  {exp_name:<30} {cafl_acc:>8.4f} {favg_acc:>8.4f} "
              f"{fprox_acc:>8.4f} {improve:>+7.1f}%")

    print(f"{'='*70}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Context-Aware Adaptive Federated Learning — Experiments")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device : {device}")
    print(f"  Rounds : {CONFIG['num_rounds']}")
    print(f"  Clients: {CONFIG['num_clients']}")

    all_results = {}

    # ── EXPERIMENT 1: FD-IDS Dataset ──
    print("\n\n>>> Loading FD-IDS Dataset (UNSW-NB15)...")
    try:
        X_fd, y_fd, classes_fd = load_fd_ids()
        input_dim_fd  = X_fd.shape[1]
        num_classes_fd = len(np.unique(y_fd))
        print(f"  Features: {input_dim_fd} | Classes: {num_classes_fd}")

        # Run for each alpha value
        for alpha in CONFIG['alpha_values']:
            exp_name = f"FD-IDS_alpha{alpha}"
            all_results[exp_name] = run_experiment(
                X_fd, y_fd,
                input_dim=input_dim_fd,
                num_classes=num_classes_fd,
                alpha=alpha,
                dataset_name="FD-IDS",
                config=CONFIG,
                device=device
            )

    except Exception as e:
        print(f"  FD-IDS Error: {e}")

    # ── EXPERIMENT 2: FLY-SMOTE Dataset ──
    print("\n\n>>> Loading FLY-SMOTE Dataset (CICIDS-2017)...")
    try:
        X_fly, y_fly, classes_fly = load_fly_smote()
        input_dim_fly  = X_fly.shape[1]
        num_classes_fly = len(np.unique(y_fly))
        print(f"  Features: {input_dim_fly} | Classes: {num_classes_fly}")

        # Run for each alpha value
        for alpha in CONFIG['alpha_values']:
            exp_name = f"FLY-SMOTE_alpha{alpha}"
            all_results[exp_name] = run_experiment(
                X_fly, y_fly,
                input_dim=input_dim_fly,
                num_classes=num_classes_fly,
                alpha=alpha,
                dataset_name="FLY-SMOTE",
                config=CONFIG,
                device=device
            )

    except Exception as e:
        print(f"  FLY-SMOTE Error: {e}")

    # ── Save and summarize ──
    if all_results:
        save_results(all_results, CONFIG)
        print_final_summary(all_results)

    print("\n\nAll experiments complete!")
