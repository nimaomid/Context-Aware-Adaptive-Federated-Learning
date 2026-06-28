"""
Ablation Study for Context-Aware Adaptive Federated Learning
Tests:
  1. Phase Removal  — remove each phase one by one
  2. Cluster Count  — test different numbers of clusters
  3. Alpha Values   — test different Non-IID levels
"""

import numpy as np
import torch
import json
import os
import time
import copy
from data_loader import load_fd_ids, load_fly_smote, split_non_iid
from federated import (CAFLFramework, extract_context, cluster_clients,
                       get_cluster_groups, compute_client_weights,
                       aggregate_cluster, aggregate_global, DriftDetector)
from baselines import fedavg_aggregate
from model import get_model, train_local, evaluate_local


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
CONFIG = {
    'num_clients'  : 10,
    'num_rounds'   : 20,
    'local_epochs' : 5,
    'lr'           : 0.001,
    'max_samples'  : 50000,
    'results_dir'  : 'results/'
}


# ─────────────────────────────────────────────
# HELPER: Subsample client data
# ─────────────────────────────────────────────
def subsample_clients(client_data, max_samples_total):
    num_clients = len(client_data)
    per_client  = max_samples_total // num_clients
    subsampled  = {}
    for i in range(num_clients):
        X = client_data[i]["X"]
        y = client_data[i]["y"]
        if len(y) > per_client:
            idx = np.random.choice(len(y), per_client, replace=False)
            subsampled[i] = {"X": X[idx], "y": y[idx]}
        else:
            subsampled[i] = {"X": X, "y": y}
    return subsampled


# ─────────────────────────────────────────────
# ABLATION 1: PHASE REMOVAL
# Tests what happens when each phase is removed
# ─────────────────────────────────────────────
class CAFLNoPhase2(CAFLFramework):
    """CAFL without Phase 2 (no clustering — all clients in one group)"""
    def run(self, client_data, verbose=True):
        if verbose:
            print(f"\n  [No Phase 2] Running without Adaptive Clustering...")

        for round_num in range(self.num_rounds):
            # Phase 1: Context extraction
            client_contexts = {}
            for i in range(self.num_clients):
                client_contexts[i] = extract_context(
                    i, client_data[i], round_num, self.num_clients
                )

            # NO Phase 2: All clients in one cluster
            client_models = {}
            client_losses = {}
            for i in range(self.num_clients):
                local_model = copy.deepcopy(self.global_model)
                local_model, loss = train_local(
                    local_model,
                    client_data[i]["X"], client_data[i]["y"],
                    epochs=self.local_epochs, lr=self.lr, device=self.device
                )
                client_models[i] = local_model
                client_losses[i] = loss
                self.drift_detector.update(i, client_contexts[i], loss)

            # Phase 4: Simple FedAvg aggregation (no clustering)
            models  = list(client_models.values())
            weights = [1.0 / self.num_clients] * self.num_clients
            self.global_model = aggregate_cluster(models, weights)

            # Evaluate
            all_X = np.concatenate([client_data[i]["X"]
                                    for i in range(self.num_clients)])
            all_y = np.concatenate([client_data[i]["y"]
                                    for i in range(self.num_clients)])
            if len(all_y) > 10000:
                idx = np.random.choice(len(all_y), 10000, replace=False)
                all_X, all_y = all_X[idx], all_y[idx]

            metrics  = evaluate_local(self.global_model, all_X, all_y,
                                      device=self.device)
            avg_loss = np.mean(list(client_losses.values()))

            self.history['round'].append(round_num + 1)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['precision'].append(metrics['precision'])
            self.history['recall'].append(metrics['recall'])
            self.history['f1'].append(metrics['f1'])
            self.history['loss'].append(avg_loss)
            self.history['drift_clients'].append(0)

            if verbose:
                print(f"    Round {round_num+1:02d} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f}")

        return self.history


class CAFLNoPhase3(CAFLFramework):
    """CAFL without Phase 3 (no drift detection)"""
    def run(self, client_data, verbose=True):
        if verbose:
            print(f"\n  [No Phase 3] Running without Drift Detection...")

        for round_num in range(self.num_rounds):
            # Phase 1 & 2 normal
            client_contexts = {}
            for i in range(self.num_clients):
                client_contexts[i] = extract_context(
                    i, client_data[i], round_num, self.num_clients
                )

            assignments = cluster_clients(
                client_contexts, num_clusters=self.num_clusters
            )
            groups = get_cluster_groups(assignments, self.num_clusters)

            client_models = {}
            client_losses = {}
            for i in range(self.num_clients):
                local_model = copy.deepcopy(self.global_model)
                local_model, loss = train_local(
                    local_model,
                    client_data[i]["X"], client_data[i]["y"],
                    epochs=self.local_epochs, lr=self.lr, device=self.device
                )
                client_models[i] = local_model
                client_losses[i] = loss
                # NO drift detection update

            # Phase 4: Equal weights (no drift penalty)
            cluster_models = []
            cluster_sizes  = []
            for cluster_id, members in groups.items():
                if not members:
                    continue
                models  = [client_models[i] for i in members]
                weights = [1.0 / len(members)] * len(members)
                cluster_model = aggregate_cluster(models, weights)
                cluster_models.append(cluster_model)
                cluster_sizes.append(sum(len(client_data[i]["y"])
                                        for i in members))

            if cluster_models:
                self.global_model = aggregate_global(cluster_models,
                                                     cluster_sizes)

            all_X = np.concatenate([client_data[i]["X"]
                                    for i in range(self.num_clients)])
            all_y = np.concatenate([client_data[i]["y"]
                                    for i in range(self.num_clients)])
            if len(all_y) > 10000:
                idx = np.random.choice(len(all_y), 10000, replace=False)
                all_X, all_y = all_X[idx], all_y[idx]

            metrics  = evaluate_local(self.global_model, all_X, all_y,
                                      device=self.device)
            avg_loss = np.mean(list(client_losses.values()))

            self.history['round'].append(round_num + 1)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['precision'].append(metrics['precision'])
            self.history['recall'].append(metrics['recall'])
            self.history['f1'].append(metrics['f1'])
            self.history['loss'].append(avg_loss)
            self.history['drift_clients'].append(0)

            if verbose:
                print(f"    Round {round_num+1:02d} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f}")

        return self.history


class CAFLNoPhase4(CAFLFramework):
    """CAFL without Phase 4 (uses standard FedAvg aggregation instead)"""
    def run(self, client_data, verbose=True):
        if verbose:
            print(f"\n  [No Phase 4] Running without Lightweight Aggregation...")

        for round_num in range(self.num_rounds):
            client_contexts = {}
            for i in range(self.num_clients):
                client_contexts[i] = extract_context(
                    i, client_data[i], round_num, self.num_clients
                )

            client_models = {}
            client_losses = {}
            for i in range(self.num_clients):
                local_model = copy.deepcopy(self.global_model)
                local_model, loss = train_local(
                    local_model,
                    client_data[i]["X"], client_data[i]["y"],
                    epochs=self.local_epochs, lr=self.lr, device=self.device
                )
                client_models[i] = local_model
                client_losses[i] = loss
                self.drift_detector.update(i, client_contexts[i], loss)

            # NO Phase 4: Use standard FedAvg instead of lightweight agg
            self.global_model = fedavg_aggregate(
                list(client_models.values()), client_data, self.num_clients
            )

            all_X = np.concatenate([client_data[i]["X"]
                                    for i in range(self.num_clients)])
            all_y = np.concatenate([client_data[i]["y"]
                                    for i in range(self.num_clients)])
            if len(all_y) > 10000:
                idx = np.random.choice(len(all_y), 10000, replace=False)
                all_X, all_y = all_X[idx], all_y[idx]

            metrics  = evaluate_local(self.global_model, all_X, all_y,
                                      device=self.device)
            avg_loss = np.mean(list(client_losses.values()))

            self.history['round'].append(round_num + 1)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['precision'].append(metrics['precision'])
            self.history['recall'].append(metrics['recall'])
            self.history['f1'].append(metrics['f1'])
            self.history['loss'].append(avg_loss)
            self.history['drift_clients'].append(0)

            if verbose:
                print(f"    Round {round_num+1:02d} | "
                      f"Acc: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f}")

        return self.history


# ─────────────────────────────────────────────
# RUN PHASE ABLATION
# ─────────────────────────────────────────────
def run_phase_ablation(X, y, input_dim, num_classes,
                       alpha, dataset_name, config, device):
    print(f"\n{'='*60}")
    print(f"  Phase Ablation — {dataset_name} (alpha={alpha})")
    print(f"{'='*60}")

    client_data = split_non_iid(X, y, num_clients=config['num_clients'],
                                alpha=alpha)
    client_data = subsample_clients(client_data, config['max_samples'])

    results = {}
    variants = {
        'CAFL_Full'    : CAFLFramework,
        'CAFL_NoPhase2': CAFLNoPhase2,
        'CAFL_NoPhase3': CAFLNoPhase3,
        'CAFL_NoPhase4': CAFLNoPhase4,
    }

    for name, cls in variants.items():
        print(f"\n  Running {name}...")
        framework = cls(
            input_dim=input_dim, num_classes=num_classes,
            num_clients=config['num_clients'], num_clusters=3,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            lr=config['lr'], device=device
        )
        results[name] = framework.run(client_data, verbose=True)

    # Print table
    print(f"\n  {'Variant':<20} {'Accuracy':>10} {'F1-Score':>10}")
    print(f"  {'-'*42}")
    for name, history in results.items():
        acc = history['accuracy'][-1]
        f1  = history['f1'][-1]
        print(f"  {name:<20} {acc:>10.4f} {f1:>10.4f}")

    return results


# ─────────────────────────────────────────────
# RUN CLUSTER COUNT ABLATION
# ─────────────────────────────────────────────
def run_cluster_ablation(X, y, input_dim, num_classes,
                         alpha, dataset_name, config, device):
    print(f"\n{'='*60}")
    print(f"  Cluster Count Ablation — {dataset_name} (alpha={alpha})")
    print(f"{'='*60}")

    client_data = split_non_iid(X, y, num_clients=config['num_clients'],
                                alpha=alpha)
    client_data = subsample_clients(client_data, config['max_samples'])

    results = {}
    cluster_counts = [2, 3, 5, 7]

    for k in cluster_counts:
        name = f"K={k}"
        print(f"\n  Running CAFL with {k} clusters...")
        framework = CAFLFramework(
            input_dim=input_dim, num_classes=num_classes,
            num_clients=config['num_clients'], num_clusters=k,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            lr=config['lr'], device=device
        )
        results[name] = framework.run(client_data, verbose=False)
        acc = results[name]['accuracy'][-1]
        f1  = results[name]['f1'][-1]
        print(f"  K={k} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return results


# ─────────────────────────────────────────────
# RUN ALPHA ABLATION
# ─────────────────────────────────────────────
def run_alpha_ablation(X, y, input_dim, num_classes,
                       dataset_name, config, device):
    print(f"\n{'='*60}")
    print(f"  Alpha Ablation — {dataset_name}")
    print(f"{'='*60}")

    alpha_values = [0.1, 0.3, 0.5]
    results = {}

    for alpha in alpha_values:
        name = f"alpha={alpha}"
        print(f"\n  Running CAFL with alpha={alpha}...")
        client_data = split_non_iid(X, y, num_clients=config['num_clients'],
                                    alpha=alpha)
        client_data = subsample_clients(client_data, config['max_samples'])

        framework = CAFLFramework(
            input_dim=input_dim, num_classes=num_classes,
            num_clients=config['num_clients'], num_clusters=3,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            lr=config['lr'], device=device
        )
        results[name] = framework.run(client_data, verbose=False)
        acc = results[name]['accuracy'][-1]
        f1  = results[name]['f1'][-1]
        print(f"  alpha={alpha} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return results


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_ablation(all_results, config):
    os.makedirs(config['results_dir'], exist_ok=True)
    path = os.path.join(config['results_dir'], 'ablation_results.json')

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = {}
    for exp, results in all_results.items():
        clean[exp] = {}
        for method, history in results.items():
            clean[exp][method] = {
                k: [convert(v) for v in vals]
                for k, vals in history.items()
            }

    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)

    print(f"\n  Ablation results saved to: {path}")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Ablation Study — Context-Aware Federated Learning")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    all_ablation = {}

    # Load FLY-SMOTE (most interesting dataset for ablation)
    print("\n>>> Loading FLY-SMOTE dataset...")
    X_fly, y_fly, _ = load_fly_smote()
    input_dim_fly   = X_fly.shape[1]
    num_classes_fly = len(np.unique(y_fly))
    print(f"  Features: {input_dim_fly} | Classes: {num_classes_fly}")

    # Load FD-IDS
    print("\n>>> Loading FD-IDS dataset...")
    X_fd, y_fd, _  = load_fd_ids()
    input_dim_fd   = X_fd.shape[1]
    num_classes_fd = len(np.unique(y_fd))
    print(f"  Features: {input_dim_fd} | Classes: {num_classes_fd}")

    # ── 1. Phase Ablation ──
    print("\n\n>>> ABLATION 1: Phase Removal")
    all_ablation['phase_flysmote'] = run_phase_ablation(
        X_fly, y_fly, input_dim_fly, num_classes_fly,
        alpha=0.1, dataset_name="FLY-SMOTE",
        config=CONFIG, device=device
    )
    all_ablation['phase_fdids'] = run_phase_ablation(
        X_fd, y_fd, input_dim_fd, num_classes_fd,
        alpha=0.1, dataset_name="FD-IDS",
        config=CONFIG, device=device
    )

    # ── 2. Cluster Count Ablation ──
    print("\n\n>>> ABLATION 2: Cluster Count")
    all_ablation['clusters_flysmote'] = run_cluster_ablation(
        X_fly, y_fly, input_dim_fly, num_classes_fly,
        alpha=0.1, dataset_name="FLY-SMOTE",
        config=CONFIG, device=device
    )
    all_ablation['clusters_fdids'] = run_cluster_ablation(
        X_fd, y_fd, input_dim_fd, num_classes_fd,
        alpha=0.1, dataset_name="FD-IDS",
        config=CONFIG, device=device
    )

    # ── 3. Alpha Ablation ──
    print("\n\n>>> ABLATION 3: Alpha Values (Non-IID Level)")
    all_ablation['alpha_flysmote'] = run_alpha_ablation(
        X_fly, y_fly, input_dim_fly, num_classes_fly,
        dataset_name="FLY-SMOTE", config=CONFIG, device=device
    )
    all_ablation['alpha_fdids'] = run_alpha_ablation(
        X_fd, y_fd, input_dim_fd, num_classes_fd,
        dataset_name="FD-IDS", config=CONFIG, device=device
    )

    # Save all results
    save_ablation(all_ablation, CONFIG)

    print("\n\n" + "=" * 60)
    print("  Ablation Study Complete!")
    print("=" * 60)
