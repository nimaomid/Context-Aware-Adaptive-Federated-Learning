"""
Context-Aware Adaptive Federated Learning Framework
Implements all 4 phases from the paper:
  Phase 1: Context Acquisition & Modeling
  Phase 2: Adaptive Clustering & Personalization
  Phase 3: Context Drift Detection & Adaptation
  Phase 4: Lightweight Context-Aware Aggregation
"""

import numpy as np
import torch
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from model import get_model, train_local, evaluate_local, get_model_weights, set_model_weights


# ─────────────────────────────────────────────
# PHASE 1: CONTEXT ACQUISITION & MODELING
# Each IoT node extracts lightweight context
# features to model its local environment
# (Section 3, Step 1 of the paper)
# ─────────────────────────────────────────────
def extract_context(client_id, client_data, round_num, num_clients):
    """
    Extract lightweight context vector for each client.
    Context includes: data statistics, temporal info,
    resource simulation, and mobility status.
    Implements: c_k = [c_k^(1), c_k^(2), ..., c_k^(m)]
    """
    X = client_data["X"]
    y = client_data["y"]

    # Data statistics context
    mean_features  = np.mean(X, axis=0)[:5]   # first 5 feature means
    std_features   = np.std(X, axis=0)[:5]    # first 5 feature stds
    class_counts   = np.bincount(y, minlength=2)
    class_ratio    = class_counts / (len(y) + 1e-8)

    # Temporal context (simulated)
    temporal       = np.array([
        np.sin(2 * np.pi * round_num / 10),   # cyclic time signal
        np.cos(2 * np.pi * round_num / 10)
    ])

    # Resource context (simulated per client)
    np.random.seed(client_id + round_num)
    energy_level   = np.random.uniform(0.5, 1.0)   # battery/power level
    bandwidth      = np.random.uniform(0.6, 1.0)   # available bandwidth
    compute_power  = np.random.uniform(0.4, 1.0)   # CPU/GPU availability

    resource_ctx   = np.array([energy_level, bandwidth, compute_power])

    # Combine into context vector
    context = np.concatenate([
        mean_features,
        std_features,
        class_ratio[:2],
        temporal,
        resource_ctx
    ])

    return context


# ─────────────────────────────────────────────
# PHASE 2: ADAPTIVE CLUSTERING & PERSONALIZATION
# Clients grouped by data + context similarity
# Implements: d(k,j) = α‖μ(D_k)−μ(D_j)‖² + (1−α)‖c_k−c_j‖²
# (Section 3, Step 2 of the paper)
# ─────────────────────────────────────────────
def cluster_clients(client_contexts, num_clusters=3, alpha=0.5):
    """
    Cluster clients based on data and context similarity.
    alpha controls balance between data vs context similarity.
    Returns cluster assignments for each client.
    """
    num_clients = len(client_contexts)

    if num_clients <= num_clusters:
        # If fewer clients than clusters, each client is its own cluster
        return {i: i for i in range(num_clients)}

    # Stack context vectors
    context_matrix = np.array(list(client_contexts.values()))

    # Normalize context matrix
    norms = np.linalg.norm(context_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    context_normalized = context_matrix / norms

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(context_normalized)

    # Map client_id -> cluster_id
    client_ids = list(client_contexts.keys())
    assignments = {client_ids[i]: int(cluster_labels[i])
                   for i in range(num_clients)}

    return assignments


def get_cluster_groups(assignments, num_clusters):
    """Convert assignments dict to groups dict: cluster_id -> [client_ids]"""
    groups = {c: [] for c in range(num_clusters)}
    for client_id, cluster_id in assignments.items():
        groups[cluster_id].append(client_id)
    return groups


# ─────────────────────────────────────────────
# PHASE 3: CONTEXT DRIFT DETECTION
# Monitors statistical changes in data and
# model performance to detect drift
# Implements Equations 3 from the paper:
# Δ_k^t = ‖c_k^t − c_k^{t−1}‖²
# δ_k^t = |L_k^t(w) − L_k^{t-1}(w)|
# ─────────────────────────────────────────────
class DriftDetector:
    def __init__(self, threshold_context=0.05, threshold_loss=0.05, window=10):
        """
        threshold_context : τ_c — context change threshold
        threshold_loss    : τ_l — loss change threshold
        window            : W  — sliding window size
        """
        self.tau_c   = threshold_context
        self.tau_l   = threshold_loss
        self.window  = window

        self.context_history = {}   # client_id -> list of contexts
        self.loss_history    = {}   # client_id -> list of losses
        self.drift_flags     = {}   # client_id -> bool

    def update(self, client_id, context, loss):
        """Update history and check for drift"""
        # Initialize histories
        if client_id not in self.context_history:
            self.context_history[client_id] = []
            self.loss_history[client_id]    = []
            self.drift_flags[client_id]     = False

        self.context_history[client_id].append(context)
        self.loss_history[client_id].append(loss)

        # Keep only window size
        if len(self.context_history[client_id]) > self.window:
            self.context_history[client_id].pop(0)
            self.loss_history[client_id].pop(0)

        # Need at least 2 observations to detect drift
        if len(self.context_history[client_id]) < 2:
            self.drift_flags[client_id] = False
            return False

        # Context drift: Δ_k^t = ‖c_k^t − c_k^{t-1}‖²
        ctx_current  = self.context_history[client_id][-1]
        ctx_previous = self.context_history[client_id][-2]
        delta_context = np.linalg.norm(ctx_current - ctx_previous)

        # Loss drift: δ_k^t = |L_k^t − L_k^{t-1}|
        loss_current  = self.loss_history[client_id][-1]
        loss_previous = self.loss_history[client_id][-2]
        delta_loss    = abs(loss_current - loss_previous)

        # Drift condition from paper
        drift_detected = (delta_context > self.tau_c) or (delta_loss > self.tau_l)
        self.drift_flags[client_id] = drift_detected

        return drift_detected

    def get_drifting_clients(self):
        """Return list of clients experiencing drift"""
        return [cid for cid, flag in self.drift_flags.items() if flag]

    def any_drift(self):
        """Check if any client is drifting"""
        return any(self.drift_flags.values())


# ─────────────────────────────────────────────
# PHASE 4: LIGHTWEIGHT CONTEXT-AWARE AGGREGATION
# Fair weighted aggregation considering data
# quality, context consistency, and resources
# Implements Equation from paper:
# ω_k = (n_k/Σn_j) · (1/(1+Δ_k^t)) · (E_k/E_max)
# ─────────────────────────────────────────────
def compute_client_weights(client_data, client_contexts,
                           drift_detector, num_clients):
    """
    Compute participation weight for each client.
    Balances data size, drift penalty, and resource level.
    """
    weights = {}
    total_samples = sum(len(client_data[i]["y"]) for i in range(num_clients))

    # Get max energy across clients for normalization
    energy_levels = {}
    for i in range(num_clients):
        ctx = client_contexts.get(i, np.ones(17))
        energy_levels[i] = ctx[-3]   # energy is 3rd from last in context vector
    E_max = max(energy_levels.values()) + 1e-8

    for i in range(num_clients):
        n_k = len(client_data[i]["y"])

        # Data size component
        size_weight = n_k / (total_samples + 1e-8)

        # Drift penalty component — drifting clients get lower weight
        if i in drift_detector.context_history and \
           len(drift_detector.context_history[i]) >= 2:
            ctx_curr = drift_detector.context_history[i][-1]
            ctx_prev = drift_detector.context_history[i][-2]
            delta    = np.linalg.norm(ctx_curr - ctx_prev)
        else:
            delta = 0.0

        drift_weight = 1.0 / (1.0 + delta)

        # Resource/energy component
        E_k = energy_levels[i]
        resource_weight = E_k / E_max

        # Final weight (Equation from paper)
        weights[i] = size_weight * drift_weight * resource_weight

    # Normalize weights to sum to 1
    total_weight = sum(weights.values()) + 1e-8
    weights = {i: w / total_weight for i, w in weights.items()}

    return weights


def aggregate_cluster(models, weights):
    """
    Aggregate models within a cluster using weighted averaging.
    Implements: w_m^{t+1} = Σ_{k∈C_m} ω_k · w_k^t
    """
    if not models:
        return None

    # Start with zero weights
    aggregated = copy.deepcopy(models[0])
    agg_state  = aggregated.state_dict()

    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)

    # Weighted sum
    for model, weight in zip(models, weights):
        state = model.state_dict()
        for key in agg_state:
            agg_state[key] += weight * state[key].float()

    aggregated.load_state_dict(agg_state)
    return aggregated


def aggregate_global(cluster_models, cluster_sizes):
    """
    Aggregate cluster models into one global model.
    Weighted by cluster size.
    """
    total_size = sum(cluster_sizes) + 1e-8
    global_weights = [s / total_size for s in cluster_sizes]

    agg_state = cluster_models[0].state_dict()
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)

    for model, weight in zip(cluster_models, global_weights):
        state = model.state_dict()
        for key in agg_state:
            agg_state[key] += weight * state[key].float()

    global_model = copy.deepcopy(cluster_models[0])
    global_model.load_state_dict(agg_state)
    return global_model


# ─────────────────────────────────────────────
# MAIN FL FRAMEWORK
# Puts all 4 phases together
# ─────────────────────────────────────────────
class CAFLFramework:
    """
    Context-Aware Adaptive Federated Learning Framework
    Combines all 4 phases into a complete FL system
    """
    def __init__(self, input_dim, num_classes, num_clients=10,
                 num_clusters=3, num_rounds=20, local_epochs=5,
                 lr=0.001, alpha_cluster=0.5, device='cpu'):

        self.input_dim    = input_dim
        self.num_classes  = num_classes
        self.num_clients  = num_clients
        self.num_clusters = num_clusters
        self.num_rounds   = num_rounds
        self.local_epochs = local_epochs
        self.lr           = lr
        self.alpha        = alpha_cluster
        self.device       = device

        # Initialize global model
        self.global_model = get_model(input_dim, num_classes, device)

        # Initialize drift detector (Phase 3)
        self.drift_detector = DriftDetector(
            threshold_context=0.05,
            threshold_loss=0.05,
            window=10
        )

        # Track results per round
        self.history = {
            'round'    : [],
            'accuracy' : [],
            'precision': [],
            'recall'   : [],
            'f1'       : [],
            'loss'     : [],
            'drift_clients': []
        }

    def run(self, client_data, verbose=True):
        """
        Run the full FL training for num_rounds rounds
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  CAFL Framework Starting")
            print(f"  Clients: {self.num_clients} | Clusters: {self.num_clusters}")
            print(f"  Rounds: {self.num_rounds} | Device: {self.device}")
            print(f"{'='*60}")

        for round_num in range(self.num_rounds):
            if verbose:
                print(f"\n  Round {round_num+1}/{self.num_rounds}")

            # ── PHASE 1: Extract context for all clients ──
            client_contexts = {}
            for i in range(self.num_clients):
                client_contexts[i] = extract_context(
                    i, client_data[i], round_num, self.num_clients
                )

            # ── PHASE 2: Cluster clients ──
            assignments = cluster_clients(
                client_contexts,
                num_clusters=self.num_clusters,
                alpha=self.alpha
            )
            groups = get_cluster_groups(assignments, self.num_clusters)

            # ── Local training for all clients ──
            client_models = {}
            client_losses = {}

            for i in range(self.num_clients):
                # Initialize client model from global model
                local_model = copy.deepcopy(self.global_model)

                # Train locally (Phase 1 - local update equation)
                local_model, loss = train_local(
                    local_model,
                    client_data[i]["X"],
                    client_data[i]["y"],
                    epochs=self.local_epochs,
                    lr=self.lr,
                    device=self.device
                )
                client_models[i] = local_model
                client_losses[i] = loss

                # ── PHASE 3: Update drift detector ──
                self.drift_detector.update(i, client_contexts[i], loss)

            # Get drifting clients
            drifting = self.drift_detector.get_drifting_clients()
            if verbose and drifting:
                print(f"    Drift detected in clients: {drifting}")

            # ── PHASE 4: Cluster-level aggregation ──
            client_weights = compute_client_weights(
                client_data, client_contexts,
                self.drift_detector, self.num_clients
            )

            cluster_models = []
            cluster_sizes  = []

            for cluster_id, members in groups.items():
                if not members:
                    continue

                models  = [client_models[i] for i in members]
                weights = [client_weights[i] for i in members]

                # Normalize within cluster
                total_w = sum(weights) + 1e-8
                weights = [w / total_w for w in weights]

                cluster_model = aggregate_cluster(models, weights)
                cluster_models.append(cluster_model)
                cluster_sizes.append(sum(len(client_data[i]["y"])
                                        for i in members))

            # Global aggregation
            if cluster_models:
                self.global_model = aggregate_global(cluster_models, cluster_sizes)

            # ── Evaluate global model ──
            all_X = np.concatenate([client_data[i]["X"]
                                    for i in range(self.num_clients)], axis=0)
            all_y = np.concatenate([client_data[i]["y"]
                                    for i in range(self.num_clients)], axis=0)

            # Sample for speed (max 10000 samples)
            if len(all_y) > 10000:
                idx   = np.random.choice(len(all_y), 10000, replace=False)
                all_X = all_X[idx]
                all_y = all_y[idx]

            metrics = evaluate_local(
                self.global_model, all_X, all_y, device=self.device
            )
            avg_loss = np.mean(list(client_losses.values()))

            # Store history
            self.history['round'].append(round_num + 1)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['precision'].append(metrics['precision'])
            self.history['recall'].append(metrics['recall'])
            self.history['f1'].append(metrics['f1'])
            self.history['loss'].append(avg_loss)
            self.history['drift_clients'].append(len(drifting))

            if verbose:
                print(f"    Accuracy : {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Drifting: {len(drifting)}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training Complete!")
            print(f"  Final Accuracy : {self.history['accuracy'][-1]:.4f}")
            print(f"  Final F1-Score : {self.history['f1'][-1]:.4f}")
            print(f"{'='*60}")

        return self.history


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CAFL Framework with dummy data...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create dummy client data
    num_clients = 10
    client_data = {}
    for i in range(num_clients):
        n = np.random.randint(500, 2000)
        client_data[i] = {
            "X": np.random.randn(n, 39).astype(np.float32),
            "y": np.random.randint(0, 2, n)
        }

    # Run framework
    framework = CAFLFramework(
        input_dim=39, num_classes=2,
        num_clients=num_clients, num_clusters=3,
        num_rounds=3, local_epochs=2,
        device=device
    )

    history = framework.run(client_data, verbose=True)
    print("\nFramework test complete!")
