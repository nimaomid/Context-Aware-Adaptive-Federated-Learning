"""
Baseline Federated Learning Algorithms
Implements FedAvg and FedProx for comparison against CAFL framework
Referenced in paper Section 4.5 as baseline methods
"""

import numpy as np
import torch
import torch.nn as nn
import copy
from model import get_model, evaluate_local, get_model_weights, set_model_weights


# ─────────────────────────────────────────────
# FEDAVG BASELINE
# McMahan et al. — Communication-Efficient
# Learning of Deep Networks from
# Decentralized Data (2017)
# Implements Equation 2 from the paper:
# w^{t+1} = Σ(|D_i|/Σ|D_j|) · w_i^{t+1}
# ─────────────────────────────────────────────
def fedavg_train_local(model, X, y, epochs=5,
                       lr=0.001, batch_size=256, device='cpu'):
    """Standard local training for FedAvg"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return model, total_loss / max(num_batches, 1)


def fedavg_aggregate(client_models, client_data, num_clients):
    """
    Standard FedAvg aggregation — weighted by dataset size
    w^{t+1} = Σ(n_k / Σn_j) · w_k
    """
    total_samples = sum(len(client_data[i]["y"]) for i in range(num_clients))

    # Get aggregated state dict
    agg_state = copy.deepcopy(client_models[0].state_dict())
    for key in agg_state:
        agg_state[key] = torch.zeros_like(agg_state[key], dtype=torch.float32)

    for i in range(num_clients):
        weight = len(client_data[i]["y"]) / total_samples
        state  = client_models[i].state_dict()
        for key in agg_state:
            agg_state[key] += weight * state[key].float()

    global_model = copy.deepcopy(client_models[0])
    global_model.load_state_dict(agg_state)
    return global_model


class FedAvg:
    """
    FedAvg: Standard Federated Averaging
    Baseline comparison for CAFL framework
    """
    def __init__(self, input_dim, num_classes, num_clients=10,
                 num_rounds=20, local_epochs=5, lr=0.001, device='cpu'):

        self.input_dim    = input_dim
        self.num_classes  = num_classes
        self.num_clients  = num_clients
        self.num_rounds   = num_rounds
        self.local_epochs = local_epochs
        self.lr           = lr
        self.device       = device

        self.global_model = get_model(input_dim, num_classes, device)

        self.history = {
            'round'    : [],
            'accuracy' : [],
            'precision': [],
            'recall'   : [],
            'f1'       : [],
            'loss'     : []
        }

    def run(self, client_data, verbose=True):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  FedAvg Baseline")
            print(f"  Clients: {self.num_clients} | Rounds: {self.num_rounds}")
            print(f"{'='*60}")

        for round_num in range(self.num_rounds):
            client_models = []
            client_losses = []

            # Local training
            for i in range(self.num_clients):
                local_model = copy.deepcopy(self.global_model)
                local_model, loss = fedavg_train_local(
                    local_model,
                    client_data[i]["X"],
                    client_data[i]["y"],
                    epochs=self.local_epochs,
                    lr=self.lr,
                    device=self.device
                )
                client_models.append(local_model)
                client_losses.append(loss)

            # FedAvg aggregation
            self.global_model = fedavg_aggregate(
                client_models, client_data, self.num_clients
            )

            # Evaluate
            all_X = np.concatenate([client_data[i]["X"]
                                    for i in range(self.num_clients)], axis=0)
            all_y = np.concatenate([client_data[i]["y"]
                                    for i in range(self.num_clients)], axis=0)

            if len(all_y) > 10000:
                idx   = np.random.choice(len(all_y), 10000, replace=False)
                all_X = all_X[idx]
                all_y = all_y[idx]

            metrics  = evaluate_local(self.global_model, all_X, all_y,
                                      device=self.device)
            avg_loss = np.mean(client_losses)

            self.history['round'].append(round_num + 1)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['precision'].append(metrics['precision'])
            self.history['recall'].append(metrics['recall'])
            self.history['f1'].append(metrics['f1'])
            self.history['loss'].append(avg_loss)

            if verbose:
                print(f"  Round {round_num+1:02d} | "
                      f"Accuracy: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"Loss: {avg_loss:.4f}")

        if verbose:
            print(f"\n  FedAvg Final Accuracy : {self.history['accuracy'][-1]:.4f}")
            print(f"  FedAvg Final F1-Score : {self.history['f1'][-1]:.4f}")

        return self.history


# ─────────────────────────────────────────────
# FEDPROX BASELINE
# Li et al. — Federated Optimization in
# Heterogeneous Networks (2020)
# Adds proximal term to local objective:
# min L_k(w) + (μ/2)‖w − w_global‖²
# ─────────────────────────────────────────────
def fedprox_train_local(model, global_model, X, y, epochs=5,
                        lr=0.001, mu=0.01, batch_size=256, device='cpu'):
    """
    FedProx local training with proximal term.
    mu: proximal coefficient (controls deviation from global model)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Store global model weights for proximal term
    global_weights = {k: v.clone().detach()
                      for k, v in global_model.state_dict().items()}

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    total_loss  = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            # Standard cross-entropy loss
            outputs = model(batch_X)
            ce_loss = criterion(outputs, batch_y)

            # Proximal term: (μ/2)‖w − w_global‖²
            prox_loss = 0.0
            for name, param in model.named_parameters():
                if name in global_weights:
                    prox_loss += torch.sum(
                        (param - global_weights[name].to(device)) ** 2
                    )
            prox_loss = (mu / 2) * prox_loss

            # Total FedProx loss
            loss = ce_loss + prox_loss
            loss.backward()
            optimizer.step()

            total_loss  += ce_loss.item()
            num_batches += 1

    return model, total_loss / max(num_batches, 1)


class FedProx:
    """
    FedProx: Federated Learning with Proximal Term
    Baseline comparison for CAFL framework
    Better than FedAvg for Non-IID data
    """
    def __init__(self, input_dim, num_classes, num_clients=10,
                 num_rounds=20, local_epochs=5, lr=0.001,
                 mu=0.01, device='cpu'):

        self.input_dim    = input_dim
        self.num_classes  = num_classes
        self.num_clients  = num_clients
        self.num_rounds   = num_rounds
        self.local_epochs = local_epochs
        self.lr           = lr
        self.mu           = mu
        self.device       = device

        self.global_model = get_model(input_dim, num_classes, device)

        self.history = {
            'round'    : [],
            'accuracy' : [],
            'precision': [],
            'recall'   : [],
            'f1'       : [],
            'loss'     : []
        }

    def run(self, client_data, verbose=True):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  FedProx Baseline (mu={self.mu})")
            print(f"  Clients: {self.num_clients} | Rounds: {self.num_rounds}")
            print(f"{'='*60}")

        for round_num in range(self.num_rounds):
            client_models = []
            client_losses = []

            # Local training with proximal term
            for i in range(self.num_clients):
                local_model = copy.deepcopy(self.global_model)
                local_model, loss = fedprox_train_local(
                    local_model,
                    self.global_model,
                    client_data[i]["X"],
                    client_data[i]["y"],
                    epochs=self.local_epochs,
                    lr=self.lr,
                    mu=self.mu,
                    device=self.device
                )
                client_models.append(local_model)
                client_losses.append(loss)

            # Same aggregation as FedAvg
            self.global_model = fedavg_aggregate(
                client_models, client_data, self.num_clients
            )

            # Evaluate
            all_X = np.concatenate([client_data[i]["X"]
                                    for i in range(self.num_clients)], axis=0)
            all_y = np.concatenate([client_data[i]["y"]
                                    for i in range(self.num_clients)], axis=0)

            if len(all_y) > 10000:
                idx   = np.random.choice(len(all_y), 10000, replace=False)
                all_X = all_X[idx]
                all_y = all_y[idx]

            metrics  = evaluate_local(self.global_model, all_X, all_y,
                                      device=self.device)
            avg_loss = np.mean(client_losses)

            self.history['round'].append(round_num + 1)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['precision'].append(metrics['precision'])
            self.history['recall'].append(metrics['recall'])
            self.history['f1'].append(metrics['f1'])
            self.history['loss'].append(avg_loss)

            if verbose:
                print(f"  Round {round_num+1:02d} | "
                      f"Accuracy: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1']:.4f} | "
                      f"Loss: {avg_loss:.4f}")

        if verbose:
            print(f"\n  FedProx Final Accuracy : {self.history['accuracy'][-1]:.4f}")
            print(f"  FedProx Final F1-Score : {self.history['f1'][-1]:.4f}")

        return self.history


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Baselines with dummy data...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Dummy client data
    num_clients = 10
    client_data = {}
    for i in range(num_clients):
        n = 500
        client_data[i] = {
            "X": np.random.randn(n, 39).astype(np.float32),
            "y": np.random.randint(0, 2, n)
        }

    # Test FedAvg
    fedavg = FedAvg(
        input_dim=39, num_classes=2,
        num_clients=num_clients,
        num_rounds=3, local_epochs=2,
        device=device
    )
    history_avg = fedavg.run(client_data, verbose=True)

    # Test FedProx
    fedprox = FedProx(
        input_dim=39, num_classes=2,
        num_clients=num_clients,
        num_rounds=3, local_epochs=2,
        mu=0.01, device=device
    )
    history_prox = fedprox.run(client_data, verbose=True)

    print("\nBaselines test complete!")
