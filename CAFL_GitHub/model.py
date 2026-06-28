"""
Neural Network Model for Context-Aware Adaptive Federated Learning
Each IoT client trains this model locally on its private data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# MAIN MODEL — Lightweight MLP for IoT devices
# Designed to be small enough for edge devices
# but accurate enough for intrusion detection
# ─────────────────────────────────────────────
class IoTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        """
        Args:
            input_dim  : number of input features (39 for FD-IDS, 78 for FLY-SMOTE)
            num_classes: number of output classes (2 for FD-IDS, 15 for FLY-SMOTE)
            hidden_dim : size of hidden layers (kept small for IoT devices)
        """
        super(IoTClassifier, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),

            # Output layer
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────
def get_model(input_dim, num_classes, device='cpu'):
    """Create and return a model on the specified device"""
    model = IoTClassifier(input_dim, num_classes)
    model = model.to(device)
    return model


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_weights(model):
    """Extract model weights as a dictionary (for FL aggregation)"""
    return {k: v.clone() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    """Load weights into a model (for FL aggregation)"""
    model.load_state_dict(weights)
    return model


def copy_model_weights(source_model, target_model):
    """Copy weights from source model to target model"""
    target_model.load_state_dict(source_model.state_dict())
    return target_model


# ─────────────────────────────────────────────
# LOCAL TRAINING — Phase 1 of FL
# Each client trains on its local data
# ─────────────────────────────────────────────
def train_local(model, X, y, epochs=5, lr=0.001, batch_size=256, device='cpu'):
    """
    Train model on local client data
    Implements Equation 1 from the paper:
    w_i^{t+1} = w_i^t - η ∇L(w_i^t; D_i)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        total_loss += epoch_loss

    avg_loss = total_loss / max(num_batches, 1)
    return model, avg_loss


# ─────────────────────────────────────────────
# LOCAL EVALUATION
# ─────────────────────────────────────────────
def evaluate_local(model, X, y, batch_size=512, device='cpu'):
    """
    Evaluate model on local client data
    Returns accuracy, precision, recall, f1
    """
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'accuracy' : accuracy,
        'precision': precision,
        'recall'   : recall,
        'f1'       : f1
    }


# ─────────────────────────────────────────────
# TEST — Verify model works
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("  Model Test")
    print("=" * 60)

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Test with FD-IDS dimensions
    print("\n  Testing FD-IDS model (39 features, 2 classes)...")
    model_fd = get_model(input_dim=39, num_classes=2, device=device)
    print(f"  Parameters: {count_parameters(model_fd):,}")

    # Dummy data test
    X_dummy = np.random.randn(1000, 39).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 1000)
    model_fd, loss = train_local(model_fd, X_dummy, y_dummy, epochs=2, device=device)
    metrics = evaluate_local(model_fd, X_dummy, y_dummy, device=device)
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")

    # Test with FLY-SMOTE dimensions
    print("\n  Testing FLY-SMOTE model (78 features, 15 classes)...")
    model_fly = get_model(input_dim=78, num_classes=15, device=device)
    print(f"  Parameters: {count_parameters(model_fly):,}")

    X_dummy2 = np.random.randn(1000, 78).astype(np.float32)
    y_dummy2 = np.random.randint(0, 15, 1000)
    model_fly, loss2 = train_local(model_fly, X_dummy2, y_dummy2, epochs=2, device=device)
    metrics2 = evaluate_local(model_fly, X_dummy2, y_dummy2, device=device)
    print(f"  Loss: {loss2:.4f}")
    print(f"  Accuracy: {metrics2['accuracy']:.4f}")
    print(f"  F1-Score: {metrics2['f1']:.4f}")

    print("\n" + "=" * 60)
    print("  Model test complete!")
    print("=" * 60)