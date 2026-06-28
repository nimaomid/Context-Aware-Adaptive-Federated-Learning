"""
Data Loader for Context-Aware Adaptive Federated Learning
Handles both FD-IDS (UNSW-NB15) and FLY-SMOTE (CICIDS-2017) datasets
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
FD_IDS_PATH    = "data/fd_ids/"
FLY_SMOTE_PATH = "data/fly_smote/"
NUM_CLIENTS    = 10
RANDOM_SEED    = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# 1. LOAD FD-IDS DATASET (UNSW-NB15)
# ─────────────────────────────────────────────
def load_fd_ids():
    print("\n[FD-IDS] Loading UNSW-NB15 dataset...")

    col_names = [
        'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
        'sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts',
        'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len',
        'Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
        'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
        'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
        'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label'
    ]

    all_files = [f for f in os.listdir(FD_IDS_PATH) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {FD_IDS_PATH}")

    dfs = []
    for f in sorted(all_files):
        path = os.path.join(FD_IDS_PATH, f)
        df = pd.read_csv(path, header=None, names=col_names,
                         low_memory=False, dtype={'label': str})
        dfs.append(df)
        print(f"  Loaded {f}: {df.shape[0]} rows")

    data = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {data.shape[0]}")

    # Clean label column — keep only rows with '0' or '1'
    data['label'] = data['label'].astype(str).str.strip()
    data = data[data['label'].isin(['0', '1'])]
    data['label'] = data['label'].astype(int)

    print(f"  Rows after label cleaning: {data.shape[0]}")
    print(f"  Label column: 'label'")
    print(f"  Classes: {sorted(data['label'].unique())}")

    # Drop non-numeric and identifier columns
    drop_cols = ['label', 'attack_cat', 'srcip', 'dstip', 'proto', 'state', 'service']
    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    y = data['label'].values

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Replace inf and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Features shape: {X_scaled.shape}")
    return X_scaled, y, np.array(['Normal', 'Attack'])


# ─────────────────────────────────────────────
# 2. LOAD FLY-SMOTE DATASET (CICIDS-2017)
# ─────────────────────────────────────────────
def load_fly_smote():
    print("\n[FLY-SMOTE] Loading CICIDS-2017 dataset...")

    all_files = [f for f in os.listdir(FLY_SMOTE_PATH) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {FLY_SMOTE_PATH}")

    dfs = []
    for f in sorted(all_files):
        path = os.path.join(FLY_SMOTE_PATH, f)
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
        print(f"  Loaded {f}: {df.shape[0]} rows")

    data = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows: {data.shape[0]}")

    # Clean column names
    data.columns = data.columns.str.strip()

    # Replace inf and NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Find label column
    label_col = None
    for col in data.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break
    if label_col is None:
        label_col = data.columns[-1]
    print(f"  Label column: '{label_col}'")

    # Encode labels
    le = LabelEncoder()
    data[label_col] = le.fit_transform(data[label_col].astype(str))

    X = data.drop(columns=[label_col])
    y = data[label_col].values
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Features shape: {X_scaled.shape}")
    print(f"  Classes: {le.classes_}")
    return X_scaled, y, le.classes_


# ─────────────────────────────────────────────
# 3. NON-IID SPLIT (Dirichlet — Equation 4)
# ─────────────────────────────────────────────
def split_non_iid(X, y, num_clients=NUM_CLIENTS, alpha=0.3):
    print(f"\n[Non-IID Split] {num_clients} clients, alpha={alpha}...")

    num_classes = len(np.unique(y))
    client_data = {i: {"X": [], "y": []} for i in range(num_clients)}

    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_idx)).astype(int)
        proportions[-1] = len(class_idx) - proportions[:-1].sum()

        start = 0
        for cid, count in enumerate(proportions):
            end = start + count
            idx = class_idx[start:end]
            client_data[cid]["X"].append(X[idx])
            client_data[cid]["y"].append(y[idx])
            start = end

    for i in range(num_clients):
        client_data[i]["X"] = np.concatenate(client_data[i]["X"], axis=0)
        client_data[i]["y"] = np.concatenate(client_data[i]["y"], axis=0)
        print(f"  Client {i:02d}: {len(client_data[i]['y'])} samples, "
              f"classes: {np.unique(client_data[i]['y'])}")

    return client_data


# ─────────────────────────────────────────────
# 4. MAIN TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Context-Aware Federated Learning — Data Loader Test")
    print("=" * 60)

    try:
        X_fd, y_fd, classes_fd = load_fd_ids()
        clients_fd = split_non_iid(X_fd, y_fd)
        print("\n[FD-IDS] Data loaded and split successfully!")
    except Exception as e:
        print(f"\n[FD-IDS] Error: {e}")

    try:
        X_fly, y_fly, classes_fly = load_fly_smote()
        clients_fly = split_non_iid(X_fly, y_fly)
        print("\n[FLY-SMOTE] Data loaded and split successfully!")
    except Exception as e:
        print(f"\n[FLY-SMOTE] Error: {e}")

    print("\n" + "=" * 60)
    print("  Data loader test complete!")
    print("=" * 60)