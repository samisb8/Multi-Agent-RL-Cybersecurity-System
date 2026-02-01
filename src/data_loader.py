"""
Data loading and preprocessing module
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import datasets, transforms


def load_nsl_kdd():
    """Load NSL-KDD dataset from GitHub."""
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
    ]

    try:
        url_train = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
        print("Downloading KDDTrain+.txt...")
        response = requests.get(url_train, timeout=30)

        if response.status_code == 200:
            df_full = pd.read_csv(StringIO(response.text), names=columns, header=None)
            print(f"✓ Downloaded: {df_full.shape}")
        else:
            raise Exception("Download failed")

    except Exception as e:
        print(f"✗ Failed to download from GitHub: {e}")
        print("\n⚠️  CRITICAL: Cannot proceed without real data.")
        raise

    # Remove level column
    df_full = df_full.drop('level', axis=1)

    # Show attack types
    print(f"\n✓ Attack types distribution:")
    print(df_full['attack'].value_counts())

    # Map attack types to categories
    attack_mapping = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
        'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r'
    }

    df_full['category'] = df_full['attack'].map(attack_mapping)
    df_full['category'] = df_full['category'].fillna('unknown')

    print(f"\n✓ Category distribution:")
    print(df_full['category'].value_counts())

    return df_full


def preprocess_nsl_kdd(df_full):
    """Preprocess NSL-KDD data."""
    print("\n[PREPROCESSING] NSL-KDD...\n")

    # One-hot encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_encoded = pd.get_dummies(df_full, columns=categorical_cols, drop_first=True)

    # Extract features and labels
    feature_cols = [col for col in df_encoded.columns if col not in ['attack', 'category']]
    X_full = df_encoded[feature_cols].values
    y_category = df_encoded['category'].values
    y_attack = df_encoded['attack'].values

    # Normalize features
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    print(f"✓ Features shape: {X_full_scaled.shape}")
    print(f"✓ Total samples: {len(X_full_scaled)}")

    return X_full_scaled, y_category, y_attack, df_encoded, scaler


def create_agent_datasets(X_full_scaled, y_category, y_attack, df_encoded):
    """Create specialized datasets for each agent."""
    print("\n[CREATING AGENT DATASETS]...\n")

    # Agent 2: DoS vs Normal
    mask_dos = (y_category == 'dos') | (y_category == 'normal')
    X_dos = X_full_scaled[mask_dos]
    y_dos = (y_category[mask_dos] == 'dos').astype(int)
    print(f"Agent 2 (DoS): {X_dos.shape}, Distribution: {np.bincount(y_dos)}")

    # Agent 3: Full multi-class (all 23 attack types)
    le_full = LabelEncoder()
    y_full = le_full.fit_transform(y_attack)
    X_full_agent3 = X_full_scaled
    y_full_agent3 = y_full
    n_classes_full = len(np.unique(y_full_agent3))
    print(f"Agent 3 (Full): {X_full_agent3.shape}, Classes: {n_classes_full}")

    # Agent 4: Probe vs Normal
    mask_probe = (y_category == 'probe') | (y_category == 'normal')
    X_probe = X_full_scaled[mask_probe]
    y_probe = (y_category[mask_probe] == 'probe').astype(int)
    print(f"Agent 4 (Probe): {X_probe.shape}, Distribution: {np.bincount(y_probe)}")

    # Agent 5: R2L/U2R vs Normal
    mask_r2l_u2r = np.isin(y_category, ['r2l', 'u2r', 'normal'])
    X_r2l_u2r = X_full_scaled[mask_r2l_u2r]
    y_r2l_u2r = np.isin(y_category[mask_r2l_u2r], ['r2l', 'u2r']).astype(int)
    print(f"Agent 5 (R2L/U2R): {X_r2l_u2r.shape}, Distribution: {np.bincount(y_r2l_u2r)}")

    print("\n✓ All agent datasets created from REAL NSL-KDD data")

    return {
        2: (X_dos, y_dos),
        3: (X_full_agent3, y_full_agent3, n_classes_full),
        4: (X_probe, y_probe),
        5: (X_r2l_u2r, y_r2l_u2r)
    }


def load_mnist(train_size=10000, test_size=2000):
    """Load MNIST dataset."""
    print("\n[MNIST] Loading for Agent 1...\n")

    # Load using torchvision
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Convert to numpy
    X_train_mnist = train_dataset.data[:train_size].numpy() / 255.0
    y_train_mnist = train_dataset.targets[:train_size].numpy()
    
    X_test_mnist = test_dataset.data[:test_size].numpy() / 255.0
    y_test_mnist = test_dataset.targets[:test_size].numpy()

    # Reshape for PyTorch (add channel dimension)
    X_train_mnist = X_train_mnist.reshape(-1, 1, 28, 28)
    X_test_mnist = X_test_mnist.reshape(-1, 1, 28, 28)

    # 3-way classification: Allow (0-4) / Alert (5-7) / Block (8-9)
    y_train_adv = np.where(y_train_mnist < 5, 0, np.where(y_train_mnist < 8, 1, 2))
    y_test_adv = np.where(y_test_mnist < 5, 0, np.where(y_test_mnist < 8, 1, 2))

    print(f"✓ MNIST loaded: Train={X_train_mnist.shape}, Test={X_test_mnist.shape}")
    print(f"✓ 3-way labels: {np.bincount(y_train_adv)}")

    return X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, y_train_adv, y_test_adv


def create_train_val_test_splits(X_train_mnist, y_train_adv, X_test_mnist, y_test_adv, agent_datasets):
    """Create train/val/test splits for all agents."""
    print("\n" + "="*80)
    print("CREATING SPLITS")
    print("="*80 + "\n")

    splits = {}

    # Agent 1: MNIST
    X_train_a1, X_val_a1, y_train_a1, y_val_a1 = train_test_split(
        X_train_mnist, y_train_adv, test_size=0.2, random_state=42, stratify=y_train_adv
    )
    X_test_a1, y_test_a1 = X_test_mnist, y_test_adv
    splits[1] = (X_train_a1, X_val_a1, X_test_a1, y_train_a1, y_val_a1, y_test_a1)
    print(f"Agent 1: Train={X_train_a1.shape}, Val={X_val_a1.shape}, Test={X_test_a1.shape}")

    # Agent 2: DoS
    X_dos, y_dos = agent_datasets[2]
    X_train_a2, X_temp, y_train_a2, y_temp = train_test_split(
        X_dos, y_dos, test_size=0.3, random_state=42, stratify=y_dos
    )
    X_val_a2, X_test_a2, y_val_a2, y_test_a2 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    splits[2] = (X_train_a2, X_val_a2, X_test_a2, y_train_a2, y_val_a2, y_test_a2)
    print(f"Agent 2: Train={X_train_a2.shape}, Val={X_val_a2.shape}, Test={X_test_a2.shape}")

    # Agent 3: Full multi-class
    X_full_agent3, y_full_agent3, n_classes = agent_datasets[3]
    X_train_a3, X_temp, y_train_a3, y_temp = train_test_split(
        X_full_agent3, y_full_agent3, test_size=0.3, random_state=42, stratify=y_full_agent3
    )
    X_val_a3, X_test_a3, y_val_a3, y_test_a3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    splits[3] = (X_train_a3, X_val_a3, X_test_a3, y_train_a3, y_val_a3, y_test_a3, n_classes)
    print(f"Agent 3: Train={X_train_a3.shape}, Val={X_val_a3.shape}, Test={X_test_a3.shape}")

    # Agent 4: Probe
    X_probe, y_probe = agent_datasets[4]
    X_train_a4, X_temp, y_train_a4, y_temp = train_test_split(
        X_probe, y_probe, test_size=0.3, random_state=42, stratify=y_probe
    )
    X_val_a4, X_test_a4, y_val_a4, y_test_a4 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    splits[4] = (X_train_a4, X_val_a4, X_test_a4, y_train_a4, y_val_a4, y_test_a4)
    print(f"Agent 4: Train={X_train_a4.shape}, Val={X_val_a4.shape}, Test={X_test_a4.shape}")

    # Agent 5: R2L/U2R
    X_r2l_u2r, y_r2l_u2r = agent_datasets[5]
    X_train_a5, X_temp, y_train_a5, y_temp = train_test_split(
        X_r2l_u2r, y_r2l_u2r, test_size=0.3, random_state=42, stratify=y_r2l_u2r
    )
    X_val_a5, X_test_a5, y_val_a5, y_test_a5 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    splits[5] = (X_train_a5, X_val_a5, X_test_a5, y_train_a5, y_val_a5, y_test_a5)
    print(f"Agent 5: Train={X_train_a5.shape}, Val={X_val_a5.shape}, Test={X_test_a5.shape}")

    print("\n✓ All splits ready")

    return splits
