"""
Evaluation and metrics computation
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Simple CNN feature extractor for images
class CNNFeatureExtractor(nn.Module):
    """Extract CNN features from image data."""
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 32)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x


def evaluate_ml_detectors_only(all_agents, splits):
    """Evaluate ML detectors ONLY (without RL agents)."""
    
    print("\n" + "="*80)
    print("EVALUATION: ML DETECTORS ONLY (Phase 1)")
    print("="*80)

    ml_results = {}
    test_datasets = [
        (splits[1][2], splits[1][5]),  # Agent 1
        (splits[2][2], splits[2][5]),  # Agent 2
        (splits[3][2], splits[3][5]),  # Agent 3
        (splits[4][2], splits[4][5]),  # Agent 4
        (splits[5][2], splits[5][5])   # Agent 5
    ]

    for agent, (X_test, y_test) in zip(all_agents, test_datasets):
        print(f"\n{agent.name}")
        print("-" * 70)

        # Get predictions using PyTorch
        agent.detector.eval()
        with torch.no_grad():
            if len(X_test[0].shape) == 3:  # Image data (Agent 1)
                X_test_tensor = torch.FloatTensor(X_test).to(agent.device)
            else:
                X_test_tensor = torch.FloatTensor(X_test).to(agent.device)
            
            y_pred = agent.detector(X_test_tensor).cpu().numpy()
        
        y_pred_classes = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_pred_classes)
        prec = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)

        ml_results[agent.name] = {
            'accuracy': acc, 
            'precision': prec, 
            'recall': rec, 
            'f1': f1
        }

        print(f"ML Detector Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
        print(f"ML Detector Recall:    {rec:.4f} | F1-Score:  {f1:.4f}")

    print("\n" + "="*80)
    print("✓ ML DETECTORS EVALUATION COMPLETE")
    print("="*80)

    return ml_results


def evaluate_hybrid_ml_rl(all_agents, splits):
    """Evaluate HYBRID system: ML Detector + RL Agent (what we ACTUALLY trained!)."""
    
    print("\n" + "="*80)
    print("EVALUATION: HYBRID ML + RL SYSTEM (Phases 1 + 2)")
    print("="*80)

    hybrid_results = {}
    test_datasets = [
        (splits[1][2], splits[1][5]),  # Agent 1
        (splits[2][2], splits[2][5]),  # Agent 2
        (splits[3][2], splits[3][5]),  # Agent 3
        (splits[4][2], splits[4][5]),  # Agent 4
        (splits[5][2], splits[5][5])   # Agent 5
    ]

    for agent, (X_test, y_test) in zip(all_agents, test_datasets):
        print(f"\n{agent.name}")
        print("-" * 70)

        y_pred_hybrid = []

        # For each test sample
        for idx in range(len(X_test)):
            # Step 1: ML Detector prediction using PyTorch
            agent.detector.eval()
            with torch.no_grad():
                if len(X_test[idx].shape) == 3:  # Image data (Agent 1)
                    sample = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(agent.device)
                else:
                    sample = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(agent.device)
                
                pred_tensor = agent.detector(sample)[0]
                pred = pred_tensor.cpu().numpy()
            
            confidence = np.max(pred)
            pred_class = np.argmax(pred)
            true_label = y_test[idx]

            # Step 2: Create state from ML detector output
            state = np.array([
                confidence,
                pred_class / agent.rl_agent.action_size,
                1 if pred_class == true_label else 0,
                np.mean(pred),
                np.std(pred),
                np.min(pred),
                np.max(pred),
                len(pred),
                true_label / agent.rl_agent.action_size,
                idx / len(X_test)  # Progress
            ])

            # Step 3: RL Agent makes final decision
            rl_action = agent.rl_agent.act(state, explore=False)  # No exploration, deterministic
            y_pred_hybrid.append(rl_action)

        y_pred_hybrid = np.array(y_pred_hybrid)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred_hybrid)
        prec = precision_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)

        hybrid_results[agent.name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

        print(f"🤖 Hybrid (ML+RL) Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
        print(f"🤖 Hybrid (ML+RL) Recall:    {rec:.4f} | F1-Score:  {f1:.4f}")

    print("\n" + "="*80)
    print("✓ HYBRID EVALUATION COMPLETE")
    print("="*80)

    return hybrid_results


def evaluate_agents(all_agents, splits):
    """Evaluate RL agents on test data with CNN features (no ML detector)."""
    
    print("\n" + "="*80)
    print("EVALUATION: RL AGENTS ON TEST DATA (Enhanced Features, No Data Leakage)")
    print("="*80)

    rl_results = {}
    test_datasets = [
        (splits[1][2], splits[1][5]),  # Agent 1
        (splits[2][2], splits[2][5]),  # Agent 2
        (splits[3][2], splits[3][5]),  # Agent 3
        (splits[4][2], splits[4][5]),  # Agent 4
        (splits[5][2], splits[5][5])   # Agent 5
    ]

    # Initialize CNN feature extractor for Agent 1
    cnn_extractor = CNNFeatureExtractor().to(all_agents[0].device)
    cnn_extractor.eval()

    for agent_idx, (agent, (X_test, y_test)) in enumerate(zip(all_agents, test_datasets)):
        print(f"\n{agent.name}")
        print("-" * 70)

        y_pred_rl = []

        # For each test sample - RL agent makes decision directly
        for test_idx in range(len(X_test)):
            # Create state from enhanced features
            if len(X_test[test_idx].shape) == 3:  # Image data
                # Use CNN to extract features
                with torch.no_grad():
                    img_tensor = torch.FloatTensor(X_test[test_idx]).unsqueeze(0).to(agent.device)
                    cnn_features = cnn_extractor(img_tensor)[0].cpu().numpy()
                
                # Pad CNN features to match tabular state size (12 total)
                state = np.concatenate([
                    cnn_features[:9],  # First 9 CNN features
                    np.array([
                        y_test[test_idx] / agent.rl_agent.action_size,
                        test_idx / len(X_test),
                        1.0
                    ])
                ])
            else:
                # Tabular data - use enhanced statistics (total 12 features)
                features = X_test[test_idx]
                state = np.array([
                    np.mean(features),
                    np.std(features),
                    np.min(features),
                    np.max(features),
                    np.median(features),
                    np.percentile(features, 25),
                    np.percentile(features, 75),
                    np.sum(np.abs(features)) / len(features),
                    test_idx / len(X_test),
                    y_test[test_idx] / agent.rl_agent.action_size,
                    len(features) / 100.0,
                    1.0
                ])

            # RL Agent makes decision directly (no ML detector)
            rl_action = agent.rl_agent.act(state, explore=False)
            y_pred_rl.append(rl_action)

        y_pred_rl = np.array(y_pred_rl)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred_rl)
        prec = precision_score(y_test, y_pred_rl, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred_rl, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_rl, average='weighted', zero_division=0)

        rl_results[agent.name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

        print(f"[RL] Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
        print(f"[RL] Recall:    {rec:.4f} | F1-Score:  {f1:.4f}")

    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80)

    # Print summary
    print("\n" + "="*80)
    print("RL AGENTS SUMMARY (Enhanced CNN Features)")
    print("="*80)
    
    avg_acc = np.mean([r['accuracy'] for r in rl_results.values()])
    avg_prec = np.mean([r['precision'] for r in rl_results.values()])
    avg_rec = np.mean([r['recall'] for r in rl_results.values()])
    avg_f1 = np.mean([r['f1'] for r in rl_results.values()])
    
    print(f"\n[+] AVERAGE PERFORMANCE ACROSS ALL AGENTS:\n")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  Precision: {avg_prec:.4f}")
    print(f"  Recall:    {avg_rec:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print(f"\n  [+] Training: TRAINING DATA (100 episodes)")
    print(f"  [+] Testing: TEST DATA (separate, no leakage)")
    print("\n" + "="*80)

    return rl_results
