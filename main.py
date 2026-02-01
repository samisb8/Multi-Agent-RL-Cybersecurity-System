"""
Main training pipeline - Entry point from root directory
Imports modules from src/ and saves outputs to results/
"""

import sys
import os
from pathlib import Path

# Add src/ to path so imports work
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import torch
import random
import warnings

from config import *
from data_loader import (
    load_nsl_kdd, preprocess_nsl_kdd, create_agent_datasets,
    load_mnist, create_train_val_test_splits
)
from agents import Agent1_Adversarial, BinaryAgent, Agent3_MultiClass
from rl_trainer import train_rl_agents
from evaluator import evaluate_agents
from visualizer import (
    plot_rl_training_progress,
    plot_learning_curves
)
from gpu_utils import setup_gpu, get_device_info, get_device

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("="*80)
print("MULTI-AGENT RL CYBERSECURITY - PURE RL (No ML, No Data Leakage)")
print("="*80)
print(f"[+] PyTorch Version: {torch.__version__}")
print(f"[+] NumPy: {np.__version__}")
print(f"[+] Mode: RL ONLY - Pure Reinforcement Learning Agents")

# GPU Configuration
gpu_available = setup_gpu()
get_device_info()
device = get_device()


def main():
    """Main training pipeline."""
    
    # ==================== LOAD DATA ====================
    print("\n[DATASET] Loading NSL-KDD from GitHub...\n")
    df_full = load_nsl_kdd()
    
    X_full_scaled, y_category, y_attack, df_encoded, scaler = preprocess_nsl_kdd(df_full)
    agent_datasets = create_agent_datasets(X_full_scaled, y_category, y_attack, df_encoded)
    
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, y_train_adv, y_test_adv = load_mnist(
        train_size=MNIST_TRAIN_SIZE,
        test_size=MNIST_TEST_SIZE
    )
    
    # ==================== CREATE SPLITS ====================
    splits = create_train_val_test_splits(
        X_train_mnist, y_train_adv, X_test_mnist, y_test_adv, agent_datasets
    )
    
    # ==================== CREATE AGENTS ====================
    agent1 = Agent1_Adversarial(device=device)
    agent2 = BinaryAgent(2, "Agent 2: DoS Detection", splits[2][0].shape[1], device=device)
    agent3 = Agent3_MultiClass(splits[3][6], splits[3][0].shape[1], device=device)  # n_classes from splits
    agent4 = BinaryAgent(4, "Agent 4: Probe Detection", splits[4][0].shape[1], device=device)
    agent5 = BinaryAgent(5, "Agent 5: R2L/U2R Detection", splits[5][0].shape[1], device=device)

    all_agents = [agent1, agent2, agent3, agent4, agent5]

    print("\n✓ All 5 agents created")
    for agent in all_agents:
        print(f"  - {agent.name}")
    
    # ==================== RL TRAINING ONLY ====================
    print("\n" + "="*80)
    print("REINFORCEMENT LEARNING TRAINING (VALIDATION DATA)")
    print("="*80)
    print("Training RL agents on VALIDATION data (no leakage to test)")
    print("="*80)
    
    rl_curves = train_rl_agents(all_agents, splits, n_episodes=N_EPISODES, 
                               max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                               update_target_every=UPDATE_TARGET_EVERY)
    
    # ==================== EVALUATION ON TEST DATA ====================
    rl_results = evaluate_agents(all_agents, splits)
    
    # ==================== VISUALIZATION ====================
    plot_rl_training_progress(all_agents, N_EPISODES, MAX_STEPS_PER_EPISODE)
    plot_learning_curves(rl_curves, N_EPISODES)
    
    return all_agents, rl_results, rl_curves


if __name__ == "__main__":
    all_agents, rl_results, rl_curves = main()
