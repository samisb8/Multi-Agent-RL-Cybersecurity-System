"""
Configuration settings for Multi-Agent RL Cybersecurity System
"""

# GPU Configuration
USE_GPU = True
MIXED_PRECISION = True  # Faster training with float16
MEMORY_GROWTH = True    # Prevent OOM errors

# RL Training Configuration
N_EPISODES = 100  # Full training with 100 episodes
MAX_STEPS_PER_EPISODE = 50
UPDATE_TARGET_EVERY = 20  # Increased from 10 to stabilize training

# Training Mode
TRAINING_MODE = 'ML_ONLY'  # 'ML_ONLY' or 'RL' or 'HYBRID'

# DQN Hyperparameters
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000

# Dataset Configuration
MNIST_TRAIN_SIZE = 10000
MNIST_TEST_SIZE = 2000

# Agent Configuration
AGENTS_CONFIG = {
    1: {
        'name': 'Agent 1: Adversarial (3-way)',
        'type': 'image',
        'action_size': 3,
        'detector_epochs': 5
    },
    2: {
        'name': 'Agent 2: DoS Detection',
        'type': 'tabular',
        'action_size': 2,
        'detector_epochs': 15
    },
    3: {
        'name': 'Agent 3: Intrusion Detection',
        'type': 'tabular',
        'action_size': 23,  # Will be set dynamically
        'detector_epochs': 20
    },
    4: {
        'name': 'Agent 4: Probe Detection',
        'type': 'tabular',
        'action_size': 2,
        'detector_epochs': 15
    },
    5: {
        'name': 'Agent 5: R2L/U2R Detection',
        'type': 'tabular',
        'action_size': 2,
        'detector_epochs': 15
    }
}

# Paths
DATA_DIR = './data'
MODELS_DIR = './models'
RESULTS_DIR = './results'

# Random seeds
RANDOM_SEED = 42
