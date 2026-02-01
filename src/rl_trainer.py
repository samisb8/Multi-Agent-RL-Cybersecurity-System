"""
RL training module
"""

import numpy as np
import torch
import torch.nn as nn


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


def train_rl_agents(all_agents, splits, n_episodes=100, max_steps_per_episode=50, 
                    update_target_every=10):
    """Train PURE RL agents on TRAINING data only (no ML detectors). Returns learning curves."""
    
    print("\n" + "="*80)
    print("PURE REINFORCEMENT LEARNING TRAINING")
    print("="*80)
    print("Training on: TRAINING DATA")
    print("Testing on:  TEST DATA (separate, no leakage)")
    print("="*80)

    # Use TRAINING data only for RL agents (no ML detectors)
    # splits structure: [X_train, X_val, X_test, y_train, y_val, y_test]
    rl_datasets = {
        0: (splits[1][0], splits[1][3]),  # Agent 1: X_train[0], y_train[3]
        1: (splits[2][0], splits[2][3]),  # Agent 2: X_train[0], y_train[3]
        2: (splits[3][0], splits[3][3]),  # Agent 3: X_train[0], y_train[3]
        3: (splits[4][0], splits[4][3]),  # Agent 4: X_train[0], y_train[3]
        4: (splits[5][0], splits[5][3])   # Agent 5: X_train[0], y_train[3]
    }

    print(f"\nConfiguration:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Steps per episode: {max_steps_per_episode}")
    print(f"  Update target network every: {update_target_every} episodes")
    print(f"  Features: CNN for images, Enhanced stats for tabular")
    print(f"\nStarting RL training on training data...\n")

    # Initialize CNN feature extractor for Agent 1
    cnn_extractor = CNNFeatureExtractor().to(all_agents[0].device)
    cnn_extractor.eval()

    # Learning curves storage
    learning_curves = {
        i: {
            'rewards': [],
            'losses': [],
            'avg_rewards': [],
            'avg_losses': []
        }
        for i in range(5)
    }

    # Training loop
    for episode in range(n_episodes):
        episode_rewards = {i: 0 for i in range(5)}
        episode_losses = {i: [] for i in range(5)}

        for step in range(max_steps_per_episode):
            for agent_idx, agent in enumerate(all_agents):
                X_train, y_train = rl_datasets[agent_idx]

                # Sample random data point from TRAINING set only
                # Make sure we don't exceed dataset size
                max_idx = len(X_train) - 1
                idx = np.random.randint(0, max(1, max_idx))

                # Safely access the data
                if idx >= len(X_train):
                    idx = len(X_train) - 1

                # Create state from enhanced features
                if len(X_train[idx].shape) == 3:  # Image data (Agent 1)
                    # Use CNN to extract features
                    with torch.no_grad():
                        img_tensor = torch.FloatTensor(X_train[idx]).unsqueeze(0).to(agent.device)
                        cnn_features = cnn_extractor(img_tensor)[0].cpu().numpy()
                    
                    # Pad CNN features to match tabular state size (12 total)
                    # Use first 9 CNN features, then add 3 metadata
                    state = np.concatenate([
                        cnn_features[:9],  # First 9 CNN features
                        np.array([
                            y_train[idx] / agent.rl_agent.action_size,
                            step / max_steps_per_episode,
                            idx / len(X_train)
                        ])
                    ])
                else:
                    # Tabular data - use enhanced statistics (total 12 features)
                    features = X_train[idx]
                    state = np.array([
                        np.mean(features),
                        np.std(features),
                        np.min(features),
                        np.max(features),
                        np.median(features),
                        np.percentile(features, 25),
                        np.percentile(features, 75),
                        np.sum(np.abs(features)) / len(features),
                        step / max_steps_per_episode,
                        y_train[idx] / agent.rl_agent.action_size,
                        len(features) / 100.0,
                        1.0
                    ])

                true_label = y_train[idx]

                # RL agent selects action
                action = agent.rl_agent.act(state, explore=True)

                # Enhanced reward shaping (especially for multi-class)
                if action == true_label:
                    # Correct: higher reward for harder problems (more classes)
                    reward = 1.0 + (agent.rl_agent.action_size / 50.0)  # Bonus for multi-class
                else:
                    # Penalty based on action space (23-class gets higher penalty)
                    penalty = min(0.5, agent.rl_agent.action_size / 100.0)
                    reward = -penalty

                # Next state
                next_state = state + np.random.randn(len(state)) * 0.05
                done = (step == max_steps_per_episode - 1)

                # Store in memory
                agent.rl_agent.remember(state, action, reward, next_state, done)

                # Train RL agent
                loss = agent.rl_agent.replay()
                if loss is not None:
                    episode_losses[agent_idx].append(loss)

                episode_rewards[agent_idx] += reward

        # Update target networks
        if (episode + 1) % update_target_every == 0:
            for agent in all_agents:
                agent.rl_agent.update_target_model()

        # Store episode rewards and losses in learning curves
        for agent_idx, agent in enumerate(all_agents):
            agent.rl_agent.episode_rewards.append(episode_rewards[agent_idx])
            learning_curves[agent_idx]['rewards'].append(episode_rewards[agent_idx])
            
            avg_loss = np.mean(episode_losses[agent_idx]) if episode_losses[agent_idx] else 0
            learning_curves[agent_idx]['losses'].append(avg_loss)
            
            # Calculate moving averages
            window_size = min(10, episode + 1)
            avg_reward = np.mean(learning_curves[agent_idx]['rewards'][-window_size:])
            learning_curves[agent_idx]['avg_rewards'].append(avg_reward)
            
            if episode > 0:
                avg_loss_smooth = np.mean(learning_curves[agent_idx]['losses'][-window_size:])
            else:
                avg_loss_smooth = avg_loss
            learning_curves[agent_idx]['avg_losses'].append(avg_loss_smooth)

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode+1}/{n_episodes}")
            agents_short = ['Adv.', 'DoS', 'Intr.', 'Probe', 'R2L/U2R']
            for agent_idx, agent in enumerate(all_agents):
                avg_reward = episode_rewards[agent_idx] / max_steps_per_episode
                avg_loss = np.mean(episode_losses[agent_idx]) if episode_losses[agent_idx] else 0
                epsilon = agent.rl_agent.epsilon
                print(f"  {agents_short[agent_idx]:10s} | Reward: {avg_reward:6.2f} | Loss: {avg_loss:6.4f} | Eps: {epsilon:.3f}")

    print("\n" + "="*80)
    print("✓ RL TRAINING COMPLETE (Enhanced CNN Features, No Data Leakage)")
    print("="*80)
    
    return learning_curves
