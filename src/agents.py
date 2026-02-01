"""
DQN Agent and specialized agent implementations with PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    """DQN Neural Network for Q-value estimation with multi-class support."""
    
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        # Scale network size based on action space (more classes = bigger network)
        hidden1 = max(128, min(256, action_size * 10))  # Adaptive hidden layer size
        hidden2 = max(64, min(128, action_size * 5))
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.2 to 0.4
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.2 to 0.4
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout before output
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Complete DQN Agent with Experience Replay using PyTorch."""

    def __init__(self, state_size, action_size, name="Agent", device='cpu',
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Adaptive learning rate: higher for multi-class problems
        if action_size > 10:  # Multi-class (e.g., 23 classes)
            self.learning_rate = learning_rate * 1.2  # 20% higher (reduced from 50%)
        else:
            self.learning_rate = learning_rate
            
        self.batch_size = batch_size

        # Experience Replay
        self.memory = deque(maxlen=10000)

        # Build networks
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.001)  # L2 regularization
        self.loss_fn = nn.MSELoss()

        # Statistics
        self.losses = []
        self.episode_rewards = []
        self.updates = 0  # Track number of updates for learning rate decay

    def update_target_model(self):
        """Update target network weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, explore=True):
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)[0]
        return int(torch.argmax(q_values).item())

    def replay(self):
        """Train the agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.LongTensor(np.array([exp[1] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp[4] for exp in batch])).to(self.device)

        # Current Q-values
        current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and backprop
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.updates += 1

        # Learning rate decay every 5000 updates
        if self.updates % 5000 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.95
                print(f"[LR Decay] New LR: {param_group['lr']:.6f}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


class CNNNetwork(nn.Module):
    """CNN for image classification (Agent 1)."""
    
    def __init__(self, num_classes):
        super(CNNNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNetwork(nn.Module):
    """Dense network for tabular data."""
    
    def __init__(self, input_size, num_classes):
        super(DenseNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class Agent1_Adversarial:
    """Agent 1: Adversarial MNIST Detection (3-way)"""

    def __init__(self, device='cpu'):
        self.name = "Agent 1: Adversarial (3-way)"
        self.device = device
        self.detector = CNNNetwork(num_classes=3).to(device)
        self.optimizer = optim.Adam(self.detector.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.rl_agent = DQNAgent(state_size=12, action_size=3, name=self.name, device=device)

    def train_detector(self, X_train, y_train, X_val, y_val, epochs=5):
        print(f"\n{self.name} - Training detector...")
        self.detector.train()
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.detector(X_train)
            loss = self.loss_fn(outputs, y_train)
            loss.backward()
            self.optimizer.step()
        
        # Validation
        self.detector.eval()
        with torch.no_grad():
            val_outputs = self.detector(X_val)
            val_loss = self.loss_fn(val_outputs, y_val)
            val_acc = (torch.argmax(val_outputs, dim=1) == y_val).float().mean().item()
        
        print(f"  Validation Accuracy: {val_acc:.4f}")
        self.detector.train()


class BinaryAgent:
    """Binary classification agent for NSL-KDD subsets"""

    def __init__(self, agent_id, name, input_dim, device='cpu'):
        self.name = name
        self.device = device
        self.detector = DenseNetwork(input_dim, num_classes=2).to(device)
        self.optimizer = optim.Adam(self.detector.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.rl_agent = DQNAgent(state_size=12, action_size=2, name=name, device=device)

    def train_detector(self, X_train, y_train, X_val, y_val, epochs=15):
        print(f"\n{self.name} - Training detector...")
        self.detector.train()
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.detector(X_train)
            loss = self.loss_fn(outputs, y_train)
            loss.backward()
            self.optimizer.step()
        
        # Validation
        self.detector.eval()
        with torch.no_grad():
            val_outputs = self.detector(X_val)
            val_loss = self.loss_fn(val_outputs, y_val)
            val_acc = (torch.argmax(val_outputs, dim=1) == y_val).float().mean().item()
        
        print(f"  Validation Accuracy: {val_acc:.4f}")
        self.detector.train()


class Agent3_MultiClass:
    """Multi-class agent for full NSL-KDD intrusion detection"""

    def __init__(self, n_classes, input_dim, device='cpu'):
        self.name = f"Agent 3: Intrusion ({n_classes} classes)"
        self.n_classes = n_classes
        self.device = device
        self.detector = DenseNetwork(input_dim, num_classes=n_classes).to(device)
        self.optimizer = optim.Adam(self.detector.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.rl_agent = DQNAgent(state_size=12, action_size=n_classes, name=self.name, device=device)

    def train_detector(self, X_train, y_train, X_val, y_val, epochs=20):
        print(f"\n{self.name} - Training detector...")
        self.detector.train()
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.detector(X_train)
            loss = self.loss_fn(outputs, y_train)
            loss.backward()
            self.optimizer.step()
        
        # Validation
        self.detector.eval()
        with torch.no_grad():
            val_outputs = self.detector(X_val)
            val_loss = self.loss_fn(val_outputs, y_val)
            val_acc = (torch.argmax(val_outputs, dim=1) == y_val).float().mean().item()
        
        print(f"  Validation Accuracy: {val_acc:.4f}")
        self.detector.train()
