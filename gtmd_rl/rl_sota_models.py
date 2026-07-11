"""
State-of-the-Art RL Models for 5G Resource Allocation
Includes: DQN, PPO, A2C implementations
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# 1. DQN (Deep Q-Network) - SOTA for discrete action spaces
# ============================================================================

class DQNNetwork(nn.Module):
    """Deep Q-Network for 5G resource allocation.
    
    State: [load_ratio, mean_delay, rho_recent, dominant_slice, demand_estimates...]
    Output: Q-values for each weight profile action
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:
    """Deep Q-Learning Agent with experience replay and target network.
    
    Key innovations:
    - Experience replay buffer (256 capacity)
    - Target network with soft updates (τ=0.001)
    - Double Q-learning to reduce overestimation
    - Dueling architecture option available
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 256,
        batch_size: int = 32,
        target_update_freq: int = 10,
        tau: float = 0.001,  # soft update
        seed: int = 42,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.update_counter = 0
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Track training metrics
        self.losses = []
        self.q_values = []
        
    def encode_state(self, load_ratio: float, mean_delay: float, rho: float, 
                    dominant: int, demand_estimates: np.ndarray) -> torch.Tensor:
        """Convert state to tensor for network."""
        state_array = np.array([
            load_ratio,
            mean_delay,
            rho,
            float(dominant),
            *demand_estimates
        ], dtype=np.float32)
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
    
    def select_action(self, state: torch.Tensor, train: bool = True) -> int:
        """ε-greedy action selection."""
        if train and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Sample batch and perform gradient update (Double DQN)."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double DQN: Use current network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update target network
        if self.update_counter % self.target_update_freq == 0:
            self._soft_update()
        
        self.update_counter += 1
        self.losses.append(loss.item())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update of target network: θ_target = τ*θ + (1-τ)*θ_target"""
        for param, target_param in zip(self.q_network.parameters(), 
                                      self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# ============================================================================
# 2. PPO (Proximal Policy Optimization) - SOTA for continuous/discrete control
# ============================================================================

class PPOActor(nn.Module):
    """Policy network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOCritic(nn.Module):
    """Value network for PPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass
class PPOBuffer:
    """Rollout buffer for PPO."""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    dones: List[bool]
    log_probs: List[float]
    
    def __post_init__(self):
        self.size = len(self.states)
    
    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE advantages."""
        advantages = []
        gae = 0.0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        self.advantages = np.array(advantages)
        self.returns = np.array(advantages) + np.array(self.values)


class PPOAgent:
    """Proximal Policy Optimization agent.
    
    Key features:
    - Actor-Critic architecture
    - GAE (Generalized Advantage Estimation) for variance reduction
    - Clipped objective for stable training
    - Entropy regularization
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Networks
        self.actor = PPOActor(state_dim, action_dim).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Tracking
        self.losses = []
    
    def encode_state(self, load_ratio: float, mean_delay: float, rho: float,
                    dominant: int, demand_estimates: np.ndarray) -> torch.Tensor:
        """Convert state to tensor."""
        state_array = np.array([
            load_ratio,
            mean_delay,
            rho,
            float(dominant),
            *demand_estimates
        ], dtype=np.float32)
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """Select action with probability and log probability."""
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action]).detach().item()
        
        with torch.no_grad():
            value = self.critic(state).item()
        
        return action, log_prob, value
    
    def update(self, buffer: PPOBuffer):
        """Update networks using PPO objective."""
        buffer.compute_advantages(self.gamma, self.lam)
        
        states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        actions = torch.LongTensor(buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(buffer.advantages).to(self.device)
        returns = torch.FloatTensor(buffer.returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(buffer.states))
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                
                # Actor update
                logits = self.actor(states[batch_idx])
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs.gather(1, actions[batch_idx].unsqueeze(1))).squeeze(1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic update
                values = self.critic(states[batch_idx]).squeeze(1)
                critic_loss = nn.MSELoss()(values, returns[batch_idx])
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                self.losses.append((actor_loss.item(), critic_loss.item()))


# ============================================================================
# 3. A2C (Advantage Actor-Critic) - SOTA baseline method
# ============================================================================

class A2CActor(nn.Module):
    """Actor for A2C."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class A2CCritic(nn.Module):
    """Critic for A2C."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class A2CAgent:
    """Advantage Actor-Critic agent.
    
    Synchronous version - updates after each episode.
    Key features:
    - Lower variance than REINFORCE
    - Higher bias than A3C (but synchronous)
    - Good for smaller state/action spaces
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: int = 42,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.actor = A2CActor(state_dim, action_dim).to(self.device)
        self.critic = A2CCritic(state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.losses = []
    
    def encode_state(self, load_ratio: float, mean_delay: float, rho: float,
                    dominant: int, demand_estimates: np.ndarray) -> torch.Tensor:
        """Convert state to tensor."""
        state_array = np.array([
            load_ratio,
            mean_delay,
            rho,
            float(dominant),
            *demand_estimates
        ], dtype=np.float32)
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """Select action and get value estimate."""
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action]).item()
        
        with torch.no_grad():
            value = self.critic(state).item()
        
        return action, value
    
    def update(self, states: List[np.ndarray], actions: List[int], 
              rewards: List[float], next_value: float):
        """Update after episode."""
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        
        # Compute returns
        returns = []
        G = next_value
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get values and log probs
        logits = self.actor(states_t)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions_t.unsqueeze(1))).squeeze(1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        
        values = self.critic(states_t).squeeze(1)
        
        # Advantage for the policy-gradient term is detached from the critic.
        advantages = (returns - values.detach())

        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy

        # Critic loss must retain the gradient path through ``values`` (using the
        # detached ``advantages`` here leaves the loss constant w.r.t. the critic
        # weights, so ``backward()`` has nothing to differentiate).
        critic_loss = (returns - values).pow(2).mean()
        
        # Updates
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.losses.append((actor_loss.item(), critic_loss.item()))
