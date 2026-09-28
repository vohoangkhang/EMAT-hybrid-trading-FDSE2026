"""
Phase 4 - Hierarchical DQN-PPO Agent Architecture
===================================================
DQN Head (Strategic): Chọn direction Buy(0)/Hold(1)/Sell(2) — Discrete
PPO Actor (Tactical): Chọn allocation ∈ (0,1) qua Beta distribution — Continuous
PPO Critic: V(s) → scalar (shared)
Shared Encoder: φ(s) → feature vector

Luồng quyết định:
  1. DQN chọn direction (ε-greedy hoặc argmax Q)
  2. Nếu Hold → không giao dịch
  3. Nếu Buy/Sell → PPO chọn allocation ~ Beta(α,β)
  4. Env nhận (direction, fraction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


class SharedEncoder(nn.Module):
    """Shared feature extractor φ(s) cho cả DQN và PPO."""
    def __init__(self, state_dim, hidden1=config.ENCODER_HIDDEN_1,
                 hidden2=config.ENCODER_HIDDEN_2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.LayerNorm(hidden2),
        )

    def forward(self, x):
        return self.net(x)


class DQNHead(nn.Module):
    """DQN Strategic: Output Q-values cho 3 directions (Buy/Hold/Sell)."""
    def __init__(self, input_dim=config.ENCODER_HIDDEN_2,
                 hidden=config.DQN_HEAD_HIDDEN,
                 n_directions=config.DQN_NUM_DIRECTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_directions),
        )

    def forward(self, x):
        return self.net(x)  # (batch, 3) — Q-values cho Buy/Hold/Sell


class PPOContinuousActor(nn.Module):
    """
    PPO Tactical: Output α, β cho Beta distribution → allocation ∈ (0, 1).

    Beta(α, β) là distribution bounded tự nhiên trong (0, 1),
    phù hợp cho tỷ lệ phân bổ vốn.
    - α > 1, β > 1 → bell-shaped (tập trung quanh mean)
    - α = β = 1 → uniform
    - α > β → skew phải (ưu tiên allocation cao)
    """
    def __init__(self, input_dim=config.ENCODER_HIDDEN_2,
                 hidden=config.PPO_HEAD_HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(hidden, 1)
        self.beta_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        # Softplus + 1.0 đảm bảo α, β > 1 → distribution bell-shaped, ổn định
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        return alpha.squeeze(-1), beta.squeeze(-1)  # (batch,)


class PPOCriticHead(nn.Module):
    """PPO Critic: Output state value V(s)."""
    def __init__(self, input_dim=config.ENCODER_HIDDEN_2,
                 hidden=config.PPO_HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


class HybridAgent(nn.Module):
    """
    Hierarchical DQN-PPO Agent.

    DQN Head → direction (Buy=0, Hold=1, Sell=2) — Discrete, ε-greedy
    PPO Actor → allocation ∈ (0, 1) qua Beta(α, β) — Continuous
    PPO Critic → V(s) — Baseline cho advantage estimation
    """
    def __init__(self, state_dim):
        super().__init__()
        self.encoder = SharedEncoder(state_dim)
        self.dqn_head = DQNHead()
        self.ppo_actor = PPOContinuousActor()
        self.ppo_critic = PPOCriticHead()

    def forward(self, state):
        """Forward pass đầy đủ cho debugging/visualization."""
        encoded = self.encoder(state)
        q_values = self.dqn_head(encoded)
        alpha, beta = self.ppo_actor(encoded)
        value = self.ppo_critic(encoded)
        return q_values, alpha, beta, value

    def get_q_values(self, state):
        """DQN: Tính Q-values cho 3 directions."""
        encoded = self.encoder(state)
        return self.dqn_head(encoded)

    def get_allocation_dist(self, state):
        """PPO: Tính Beta distribution parameters."""
        encoded = self.encoder(state)
        alpha, beta = self.ppo_actor(encoded)
        return alpha, beta

    def get_value(self, state):
        """PPO Critic: Tính V(s)."""
        encoded = self.encoder(state)
        return self.ppo_critic(encoded)

    # =========================================================
    # ACTION SELECTION METHODS
    # =========================================================
    def select_direction_dqn(self, state, epsilon):
        """DQN ε-greedy: chọn Buy(0) / Hold(1) / Sell(2)."""
        if random.random() < epsilon:
            return random.randint(0, config.DQN_NUM_DIRECTIONS - 1)
        with torch.no_grad():
            q = self.get_q_values(state)
            return q.argmax(dim=-1).item()

    def select_allocation_ppo(self, state):
        """
        PPO: sample allocation ~ Beta(α, β).

        Returns:
            fraction: float ∈ (0, 1)
            log_prob: float — log probability cho PPO loss
            value: float — V(s) cho advantage estimation
        """
        with torch.no_grad():
            encoded = self.encoder(state)
            alpha, beta = self.ppo_actor(encoded)
            dist = torch.distributions.Beta(alpha, beta)
            fraction = dist.sample()
            log_prob = dist.log_prob(fraction)
            value = self.ppo_critic(encoded)
        return fraction.item(), log_prob.item(), value.item()

    def select_hierarchical_action(self, state, epsilon=0.0):
        """
        Full hierarchical inference: DQN chọn direction, PPO chọn allocation.

        Args:
            state: tensor (1, state_dim)
            epsilon: float — exploration rate cho DQN

        Returns:
            direction: int — 0=Buy, 1=Hold, 2=Sell
            fraction: float — tỷ lệ vốn ∈ (0, 1), 0.0 nếu Hold
        """
        direction = self.select_direction_dqn(state, epsilon)
        if direction == config.DIRECTION_HOLD:
            return direction, 0.0
        fraction, _, _ = self.select_allocation_ppo(state)
        # Clamp vào khoảng hợp lệ
        fraction = max(config.PPO_ALLOCATION_MIN,
                       min(config.PPO_ALLOCATION_MAX, fraction))
        return direction, fraction

    def evaluate_allocation(self, states, fractions):
        """
        Đánh giá lại allocation cho PPO update (cần gradient).

        Args:
            states: tensor (batch, state_dim)
            fractions: tensor (batch,) — allocation đã chọn trước đó

        Returns:
            log_probs: tensor (batch,) — log π(fraction|state)
            entropy: tensor scalar — entropy trung bình
            values: tensor (batch,) — V(s)
        """
        encoded = self.encoder(states)
        alpha, beta = self.ppo_actor(encoded)
        dist = torch.distributions.Beta(alpha, beta)
        log_probs = dist.log_prob(fractions)
        entropy = dist.entropy().mean()
        values = self.ppo_critic(encoded)
        return log_probs, entropy, values


class ReplayBuffer:
    """Experience replay buffer cho DQN (off-policy)."""
    def __init__(self, capacity=config.DQN_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PPOContinuousRolloutBuffer:
    """
    On-policy rollout buffer cho PPO với continuous allocation.

    Lưu trữ: state, fraction (continuous), log_prob, reward, value, done.
    Hỗ trợ GAE (Generalized Advantage Estimation).
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.fractions = []    # Continuous allocation ∈ (0, 1)
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, fraction, log_prob, reward, value, done):
        self.states.append(state)
        self.fractions.append(fraction)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=config.PPO_GAMMA,
                    lam=config.PPO_GAE_LAMBDA):
        """Tính Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0
        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t]
                     + gamma * values[t + 1] * (1 - self.dones[t])
                     - values[t])
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_batches(self, advantages, returns,
                    batch_size=config.PPO_BATCH_SIZE):
        """Chia data thành mini-batches cho PPO update."""
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)

        states = np.array(self.states)
        fractions = np.array(self.fractions, dtype=np.float32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                torch.tensor(states[idx], dtype=torch.float32),
                torch.tensor(fractions[idx], dtype=torch.float32),
                torch.tensor(old_log_probs[idx], dtype=torch.float32),
                torch.tensor(advantages[idx], dtype=torch.float32),
                torch.tensor(returns[idx], dtype=torch.float32),
            )


if __name__ == "__main__":
    # Test kiến trúc
    print("Testing HybridAgent (Hierarchical DQN-PPO)...")
    state_dim = config.STATE_DIM_TOTAL
    agent = HybridAgent(state_dim)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder: {sum(p.numel() for p in agent.encoder.parameters()):,}")
    print(f"  DQN Head: {sum(p.numel() for p in agent.dqn_head.parameters()):,}")
    print(f"  PPO Actor: {sum(p.numel() for p in agent.ppo_actor.parameters()):,}")
    print(f"  PPO Critic: {sum(p.numel() for p in agent.ppo_critic.parameters()):,}")

    # Test inference
    state = torch.randn(1, state_dim)
    direction, fraction = agent.select_hierarchical_action(state, epsilon=0.0)
    dir_names = {0: "Buy", 1: "Hold", 2: "Sell"}
    print(f"\n  Test action: {dir_names[direction]}, Allocation: {fraction*100:.1f}%")

    # Test batch forward
    batch = torch.randn(4, state_dim)
    q_vals, alpha, beta, values = agent(batch)
    print(f"  Q-values shape: {q_vals.shape}")  # (4, 3)
    print(f"  Alpha shape: {alpha.shape}")       # (4,)
    print(f"  Beta shape: {beta.shape}")         # (4,)
    print(f"  Values shape: {values.shape}")     # (4,)
    print("  ✔ All tests passed!")
