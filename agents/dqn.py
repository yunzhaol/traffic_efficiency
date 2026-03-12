"""
agents/dqn.py — Deep Q-Network agent for traffic signal control.

Architecture: 3-layer MLP with experience replay, target network, and ε-greedy exploration.
Reference: Mnih et al. (2015); Zhu et al. (2025) Section 3.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, Tuple


class QNetwork(nn.Module):
    """3-layer MLP Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, ...] = (128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with:
    - Experience replay (capacity 50,000)
    - Target network with hard update every `target_update_freq` steps
    - ε-greedy exploration annealed from eps_start → eps_end over eps_decay steps
    - Huber loss for robustness to outlier rewards
    - Action masking: cannot re-select the currently active phase
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = cfg.get("gamma", 0.95)
        self.lr = cfg.get("lr", 1e-3)
        self.batch_size = cfg.get("batch_size", 64)
        self.buffer_capacity = cfg.get("buffer_capacity", 50_000)
        self.target_update_freq = cfg.get("target_update_freq", 500)
        self.eps_start = cfg.get("eps_start", 1.0)
        self.eps_end = cfg.get("eps_end", 0.05)
        self.eps_decay_steps = cfg.get("eps_decay_steps", 10_000)
        self.min_buffer_size = cfg.get("min_buffer_size", 1_000)
        self.hidden = tuple(cfg.get("hidden", [128, 64]))

        # Networks
        self.q_net = QNetwork(obs_dim, n_actions, self.hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, self.hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.steps = 0
        self.epsilon = self.eps_start

        # Track current phase for action masking
        self.current_phase: Optional[int] = None

    @property
    def _epsilon(self) -> float:
        """Linearly annealed epsilon."""
        frac = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray, valid_actions: Optional[list] = None) -> int:
        """ε-greedy with optional action masking."""
        self.epsilon = self._epsilon

        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
            # Action masking: avoid staying in current phase unnecessarily
            if self.current_phase is not None and len(valid_actions) > 1:
                valid_actions = [a for a in valid_actions if a != self.current_phase]

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_net(obs_t).squeeze(0)

            # Mask invalid actions: set all to -inf, valid to 0
            import numpy as _np
            mask_arr = _np.full(self.n_actions, -1e9, dtype=_np.float32)
            for a in valid_actions:
                mask_arr[a] = 0.0
            mask = torch.FloatTensor(mask_arr).to(self.device)
            q_values = q_values + mask

            action = int(q_values.argmax().item())

        self.current_phase = action
        return action

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        self.steps += 1

    def update(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient update."""
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Hard target update
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }, path)
        print(f"[DQN] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        print(f"[DQN] Loaded checkpoint ← {path}")
