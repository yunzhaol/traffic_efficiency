"""
agents/ppo.py — Proximal Policy Optimization with GAE.

Reference: Schulman et al. (2017) arXiv:1707.06347
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, List, Tuple


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic: trunk → actor head + critic head."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, ...] = (128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.actor  = nn.Linear(in_dim, n_actions)
        self.critic = nn.Linear(in_dim, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class RolloutBuffer:
    """On-policy rollout buffer for PPO."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE advantage estimation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        values_ext = np.array(self.values + [last_value], dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            delta = self.rewards[t] + gamma * values_ext[t + 1] * (1 - self.dones[t]) - values_ext[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


class PPOAgent:
    """
    PPO agent for single-intersection traffic control.
    Collects on-policy rollouts of length `rollout_steps`, then performs
    `ppo_epochs` passes of mini-batch updates.
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.obs_dim    = obs_dim
        self.n_actions  = n_actions
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma          = cfg.get("gamma",          0.95)
        self.gae_lambda     = cfg.get("gae_lambda",     0.95)
        self.clip_eps       = cfg.get("clip_eps",       0.2)
        self.ppo_epochs     = cfg.get("ppo_epochs",     10)
        self.mini_batch_size= cfg.get("mini_batch_size",64)
        self.rollout_steps  = cfg.get("rollout_steps",  2048)
        self.vf_coef        = cfg.get("vf_coef",        0.5)
        self.ent_coef       = cfg.get("ent_coef",       0.01)
        self.lr             = cfg.get("lr",             3e-4)
        self.max_grad_norm  = cfg.get("max_grad_norm",  0.5)
        self.hidden         = tuple(cfg.get("hidden",   [128, 64]))

        self.ac        = ActorCritic(obs_dim, n_actions, self.hidden).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)

        self.buffer = RolloutBuffer()
        self.steps  = 0

        # Store last obs/action/log_prob/value between select_action and store_transition
        self._last_obs:      Optional[np.ndarray] = None
        self._last_action:   Optional[int]        = None
        self._last_log_prob: Optional[float]      = None
        self._last_value:    Optional[float]      = None
        self._last_next_obs: Optional[np.ndarray] = None  # for bootstrapping

    def select_action(self, obs: np.ndarray) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.ac.get_action_and_value(obs_t)
        self._last_obs      = obs
        self._last_action   = action.item()
        self._last_log_prob = log_prob.item()
        self._last_value    = value.item()
        return self._last_action

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store the transition produced after the last select_action call.
        `obs`, `action` args are accepted for API compatibility but the
        internally cached values (from select_action) are authoritative.
        `next_obs` is kept for bootstrapping the final value estimate.
        """
        self.buffer.add(
            self._last_obs,
            self._last_action,
            self._last_log_prob,
            reward,
            self._last_value,
            done,
        )
        self._last_next_obs = np.array(next_obs, dtype=np.float32)
        self.steps += 1

    def update(self) -> Optional[float]:
        """Perform PPO update once rollout_steps transitions are collected."""
        if len(self.buffer) < self.rollout_steps:
            return None

        # Bootstrap last value from the actual last next_obs
        if self._last_next_obs is not None:
            last_obs_t = torch.FloatTensor(self._last_next_obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, last_value = self.ac(last_obs_t)
            last_value = last_value.item()
        else:
            last_value = 0.0

        advantages, returns = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t       = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
        actions_t   = torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp_t    = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns_t   = torch.FloatTensor(returns).to(self.device)
        adv_t       = torch.FloatTensor(advantages).to(self.device)

        total_loss, n_updates = 0.0, 0
        n = len(self.buffer)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start : start + self.mini_batch_size]

                _, log_probs, entropy, values = self.ac.get_action_and_value(
                    obs_t[idx], actions_t[idx]
                )

                ratio    = (log_probs - old_lp_t[idx]).exp()
                adv_mb   = adv_t[idx]
                pg_loss  = -torch.min(
                    ratio * adv_mb,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                ).mean()
                vf_loss  = 0.5 * (values - returns_t[idx]).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        self.buffer.reset()
        return total_loss / max(n_updates, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"ac": self.ac.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "steps": self.steps}, path)
        print(f"[PPO] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        print(f"[PPO] Loaded ← {path}")
