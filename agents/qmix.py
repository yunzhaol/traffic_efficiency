"""
agents/qmix.py — QMIX multi-agent RL agent.

QMIX learns a monotonic joint Q-value from individual agent Q-values,
enabling Centralised Training with Decentralised Execution (CTDE).

Reference: Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation
           for Deep Multi-Agent Reinforcement Learning. ICML 2018.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Optional, Tuple


# ─── Individual Agent Network ─────────────────────────────────────────────────

class AgentQNetwork(nn.Module):
    """Per-agent Q-network (weights shared across agents)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, ...] = (128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─── Mixing Network ───────────────────────────────────────────────────────────

class MixingNetwork(nn.Module):
    """
    QMIX mixing network.
    Combines individual Q_i values into Q_tot via a state-conditioned
    hypernetwork with non-negative weights (guarantees monotonicity / IGM).
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents  = n_agents
        self.embed_dim = embed_dim

        # Hypernetworks → weights for layer 1 and layer 2
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs : (B, N)       — per-agent Q at chosen action
            state    : (B, state_dim)
        Returns:
            q_tot    : (B, 1)
        """
        B = agent_qs.size(0)
        qs = agent_qs.view(B, 1, self.n_agents)

        # Layer 1  (non-negative weights → monotonic)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        h  = torch.relu(torch.bmm(qs, w1) + b1)             # (B, 1, embed)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(h, w2) + b2                        # (B, 1, 1)

        return q_tot.view(B, 1)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class MAReplayBuffer:
    """
    Multi-agent replay buffer.
    Stores joint transitions: (obs_list, actions, rewards, next_obs_list,
                                done, global_state, next_global_state)
    All list/array fields are converted to numpy at push time to avoid
    silent type errors during sampling.
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self,
             obs_list:        List[np.ndarray],
             actions:         List[int],
             rewards:         List[float],
             next_obs_list:   List[np.ndarray],
             done:            bool,
             global_state:    np.ndarray,
             next_global_state: np.ndarray):
        self.buffer.append((
            np.array(obs_list,       dtype=np.float32),   # (N, obs_dim)
            np.array(actions,        dtype=np.int64),      # (N,)
            np.array(rewards,        dtype=np.float32),    # (N,)
            np.array(next_obs_list,  dtype=np.float32),   # (N, obs_dim)
            float(done),
            np.array(global_state,   dtype=np.float32),   # (state_dim,)
            np.array(next_global_state, dtype=np.float32),# (state_dim,)
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones, states, next_states = zip(*batch)
        return (
            np.stack(obs),         # (B, N, obs_dim)
            np.stack(acts),        # (B, N)
            np.stack(rews),        # (B, N)
            np.stack(next_obs),    # (B, N, obs_dim)
            np.array(dones, dtype=np.float32),    # (B,)
            np.stack(states),      # (B, state_dim)
            np.stack(next_states), # (B, state_dim)
        )

    def __len__(self):
        return len(self.buffer)


# ─── QMIX Agent ───────────────────────────────────────────────────────────────

class QMIXAgent:
    """
    QMIX multi-intersection traffic agent.

    - Shared agent Q-network (parameter sharing for sample efficiency)
    - Centralised mixing network conditioned on concatenated global state
    - CTDE: acts on local obs; mixer only used during training
    - Joint ε-greedy exploration with shared epsilon
    """

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int,
                 state_dim: int, cfg: dict):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.n_agents  = n_agents
        self.state_dim = state_dim
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyper-parameters
        self.gamma             = cfg.get("gamma",             0.95)
        self.lr                = cfg.get("lr",                5e-4)
        self.batch_size        = cfg.get("batch_size",        64)
        self.buffer_capacity   = cfg.get("buffer_capacity",   50_000)
        self.target_update_freq= cfg.get("target_update_freq",200)
        self.eps_start         = cfg.get("eps_start",         1.0)
        self.eps_end           = cfg.get("eps_end",           0.05)
        self.eps_decay_steps   = cfg.get("eps_decay_steps",   20_000)
        self.min_buffer_size   = cfg.get("min_buffer_size",   1_000)
        self.embed_dim         = cfg.get("embed_dim",         32)
        self.hidden            = tuple(cfg.get("hidden",      [128, 64]))

        # Networks
        self.agent_net        = AgentQNetwork(obs_dim, n_actions, self.hidden).to(self.device)
        self.target_agent_net = AgentQNetwork(obs_dim, n_actions, self.hidden).to(self.device)
        self.mixer            = MixingNetwork(n_agents, state_dim, self.embed_dim).to(self.device)
        self.target_mixer     = MixingNetwork(n_agents, state_dim, self.embed_dim).to(self.device)
        self._hard_update_targets()

        self.optimizer = optim.RMSprop(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=self.lr, alpha=0.99, eps=1e-5
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = MAReplayBuffer(self.buffer_capacity)
        self.steps    = 0
        self.epsilon  = self.eps_start

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hard_update_targets(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _current_epsilon(self) -> float:
        frac = min(self.steps / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_actions(self, obs_list: List[np.ndarray]) -> List[int]:
        """ε-greedy joint action selection (shared epsilon)."""
        self.epsilon = self._current_epsilon()

        if random.random() < self.epsilon:
            return [random.randrange(self.n_actions) for _ in range(self.n_agents)]

        obs_t = torch.FloatTensor(np.array(obs_list, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            q_vals = self.agent_net(obs_t)   # (N, n_actions)
        return q_vals.argmax(dim=1).cpu().tolist()

    # ── Store transition ──────────────────────────────────────────────────────

    def store_transition(self,
                         obs_list:         List[np.ndarray],
                         actions:          List[int],
                         rewards:          List[float],
                         next_obs_list:    List[np.ndarray],
                         done:             bool,
                         global_state:     np.ndarray,
                         next_global_state: Optional[np.ndarray] = None):
        if next_global_state is None:
            next_global_state = global_state
        self.replay_buffer.push(
            obs_list, actions, rewards, next_obs_list,
            done, global_state, next_global_state
        )
        self.steps += 1

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        obs, actions, rewards, next_obs, dones, states, next_states = \
            self.replay_buffer.sample(self.batch_size)

        B, N = actions.shape   # guaranteed by MAReplayBuffer

        obs_t        = torch.FloatTensor(obs).to(self.device)          # (B,N,obs)
        actions_t    = torch.LongTensor(actions).to(self.device)       # (B,N)
        rewards_t    = torch.FloatTensor(rewards).to(self.device)      # (B,N)
        next_obs_t   = torch.FloatTensor(next_obs).to(self.device)     # (B,N,obs)
        dones_t      = torch.FloatTensor(dones).to(self.device)        # (B,)
        states_t     = torch.FloatTensor(states).to(self.device)       # (B,state)
        next_states_t= torch.FloatTensor(next_states).to(self.device)  # (B,state)

        # ── Current Q_tot ─────────────────────────────────────────────────────
        obs_flat  = obs_t.view(B * N, self.obs_dim)
        q_all     = self.agent_net(obs_flat).view(B, N, self.n_actions)
        chosen_q  = q_all.gather(2, actions_t.unsqueeze(2)).squeeze(2)  # (B,N)
        q_tot     = self.mixer(chosen_q, states_t)                       # (B,1)

        # ── Target Q_tot ──────────────────────────────────────────────────────
        with torch.no_grad():
            next_flat    = next_obs_t.view(B * N, self.obs_dim)
            next_q_all   = self.target_agent_net(next_flat).view(B, N, self.n_actions)
            next_chosen_q= next_q_all.max(dim=2)[0]                       # (B,N)
            next_q_tot   = self.target_mixer(next_chosen_q, next_states_t) # (B,1)

            joint_reward = rewards_t.sum(dim=1, keepdim=True)              # (B,1)
            targets      = joint_reward + self.gamma * next_q_tot * (1.0 - dones_t.unsqueeze(1))

        loss = self.loss_fn(q_tot, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            max_norm=10.0
        )
        self.optimizer.step()

        if self.steps % self.target_update_freq == 0:
            self._hard_update_targets()

        return loss.item()

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "agent_net": self.agent_net.state_dict(),
            "mixer":     self.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps":     self.steps,
        }, path)
        print(f"[QMIX] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]
        self._hard_update_targets()
        print(f"[QMIX] Loaded ← {path}")
