"""
agents/iql.py — Independent Q-Learning (IQL) for multi-agent traffic control.

Each agent maintains its own DQN and treats other agents as part of the environment.
Simple and scalable, but cannot model inter-agent dependencies.
Serves as a lower-bound MARL baseline against QMIX.
"""

import os
import numpy as np
import torch
from typing import List, Optional
from agents.dqn import DQNAgent


class IQLAgent:
    """
    Independent Q-Learning: N independent DQN agents with shared architecture
    but separate parameters and replay buffers.

    Design choices:
    - Parameter sharing OFF by default (set share_params=True to compare)
    - Each agent maintains its own epsilon for independent exploration
    - Joint update: all agents update simultaneously each step
    """

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, cfg: dict):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.share_params = cfg.get("share_params", False)

        # Create N independent DQN agents
        self.agents: List[DQNAgent] = [
            DQNAgent(obs_dim, n_actions, cfg) for _ in range(n_agents)
        ]

        # Optional: weight sharing (all agents point to agent 0's network)
        if self.share_params:
            for i in range(1, n_agents):
                self.agents[i].q_net = self.agents[0].q_net
                self.agents[i].target_net = self.agents[0].target_net
                self.agents[i].optimizer = self.agents[0].optimizer

        self.steps = 0

    def select_actions(self, obs_list: List[np.ndarray]) -> List[int]:
        """Each agent independently selects an action."""
        return [self.agents[i].select_action(obs_list[i])
                for i in range(self.n_agents)]

    def store_transition(self, obs_list, actions, rewards, next_obs_list,
                         done, global_state=None, next_global_state=None):
        """Store individual transitions for each agent."""
        for i in range(self.n_agents):
            self.agents[i].store_transition(
                obs_list[i], actions[i], rewards[i], next_obs_list[i], done
            )
        self.steps += 1

    def update(self) -> Optional[float]:
        """Update all agents and return mean loss."""
        losses = []
        for agent in self.agents:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
        return np.mean(losses) if losses else None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            f"agent_{i}": {
                "q_net": self.agents[i].q_net.state_dict(),
                "target_net": self.agents[i].target_net.state_dict(),
                "steps": self.agents[i].steps,
            }
            for i in range(self.n_agents)
        }
        torch.save(state, path)
        print(f"[IQL] Saved {self.n_agents} agent checkpoints → {path}")

    def load(self, path: str):
        device = self.agents[0].device
        state = torch.load(path, map_location=device)
        for i in range(self.n_agents):
            key = f"agent_{i}"
            if key in state:
                self.agents[i].q_net.load_state_dict(state[key]["q_net"])
                self.agents[i].target_net.load_state_dict(state[key]["target_net"])
                self.agents[i].steps = state[key]["steps"]
        print(f"[IQL] Loaded {self.n_agents} agent checkpoints ← {path}")

    @property
    def epsilon(self) -> float:
        return self.agents[0].epsilon
