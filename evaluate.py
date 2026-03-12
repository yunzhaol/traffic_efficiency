"""
evaluate.py — Evaluate a trained agent and report metrics.

Usage:
    python evaluate.py --agent dqn  --checkpoint checkpoints/best_dqn.pt
    python evaluate.py --agent ppo  --checkpoint checkpoints/best_ppo.pt
    python evaluate.py --agent qmix --checkpoint checkpoints/best_qmix.pt --n_agents 16
    python evaluate.py --agent iql  --checkpoint checkpoints/best_iql.pt  --n_agents 16
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config
from env.cityflow_env import CityFlowEnv
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.iql import IQLAgent
from agents.qmix import QMIXAgent


def _set_greedy(agent):
    """
    Force the agent into purely greedy (no exploration) mode.
    DQN/IQL: override the step counter so epsilon collapses to eps_end.
    PPO:     no epsilon — already greedy via argmax during eval.
    QMIX:    same as DQN/IQL, applied to the shared agent network.
    """
    if isinstance(agent, (DQNAgent,)):
        agent.steps = agent.eps_decay_steps  # forces epsilon → eps_end
    elif isinstance(agent, IQLAgent):
        for a in agent.agents:
            a.steps = a.eps_decay_steps
    elif isinstance(agent, QMIXAgent):
        agent.steps = agent.eps_decay_steps


# ─── Single-agent evaluation ──────────────────────────────────────────────────

def evaluate_single(agent, env, n_episodes: int) -> dict:
    _set_greedy(agent)
    delays, throughputs, ep_rewards = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        delays.append(info.get("avg_delay", 0.0))
        throughputs.append(info.get("throughput", 0))
        ep_rewards.append(total_reward)

    return {
        "avg_delay_mean":    float(np.mean(delays)),
        "avg_delay_std":     float(np.std(delays)),
        "throughput_mean":   float(np.mean(throughputs)),
        "throughput_std":    float(np.std(throughputs)),
        "reward_mean":       float(np.mean(ep_rewards)),
        "reward_std":        float(np.std(ep_rewards)),
    }


# ─── Multi-agent evaluation ───────────────────────────────────────────────────

def evaluate_multi(agent, env, n_episodes: int) -> dict:
    _set_greedy(agent)
    delays, throughputs, ep_rewards = [], [], []

    for ep in range(n_episodes):
        obs_list, _ = env.reset_multi()
        total_reward = 0.0
        done = False

        while not done:
            actions = agent.select_actions(obs_list)
            obs_list, rewards, terminated, truncated, info = env.step_multi(actions)
            done = terminated or truncated
            total_reward += sum(rewards)

        delays.append(info.get("avg_delay", 0.0))
        throughputs.append(info.get("throughput", 0))
        ep_rewards.append(total_reward)

    return {
        "avg_delay_mean":        float(np.mean(delays)),
        "avg_delay_std":         float(np.std(delays)),
        "network_throughput_mean": float(np.mean(throughputs)),
        "network_throughput_std":  float(np.std(throughputs)),
        "total_reward_mean":     float(np.mean(ep_rewards)),
        "total_reward_std":      float(np.std(ep_rewards)),
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL traffic agent")
    parser.add_argument("--agent",      choices=["dqn", "ppo", "iql", "qmix"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file saved by train.py")
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--n_agents",   type=int, default=1,
                        help="Number of intersections (match the training setting)")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    env = CityFlowEnv(
        config_path=cfg["cityflow"]["config_path"],
        num_intersections=args.n_agents,
        delta_time=cfg["env"]["delta_time"],
        yellow_time=cfg["env"]["yellow_time"],
        min_green=cfg["env"]["min_green"],
        max_green=cfg["env"]["max_green"],
        reward_type=cfg["env"]["reward_type"],
    )
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if args.agent == "dqn":
        agent = DQNAgent(obs_dim, n_actions, cfg["dqn"])
    elif args.agent == "ppo":
        agent = PPOAgent(obs_dim, n_actions, cfg["ppo"])
    elif args.agent == "iql":
        agent = IQLAgent(obs_dim, n_actions, args.n_agents, cfg["iql"])
    elif args.agent == "qmix":
        agent = QMIXAgent(obs_dim, n_actions, args.n_agents, env.state_dim, cfg["qmix"])
    else:
        raise ValueError(f"Unknown agent: {args.agent!r}")

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    agent.load(args.checkpoint)

    is_multi = args.agent in ("iql", "qmix")
    results  = evaluate_multi(agent, env, args.episodes) if is_multi \
               else evaluate_single(agent, env, args.episodes)

    env.close()

    print(f"\n{'='*55}")
    print(f"  Evaluation: {args.agent.upper()}   ({args.episodes} episodes)")
    print(f"{'='*55}")
    for k, v in results.items():
        print(f"  {k:<35s}: {v:>10.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
