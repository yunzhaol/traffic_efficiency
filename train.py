"""
train.py — Unified training entry point for all RL agents.

Usage:
    python train.py --agent dqn  --env cityflow --episodes 200
    python train.py --agent ppo  --env cityflow --episodes 200
    python train.py --agent iql  --env cityflow --episodes 300 --n_agents 16
    python train.py --agent qmix --env cityflow --episodes 300 --n_agents 16
"""

import argparse
import os
import sys
import random
import numpy as np
import torch

# Allow running from repo root: python train.py  OR  python rl_traffic/train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import Logger
from utils.config import load_config
from env.cityflow_env import CityFlowEnv
from env.sumo_env import SUMOEnv
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.iql import IQLAgent
from agents.qmix import QMIXAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(args, cfg):
    if args.env == "cityflow":
        return CityFlowEnv(
            config_path=cfg["cityflow"]["config_path"],
            num_intersections=args.n_agents,
            delta_time=cfg["env"]["delta_time"],
            yellow_time=cfg["env"]["yellow_time"],
            min_green=cfg["env"]["min_green"],
            max_green=cfg["env"]["max_green"],
            reward_type=cfg["env"]["reward_type"],
        )
    elif args.env == "sumo":
        return SUMOEnv(
            net_file=cfg["sumo"]["net_file"],
            route_file=cfg["sumo"]["route_file"],
            delta_time=cfg["env"]["delta_time"],
            yellow_time=cfg["env"]["yellow_time"],
            min_green=cfg["env"]["min_green"],
            max_green=cfg["env"]["max_green"],
            reward_type=cfg["env"]["reward_type"],
        )
    else:
        raise ValueError(f"Unknown env: {args.env!r}")


def make_agent(args, env, cfg):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if args.agent == "dqn":
        return DQNAgent(obs_dim, n_actions, cfg["dqn"])
    elif args.agent == "ppo":
        return PPOAgent(obs_dim, n_actions, cfg["ppo"])
    elif args.agent == "iql":
        return IQLAgent(obs_dim, n_actions, args.n_agents, cfg["iql"])
    elif args.agent == "qmix":
        return QMIXAgent(obs_dim, n_actions, args.n_agents, env.state_dim, cfg["qmix"])
    else:
        raise ValueError(f"Unknown agent: {args.agent!r}")


# ─── Single-agent training loop (DQN / PPO) ───────────────────────────────────

def train_single_agent(agent, env, args, logger):
    best_reward = -float("inf")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        last_loss = None

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.update()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            episode_reward += reward

        metrics = {
            "episode": episode,
            "reward": episode_reward,
            "avg_delay": info.get("avg_delay", 0.0),
            "throughput": info.get("throughput", 0),
            "epsilon": float(getattr(agent, "epsilon", 0.0)),
            "loss": last_loss if last_loss is not None else 0.0,
        }
        logger.log(metrics)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.save_dir, f"best_{args.agent}.pt"))

        if (episode + 1) % 10 == 0:
            print(
                f"[{args.agent.upper()}] Ep {episode+1:>4}/{args.episodes} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Delay: {metrics['avg_delay']:6.1f}s | "
                f"ε: {metrics['epsilon']:.3f} | "
                f"Loss: {metrics['loss']:.4f}"
            )

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    logger.save(os.path.join(args.save_dir, f"{args.agent}_log.csv"))


# ─── Multi-agent training loop (IQL / QMIX) ───────────────────────────────────

def train_multi_agent(agent, env, args, logger):
    best_reward = -float("inf")

    for episode in range(args.episodes):
        # Use reset_multi to get per-agent observation list
        obs_list, _ = env.reset_multi()
        episode_reward = 0.0
        done = False
        last_loss = None

        while not done:
            global_state = env.get_global_state()           # global state BEFORE step
            actions = agent.select_actions(obs_list)
            next_obs_list, rewards, terminated, truncated, info = env.step_multi(actions)
            done = terminated or truncated
            next_global_state = env.get_global_state()      # global state AFTER step

            agent.store_transition(
                obs_list, actions, rewards, next_obs_list,
                done, global_state, next_global_state
            )
            loss = agent.update()
            if loss is not None:
                last_loss = loss

            obs_list = next_obs_list
            episode_reward += sum(rewards)

        metrics = {
            "episode": episode,
            "total_reward": episode_reward,
            "avg_delay": info.get("avg_delay", 0.0),
            "network_throughput": info.get("throughput", 0),
            "loss": last_loss if last_loss is not None else 0.0,
        }
        logger.log(metrics)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.save_dir, f"best_{args.agent}.pt"))

        if (episode + 1) % 10 == 0:
            print(
                f"[{args.agent.upper()}] Ep {episode+1:>4}/{args.episodes} | "
                f"Total Reward: {episode_reward:10.1f} | "
                f"Avg Delay: {metrics['avg_delay']:6.1f}s | "
                f"Throughput: {metrics['network_throughput']:>6} veh/h"
            )

    print(f"\nTraining complete. Best total reward: {best_reward:.2f}")
    logger.save(os.path.join(args.save_dir, f"{args.agent}_log.csv"))


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL Traffic Signal Control Training")
    parser.add_argument("--agent",    choices=["dqn", "ppo", "iql", "qmix"], default="dqn")
    parser.add_argument("--env",      choices=["cityflow", "sumo"], default="cityflow")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--n_agents", type=int, default=1,
                        help="Number of intersections (1=single, 16=4×4 grid)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    # Enforce: multi-agent algorithms require n_agents > 1
    if args.agent in ("iql", "qmix") and args.n_agents == 1:
        print(f"[WARNING] {args.agent.upper()} is a multi-agent algorithm. "
              f"Setting --n_agents 4 automatically.")
        args.n_agents = 4

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    cfg = load_config(args.config)
    env = make_env(args, cfg)
    agent = make_agent(args, env, cfg)
    logger = Logger()

    print(f"\n{'='*60}")
    print(f"  Agent   : {args.agent.upper()}")
    print(f"  Env     : {args.env}  ({args.n_agents} intersection(s))")
    print(f"  Episodes: {args.episodes}   Seed: {args.seed}")
    print(f"  obs_dim : {env.observation_space.shape[0]}   n_actions: {env.action_space.n}")
    print(f"{'='*60}\n")

    if args.agent in ("iql", "qmix"):
        train_multi_agent(agent, env, args, logger)
    else:
        train_single_agent(agent, env, args, logger)

    env.close()


if __name__ == "__main__":
    main()
