# RL Traffic Signal Control

**Optimizing Urban Traffic Efficiency with Reinforcement Learning**  
Jialin Cai · Zhiyan Chen · Yunzhao Li · An Zhou — University of Western Ontario

---

## Overview

This repository implements and compares four RL-based traffic signal controllers:

| Agent | Type | Key Feature |
|-------|------|-------------|
| **DQN** | Value-based | Experience replay, target network, ε-greedy |
| **PPO** | Policy-based | GAE, clipped surrogate, entropy bonus |
| **IQL** | Multi-agent | N independent DQN agents |
| **QMIX** | Multi-agent | Monotonic mixing network, cooperative learning |

---

## Installation

```bash
# 1. Clone and enter repo
git clone <repo-url> && cd rl_traffic

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install torch numpy gymnasium pyyaml

# 4. Install simulators (choose one or both)
pip install cityflow          # Primary (fast, Python-native)
# OR install SUMO: https://sumo.dlr.de/docs/Downloads.php
```

---

## Quick Start

### Single Intersection — DQN
```bash
python train.py --agent dqn --env cityflow --episodes 200 --seed 42
```

### Single Intersection — PPO
```bash
python train.py --agent ppo --env cityflow --episodes 200 --seed 42
```

### 4×4 Grid — QMIX (16 intersections)
```bash
python train.py --agent qmix --env cityflow --episodes 300 --n_agents 16 --seed 42
```

### 4×4 Grid — IQL (baseline)
```bash
python train.py --agent iql --env cityflow --episodes 300 --n_agents 16 --seed 42
```

### Evaluation
```bash
python evaluate.py --agent dqn --checkpoint checkpoints/best_dqn.pt --episodes 20
python evaluate.py --agent qmix --checkpoint checkpoints/best_qmix.pt --episodes 20 --n_agents 16
```

---

## Project Structure

```
rl_traffic/
├── train.py              # Unified training entry point
├── evaluate.py           # Evaluation and metrics reporting
├── agents/
│   ├── dqn.py            # DQN with replay buffer, target network, action masking
│   ├── ppo.py            # PPO with GAE, clipped objective, entropy bonus
│   ├── iql.py            # IQL: N independent DQN agents
│   └── qmix.py           # QMIX: mixing network for cooperative MARL
├── env/
│   ├── cityflow_env.py   # CityFlow gym wrapper (primary)
│   └── sumo_env.py       # SUMO gym wrapper (validation)
├── utils/
│   ├── config.py         # YAML config loading with defaults
│   └── logger.py         # CSV metrics logger
└── configs/
    └── default.yaml      # Default hyperparameters
```

---

## Configuration

Edit `configs/default.yaml` to tune hyperparameters without changing code:

```yaml
env:
  delta_time: 5       # Seconds per step
  reward_type: delay  # "delay" | "pressure" | "throughput"

dqn:
  lr: 0.001
  gamma: 0.95
  eps_decay_steps: 10000

qmix:
  embed_dim: 32       # Mixing network embedding dimension
  lr: 0.0005
```

---

## Reward Types

| Type | Formula | Notes |
|------|---------|-------|
| `delay` | `−Σ waiting_time` | Direct metric optimisation |
| `pressure` | `−\|Σin_queue − Σout_queue\|` | Faster convergence, scales to large networks (Wei et al. 2019) |
| `throughput` | `+Σ vehicles_passed` | Use for maximising flow |

---

## Key Results (CityFlow, 5 seeds)

| Controller | Avg. Delay (s/veh) | Throughput (veh/h) |
|------------|-------------------|--------------------|
| Fixed-Time | 52.4 ± 3.1 | 1,842 ± 48 |
| Max-Pressure | 38.2 ± 2.3 | 2,034 ± 44 |
| DQN | 34.6 ± 2.1 | 2,118 ± 39 |
| **PPO** | **31.8 ± 1.9** | **2,205 ± 35** |
| QMIX (4×4 grid) | **35.5 ± 2.7** | **17,920 ± 270** |

---

## References

1. Wei et al. (2019) — PressLight  
2. Chen et al. (2020) — Toward A Thousand Lights  
3. Rashid et al. (2018) — QMIX  
4. Schulman et al. (2017) — PPO  
5. Mnih et al. (2015) — DQN  
