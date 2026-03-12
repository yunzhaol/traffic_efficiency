"""utils/config.py"""
import yaml
import os

DEFAULTS = {
    "env": {
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 60,
        "reward_type": "delay",
    },
    "cityflow": {
        "config_path": "configs/cityflow_1x1.json",
    },
    "sumo": {
        "net_file": "configs/sumo/single.net.xml",
        "route_file": "configs/sumo/single.rou.xml",
    },
    "dqn": {
        "gamma": 0.95, "lr": 1e-3, "batch_size": 64,
        "buffer_capacity": 50000, "target_update_freq": 500,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 10000,
        "min_buffer_size": 1000, "hidden": [128, 64],
    },
    "ppo": {
        "gamma": 0.95, "gae_lambda": 0.95, "clip_eps": 0.2,
        "ppo_epochs": 10, "mini_batch_size": 64, "rollout_steps": 2048,
        "vf_coef": 0.5, "ent_coef": 0.01, "lr": 3e-4,
        "max_grad_norm": 0.5, "hidden": [128, 64],
    },
    "iql": {
        "gamma": 0.95, "lr": 1e-3, "batch_size": 64,
        "buffer_capacity": 50000, "target_update_freq": 500,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 20000,
        "min_buffer_size": 1000, "hidden": [128, 64], "share_params": False,
    },
    "qmix": {
        "gamma": 0.95, "lr": 5e-4, "batch_size": 64,
        "buffer_capacity": 50000, "target_update_freq": 200,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 20000,
        "min_buffer_size": 1000, "embed_dim": 32, "hidden": [128, 64],
    },
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        for key, val in user_cfg.items():
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg
