"""
env/sumo_env.py — SUMO gymnasium wrapper for single-intersection experiments.

Requires SUMO installation: https://sumo.dlr.de/docs/Downloads.php
Set SUMO_HOME environment variable before running.

pip install traci sumolib
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

# SUMO_HOME must be set
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
if os.path.exists(os.path.join(SUMO_HOME, "tools")):
    sys.path += [os.path.join(SUMO_HOME, "tools")]


class SUMOEnv(gym.Env):
    """
    SUMO-based single-intersection environment.

    Signal phases (4-phase default):
        0: North-South straight
        1: North-South left turns
        2: East-West straight
        3: East-West left turns
    """

    metadata = {"render_modes": ["human"]}

    PHASE_DEFINITIONS = {
        # phase_index: (SUMO_phase_state, description)
        0: ("GGrrGGrr", "NS straight"),
        1: ("rrGGrrGG", "EW straight"),
        2: ("GGGGrrrr", "NS all"),
        3: ("rrrrGGGG", "EW all"),
    }

    def __init__(
        self,
        net_file: str = "configs/sumo/single.net.xml",
        route_file: str = "configs/sumo/single.rou.xml",
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 60,
        reward_type: str = "delay",
        use_gui: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_type = reward_type
        self.use_gui = use_gui

        self._n_phases = len(self.PHASE_DEFINITIONS)
        self._lanes = []  # populated at reset
        self._tl_id = "TL_0"  # default traffic light ID
        self._current_phase = 0
        self._phase_duration = 0
        self._step_count = 0
        self._yellow_active = False
        self._traci = None

        # Observation: 8 queue lengths + 8 wait times + 1 phase = 17
        obs_dim = 17
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._n_phases)
        self.state_dim = obs_dim

    def _start_simulation(self):
        """Start or restart SUMO simulation."""
        try:
            import traci
            self._traci = traci
        except ImportError:
            print("[SUMOEnv] WARNING: traci not available. Using mock mode.")
            self._traci = None
            return

        if traci.isLoaded():
            traci.close()

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--seed", "42",
        ]
        traci.start(sumo_cmd)

        # Discover lanes controlled by our traffic light
        try:
            self._lanes = list(set(
                lane for _, lane in traci.trafficlight.getControlledLinks(self._tl_id)
                if lane
            ))[:8]  # limit to 8 incoming lanes
        except Exception:
            self._lanes = [f"lane_{i}" for i in range(8)]

    def _get_obs(self) -> np.ndarray:
        if self._traci is None:
            # Mock observations
            queues = np.random.rand(8) * 0.3
            waits = np.random.rand(8) * 0.2
            phase_norm = np.array([self._current_phase / self._n_phases])
            return np.clip(np.concatenate([queues, waits, phase_norm]), 0, 1).astype(np.float32)

        queues, waits = [], []
        for lane in self._lanes:
            try:
                q = self._traci.lane.getLastStepHaltingNumber(lane) / 50.0
                w = self._traci.lane.getWaitingTime(lane) / 300.0
            except Exception:
                q, w = 0.0, 0.0
            queues.append(q)
            waits.append(w)

        # Pad to length 8 if fewer lanes
        while len(queues) < 8:
            queues.append(0.0)
            waits.append(0.0)
        queues, waits = queues[:8], waits[:8]

        phase_norm = [self._current_phase / self._n_phases]
        obs = np.array(queues + waits + phase_norm, dtype=np.float32)
        return np.clip(obs, 0, 1)

    def _compute_reward(self) -> float:
        if self._traci is None:
            return -np.random.uniform(10, 50)

        if self.reward_type == "delay":
            total_wait = sum(
                self._traci.lane.getWaitingTime(l) for l in self._lanes
            )
            return -total_wait / 300.0  # normalize

        elif self.reward_type == "pressure":
            in_count = sum(
                self._traci.lane.getLastStepHaltingNumber(l)
                for l in self._lanes[:len(self._lanes)//2]
            )
            out_count = sum(
                self._traci.lane.getLastStepHaltingNumber(l)
                for l in self._lanes[len(self._lanes)//2:]
            )
            return -abs(in_count - out_count) / 50.0

        elif self.reward_type == "throughput":
            return sum(
                self._traci.lane.getLastStepVehicleNumber(l) for l in self._lanes
            ) / 50.0

        return 0.0

    def _apply_action(self, action: int):
        """Set traffic light phase with yellow-light transition."""
        if action == self._current_phase:
            self._phase_duration += self.delta_time
            return

        if self._phase_duration < self.min_green:
            return  # enforce minimum green

        # Set yellow phase if transitioning
        if self._traci is not None and not self._yellow_active:
            yellow_state = "y" * len(self.PHASE_DEFINITIONS[self._current_phase][0])
            try:
                self._traci.trafficlight.setRedYellowGreenState(self._tl_id, yellow_state)
            except Exception:
                pass

        self._current_phase = action
        self._phase_duration = 0

        if self._traci is not None:
            phase_state = self.PHASE_DEFINITIONS.get(action, ("GGrrGGrr", ""))[0]
            try:
                self._traci.trafficlight.setRedYellowGreenState(self._tl_id, phase_state)
            except Exception:
                pass

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass
        self._start_simulation()
        self._current_phase = 0
        self._phase_duration = 0
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._apply_action(action)

        if self._traci is not None:
            for _ in range(self.delta_time):
                try:
                    self._traci.simulationStep()
                except Exception:
                    break

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        info = self._get_info()

        terminated = False
        truncated = self._step_count >= 720

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        if self._traci is None:
            return {"avg_delay": np.random.uniform(30, 70),
                    "throughput": np.random.randint(1500, 2200)}
        try:
            avg_delay = self._traci.simulation.getParameter("", "avgTravelTime")
            return {"avg_delay": float(avg_delay) if avg_delay else 0,
                    "throughput": 0, "step": self._step_count}
        except Exception:
            return {"avg_delay": 0, "throughput": 0, "step": self._step_count}

    def get_global_state(self) -> np.ndarray:
        return self._get_obs()

    def close(self):
        if self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass

    def render(self):
        pass
