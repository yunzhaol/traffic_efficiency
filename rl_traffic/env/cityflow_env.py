"""
env/cityflow_env.py — CityFlow gym-compatible environment wrapper.

CityFlow: A fast, scalable traffic simulator.
Install: pip install cityflow
GitHub:  https://github.com/cityflow-project/CityFlow

Supports both single-intersection and multi-intersection grid scenarios.
Falls back to a deterministic mock engine when CityFlow is not installed,
so all training/evaluation code can be tested without the simulator.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import json
import os


# ─── Deterministic mock engine ────────────────────────────────────────────────

class _MockEngine:
    """
    Lightweight deterministic traffic mock.
    Vehicles accumulate when signal is red, drain when green.
    Produces realistic-looking obs/reward without CityFlow installed.
    """

    def __init__(self, n_intersections: int, lanes_per: int, n_phases: int, seed: int = 0):
        self.n_intersections = n_intersections
        self.lanes_per = lanes_per
        self.n_phases = n_phases
        self._rng = np.random.RandomState(seed)
        self._queues: np.ndarray = np.zeros((n_intersections, lanes_per), dtype=np.float32)
        self._waits: np.ndarray = np.zeros((n_intersections, lanes_per), dtype=np.float32)
        self._sim_time: int = 0

    def reset(self):
        self._queues[:] = 0.0
        self._waits[:] = 0.0
        self._sim_time = 0

    def next_step(self, current_phases: List[int]):
        """Advance simulation by 1 second."""
        self._sim_time += 1
        arrival_rate = 0.3 + 0.2 * np.sin(self._sim_time / 600.0)  # time-varying demand
        for i in range(self.n_intersections):
            for l in range(self.lanes_per):
                # Vehicles arrive stochastically
                if self._rng.rand() < arrival_rate:
                    self._queues[i, l] = min(self._queues[i, l] + 1, 50)
                # Green phase drains the assigned lanes
                green_lanes = self._get_green_lanes(current_phases[i])
                if l in green_lanes and self._queues[i, l] > 0:
                    drained = min(self._queues[i, l], self._rng.randint(1, 3))
                    self._queues[i, l] -= drained
                    self._waits[i, l] = 0.0
                else:
                    self._waits[i, l] += 1.0

    def _get_green_lanes(self, phase: int) -> List[int]:
        """Map phase index → lanes that get green."""
        # Phase 0→lanes 0,1; Phase 1→lanes 2,3; etc.
        lanes_per_phase = max(1, self.lanes_per // self.n_phases)
        start = phase * lanes_per_phase
        return list(range(start, min(start + lanes_per_phase, self.lanes_per)))

    def get_lane_vehicle_count(self, intersection_idx: int) -> np.ndarray:
        return self._queues[intersection_idx].copy()

    def get_lane_waiting_vehicle_count(self, intersection_idx: int) -> np.ndarray:
        return self._waits[intersection_idx].copy()

    def get_average_travel_time(self) -> float:
        total_wait = self._waits.sum()
        n_vehicles = max(self._queues.sum(), 1)
        return float(total_wait / n_vehicles)

    def set_tl_phase(self, intersection_idx: int, phase: int):
        pass  # phase is read from current_phases each step


# ─── CityFlow wrapper ─────────────────────────────────────────────────────────

class CityFlowEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around CityFlow for RL training.

    Observation (per intersection, fixed 17-dim):
        [queue_l0..l7,   # normalised queue lengths  (8 values)
         wait_l0..l7,    # normalised waiting times   (8 values)
         current_phase]  # normalised phase index     (1 value)

    Action:  discrete phase index  (0 .. n_phases-1)

    Reward:
        'delay'     : -sum(waiting vehicles per lane)
        'pressure'  : -|sum(in_queues) - sum(out_queues)|  (Wei et al. 2019)
        'throughput': +vehicles that cleared their lane this step
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Fixed obs layout constants
    N_LANES = 8   # lanes per intersection in observation
    OBS_DIM = N_LANES * 2 + 1   # 17

    def __init__(
        self,
        config_path: str = "configs/cityflow_1x1.json",
        num_intersections: int = 1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 60,
        reward_type: str = "delay",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config_path = config_path
        self.num_intersections = num_intersections
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_type = reward_type
        self.render_mode = render_mode

        # Try to import real CityFlow; fall back to deterministic mock
        self._mock = False
        self.eng = None
        self._cf_intersections: List[str] = []
        self._cf_lanes: Dict[str, List[str]] = {}   # intersection_id → lane list
        self._n_phases: int = 4

        try:
            import cityflow as cf
            self.eng = cf.Engine(config_path, thread_num=1)
            self._mock = False
            self._cf_intersections, self._cf_lanes, self._n_phases = \
                self._parse_cityflow_config(config_path)
            print(f"[CityFlowEnv] CityFlow loaded: "
                  f"{len(self._cf_intersections)} intersections, "
                  f"{self._n_phases} phases")
        except ImportError:
            print("[CityFlowEnv] cityflow not installed — using mock engine.")
            self._mock = True
        except Exception as e:
            print(f"[CityFlowEnv] CityFlow init failed ({e}) — using mock engine.")
            self._mock = True

        if self._mock:
            self._mock_eng = _MockEngine(
                n_intersections=num_intersections,
                lanes_per=self.N_LANES,
                n_phases=self._n_phases,
                seed=0,
            )
            self._cf_intersections = [f"intersection_{i}" for i in range(num_intersections)]

        # Spaces — fixed size regardless of real/mock
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._n_phases)
        # Global state dimension used by the QMIX mixing network
        self.state_dim = self.OBS_DIM * num_intersections

        # Episode state
        self._current_phases: List[int] = [0] * num_intersections
        self._phase_durations: List[int] = [0] * num_intersections
        self._step_count: int = 0
        # Track previous queue per intersection for throughput reward
        self._prev_queues: List[np.ndarray] = [
            np.zeros(self.N_LANES) for _ in range(num_intersections)
        ]

    # ── Config parsing ────────────────────────────────────────────────────────

    def _parse_cityflow_config(self, config_path: str
                               ) -> Tuple[List[str], Dict[str, List[str]], int]:
        """
        Read CityFlow roadnet JSON and return:
          - list of non-virtual intersection IDs (in order)
          - dict mapping intersection_id → list of incoming lane IDs
          - number of signal phases (from first intersection's lightphases)
        """
        with open(config_path) as f:
            top = json.load(f)

        roadnet_rel = top.get("roadnetFile", "roadnet.json")
        roadnet_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), roadnet_rel)
        with open(roadnet_path) as f:
            roadnet = json.load(f)

        intersections: List[str] = []
        lanes_map: Dict[str, List[str]] = {}
        n_phases = 4

        # Build road-id → lane-ids lookup from roads
        road_lanes: Dict[str, List[str]] = {}
        for road in roadnet.get("roads", []):
            rid = road["id"]
            n_lanes = road.get("lanes", 1) if isinstance(road.get("lanes"), int) \
                else len(road.get("lanes", [1]))
            road_lanes[rid] = [f"{rid}_{k}" for k in range(n_lanes)]

        for inter in roadnet.get("intersections", []):
            if inter.get("virtual", False):
                continue
            iid = inter["id"]
            intersections.append(iid)

            # Collect incoming lanes (roads where this intersection is the endpoint)
            incoming: List[str] = []
            for road in roadnet.get("roads", []):
                if road.get("endIntersection") == iid:
                    incoming.extend(road_lanes.get(road["id"], []))
            # Pad or truncate to exactly N_LANES
            while len(incoming) < self.N_LANES:
                incoming.append(f"{iid}_pad_{len(incoming)}")
            lanes_map[iid] = incoming[:self.N_LANES]

            # Number of phases from first intersection
            if len(intersections) == 1:
                n_phases = max(len(inter.get("lightphases", [])), 1)

        if not intersections:
            raise ValueError("No non-virtual intersections found in roadnet.")

        return intersections, lanes_map, n_phases

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self, inter_idx: int) -> np.ndarray:
        """Build 17-dim observation vector for intersection `inter_idx`."""
        if self._mock:
            q = self._mock_eng.get_lane_vehicle_count(inter_idx) / 50.0
            w = self._mock_eng.get_lane_waiting_vehicle_count(inter_idx) / 300.0
        else:
            iid = self._cf_intersections[inter_idx]
            lanes = self._cf_lanes[iid]
            lane_q = self.eng.get_lane_vehicle_count()
            lane_w = self.eng.get_lane_waiting_vehicle_count()
            q = np.array([lane_q.get(l, 0) for l in lanes], dtype=np.float32) / 50.0
            w = np.array([lane_w.get(l, 0) for l in lanes], dtype=np.float32) / 300.0

        phase = np.array(
            [self._current_phases[inter_idx] / max(self._n_phases - 1, 1)],
            dtype=np.float32
        )
        return np.clip(np.concatenate([q, w, phase]), 0.0, 1.0).astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, inter_idx: int) -> float:
        if self._mock:
            q = self._mock_eng.get_lane_vehicle_count(inter_idx)
            w = self._mock_eng.get_lane_waiting_vehicle_count(inter_idx)
        else:
            iid = self._cf_intersections[inter_idx]
            lanes = self._cf_lanes[iid]
            lane_q = self.eng.get_lane_vehicle_count()
            lane_w = self.eng.get_lane_waiting_vehicle_count()
            q = np.array([lane_q.get(l, 0) for l in lanes], dtype=np.float32)
            w = np.array([lane_w.get(l, 0) for l in lanes], dtype=np.float32)

        if self.reward_type == "delay":
            return -float(w.sum()) / 50.0      # normalise

        elif self.reward_type == "pressure":
            half = len(q) // 2
            return -float(abs(q[:half].sum() - q[half:].sum())) / 25.0

        elif self.reward_type == "throughput":
            prev = self._prev_queues[inter_idx]
            cleared = np.maximum(prev - q, 0).sum()
            self._prev_queues[inter_idx] = q.copy()
            return float(cleared) / 10.0

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type!r}")

    # ── Phase control ─────────────────────────────────────────────────────────

    def _apply_action(self, inter_idx: int, action: int):
        """
        Apply agent action with minimum-green enforcement.
        - Increments phase_duration every step regardless of action.
        - Switches phase only if min_green has been served.
        - Forces phase cycle if max_green exceeded (no recursion).
        """
        # Always tick duration first
        self._phase_durations[inter_idx] += self.delta_time

        # Force phase advance if max_green exceeded, ignoring agent action
        if self._phase_durations[inter_idx] >= self.max_green:
            action = (self._current_phases[inter_idx] + 1) % self._n_phases

        # Switch only if min_green served and action differs
        if (self._phase_durations[inter_idx] >= self.min_green
                and action != self._current_phases[inter_idx]):
            self._current_phases[inter_idx] = int(action)
            self._phase_durations[inter_idx] = 0
            if not self._mock:
                iid = self._cf_intersections[inter_idx]
                try:
                    self.eng.set_tl_phase(iid, action)
                except Exception:
                    pass

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None
              ) -> Tuple[np.ndarray, Dict]:
        """Single-agent reset. Returns (obs, info)."""
        super().reset(seed=seed)
        self._reset_state(seed)
        return self._get_obs(0), {}

    def reset_multi(self, seed: Optional[int] = None
                    ) -> Tuple[List[np.ndarray], Dict]:
        """Multi-agent reset. Returns (list_of_obs, info)."""
        super().reset(seed=seed)
        self._reset_state(seed)
        return [self._get_obs(i) for i in range(self.num_intersections)], {}

    def _reset_state(self, seed: Optional[int] = None):
        if self._mock:
            if seed is not None:
                self._mock_eng._rng = np.random.RandomState(seed)
            self._mock_eng.reset()
        else:
            self.eng.reset()
        self._current_phases = [0] * self.num_intersections
        self._phase_durations = [0] * self.num_intersections
        self._step_count = 0
        self._prev_queues = [np.zeros(self.N_LANES) for _ in range(self.num_intersections)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Single-agent step."""
        self._apply_action(0, action)
        self._advance_sim()
        self._step_count += 1

        obs = self._get_obs(0)
        reward = self._compute_reward(0)
        info = self._get_info()
        truncated = self._step_count >= 720   # 3600 s / 5 s per step
        return obs, reward, False, truncated, info

    def step_multi(self, actions: List[int]
                   ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """Multi-agent step."""
        for i, a in enumerate(actions):
            a = int(a)
            self._apply_action(i, a)
        self._advance_sim()
        self._step_count += 1

        obs_list = [self._get_obs(i) for i in range(self.num_intersections)]
        rewards = [self._compute_reward(i) for i in range(self.num_intersections)]
        info = self._get_info()
        truncated = self._step_count >= 720
        return obs_list, rewards, False, truncated, info

    def _advance_sim(self):
        """Advance simulator by delta_time seconds."""
        if self._mock:
            for _ in range(self.delta_time):
                self._mock_eng.next_step(self._current_phases)
        else:
            for _ in range(self.delta_time):
                self.eng.next_step()

    def get_global_state(self) -> np.ndarray:
        """Concatenate all agent observations → global state for QMIX mixer."""
        return np.concatenate([self._get_obs(i) for i in range(self.num_intersections)])

    def _get_info(self) -> Dict[str, Any]:
        if self._mock:
            avg_delay = self._mock_eng.get_average_travel_time()
        else:
            try:
                avg_delay = self.eng.get_average_travel_time()
            except Exception:
                avg_delay = 0.0
        throughput = int(3600.0 / max(avg_delay, 1.0))
        return {"avg_delay": avg_delay, "throughput": throughput, "step": self._step_count}

    def close(self):
        pass

    def render(self):
        pass
