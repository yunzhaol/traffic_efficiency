"""
Integration test — mocks torch/gymnasium to run without them installed.
Tests all logic in env, agents, train loop, evaluate loop, and utilities.
Run: python test_integration.py
"""
import sys, os, types, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# ── Realistic torch stub ─────────────────────────────────────────────────────
class _T:
    """Numpy-backed tensor."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
    def to(self, d):           return self
    def unsqueeze(self, d):    return _T(np.expand_dims(self.data, d))
    def squeeze(self, d=None):
        if d is None: return _T(self.data.squeeze())
        return _T(np.squeeze(self.data, axis=d))
    def item(self):            return float(self.data.flat[0])
    def view(self, *s):        return _T(self.data.reshape(s))
    def __call__(self, *a):    return self
    def argmax(self, dim=None):return _T(np.argmax(self.data, axis=dim))
    def max(self, dim=None):
        if dim is None: return _T(np.max(self.data))
        idx = np.argmax(self.data, axis=dim)
        return _T(np.max(self.data, axis=dim)), _T(idx)
    def gather(self, dim, idx):
        return _T(np.take_along_axis(self.data, idx.data.astype(int), axis=dim))
    def __sub__(self, o):      return _T(self.data-(o.data if isinstance(o,_T) else o))
    def __add__(self, o):      return _T(self.data+(o.data if isinstance(o,_T) else o))
    def __mul__(self, o):      return _T(self.data*(o.data if isinstance(o,_T) else o))
    def __rmul__(self, o):     return _T((o.data if isinstance(o,_T) else o)*self.data)
    def __radd__(self, o):     return _T((o.data if isinstance(o,_T) else o)+self.data)
    def __rsub__(self, o):     return _T((o.data if isinstance(o,_T) else o)-self.data)
    def __neg__(self):         return _T(-self.data)
    def pow(self, p):          return _T(self.data**p)
    def mean(self):            return _T(np.mean(self.data))
    def sum(self, dim=None, keepdim=False):
        return _T(np.sum(self.data,axis=dim,keepdims=keepdim))
    def exp(self):             return _T(np.exp(np.clip(self.data,-80,80)))
    def log(self):             return _T(np.log(self.data+1e-8))
    def relu(self):            return _T(np.maximum(self.data,0))
    def backward(self):        pass
    def cpu(self):             return self
    def tolist(self):          return self.data.tolist()
    def __getitem__(self, k):
        k2 = k.data.astype(int).tolist() if isinstance(k, _T) else k
        return _T(self.data[k2])
    def __setitem__(self, k, v):
        k2 = k.data.astype(int).tolist() if isinstance(k, _T) else k
        self.data[k2] = v.data if isinstance(v, _T) else v
    def size(self, d=None):    return self.data.shape[d] if d is not None else self.data.shape
    def __repr__(self):        return f"T({self.data.shape})"

class _Linear:
    def __init__(self, i, o):
        self.weight = np.random.randn(i, o).astype(np.float32)*0.1
        self.bias   = np.zeros(o, dtype=np.float32)
        self._params= [self.weight, self.bias]
    def __call__(self, x):
        return _T(x.data @ self.weight + self.bias)
    def state_dict(self): return {"w":self.weight,"b":self.bias}
    def load_state_dict(self, d): self.weight=d["w"]; self.bias=d["b"]

class _Seq:
    def __init__(self, *layers): self.layers=list(layers); self._params=[]
    def __call__(self, x):
        for l in self.layers:
            if callable(l): x = l(x)
        return x
    def state_dict(self): return {}
    def load_state_dict(self, d): pass

class _ReLU:
    def __call__(self, x): return x.relu()
class _Tanh:
    def __call__(self, x): return _T(np.tanh(x.data))
class _SmoothL1:
    def __call__(self, p, t):
        diff=np.abs(p.data-t.data)
        return _T(np.mean(np.where(diff<1, .5*diff**2, diff-.5)))

class _Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    """Base module with __call__, parameters, state_dict support."""
    def __init__(self): pass
    def to(self, d): return self
    def eval(self):  return self
    def train(self): return self
    def modules(self):
        yield self
        for v in self.__dict__.values():
            if isinstance(v, (_Module, _Linear, _Seq)):
                if hasattr(v, 'modules'):
                    yield from v.modules()
    def parameters(self):
        for m in self.modules():
            if isinstance(m, _Linear):
                yield m.weight
    def state_dict(self):
        d={}
        for k,v in self.__dict__.items():
            if hasattr(v,'state_dict'): d[k]=v.state_dict()
        return d
    def load_state_dict(self, d):
        for k,v in d.items():
            if hasattr(self,k) and hasattr(getattr(self,k),'load_state_dict'):
                getattr(self,k).load_state_dict(v)

class _Categorical:
    def __init__(self, logits=None):
        d=logits.data.flatten(); p=np.exp(d-np.max(d)); self._p=p/p.sum()
    def sample(self): return _T(np.array(np.random.choice(len(self._p),p=self._p)))
    def log_prob(self, a): return _T(np.array(np.log(self._p[int(a.item())]+1e-8)))
    def entropy(self): return _T(np.array(-np.sum(self._p*np.log(self._p+1e-8))))

class _nograd:
    def __enter__(self): return self
    def __exit__(self,*a): pass

class _optim_base:
    def __init__(self, params, **kw): pass
    def zero_grad(self): pass
    def step(self): pass
    def state_dict(self): return {}
    def load_state_dict(self,d): pass

# Assemble torch mock
torch_mod = types.ModuleType("torch")
torch_mod.Tensor      = _T
torch_mod.FloatTensor = lambda x: _T(np.array(x,dtype=np.float32))
torch_mod.LongTensor  = lambda x: _T(np.array(x,dtype=np.int64))
torch_mod.zeros       = lambda *s: _T(np.zeros(s,dtype=np.float32))
torch_mod.device      = lambda s: s
torch_mod.cuda        = types.SimpleNamespace(is_available=lambda:False, manual_seed_all=lambda s:None)
torch_mod.no_grad     = _nograd
torch_mod.save        = lambda obj,path,**kw: None
torch_mod.load        = lambda path,**kw: {}
torch_mod.abs         = lambda x: _T(np.abs(x.data))
torch_mod.relu        = lambda x: x.relu()
torch_mod.clamp       = lambda x,lo,hi: _T(np.clip(x.data,lo,hi))
torch_mod.min         = lambda a,b: _T(np.minimum(a.data,b.data))
torch_mod.bmm         = lambda a,b: _T(np.einsum('bij,bjk->bik',a.data,b.data))
torch_mod.manual_seed = lambda s: None
torch_mod.stack       = lambda ts,dim=0: _T(np.stack([t.data for t in ts],axis=dim))
torch_mod.full        = lambda shape,fill: _T(np.full(shape,fill,dtype=np.float32))

nn_mod = types.ModuleType("torch.nn")
nn_mod.Module       = _Module
nn_mod.Linear       = _Linear
nn_mod.Sequential   = _Seq
nn_mod.ReLU         = _ReLU
nn_mod.Tanh         = _Tanh
nn_mod.SmoothL1Loss = _SmoothL1
nn_mod.utils        = types.SimpleNamespace(clip_grad_norm_=lambda p,max_norm:None)
nn_mod.init         = types.SimpleNamespace(orthogonal_=lambda w,gain=1:None, zeros_=lambda b:None)
torch_mod.nn        = nn_mod

optim_mod = types.ModuleType("torch.optim")
optim_mod.Adam    = type("Adam",    (_optim_base,), {})
optim_mod.RMSprop = type("RMSprop", (_optim_base,), {})
torch_mod.optim   = optim_mod

dist_mod = types.ModuleType("torch.distributions")
dist_mod.Categorical = _Categorical
torch_mod.distributions = dist_mod

sys.modules["torch"]               = torch_mod
sys.modules["torch.nn"]            = nn_mod
sys.modules["torch.optim"]         = optim_mod
sys.modules["torch.distributions"] = dist_mod
sys.modules["torch.nn.utils"]      = types.ModuleType("torch.nn.utils")
sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")

# ── Gymnasium stub ────────────────────────────────────────────────────────────
gym_mod = types.ModuleType("gymnasium")
gym_spaces = types.ModuleType("gymnasium.spaces")
class _Box:
    def __init__(self,low,high,shape,dtype): self.shape=shape
class _Discrete:
    def __init__(self,n): self.n=n
class _GymEnv:
    def reset(self,seed=None,options=None): pass
gym_spaces.Box = _Box; gym_spaces.Discrete = _Discrete
gym_mod.Env    = _GymEnv; gym_mod.spaces = gym_spaces
sys.modules["gymnasium"]        = gym_mod
sys.modules["gymnasium.spaces"] = gym_spaces

# ── Now import project code ───────────────────────────────────────────────────
from env.cityflow_env import CityFlowEnv
from agents.dqn  import DQNAgent
from agents.ppo  import PPOAgent
from agents.iql  import IQLAgent
from agents.qmix import QMIXAgent, MAReplayBuffer
from utils.config import load_config
from utils.logger import Logger

# ── Test runner ───────────────────────────────────────────────────────────────
PASSES, FAILS = [], []
def check(name, fn):
    try:
        fn(); PASSES.append(name); print(f"  ✓  {name}")
    except Exception as e:
        import traceback
        FAILS.append((name, str(e))); print(f"  ✗  {name}: {e}")
        traceback.print_exc()

# ─── Tests ────────────────────────────────────────────────────────────────────
def t_config():
    cfg = load_config("configs/default.yaml")
    assert "dqn" in cfg and "qmix" in cfg and "ppo" in cfg
    assert cfg["env"]["delta_time"] == 5

def t_env_single_reset():
    env = CityFlowEnv(num_intersections=1)
    assert env._mock
    obs, info = env.reset(seed=0)
    assert obs.shape == (17,), f"got {obs.shape}"
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0

def t_env_single_step():
    env = CityFlowEnv(num_intersections=1)
    obs, _ = env.reset()
    obs2, rew, term, trunc, info = env.step(0)
    assert obs2.shape == (17,)
    assert isinstance(float(rew), float)
    assert not term
    assert "avg_delay" in info and "throughput" in info

def t_env_full_episode():
    env = CityFlowEnv(num_intersections=1, min_green=5, max_green=60)
    obs, _ = env.reset(seed=1)
    steps = 0; done = False
    while not done:
        obs, rew, term, trunc, info = env.step(env.action_space.n - 1)
        done = term or trunc; steps += 1
    assert steps == 720, f"Expected 720, got {steps}"

def t_env_multi_reset():
    env = CityFlowEnv(num_intersections=4)
    obs_list, _ = env.reset_multi(seed=0)
    assert len(obs_list) == 4
    assert all(o.shape == (17,) for o in obs_list)
    gs = env.get_global_state()
    assert gs.shape == (68,), f"global state shape: {gs.shape}"

def t_env_multi_step():
    env = CityFlowEnv(num_intersections=4)
    obs_list, _ = env.reset_multi()
    next_obs, rews, term, trunc, info = env.step_multi([0,1,2,3])
    assert len(next_obs) == 4 and len(rews) == 4
    assert all(isinstance(float(r), float) for r in rews)

def t_phase_duration_ticks():
    env = CityFlowEnv(num_intersections=1, delta_time=5, min_green=5, max_green=60)
    env.reset()
    assert env._phase_durations[0] == 0
    env._apply_action(0, 1)   # tick: duration → 5 ≥ min_green → switch
    assert env._current_phases[0]  == 1, "Should switch to phase 1"
    assert env._phase_durations[0] == 0, "Duration resets on switch"

def t_max_green_force():
    env = CityFlowEnv(num_intersections=1, delta_time=5, min_green=5, max_green=10)
    env.reset()
    env._phase_durations[0] = 10  # already at max
    env._apply_action(0, 0)       # try to stay on phase 0
    assert env._current_phases[0] == 1, "max_green should force advance"

def t_reward_types():
    for rt in ("delay", "pressure", "throughput"):
        env = CityFlowEnv(num_intersections=1, reward_type=rt)
        env.reset()
        _, rew, _, _, _ = env.step(0)
        assert isinstance(float(rew), float), f"reward_type={rt} broken"

def t_dqn_steps():
    cfg = load_config("configs/default.yaml")
    cfg["dqn"].update(min_buffer_size=3, batch_size=2)
    agent = DQNAgent(17, 4, cfg["dqn"])
    env   = CityFlowEnv(num_intersections=1)
    obs, _ = env.reset()
    for _ in range(10):
        action = agent.select_action(obs)
        next_obs, rew, term, trunc, info = env.step(action)
        agent.store_transition(obs, action, rew, next_obs, term or trunc)
        agent.update()
        obs = next_obs
    assert agent.steps == 10

def t_dqn_epsilon():
    cfg = load_config("configs/default.yaml")
    agent = DQNAgent(17, 4, cfg["dqn"])
    assert abs(agent._epsilon - cfg["dqn"]["eps_start"]) < 1e-6
    agent.steps = cfg["dqn"]["eps_decay_steps"]
    assert abs(agent._epsilon - cfg["dqn"]["eps_end"]) < 1e-6

def t_ppo_update_triggers():
    cfg = load_config("configs/default.yaml")
    cfg["ppo"].update(rollout_steps=5, ppo_epochs=1, mini_batch_size=3)
    agent = PPOAgent(17, 4, cfg["ppo"])
    env   = CityFlowEnv(num_intersections=1)
    obs, _ = env.reset()
    losses = []
    for _ in range(8):
        action = agent.select_action(obs)
        next_obs, rew, term, trunc, info = env.step(action)
        done = term or trunc
        agent.store_transition(obs, action, rew, next_obs, done)
        loss = agent.update()
        if loss is not None: losses.append(loss)
        obs = next_obs if not done else env.reset()[0]
    assert len(losses) >= 1, "PPO never triggered an update"

def t_iql_steps():
    cfg = load_config("configs/default.yaml")
    cfg["iql"].update(min_buffer_size=3, batch_size=2)
    N = 4
    agent = IQLAgent(17, 4, N, cfg["iql"])
    env   = CityFlowEnv(num_intersections=N)
    obs_list, _ = env.reset_multi()
    for _ in range(10):
        gs   = env.get_global_state()
        acts = agent.select_actions(obs_list)
        nobs, rews, term, trunc, info = env.step_multi(acts)
        done = term or trunc
        ngs  = env.get_global_state()
        agent.store_transition(obs_list, acts, rews, nobs, done, gs, ngs)
        agent.update()
        obs_list = nobs
    assert agent.steps == 10

def t_qmix_update_triggers():
    cfg = load_config("configs/default.yaml")
    cfg["qmix"].update(min_buffer_size=3, batch_size=2)
    N = 4
    agent = QMIXAgent(17, 4, N, 17*N, cfg["qmix"])
    env   = CityFlowEnv(num_intersections=N)
    obs_list, _ = env.reset_multi()
    losses = []
    for _ in range(10):
        gs   = env.get_global_state()
        acts = agent.select_actions(obs_list)
        nobs, rews, term, trunc, info = env.step_multi(acts)
        done = term or trunc
        ngs  = env.get_global_state()
        agent.store_transition(obs_list, acts, rews, nobs, done, gs, ngs)
        loss = agent.update()
        if loss is not None: losses.append(loss)
        obs_list = nobs
    assert len(losses) >= 1, "QMIX never triggered an update"

def t_qmix_buffer_shapes():
    buf = MAReplayBuffer(100)
    N=4; obs_dim=17; state_dim=N*obs_dim
    for _ in range(5):
        obs  = [np.random.rand(obs_dim).astype(np.float32) for _ in range(N)]
        acts = [random.randint(0,3) for _ in range(N)]
        rews = [random.random() for _ in range(N)]
        nobs = [np.random.rand(obs_dim).astype(np.float32) for _ in range(N)]
        gs   = np.random.rand(state_dim).astype(np.float32)
        ngs  = np.random.rand(state_dim).astype(np.float32)
        buf.push(obs, acts, rews, nobs, False, gs, ngs)
    o,a,r,no,d,s,ns = buf.sample(3)
    assert o.shape==(3,N,obs_dim),  f"obs: {o.shape}"
    assert a.shape==(3,N),          f"acts:{a.shape}"
    assert r.shape==(3,N),          f"rew: {r.shape}"
    assert s.shape==(3,state_dim),  f"state:{s.shape}"
    assert ns.shape==(3,state_dim), f"nstate:{ns.shape}"
    assert a.dtype==np.int64,       f"actions must be int64, got {a.dtype}"

def t_greedy_eval_dqn():
    from evaluate import _set_greedy
    cfg = load_config("configs/default.yaml")
    agent = DQNAgent(17, 4, cfg["dqn"])
    _set_greedy(agent)
    assert abs(agent._epsilon - cfg["dqn"]["eps_end"]) < 1e-6

def t_greedy_eval_iql():
    from evaluate import _set_greedy
    cfg = load_config("configs/default.yaml")
    agent = IQLAgent(17, 4, 4, cfg["iql"])
    _set_greedy(agent)
    for a in agent.agents:
        assert abs(a._epsilon - cfg["iql"]["eps_end"]) < 1e-6

def t_eval_single_pipeline():
    from evaluate import evaluate_single
    cfg = load_config("configs/default.yaml")
    cfg["dqn"].update(min_buffer_size=1)
    agent = DQNAgent(17, 4, cfg["dqn"])
    env   = CityFlowEnv(num_intersections=1)
    results = evaluate_single(agent, env, n_episodes=2)
    assert "avg_delay_mean"  in results
    assert "throughput_mean" in results
    assert "reward_mean"     in results
    assert isinstance(results["avg_delay_mean"], float)
    env.close()

def t_eval_multi_pipeline():
    from evaluate import evaluate_multi
    cfg = load_config("configs/default.yaml")
    N = 4
    agent = QMIXAgent(17, 4, N, 17*N, cfg["qmix"])
    env   = CityFlowEnv(num_intersections=N)
    results = evaluate_multi(agent, env, n_episodes=2)
    assert "avg_delay_mean"   in results
    assert "total_reward_mean" in results
    env.close()

def t_train_single_dqn():
    """Mini end-to-end: DQN train loop as called by train.py."""
    import train as tr_mod
    import argparse
    cfg = load_config("configs/default.yaml")
    cfg["dqn"].update(min_buffer_size=3, batch_size=2)
    env   = CityFlowEnv(num_intersections=1)
    agent = DQNAgent(17, 4, cfg["dqn"])
    logger = Logger()
    args  = argparse.Namespace(episodes=3, agent="dqn", save_dir="/tmp/rl_test_ckpt")
    os.makedirs(args.save_dir, exist_ok=True)
    tr_mod.train_single_agent(agent, env, args, logger)
    assert len(logger._records) == 3
    env.close()

def t_train_multi_qmix():
    """Mini end-to-end: QMIX train loop as called by train.py."""
    import train as tr_mod
    import argparse
    cfg = load_config("configs/default.yaml")
    cfg["qmix"].update(min_buffer_size=3, batch_size=2)
    N   = 4
    env = CityFlowEnv(num_intersections=N)
    agent = QMIXAgent(17, 4, N, 17*N, cfg["qmix"])
    logger = Logger()
    args = argparse.Namespace(episodes=3, agent="qmix", save_dir="/tmp/rl_test_ckpt")
    tr_mod.train_multi_agent(agent, env, args, logger)
    assert len(logger._records) == 3
    env.close()

def t_logger():
    logger = Logger()
    logger.log({"episode": 0, "reward": -100.0, "loss": 0.5})
    logger.log({"episode": 1, "reward":  -80.0, "loss": 0.3})
    assert logger.last("reward") == -80.0
    assert logger.last("missing_key", 99) == 99

# ─── Run ──────────────────────────────────────────────────────────────────────
tests = [
    ("Config loading",               t_config),
    ("Env single: reset",            t_env_single_reset),
    ("Env single: step",             t_env_single_step),
    ("Env single: full episode",     t_env_full_episode),
    ("Env multi: reset",             t_env_multi_reset),
    ("Env multi: step",              t_env_multi_step),
    ("Phase duration ticking",       t_phase_duration_ticks),
    ("Max green force-advance",      t_max_green_force),
    ("Reward types: all 3",          t_reward_types),
    ("DQN: train steps",             t_dqn_steps),
    ("DQN: epsilon schedule",        t_dqn_epsilon),
    ("PPO: update triggers",         t_ppo_update_triggers),
    ("IQL: train steps",             t_iql_steps),
    ("QMIX: update triggers",        t_qmix_update_triggers),
    ("QMIX: buffer shapes",          t_qmix_buffer_shapes),
    ("Eval: greedy DQN",             t_greedy_eval_dqn),
    ("Eval: greedy IQL",             t_greedy_eval_iql),
    ("Eval: single pipeline",        t_eval_single_pipeline),
    ("Eval: multi pipeline",         t_eval_multi_pipeline),
    ("Train loop: DQN end-to-end",   t_train_single_dqn),
    ("Train loop: QMIX end-to-end",  t_train_multi_qmix),
    ("Logger",                       t_logger),
]

print("\n" + "="*60)
print("  RL Traffic — Integration Tests")
print("="*60)
for name, fn in tests:
    check(name, fn)
print("\n" + "="*60)
print(f"  {len(PASSES)}/{len(tests)} passed   |   {len(FAILS)} failed")
print("="*60)
if FAILS:
    print("\nFailed tests:")
    for n,e in FAILS: print(f"  ✗ {n}: {e}")
    sys.exit(1)
else:
    print("  All tests passed! ✓")
