"""Gymnasium wrapper around the task envs for model-free baselines (PPO, SAC).

Reward = STL-robustness-based shaping (v2, logged in LAB_NOTEBOOK.md):
    potential  Phi(s) = rho_target(s) - 0.5 * ||pusher - block||
    r_t = 10 * (Phi(s_{t+1}) - Phi(s_t))            (potential-based progress shaping)
          + 0.5 * [rho_target(s_{t+1}) > 0]          (inside-target bonus)
          - p * #{c : rho_safety_c(s_{t+1}) < 0}     (bounded per-step safety penalty, p = config
                                                     `safety_penalty`; v1 used
                                                     5*min(0, rho), which was unbounded when the
                                                     block fell off and taught PPO to avoid it)
    final step additionally: + 10 * clip(rho_spec(trajectory), -1, 1)  (exact episode STL)
rho_target(s) = TARGET_RADIUS - ||block - target|| (the predicate inside F[0,T]).
Observation adds relative features (block - pusher, target - block) to the 12-D obs.
For the arm task the same reward is used with ee/puck in place of pusher/block, and the
observation also gets (cup - ee xy).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.registry import env_info, make_env
from verifier.verifier import Spec


def augment_obs(obs: np.ndarray, env: str = "push") -> np.ndarray:
    """Append relative features: (object - tool), (target - object xy) [, (cup - ee xy) for the arm]."""
    if env == "push":  # 12-D -> 16-D
        return np.concatenate([obs, obs[4:6] - obs[0:2], obs[10:12] - obs[4:6]]).astype(np.float32)
    # arm: state(21) ++ goal(4) -> 32-D
    return np.concatenate([obs, obs[6:9] - obs[0:3], obs[21:23] - obs[6:8], obs[12:14] - obs[0:2]]).astype(np.float32)


def aug_dim(env: str) -> int:
    return {"push": 16, "arm_push": 32}[env]


class PushGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: str = "push_safely", seed: int = 0, safety_penalty: float = 1.0):
        super().__init__()
        self.spec = Spec.load(spec)
        self.info = env_info(self.spec.env)
        self.env = make_env(self.spec.env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (aug_dim(self.spec.env),), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.info.act_dim,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.safety_penalty = safety_penalty
        self._states = []
        self._preds = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = int(seed) if seed is not None else int(self._rng.integers(0, 2**31 - 1))
        obs = self.env.reset(ep_seed)
        self._states = [self.env.get_state()]
        self._phi = self._potential(self._states[-1])
        return augment_obs(obs, self.spec.env), {}

    def _potential(self, s):
        i = self.info
        rho = float(self.spec.batch_clause_robustness(s[None, :], self.env.target)[self.spec.liveness_clause])
        return rho - 0.5 * float(np.linalg.norm(s[list(i.tool_idx)] - s[list(i.object_idx)]))

    def step(self, action):
        obs, done, info = self.env.step(action)
        s = self.env.get_state()
        self._states.append(s)
        per_step = self.spec.batch_clause_robustness(s[None, :], self.env.target)
        phi = self._potential(s)
        r = 10.0 * (phi - self._phi) + 0.5 * float(per_step[self.spec.liveness_clause] > 0)
        self._phi = phi
        for c in self.spec.safety_clauses:
            r -= self.safety_penalty * float(per_step[c] < 0)
        if done:
            ok, rob, per = self.spec.verify(np.array(self._states), self.env.target)
            r += 10.0 * float(np.clip(rob, -1.0, 1.0))
            info = dict(info, success=ok, robustness=rob, per_clause=per)
        return augment_obs(obs, self.spec.env), float(r), False, bool(done), info
