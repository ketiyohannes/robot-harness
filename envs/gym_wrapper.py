"""Gymnasium wrapper around PushEnv for model-free baselines (PPO).

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
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.push_env import ACT_DIM, OBS_DIM, TARGET_RADIUS, PushEnv
from verifier.verifier import Spec


def augment_obs(obs: np.ndarray) -> np.ndarray:
    """12-D env obs -> 16-D: append (block - pusher) and (target - block)."""
    return np.concatenate([obs, obs[4:6] - obs[0:2], obs[10:12] - obs[4:6]]).astype(np.float32)


class PushGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: str = "push_safely", seed: int = 0, safety_penalty: float = 1.0):
        super().__init__()
        self.env = PushEnv()
        self.spec = Spec.load(spec)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM + 4,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
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
        return augment_obs(obs), {}

    def _potential(self, s):
        return float(TARGET_RADIUS - np.linalg.norm(s[4:6] - self.env.target)
                     - 0.5 * np.linalg.norm(s[0:2] - s[4:6]))

    def step(self, action):
        obs, done, info = self.env.step(action)
        s = self.env.get_state()
        self._states.append(s)
        per_step = self.spec.batch_clause_robustness(s[None, :], self.env.target)
        phi = self._potential(s)
        r = 10.0 * (phi - self._phi) + 0.5 * float(per_step["reach_target"] > 0)
        self._phi = phi
        for c in self.spec.safety_clauses:
            r -= self.safety_penalty * float(per_step[c] < 0)
        if done:
            ok, rob, per = self.spec.verify(np.array(self._states), self.env.target)
            r += 10.0 * float(np.clip(rob, -1.0, 1.0))
            info = dict(info, success=ok, robustness=rob, per_clause=per)
        return augment_obs(obs), float(r), False, bool(done), info
