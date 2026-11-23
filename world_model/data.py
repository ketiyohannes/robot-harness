"""Transition collection and on-disk .npz shards."""
from __future__ import annotations

from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class OUNoisePolicy:
    """Temporally correlated random actions (random exploration that actually travels)."""

    def __init__(self, rng, theta=0.15, sigma=0.5, dim=2):
        self.rng, self.theta, self.sigma, self.dim = rng, theta, sigma, dim
        self.x = np.zeros(dim)

    def reset(self):
        self.x = self.rng.uniform(-1, 1, self.dim)

    def __call__(self, obs):
        self.x += -self.theta * self.x + self.sigma * self.rng.normal(size=self.dim)
        return np.clip(self.x, -1, 1)


class UniformPolicy:
    def __init__(self, rng, dim=2):
        self.rng, self.dim = rng, dim

    def reset(self):
        pass

    def __call__(self, obs):
        return self.rng.uniform(-1, 1, self.dim)


def collect(env, policy, n_steps: int, seed0: int):
    """Roll out full episodes until n_steps transitions. Returns dict of arrays."""
    S, A, NS, EP = [], [], [], []
    ep = 0
    while len(S) < n_steps:
        obs = env.reset(seed0 + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done and len(S) < n_steps:
            s = env.get_state()
            a = np.clip(np.asarray(policy(obs), dtype=np.float64), -1, 1)
            obs, done, _ = env.step(a)
            S.append(s); A.append(a); NS.append(env.get_state()); EP.append(seed0 + ep)
        ep += 1
    return dict(s=np.array(S), a=np.array(A), ns=np.array(NS), ep=np.array(EP))


def save_shard(data: dict, name: str):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.npz"
    np.savez_compressed(path, **data)
    return path


def load_shards(names) -> dict:
    parts = [np.load(DATA_DIR / f"{n}.npz") for n in names]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def concat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}
