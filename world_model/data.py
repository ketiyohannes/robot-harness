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


class ArmWaypointExplorer:
    """Random exploration for the arm task that actually touches things.

    OU noise in 3-D almost never contacts a 6 cm puck, so the world model never sees a push. This
    policy moves the fingertip toward random waypoints (re-drawn every 8-20 steps) with OU noise
    on top: 55 % near the puck, 15 % near the cup, 30 % anywhere in the workspace; waypoint height
    is near the table 80 % of the time. It uses only the observation (no privileged state) and is
    shared by every arm method, so it does not favour verification.
    """

    def __init__(self, rng, gain=4.0, noise=0.35):
        self.rng, self.gain, self.noise = rng, gain, noise
        self.ou = OUNoisePolicy(rng, sigma=noise, dim=3)
        self.wp, self.left = None, 0

    def reset(self):
        self.ou.reset()
        self.wp, self.left = None, 0

    def _draw(self, obs):
        r = self.rng.random()
        if r < 0.55:
            xy = obs[6:8] + self.rng.normal(0, 0.06, 2)
        elif r < 0.70:
            xy = obs[12:14] + self.rng.normal(0, 0.06, 2)
        else:
            xy = self.rng.uniform(-0.33, 0.33, 2)
        z = self.rng.uniform(0.005, 0.07) if self.rng.random() < 0.8 else self.rng.uniform(0.08, 0.3)
        return np.array([xy[0], xy[1], z])

    def __call__(self, obs):
        if self.left <= 0:
            self.wp, self.left = self._draw(obs), int(self.rng.integers(8, 21))
        self.left -= 1
        a = self.gain * (self.wp - obs[0:3]) / 0.25 + self.ou(obs)
        return np.clip(a, -1, 1)
