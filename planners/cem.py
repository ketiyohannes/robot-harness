"""CEM planning inside the learned world model.

``K`` independent CEM searches run batched in one model call; each returns its best action
sequence, giving ``K`` structurally different candidate plans. Candidates are ranked by
imagined score; the top-1 is "the model's favourite".

Imagined score of an action sequence (averaged over ensemble members):

    J = rho_reach_window                        # STL robustness of F(block_in_target) over the window
        - w_term  * ||block_H - target||        # progress beyond the window (heuristic)
        - w_app   * min_t ||pusher_t - block_t||  # approach term so plans find the block (heuristic)
        + w_safe  * sum_c min(0, rho_c_window - m)  # safety clauses, tightened by margin m
        - lambda  * mean_t ensemble_std(block_xy, pusher_xy)

The heuristic terms only shape the *search*; success/failure is always decided by the exact
STL verifier on real trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.push_env import ACT_DIM
from verifier.verifier import Spec


@dataclass
class CEMConfig:
    horizon: int = 20
    n_candidates: int = 4       # independent CEM searches (= max k for branching)
    population: int = 128
    elites: int = 12
    iterations: int = 4
    init_std: float = 0.6
    min_std: float = 0.05
    w_term: float = 1.0
    w_app: float = 0.3
    w_safe: float = 10.0
    safe_margin: float = 0.05   # plan against tightened safety predicates (rho_c - margin)
    lam: float = 2.0            # uncertainty penalty lambda
    n_ghosts: int = 16          # population samples kept per decision for visualisation


def plan_score(spec: Spec, states: np.ndarray, target: np.ndarray, cfg: CEMConfig) -> dict:
    """Score trajectories states (..., H+1, S). Returns dict with 'J' (...) and components."""
    per = spec.batch_clause_robustness(states, target)
    d_bt_end = np.linalg.norm(states[..., -1, 4:6] - target, axis=-1)
    d_pb_min = np.linalg.norm(states[..., 0:2] - states[..., 4:6], axis=-1).min(-1)
    safe = sum(np.minimum(0.0, per[c] - cfg.safe_margin) for c in spec.safety_clauses) if spec.safety_clauses else 0.0
    J = per["reach_target"] - cfg.w_term * d_bt_end - cfg.w_app * d_pb_min + cfg.w_safe * safe
    return dict(J=J, per=per)


class CEMPlanner:
    def __init__(self, model, spec: Spec, cfg: CEMConfig, seed: int = 0):
        self.model, self.spec, self.cfg = model, spec, cfg
        self.rng = np.random.default_rng(seed)
        self.means = None
        self.model_steps = 0  # imagined transitions (per ensemble member) computed

    def reset(self):
        self.means = None

    def _imagine(self, state, actions, target):
        traj = self.model.rollout(state, actions)  # (E, N, H+1, S)
        self.model_steps += actions.shape[0] * actions.shape[1]
        sc = plan_score(self.spec, traj, target, self.cfg)
        unc = traj[..., [0, 1, 4, 5]].std(0).mean(-1).mean(-1)  # (N,)
        score = sc["J"].mean(0) - self.cfg.lam * unc
        return traj, score, unc, sc

    def plan(self, state: np.ndarray, target: np.ndarray, horizon: int | None = None, shift: int = 0):
        c = self.cfg
        H = horizon or c.horizon
        K, N = c.n_candidates, c.population
        if self.means is None or self.means.shape[1] < H:
            mu = np.zeros((K, H, ACT_DIM))
        else:
            mu = np.concatenate([self.means[:, shift:], np.zeros((K, shift, ACT_DIM))], 1)[:, :H]
            mu[1:] = 0.5 * mu[1:]  # keep restarts loosely tied to the past, for diversity
        std = np.full((K, H, ACT_DIM), c.init_std)
        best_a = np.zeros((K, H, ACT_DIM))
        best_s = np.full(K, -np.inf)
        for _ in range(c.iterations):
            samples = np.clip(mu[:, None] + std[:, None] * self.rng.normal(size=(K, N, H, ACT_DIM)), -1, 1)
            samples[:, 0] = np.clip(mu, -1, 1)  # always re-evaluate the current mean
            traj, score, unc, _ = self._imagine(state, samples.reshape(K * N, H, ACT_DIM), target)
            score = score.reshape(K, N)
            order = np.argsort(-score, axis=1)
            el = order[:, :c.elites]
            elites = np.take_along_axis(samples, el[:, :, None, None], 1)
            mu = elites.mean(1)
            std = np.maximum(elites.std(1), c.min_std)
            top = order[:, 0]
            improved = score[np.arange(K), top] > best_s
            best_s = np.where(improved, score[np.arange(K), top], best_s)
            best_a[improved] = samples[np.arange(K), top][improved]
            last_traj, last_score, last_samples = traj, score, samples
        self.means = mu
        # Final imagined rollouts of the K candidates (for ranking, logging, video).
        traj, score, unc, sc = self._imagine(state, best_a, target)
        rank = np.argsort(-score)
        ghosts_idx = self.rng.choice(K * N, size=min(c.n_ghosts, K * N), replace=False)
        ghosts = last_traj.mean(0)[ghosts_idx][..., [0, 1, 4, 5]]
        return dict(
            actions=best_a[rank], imagined=traj[:, rank], score=score[rank], unc=unc[rank],
            imagined_per_clause={k: v.mean(0)[rank] for k, v in sc["per"].items()},
            ghosts=ghosts,
        )
