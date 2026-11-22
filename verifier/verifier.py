"""STL spec loading and verification for the push tasks.

A spec is a conjunction of named clauses (see ``specs/*.yaml``). ``verify`` returns
``(passed, robustness, per_clause)`` for one trajectory; ``batch_clause_robustness``
scores many (real or imagined) trajectories at once.

Trajectories are state arrays of shape ``(..., T, STATE_DIM)`` *including* the initial
state, plus the target position ``(..., 2)`` (or ``(2,)`` shared by the batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from envs.push_env import BXY, BVEL, BZ, PUSHER_REGION_HALF, PXY, TABLE_HALF, TARGET_RADIUS
from verifier.stl import Always, Eventually, Formula, Predicate

SPEC_DIR = Path(__file__).resolve().parents[1] / "specs"


def _target(ctx):
    tgt = np.asarray(ctx["target"], dtype=np.float64)
    states = ctx["states"]
    if tgt.ndim == 1:
        return tgt
    return tgt[..., None, :]  # broadcast over time


def make_predicates(params: dict) -> dict[str, Predicate]:
    vmax = float(params.get("block_vmax", 0.30))

    def block_in_target(ctx):
        d = np.linalg.norm(ctx["states"][..., BXY] - _target(ctx), axis=-1)
        return TARGET_RADIUS - d

    def pusher_in_region(ctx):
        return PUSHER_REGION_HALF - np.abs(ctx["states"][..., PXY]).max(axis=-1)

    def block_slow(ctx):
        return vmax - np.linalg.norm(ctx["states"][..., BVEL], axis=-1)

    def block_on_table(ctx):
        s = ctx["states"]
        return np.minimum(TABLE_HALF - np.abs(s[..., BXY]).max(axis=-1), 10.0 * (s[..., BZ] - 0.01))

    return {
        "block_in_target": Predicate("block_in_target", block_in_target),
        "pusher_in_region": Predicate("pusher_in_region", pusher_in_region),
        "block_slow": Predicate("block_slow", block_slow),
        "block_on_table": Predicate("block_on_table", block_on_table),
    }


@dataclass
class Clause:
    name: str
    formula: Formula
    kind: str  # "safety" | "liveness"
    text: str


@dataclass
class Spec:
    name: str
    horizon: int
    clauses: list[Clause]
    params: dict = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str) -> "Spec":
        p = Path(name_or_path)
        if not p.exists():
            p = SPEC_DIR / f"{name_or_path}.yaml"
        cfg = yaml.safe_load(p.read_text())
        params = cfg.get("params", {}) or {}
        preds = make_predicates(params)
        clauses = []
        for c in cfg["clauses"]:
            a, b = c.get("interval", [0, cfg["horizon"]])
            pred = preds[c["predicate"]]
            if c["op"] == "always":
                f = Always(pred, a, b)
            elif c["op"] == "eventually":
                f = Eventually(pred, a, b)
            else:
                raise ValueError(f"unsupported op {c['op']}")
            clauses.append(Clause(c["name"], f, c.get("kind", "safety"), str(f)))
        return cls(cfg["name"], int(cfg["horizon"]), clauses, params)

    # ------------------------------------------------------------------
    @property
    def safety_clauses(self) -> list[str]:
        return [c.name for c in self.clauses if c.kind == "safety"]

    def text(self) -> str:
        return "  AND  ".join(c.text for c in self.clauses)

    def batch_clause_robustness(self, states: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
        ctx = {"states": np.asarray(states, dtype=np.float64), "target": target}
        return {c.name: c.formula.robustness(ctx) for c in self.clauses}

    def verify(self, states: np.ndarray, target: np.ndarray, eps: float = 0.0,
               clauses: list[str] | None = None):
        """Exact check of one trajectory. Returns (passed, robustness, per_clause dict).

        ``clauses`` restricts the check to a subset (e.g. safety clauses only for partial
        branch windows). ``eps`` is the determinism margin: pass iff robustness > eps.
        """
        per = self.batch_clause_robustness(states, target)
        per = {k: float(v) for k, v in per.items() if clauses is None or k in clauses}
        rob = min(per.values())
        return rob > eps, rob, per


def verify(traj_states, target, spec: Spec | str = "push", eps: float = 0.0):
    spec = Spec.load(spec) if isinstance(spec, str) else spec
    return spec.verify(traj_states, target, eps=eps)
