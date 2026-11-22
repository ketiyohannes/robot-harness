"""Shared evaluation protocol and bookkeeping helpers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Fixed held-out evaluation episodes (identical for every method).
EVAL_SEEDS = list(range(10_000, 10_030))


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def episode_record(spec, states, target, seed, committed_steps, branch_steps=0, model_steps=0, **extra):
    ok, rob, per = spec.verify(np.asarray(states), target)
    safety_violated = {c: per[c] <= 0 for c in spec.safety_clauses}
    rec = dict(seed=int(seed), success=bool(ok), robustness=float(rob), per_clause=per,
               violated={k: bool(v <= 0) for k, v in per.items()},
               safety_violation=bool(any(safety_violated.values())),
               committed_steps=int(committed_steps), branch_steps=int(branch_steps),
               model_steps=int(model_steps))
    rec.update(extra)
    return rec


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    clauses = records[0]["per_clause"].keys()
    out = dict(
        n=len(records),
        success_rate=float(np.mean([r["success"] for r in records])),
        safety_violation_rate=float(np.mean([r["safety_violation"] for r in records])),
        mean_robustness=float(np.mean([r["robustness"] for r in records])),
        committed_steps=int(sum(r["committed_steps"] for r in records)),
        branch_steps=int(sum(r["branch_steps"] for r in records)),
        model_steps=int(sum(r["model_steps"] for r in records)),
    )
    for c in clauses:
        out[f"viol_{c}"] = float(np.mean([r["violated"][c] for r in records]))
    return out


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
