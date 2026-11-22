"""Record decision traces for the demo video from trained runs.

Loads the final world model of a C run (verified, k=4) and of a B run (no verification),
replays held-out eval seeds with full tracing, and picks:
  * c_success.pkl  : a successful C episode (prefer one that contains an overturn)
  * c_overturn.pkl : a C episode with a decision where the model's #1 failed verification and
                     a different candidate that passed was committed
  * b_same_seed.pkl: B on the same seed as c_success (for the side-by-side)
Steps spent here are demo-only and are NOT part of any reported experiment number.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.push_env import PushEnv  # noqa: E402
from experiments.common import EVAL_SEEDS  # noqa: E402
from planners.agent import IVAAgent, _train_cfg  # noqa: E402
from verifier.verifier import Spec  # noqa: E402
from world_model.ensemble import DynamicsEnsemble  # noqa: E402

OUT = ROOT / "results" / "demo_traces"


def load_agent(run_dir: Path, seed: int):
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta["config"]
    model = DynamicsEnsemble().load(run_dir / "ensemble.pt")
    return IVAAgent(model, Spec.load(cfg["task"]), _train_cfg(cfg["agent"]), seed=seed), cfg


def has_overturn(rec):
    for i, d in enumerate(rec["trace"]["decisions"]):
        br = d.get("branches")
        if br and not br[0]["passed"] and any(b["passed"] for b in br) and d["chosen"] != 0:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_run", default=str(ROOT / "results/runs/C_verify_k4/seed_0"))
    ap.add_argument("--b_run", default=str(ROOT / "results/runs/B_plan_k1/seed_0"))
    ap.add_argument("--n_seeds", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    c_agent, c_cfg = load_agent(Path(args.c_run), seed=123)
    b_agent, _ = load_agent(Path(args.b_run), seed=123)
    env = PushEnv()
    recs = []
    for sd in EVAL_SEEDS[:args.n_seeds]:
        rec = c_agent.run_episode(env, sd, dataset=None, trace=True)
        ov = has_overturn(rec)
        recs.append((sd, rec, ov))
        print(f"seed {sd}: success={rec['success']} rob={rec['robustness']:.3f} overturn_decision={ov}", flush=True)
    succ = [r for r in recs if r[1]["success"]]
    succ_ov = [r for r in succ if r[2] is not None]
    pick = (succ_ov or succ or recs)[0]
    ov_pick = next((r for r in recs if r[2] is not None), None)
    index = dict(c_run=args.c_run, b_run=args.b_run, c_success_seed=pick[0], c_success_is_success=pick[1]["success"],
                 c_overturn_seed=ov_pick[0] if ov_pick else None,
                 c_overturn_decision=ov_pick[2] if ov_pick else None,
                 c_success_rate_on_scan=float(np.mean([r[1]["success"] for r in recs])))
    with open(OUT / "c_success.pkl", "wb") as f:
        pickle.dump(pick[1], f)
    if ov_pick:
        with open(OUT / "c_overturn.pkl", "wb") as f:
            pickle.dump(ov_pick[1], f)
    b_rec = b_agent.run_episode(PushEnv(), pick[0], dataset=None, trace=True)
    index["b_same_seed_success"] = b_rec["success"]
    with open(OUT / "b_same_seed.pkl", "wb") as f:
        pickle.dump(b_rec, f)
    # Also find a seed where B fails and C succeeds, for a more informative side-by-side.
    for sd, rec, _ in recs:
        if rec["success"]:
            b2 = b_agent.run_episode(PushEnv(), sd, dataset=None, trace=True)
            if not b2["success"]:
                with open(OUT / "c_vs_b_contrast_c.pkl", "wb") as f:
                    pickle.dump(rec, f)
                with open(OUT / "c_vs_b_contrast_b.pkl", "wb") as f:
                    pickle.dump(b2, f)
                index["contrast_seed"] = sd
                break
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
