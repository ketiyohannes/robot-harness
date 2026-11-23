"""Record decision traces (with full physics states, for 3-D rendering) for the arm demo video.

Loads the final world models of a trained arm C run (verified, k = 4) and arm B run (no
verification), replays held-out eval seeds with tracing, and writes to results/arm_demo_traces/:
  * c_success.pkl  : a successful C episode, preferring one that contains an overturn
  * c_overturn.pkl : a C episode with a decision where the model's #1 failed a *safety* clause for
                     real (cup clauses preferred) and a passing alternative was committed
  * contrast_b.pkl / contrast_c.pkl : a seed where B violates a safety clause (or fails) and C
                     succeeds
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.registry import env_info, make_env  # noqa: E402
from experiments.common import EVAL_SEEDS  # noqa: E402
from planners.agent import IVAAgent, _train_cfg  # noqa: E402
from verifier.verifier import Spec  # noqa: E402
from world_model.ensemble import DynamicsEnsemble  # noqa: E402

OUT = ROOT / "results" / "arm_demo_traces"
CUP = ("cup_upright", "cup_still")


def load_agent(run_dir: Path, seed: int):
    cfg = json.loads((run_dir / "meta.json").read_text())["config"]
    spec = Spec.load(cfg["task"])
    info = env_info(spec.env)
    model = DynamicsEnsemble(state_dim=info.state_dim, act_dim=info.act_dim).load(run_dir / "ensemble.pt")
    return IVAAgent(model, spec, _train_cfg(cfg["agent"]), seed=seed), spec


def overturn_decision(rec, safety: list[str]):
    """Index of the best decision where the model's #1 failed a safety clause for real."""
    best, best_score = None, -1
    for i, d in enumerate(rec["trace"]["decisions"]):
        br = d.get("branches")
        if not br or br[0]["passed"] or d["chosen"] == 0 or not br[[b["idx"] for b in br].index(d["chosen"])]["passed"]:
            continue
        bad = set(br[0]["violated"]) & set(safety)
        if not bad:
            continue
        score = 2 * bool(bad & set(CUP)) + 1
        if score > best_score:
            best, best_score = i, score
    return best


def dump(obj, name):
    with open(OUT / name, "wb") as f:
        pickle.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_run", default=str(ROOT / "results/runs/arm_C_verify_k4/seed_0"))
    ap.add_argument("--b_run", default=str(ROOT / "results/runs/arm_B_plan_k1/seed_0"))
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    global OUT
    OUT = Path(args.out) if args.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    c_agent, spec = load_agent(Path(args.c_run), seed=123)
    b_agent, _ = load_agent(Path(args.b_run), seed=123)
    env = make_env(spec.env)
    c_recs, b_recs = {}, {}
    for sd in EVAL_SEEDS[:args.n_seeds]:
        c_recs[sd] = c_agent.run_episode(env, sd, dataset=None, trace=True)
        b_recs[sd] = b_agent.run_episode(env, sd, dataset=None, trace=True)
        c, b = c_recs[sd], b_recs[sd]
        print(f"seed {sd}: C success={c['success']} viol={c['safety_violation']} ov={overturn_decision(c, spec.safety_clauses)} | "
              f"B success={b['success']} violated={[k for k, v in b['violated'].items() if v]}", flush=True)
    seeds = list(c_recs)
    succ = [s for s in seeds if c_recs[s]["success"]]
    with_ov = [s for s in succ if overturn_decision(c_recs[s], spec.safety_clauses) is not None]
    pick = (with_ov or succ or seeds)[0]
    ov_any = [s for s in seeds if overturn_decision(c_recs[s], spec.safety_clauses) is not None]
    ov_cup = [s for s in ov_any if set(c_recs[s]["trace"]["decisions"][overturn_decision(c_recs[s], spec.safety_clauses)]
                                       ["branches"][0]["violated"]) & set(CUP)]
    ov_pick = (ov_cup or ov_any or [None])[0]
    # Contrast: B breaks a cup clause and C succeeds > B breaks any safety clause > B just fails.
    def contrast_rank(s):
        b = b_recs[s]
        if not c_recs[s]["success"] or b["success"]:
            return -1
        return 2 * any(b["violated"][c] for c in CUP) + b["safety_violation"]
    contrast = max(seeds, key=contrast_rank)
    index = dict(c_run=args.c_run, b_run=args.b_run, c_success_seed=pick, c_success_is_success=c_recs[pick]["success"],
                 c_overturn_seed=ov_pick,
                 c_overturn_decision=overturn_decision(c_recs[ov_pick], spec.safety_clauses) if ov_pick else None,
                 contrast_seed=contrast if contrast_rank(contrast) >= 0 else None,
                 scan_c_success=float(np.mean([c_recs[s]["success"] for s in seeds])),
                 scan_c_safety_violation=float(np.mean([c_recs[s]["safety_violation"] for s in seeds])),
                 scan_b_success=float(np.mean([b_recs[s]["success"] for s in seeds])),
                 scan_b_safety_violation=float(np.mean([b_recs[s]["safety_violation"] for s in seeds])))
    dump(c_recs[pick], "c_success.pkl")
    if ov_pick is not None:
        dump(c_recs[ov_pick], "c_overturn.pkl")
    if index["contrast_seed"] is not None:
        dump(c_recs[contrast], "contrast_c.pkl")
        dump(b_recs[contrast], "contrast_b.pkl")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
