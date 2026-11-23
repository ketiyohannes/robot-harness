"""Diagnostic: plan with a perfect model (private sim copy) to check the planning objective.

python scripts/oracle_check.py --spec arm_push_safely --seeds 10000 10001 --k 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.registry import make_env  # noqa: E402
from planners.agent import AgentConfig, IVAAgent  # noqa: E402
from planners.cem import CEMConfig  # noqa: E402
from planners.oracle_model import OracleModel  # noqa: E402
from verifier.verifier import Spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="arm_push_safely")
    ap.add_argument("--seeds", type=int, nargs="+", default=[10000])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--population", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()
    spec = Spec.load(args.spec)
    env = make_env(spec.env)
    cem = CEMConfig(population=args.population, iterations=args.iterations, elites=8, lam=0.0)
    agent = IVAAgent(OracleModel(env), spec, AgentConfig(k_verify=args.k, cem=cem))
    for sd in args.seeds:
        rec = agent.run_episode(env, sd, dataset=None)
        print(json.dumps(dict(seed=sd, success=rec["success"], rob=round(rec["robustness"], 3),
                              violated=[k for k, v in rec["violated"].items() if v])), flush=True)


if __name__ == "__main__":
    main()
