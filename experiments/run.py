"""Single entry point: python -m experiments.run --config experiments/configs/<name>.yaml [--seeds 0 1]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)
    seeds = args.seeds if args.seeds else cfg.get("seeds", [0])
    for seed in seeds:
        out = ROOT / "results" / "runs" / name / f"seed_{seed}"
        print(f"=== {name} seed {seed} -> {out}", flush=True)
        if cfg["method"] == "ppo":
            from baselines.ppo import run
        elif cfg["method"] == "iva":
            from planners.agent import run
        else:
            raise ValueError(cfg["method"])
        run(cfg, seed, out)


if __name__ == "__main__":
    main()
