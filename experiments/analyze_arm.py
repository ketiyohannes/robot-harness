"""Arm-task analysis: tables + plots from results/runs/arm_*, plus results/arm_summary.json.

python -m experiments.analyze_arm [--extra_evals]

--extra_evals additionally evaluates each arm C run's final world model *without* verification
(k = 1) on the same eval seeds, to separate "better data" from "verification at act time".
Results are cached in <run>/extra_eval_k1.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "runs"
PLOTS = ROOT / "results" / "plots"

METHODS = {
    "arm_A_ppo": "A: PPO",
    "arm_A2_sac": "A2: SAC",
    "arm_B_plan_k1": "B: WM+CEM, no verify (k=1)",
    "arm_C_verify_k4": "C: WM+CEM+verify (k=4)",
    "arm_D_verify_k4_lam0": "D: C with lambda=0",
    "arm_E_verify_k2": "E: verify k=2",
    "arm_G_verify_k4_stop": "G: C + stop if no branch passes",
    "arm_B_plan_k1_long": "B: no verify, 2x budget",
    "arm_C_verify_k4_long": "C: verify k=4, 2x budget",
}
COLORS = {"arm_B_plan_k1": "#d9822b", "arm_C_verify_k4": "#2b7bd9", "arm_E_verify_k2": "#7b4bd9", "arm_D_verify_k4_lam0": "#2bb0b0",
          "arm_G_verify_k4_stop": "#2b9d5b",
          "arm_B_plan_k1_long": "#f0b070", "arm_C_verify_k4_long": "#80b0f0", "arm_A2_sac": "#888888",
          "arm_A_ppo": "#444444"}


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ms(x, pct=False):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "-"
    return f"{100 * x.mean():.0f}% ± {100 * x.std():.0f}" if pct else f"{x.mean():.2f} ± {x.std():.2f}"


def rolling(x, w=10):
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def first_cross(steps, values, thr=0.5):
    return next((s for s, v in zip(steps, values) if v >= thr), None)


def extra_eval_no_verify(run_dir: Path) -> dict:
    cache = run_dir / "extra_eval_k1.json"
    if cache.exists():
        return load_json(cache)
    import torch
    from envs.registry import env_info, make_env
    from experiments.common import EVAL_SEEDS, dump_json, summarize
    from planners.agent import IVAAgent, _train_cfg
    from verifier.verifier import Spec
    from world_model.ensemble import DynamicsEnsemble
    torch.set_num_threads(1)
    meta = load_json(run_dir / "meta.json")
    cfg = meta["config"]
    spec = Spec.load(cfg["task"])
    info = env_info(spec.env)
    model = DynamicsEnsemble(state_dim=info.state_dim, act_dim=info.act_dim).load(run_dir / "ensemble.pt")
    agent = IVAAgent(model, spec, _train_cfg(dict(cfg["agent"], k_verify=1)), seed=meta["seed"])
    env = make_env(spec.env)
    out = summarize([agent.run_episode(env, sd, None) for sd in EVAL_SEEDS[:cfg.get("eval_episodes", 30)]])
    dump_json(out, cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_evals", action="store_true")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    table, train_rows, clause_rows, curves = [], [], [], {}
    clause_names = None
    for key, label in METHODS.items():
        seeds = sorted((RUNS / key).glob("seed_*")) if (RUNS / key).exists() else []
        finals = [(s, load_json(s / "final_eval.json")) for s in seeds]
        finals = [(s, f) for s, f in finals if f]
        if not finals:
            continue
        sums = [f["summary"] for _, f in finals]
        budget = sums[0]["train_real_steps"]
        n_eval = sums[0]["n"]
        steps_ep = (sums[0]["committed_steps"] + sums[0]["branch_steps"]) / n_eval
        row = dict(method=label, key=key, seeds=len(sums), budget=f"{round(budget / 1000)}k",
                   success=ms([x["success_rate"] for x in sums], pct=True),
                   safety=ms([x["safety_violation_rate"] for x in sums], pct=True),
                   steps_per_ep=f"{steps_ep:.0f}",
                   success_raw=[x["success_rate"] for x in sums],
                   safety_raw=[x["safety_violation_rate"] for x in sums])
        st = [x.get("eval_stats") for x in sums if x.get("eval_stats")]
        if st and st[0]["decisions"] and "verify" in key:
            row["overturned"] = ms([s["overturned"] / s["decisions"] for s in st], pct=True)
            row["top1_failed"] = ms([s["top1_failed"] / s["decisions"] for s in st], pct=True)
        table.append(row)
        clause_names = [k[5:] for k in sums[0] if k.startswith("viol_")]
        clause_rows.append((label, {c: ms([x[f"viol_{c}"] for x in sums], pct=True) for c in clause_names}))
        if key.startswith("arm_A"):
            for s, _ in finals:
                lc = load_json(s / "learning_curve.json") or []
                curves.setdefault(key, []).append(("eval", [r["train_steps"] for r in lc], [r["success_rate"] for r in lc], None))
            continue
        tr_viol, cross, n_eps, wall = [], [], [], []
        for s, f in finals:
            log = load_json(s / "train_log.json")
            c = log["curve"]
            steps = [r["real_steps"] for r in c]
            succ = rolling([r["success"] for r in c])
            viol = np.cumsum([r["safety_violation"] for r in c])
            curves.setdefault(key, []).append(("train", steps, succ, viol))
            tr_viol.append(int(viol[-1]) if len(viol) else 0)
            cross.append(first_cross(steps, succ))
            n_eps.append(len(c))
            wall.append(f["summary"]["train_wall_s"] / 60)
        train_rows.append(dict(method=label, episodes=f"{np.mean(n_eps):.0f}", viol=ms(tr_viol),
                               viol_frac=f"{100 * np.mean(np.array(tr_viol) / np.array(n_eps)):.0f}%",
                               cross=", ".join("never" if x is None else f"{x / 1000:.1f}k" for x in cross),
                               wall=ms(wall)))
        if args.extra_evals and "verify" in key and not key.endswith("_long"):
            ex = [extra_eval_no_verify(s) for s, _ in finals]
            table.append(dict(method=f"{label}: its model acting WITHOUT verify", key=key + "_noverify", seeds=len(ex),
                              budget=row["budget"], success=ms([x["success_rate"] for x in ex], pct=True),
                              safety=ms([x["safety_violation_rate"] for x in ex], pct=True),
                              steps_per_ep=f"{(ex[0]['committed_steps'] + ex[0]['branch_steps']) / ex[0]['n']:.0f}",
                              success_raw=[x["success_rate"] for x in ex], safety_raw=[x["safety_violation_rate"] for x in ex]))
            clause_rows.append((f"{label}: model without verify", {c: ms([x[f"viol_{c}"] for x in ex], pct=True) for c in clause_names}))

    # ------------------------------------------------------------------ markdown
    md = ["# Arm task results (arm_push_safely)", "",
          "| Method | seeds | train budget | success | episodes with a safety violation | real steps / eval episode | model's #1 overturned | model's #1 failed verification |",
          "|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['method']} | {r['seeds']} | {r['budget']} | {r['success']} | {r['safety']} | {r['steps_per_ep']} | "
                  f"{r.get('overturned', '–')} | {r.get('top1_failed', '–')} |")
    md += ["", "## Training phase", "",
           "| Method | training episodes | training episodes with a safety violation (per seed) | share | real steps to rolling-10 success ≥ 50% | train wall-clock (min) |",
           "|---|---|---|---|---|---|"]
    for r in train_rows:
        md.append(f"| {r['method']} | {r['episodes']} | {r['viol']} | {r['viol_frac']} | {r['cross']} | {r['wall']} |")
    if clause_names:
        md += ["", "## Per-clause violation rate (eval episodes)", "",
               "| Method | " + " | ".join(clause_names) + " |", "|---" * (len(clause_names) + 1) + "|"]
        for lab, v in clause_rows:
            md.append(f"| {lab} | " + " | ".join(v[c] for c in clause_names) + " |")
    (ROOT / "results" / "arm_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for key, runs in curves.items():
        col = COLORS.get(key, "k")
        for i, (kind, steps, succ, viol) in enumerate(runs):
            lab = METHODS[key] if i == 0 else None
            ax[0].plot(np.asarray(steps) / 1000, succ, color=col, alpha=0.8, lw=1.4, label=lab,
                       ls="-" if kind == "train" else "--", marker=None if kind == "train" else "o", ms=3)
            if viol is not None:
                ax[1].plot(np.asarray(steps) / 1000, viol, color=col, alpha=0.8, lw=1.4, label=lab)
    ax[0].set_xlabel("total real env steps (k), incl. verification branches")
    ax[0].set_ylabel("success (rolling 10 train episodes; SAC/PPO: eval)")
    ax[0].set_title("Learning: success vs real steps")
    ax[1].set_xlabel("total real env steps (k)")
    ax[1].set_ylabel("cumulative training episodes with a safety violation")
    ax[1].set_title("Safety during training (executed steps only)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "arm_learning_and_safety.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = [r["method"] for r in table]
    x = np.arange(len(names))
    sm = [np.mean(r["success_raw"]) for r in table]
    ss = [np.std(r["success_raw"]) for r in table]
    vm = [np.mean(r["safety_raw"]) for r in table]
    vs = [np.std(r["safety_raw"]) for r in table]
    ax.bar(x - 0.2, sm, 0.4, yerr=ss, label="success", color="#2b9d5b")
    ax.bar(x + 0.2, vm, 0.4, yerr=vs, label="episodes with safety violation", color="#d94a3b")
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Arm: held-out evaluation (30 episodes per seed)")
    fig.tight_layout()
    fig.savefig(PLOTS / "arm_eval_bars.png", dpi=130)

    summary = dict(table=[{k: v for k, v in r.items() if not k.endswith("_raw")} for r in table], train=train_rows,
                   notes=["Real steps include every verification branch step.",
                          "Same 30 held-out scenes for every method; world model frozen during evaluation."])
    (ROOT / "results" / "arm_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
