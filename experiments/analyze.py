"""Phase 6 analysis: tables + plots from results/runs/*, plus results/summary.json for the video.

python -m experiments.analyze [--extra_evals]

--extra_evals additionally evaluates each C run's final world model *without* verification
(k = 1) on the same eval seeds, to separate "better data" from "verification at act time".
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
    "A_ppo_push_safely": "A: PPO",
    "A2_sac_push_safely": "A2: SAC",
    "B_plan_k1": "B: WM+CEM, no verify (k=1)",
    "C_verify_k4": "C: WM+CEM+verify (k=4)",
    "D_verify_k4_lam0": "D: C with lambda=0",
}
BUDGET = 60_000


def ms(x, pct=False, fmt="{:.2f}"):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "-"
    if pct:
        return f"{100 * x.mean():.0f}% ± {100 * x.std():.0f}"
    return (fmt + " ± " + fmt).format(x.mean(), x.std())


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def rl_at_budget(curve, budget):
    rows = [r for r in curve if r["train_steps"] <= budget + 1]
    return rows[-1] if rows else None


def first_cross(steps, values, thr=0.5):
    for s, v in zip(steps, values):
        if v >= thr:
            return s
    return None


def rolling(x, w=10):
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def extra_eval_no_verify(run_dir: Path):
    import torch
    from envs.push_env import PushEnv
    from experiments.common import EVAL_SEEDS, summarize
    from planners.agent import IVAAgent, _train_cfg
    from verifier.verifier import Spec
    from world_model.ensemble import DynamicsEnsemble
    torch.set_num_threads(2)
    meta = load_json(run_dir / "meta.json")
    cfg = meta["config"]
    acfg = _train_cfg(dict(cfg["agent"], k_verify=1))
    agent = IVAAgent(DynamicsEnsemble().load(run_dir / "ensemble.pt"), Spec.load(cfg["task"]), acfg, seed=meta["seed"])
    env = PushEnv()
    recs = [agent.run_episode(env, sd, None) for sd in EVAL_SEEDS]
    return summarize(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_evals", action="store_true")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows_md, table = [], []
    curves = {}
    details = {}
    for key, label in METHODS.items():
        seeds = sorted((RUNS / key).glob("seed_*")) if (RUNS / key).exists() else []
        finals = [(s, load_json(s / "final_eval.json")) for s in seeds]
        finals = [(s, f) for s, f in finals if f]
        if not finals:
            continue
        if key.startswith("A"):
            # Model-free: report the learning-curve eval at the shared budget AND the final eval.
            at_b, fin = [], []
            for s, f in finals:
                lc = load_json(s / "learning_curve.json") or []
                r = rl_at_budget(lc, BUDGET)
                if r:
                    at_b.append(r)
                fin.append(f["summary"])
                curves.setdefault(key, []).append(([r["train_steps"] for r in lc], [r["success_rate"] for r in lc]))
            for name, sums, budget in [(label + f" @ {BUDGET // 1000}k", at_b, BUDGET),
                                       (label + f" @ {fin[0]['train_real_steps'] // 1000}k (final)", fin, fin[0]["train_real_steps"])]:
                if not sums or (budget == BUDGET and key == "A2_sac_push_safely" and name.endswith("(final)")):
                    continue
                table.append(dict(method=name, key=key, seeds=len(sums), budget=f"{budget // 1000}k",
                                  success=ms([x["success_rate"] for x in sums], pct=True),
                                  safety=ms([x["safety_violation_rate"] for x in sums], pct=True),
                                  steps_per_ep="100", committed_per_ep="100", branch_per_ep="0",
                                  overturned="-", top1_failed="-",
                                  viol={c: ms([x.get(f"viol_{c}", np.nan) for x in sums], pct=True)
                                        for c in ("reach_target", "pusher_in_region", "block_slow", "block_on_table")},
                                  raw_success=[x["success_rate"] for x in sums],
                                  raw_safety=[x["safety_violation_rate"] for x in sums]))
            continue
        sums = [f["summary"] for _, f in finals]
        logs = [load_json(s / "train_log.json") for s, _ in finals]
        n_ep = [x["n"] for x in sums]
        steps_ep = [(x["committed_steps"] + x["branch_steps"]) / x["n"] for x in sums]
        ov = [x["eval_stats"]["overturned"] / max(1, x["eval_stats"]["decisions"]) for x in sums]
        t1f = [x["eval_stats"]["top1_failed"] / max(1, x["eval_stats"]["decisions"]) for x in sums]
        train_viol = [np.mean([c["safety_violation"] for c in lg["curve"]]) for lg in logs]
        train_succ = [np.mean([c["success"] for c in lg["curve"]]) for lg in logs]
        to50 = [first_cross([c["real_steps"] for c in lg["curve"]], rolling([c["success"] for c in lg["curve"]])) for lg in logs]
        table.append(dict(method=label, key=key, seeds=len(sums), budget=f"{BUDGET // 1000}k",
                          success=ms([x["success_rate"] for x in sums], pct=True),
                          safety=ms([x["safety_violation_rate"] for x in sums], pct=True),
                          steps_per_ep=f"{np.mean(steps_ep):.0f}",
                          committed_per_ep=f"{np.mean([x['committed_steps'] / x['n'] for x in sums]):.0f}",
                          branch_per_ep=f"{np.mean([x['branch_steps'] / x['n'] for x in sums]):.0f}",
                          overturned=ms(ov, pct=True) if "k1" not in key else "-",
                          top1_failed=ms(t1f, pct=True) if "k1" not in key else "-",
                          viol={c: ms([x.get(f"viol_{c}", np.nan) for x in sums], pct=True)
                                for c in ("reach_target", "pusher_in_region", "block_slow", "block_on_table")},
                          train_episodes=ms([x["train_episodes"] for x in sums], fmt="{:.0f}"),
                          train_success=ms(train_succ, pct=True), train_safety_viol=ms(train_viol, pct=True),
                          steps_to_50=[t for t in to50], train_wall_min=ms([x["train_wall_s"] / 60 for x in sums], fmt="{:.1f}"),
                          raw_success=[x["success_rate"] for x in sums],
                          raw_safety=[x["safety_violation_rate"] for x in sums]))
        details[key] = logs
        if args.extra_evals and key.startswith("C"):
            ex = [extra_eval_no_verify(s) for s, _ in finals]
            table.append(dict(method="C's world model, act WITHOUT verify (k=1) at eval", key=key + "_noverify", seeds=len(ex),
                              budget=f"{BUDGET // 1000}k", success=ms([x["success_rate"] for x in ex], pct=True),
                              safety=ms([x["safety_violation_rate"] for x in ex], pct=True), steps_per_ep="100",
                              committed_per_ep="100", branch_per_ep="0", overturned="-", top1_failed="-",
                              viol={c: ms([x.get(f"viol_{c}", np.nan) for x in ex], pct=True)
                                    for c in ("reach_target", "pusher_in_region", "block_slow", "block_on_table")},
                              raw_success=[x["success_rate"] for x in ex], raw_safety=[x["safety_violation_rate"] for x in ex]))

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"B_plan_k1": "tab:blue", "C_verify_k4": "tab:green", "D_verify_k4_lam0": "tab:olive",
              "A_ppo_push_safely": "tab:red", "A2_sac_push_safely": "tab:purple"}
    for key, logs in details.items():
        for i, lg in enumerate(logs):
            st = [c["real_steps"] for c in lg["curve"]]
            ax[0].plot(st, rolling([c["success"] for c in lg["curve"]]), color=colors[key], alpha=0.8,
                       label=METHODS[key] if i == 0 else None)
            ax[1].plot(st, np.cumsum([c["safety_violation"] for c in lg["curve"]]), color=colors[key], alpha=0.8,
                       label=METHODS[key] if i == 0 else None)
    for key, cl in curves.items():
        for i, (st, sr) in enumerate(cl):
            ax[0].plot(st, sr, color=colors[key], ls="--", alpha=0.8, label=METHODS[key] + " (eval)" if i == 0 else None)
    ax[0].set_xlim(0, BUDGET * 1.05)
    ax[0].set_xlabel("total real env steps (incl. pretraining + branches)")
    ax[0].set_ylabel("success rate (rolling 10 train eps; RL: eval)")
    ax[0].set_title("Learning: success vs real steps")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("total real env steps")
    ax[1].set_ylabel("cumulative training episodes with a\ncommitted safety violation")
    ax[1].set_title("Safety violations during training (committed traj.)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "learning_and_safety.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, logs in details.items():
        for i, lg in enumerate(logs):
            w = lg["wm_err"]
            ax.plot([x["real_steps"] for x in w], [x["err_new_data_before"] for x in w], "o-", color=colors[key], alpha=0.7,
                    label=METHODS[key] if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("total real env steps")
    ax.set_ylabel("1-step error on newly collected data\n(before retrain, L2 over state)")
    ax.set_title("World-model error over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "wm_error_over_time.png", dpi=120)

    # Bar chart: eval success / safety vs real steps per eval episode.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [r["method"] for r in table]
    xs = np.arange(len(table))
    ax.bar(xs - 0.2, [np.mean(r["raw_success"]) for r in table], 0.4,
           yerr=[np.std(r["raw_success"]) for r in table], label="success", color="tab:green")
    ax.bar(xs + 0.2, [np.mean(r["raw_safety"]) for r in table], 0.4,
           yerr=[np.std(r["raw_safety"]) for r in table], label="safety violation", color="tab:red")
    ax.set_xticks(xs, labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-out eval (30 episodes x seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "eval_bars.png", dpi=120)

    # ------------------------------------------------------------------ tables
    hdr = "| Method | seeds | train budget (real steps) | eval success | eval safety-violation eps | real steps / eval ep (committed + branch) | #1 overturned | #1 failed verification |"
    lines = [hdr, "|" + "---|" * 8]
    for r in table:
        lines.append(f"| {r['method']} | {r['seeds']} | {r['budget']} | {r['success']} | {r['safety']} | "
                     f"{r['steps_per_ep']} ({r['committed_per_ep']} + {r['branch_per_ep']}) | {r['overturned']} | {r['top1_failed']} |")
    lines += ["", "Per-clause violation rate on held-out eval episodes:", "",
              "| Method | reach_target | pusher_in_region | block_slow | block_on_table |", "|---|---|---|---|---|"]
    for r in table:
        v = r["viol"]
        lines.append(f"| {r['method']} | {v['reach_target']} | {v['pusher_in_region']} | {v['block_slow']} | {v['block_on_table']} |")
    lines += ["", "Training-phase metrics (model-based methods):", "",
              "| Method | train episodes | train success | train eps w/ committed safety violation | real steps to rolling-10 success >= 50% (per seed) | train wall-clock (min) |",
              "|---|---|---|---|---|---|"]
    for r in table:
        if "train_episodes" in r:
            lines.append(f"| {r['method']} | {r['train_episodes']} | {r['train_success']} | {r['train_safety_viol']} | "
                         f"{', '.join('never' if t is None else str(t) for t in r['steps_to_50'])} | {r['train_wall_min']} |")
    (ROOT / "results" / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    summary = dict(table=[dict(method=r["method"].split(" (")[0] if len(r["method"]) > 34 else r["method"],
                               budget=r["budget"], success=r["success"], safety=r["safety"],
                               steps_per_ep=r["steps_per_ep"], overturned=r["overturned"]) for r in table],
                   notes=["Real steps per eval episode include verification branches (C/D).",
                          "Budget = total real sim steps for training, incl. world-model pretraining.",
                          "MVP scale: 3 seeds, 30 eval episodes per seed; see RESULTS.md for caveats."])
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
