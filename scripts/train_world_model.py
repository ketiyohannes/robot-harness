"""Phase 4: collect data, train the dynamics ensemble, evaluate open-loop error + calibration.

Training data: OU-noise random exploration (+ PPO-checkpoint rollouts when a checkpoint
exists). Held-out evaluation data additionally contains *scripted pushing* episodes so that
contact dynamics are actually measured (scripted data is never used for training).

Outputs: data/*.npz shards, results/world_model/{ensemble.pt, metrics.json, *.png}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.push_env import PushEnv  # noqa: E402
from world_model.data import OUNoisePolicy, collect, concat, save_shard  # noqa: E402
from world_model.ensemble import DynamicsEnsemble  # noqa: E402

OUT = ROOT / "results" / "world_model"


class ScriptedPush:
    """Hand-written pusher used ONLY to create contact-rich held-out evaluation data."""

    def __init__(self, rng, k=0.6, noise=0.3):
        self.rng, self.k, self.noise = rng, k, noise

    def reset(self):
        pass

    def __call__(self, o):
        p, b, t = o[0:2], o[4:6], o[10:12]
        d = (t - b) / (np.linalg.norm(t - b) + 1e-9)
        v = (b - 0.075 * d) - p
        if np.linalg.norm(v) < 0.02:
            v = d * np.linalg.norm(t - b)
        a = self.k * v / (0.05 + np.linalg.norm(v)) + self.noise * self.rng.normal(size=2)
        return np.clip(a, -1, 1)


class PPOPolicy:
    def __init__(self, path, rng, noise=0.2):
        from stable_baselines3 import PPO
        from envs.gym_wrapper import augment_obs
        self.m, self.aug, self.rng, self.noise = PPO.load(path, device="cpu"), augment_obs, rng, noise

    def reset(self):
        pass

    def __call__(self, o):
        a, _ = self.m.predict(self.aug(o), deterministic=True)
        return np.clip(a + self.noise * self.rng.normal(size=2), -1, 1)


def open_loop_errors(model: DynamicsEnsemble, env: PushEnv, policy, n_eps: int, seed0: int, horizons=(1, 10, 25)):
    """Run real episodes, then from every 5th step roll the model open-loop over the real actions."""
    errs = {h: [] for h in horizons}
    unc_err = []  # (horizon, ensemble std of block xy, error of mean block xy)
    H = max(horizons)
    for ep in range(n_eps):
        obs = env.reset(seed0 + ep)
        policy.reset()
        S, A = [env.get_state()], []
        done = False
        while not done:
            a = policy(obs)
            obs, done, _ = env.step(a)
            A.append(a)
            S.append(env.get_state())
        S, A = np.array(S), np.array(A)
        starts = list(range(0, len(A) - H, 5))
        acts = np.stack([A[t:t + H] for t in starts])
        pred = model.rollout(S[starts], acts)  # (E, N, H+1, S)
        mean = pred.mean(0)
        for i, t0 in enumerate(starts):
            for h in horizons:
                real = S[t0 + h]
                errs[h].append([np.linalg.norm(mean[i, h, 0:2] - real[0:2]), np.linalg.norm(mean[i, h, 4:6] - real[4:6])])
            for h in range(1, H + 1):
                std = pred[:, i, h, 4:6].std(0).mean()
                unc_err.append([h, std, np.linalg.norm(mean[i, h, 4:6] - S[t0 + h][4:6])])
    return {h: np.array(v) for h, v in errs.items()}, np.array(unc_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_steps", type=int, default=20000)
    ap.add_argument("--ppo_steps", type=int, default=10000)
    ap.add_argument("--ppo_ckpt", default=str(ROOT / "results/runs/ppo_push_safely/seed_0/checkpoint.zip"))
    ap.add_argument("--train_iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    env = PushEnv()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    parts = [collect(env, OUNoisePolicy(rng), args.random_steps, seed0=100_000)]
    save_shard(parts[0], "wm_random_ou")
    sources = {"random_ou": args.random_steps}
    if Path(args.ppo_ckpt).exists() and args.ppo_steps > 0:
        parts.append(collect(env, PPOPolicy(args.ppo_ckpt, rng), args.ppo_steps, seed0=200_000))
        save_shard(parts[-1], "wm_ppo")
        sources["ppo_checkpoint"] = args.ppo_steps
    data = concat(*parts)
    print(f"collected {len(data['s'])} transitions in {time.time() - t0:.1f}s: {sources}", flush=True)

    model = DynamicsEnsemble()
    t1 = time.time()
    loss = model.fit(data["s"], data["a"], data["ns"], epochs_steps=args.train_iters, verbose=True, seed=args.seed)
    train_s = time.time() - t1
    model.save(OUT / "ensemble.pt")
    print(f"trained in {train_s:.1f}s, final loss {loss:.4f}", flush=True)

    metrics = dict(sources=sources, n_train=int(len(data["s"])), train_iters=args.train_iters,
                   train_seconds=train_s, final_loss=loss)
    for name, pol in [("random_ou", OUNoisePolicy(np.random.default_rng(1))),
                      ("scripted_push", ScriptedPush(np.random.default_rng(2)))]:
        errs, ue = open_loop_errors(model, env, pol, n_eps=20, seed0=300_000)
        metrics[name] = {f"h{h}": dict(pusher_mean=float(v[:, 0].mean()), block_mean=float(v[:, 1].mean()),
                                        block_p90=float(np.percentile(v[:, 1], 90))) for h, v in errs.items()}
        if name == "scripted_push":
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for h in (1, 5, 10, 25):
                sel = ue[ue[:, 0] == h]
                ax[0].scatter(sel[:, 1], sel[:, 2], s=6, alpha=0.5, label=f"h={h}")
            ax[0].set_xscale("log"); ax[0].set_yscale("log")
            ax[0].set_xlabel("ensemble disagreement (std of block xy, m)")
            ax[0].set_ylabel("actual error of ensemble mean (m)")
            ax[0].set_title("Uncertainty vs error (contact-rich held-out)")
            ax[0].legend()
            hs = np.arange(1, 26)
            ax[1].plot(hs, [ue[ue[:, 0] == h][:, 2].mean() for h in hs], label="block error")
            ax[1].plot(hs, [ue[ue[:, 0] == h][:, 1].mean() for h in hs], label="ensemble std")
            ax[1].set_xlabel("open-loop horizon (steps)"); ax[1].set_ylabel("m"); ax[1].legend()
            ax[1].set_title("Open-loop block error vs horizon")
            fig.tight_layout(); fig.savefig(OUT / "uncertainty_vs_error.png", dpi=120)
            corr = np.corrcoef(np.log(ue[:, 1] + 1e-6), np.log(ue[:, 2] + 1e-6))[0, 1]
            metrics[name]["log_unc_err_corr"] = float(corr)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
