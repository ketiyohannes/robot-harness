"""PPO baseline (Stable-Baselines3) with STL-robustness reward. Checkpointed + resumable."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gym_wrapper import PushGymEnv, augment_obs
from envs.push_env import PushEnv
from experiments.common import EVAL_SEEDS, dump_json, episode_record, git_hash, summarize
from verifier.verifier import Spec


def evaluate_policy(model, spec: Spec, seeds=EVAL_SEEDS):
    env = PushEnv()
    recs = []
    for sd in seeds:
        obs = env.reset(sd)
        states = [env.get_state()]
        done = False
        while not done:
            act, _ = model.predict(augment_obs(obs), deterministic=True)
            obs, done, _ = env.step(act)
            states.append(env.get_state())
        recs.append(episode_record(spec, states, env.target, sd, committed_steps=env.t))
    return recs


class EvalCallback(BaseCallback):
    def __init__(self, spec, every, out_dir: Path, log_rows: list):
        super().__init__()
        self.spec, self.every, self.out_dir, self.rows = spec, every, out_dir, log_rows
        self.next_eval = (len(log_rows) and log_rows[-1]["train_steps"] or 0) + every

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            t0 = time.time()
            s = summarize(evaluate_policy(self.model, self.spec))
            row = dict(train_steps=int(self.num_timesteps), wall=time.time(), **s)
            self.rows.append(row)
            print(json.dumps({k: row[k] for k in ("train_steps", "success_rate", "safety_violation_rate", "mean_robustness")}), flush=True)
            dump_json(self.rows, self.out_dir / "learning_curve.json")
            self.model.save(self.out_dir / "checkpoint.zip")
            del t0
        return True


def run(cfg: dict, seed: int, out_dir: Path):
    torch.set_num_threads(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = Spec.load(cfg["task"])
    venv = DummyVecEnv([lambda i=i: PushGymEnv(cfg["task"], seed=seed * 100 + i) for i in range(cfg.get("n_envs", 4))])
    curve_path = out_dir / "learning_curve.json"
    rows = json.loads(curve_path.read_text()) if curve_path.exists() else []
    ckpt = out_dir / "checkpoint.zip"
    if ckpt.exists() and rows:
        model = PPO.load(ckpt, env=venv, device="cpu")
        print(f"resuming from {ckpt} at {model.num_timesteps} steps")
    else:
        rows = []
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                    n_steps=cfg.get("n_steps", 512), batch_size=cfg.get("batch_size", 256),
                    learning_rate=cfg.get("lr", 3e-4), gamma=cfg.get("gamma", 0.99),
                    gae_lambda=0.95, ent_coef=cfg.get("ent_coef", 0.0), n_epochs=10,
                    policy_kwargs=dict(net_arch=cfg.get("net_arch", [64, 64])))
    remaining = cfg["total_steps"] - model.num_timesteps
    meta = dict(config=cfg, seed=seed, git=git_hash(), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    dump_json(meta, out_dir / "meta.json")
    t0 = time.time()
    if remaining > 0:
        model.learn(remaining, callback=EvalCallback(spec, cfg["eval_every"], out_dir, rows),
                    reset_num_timesteps=False)
    final = evaluate_policy(model, spec)
    summ = summarize(final)
    summ.update(train_real_steps=int(model.num_timesteps), wall_clock_s=time.time() - t0)
    dump_json(dict(summary=summ, episodes=final), out_dir / "final_eval.json")
    model.save(ckpt)
    print("FINAL", json.dumps(summ), flush=True)
    return summ
