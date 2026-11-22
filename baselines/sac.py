"""SAC baseline (Stable-Baselines3): a stronger, off-policy model-free reference.

Same environment, observation features and STL-shaped reward as the PPO baseline
(envs/gym_wrapper.py). Added because PPO did not learn the task within its budgets; see
LAB_NOTEBOOK.md. Checkpointed every eval; resumable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines.ppo import EvalCallback, evaluate_policy
from envs.gym_wrapper import PushGymEnv
from experiments.common import dump_json, git_hash, summarize
from verifier.verifier import Spec


def run(cfg: dict, seed: int, out_dir: Path):
    torch.set_num_threads(cfg.get("torch_threads", 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = Spec.load(cfg["task"])
    venv = DummyVecEnv([lambda: PushGymEnv(cfg["task"], seed=seed * 100, safety_penalty=cfg.get("safety_penalty", 1.0))])
    curve_path = out_dir / "learning_curve.json"
    rows = json.loads(curve_path.read_text()) if curve_path.exists() else []
    ckpt = out_dir / "checkpoint.zip"
    if ckpt.exists() and rows:
        model = SAC.load(ckpt, env=venv, device="cpu")
        print(f"resuming from {ckpt} at {model.num_timesteps} steps (replay buffer not restored)")
    else:
        rows = []
        model = SAC("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                    learning_rate=cfg.get("lr", 3e-4), buffer_size=cfg["total_steps"],
                    learning_starts=cfg.get("learning_starts", 2000), batch_size=256,
                    gamma=cfg.get("gamma", 0.99), tau=0.005, train_freq=1, gradient_steps=1,
                    policy_kwargs=dict(net_arch=cfg.get("net_arch", [256, 256])))
    dump_json(dict(config=cfg, seed=seed, git=git_hash(), started=time.strftime("%Y-%m-%d %H:%M:%S")),
              out_dir / "meta.json")
    t0 = time.time()
    remaining = cfg["total_steps"] - model.num_timesteps
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
