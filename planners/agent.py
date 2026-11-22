"""Imagine -> Verify -> Act agent (and its no-verification ablation, k = 1).

Loop per decision point (receding horizon):
    imagine: K CEM searches in the world model -> K candidate plans ranked by imagined score
    verify : (k > 1) snapshot the sim; for each of the top-k plans: restore, execute the plan
             for real, check the STL safety clauses (exact verifier) + real plan score; restore
    act    : execute the first N steps of the best verified plan (or the model's #1 when k = 1)
All real transitions (committed + branch) go into the dataset; the world model is retrained
every M real steps with error-weighted sampling.

Accounting: env.real_steps counts every physics step. committed / branch / model steps are
tracked separately.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from envs.push_env import PushEnv
from experiments.common import EVAL_SEEDS, dump_json, episode_record, git_hash, summarize
from planners.cem import CEMConfig, CEMPlanner, plan_score
from verifier.verifier import Spec
from world_model.data import OUNoisePolicy, collect
from world_model.ensemble import DynamicsEnsemble


@dataclass
class AgentConfig:
    k_verify: int = 4               # 1 = no verification (method B)
    act_steps: int = 10             # N: committed steps per decision
    eps: float = 0.0                # verification margin (determinism is exact -> 0)
    retrain_every: int = 4000       # M real steps between world-model retrains
    retrain_iters: int = 1500
    error_weighted: bool = True
    cem: CEMConfig = field(default_factory=CEMConfig)


class Dataset:
    def __init__(self):
        self.s, self.a, self.ns = [], [], []

    def add(self, s, a, ns):
        self.s.append(s); self.a.append(a); self.ns.append(ns)

    def add_batch(self, d):
        self.s.extend(d["s"]); self.a.extend(d["a"]); self.ns.extend(d["ns"])

    def arrays(self):
        return np.array(self.s), np.array(self.a), np.array(self.ns)

    def __len__(self):
        return len(self.s)


def _exec(env: PushEnv, actions, dataset: Dataset | None):
    states = [env.get_state()]
    for a in actions:
        s = env.get_state()
        env.step(a)
        ns = env.get_state()
        if dataset is not None:
            dataset.add(s, np.clip(a, -1, 1), ns)
        states.append(ns)
    return np.array(states)


class IVAAgent:
    def __init__(self, model: DynamicsEnsemble, spec: Spec, cfg: AgentConfig, seed: int = 0):
        self.model, self.spec, self.cfg = model, spec, cfg
        self.planner = CEMPlanner(model, spec, cfg.cem, seed=seed)
        self.stats = dict(decisions=0, overturned=0, no_pass=0, top1_failed=0)

    def run_episode(self, env: PushEnv, seed: int, dataset: Dataset | None, trace: bool = False):
        cfg = self.cfg
        env.reset(seed)
        self.planner.reset()
        T = env.episode_steps
        history = [env.get_state()]
        committed = branch = 0
        model0 = self.planner.model_steps
        decisions = []
        shift = 0
        while env.t < T:
            t = env.t
            H = min(cfg.cem.horizon, T - t)
            s0 = env.get_state()
            plan = self.planner.plan(s0, env.target, horizon=H, shift=shift)
            n_act = min(cfg.act_steps, H)
            chosen, dec = 0, dict(t=t, state=s0, target=env.target.copy(),
                                  imagined=plan["imagined"].mean(0)[..., [0, 1, 4, 5]],
                                  imagined_score=plan["score"], imagined_unc=plan["unc"],
                                  ghosts=plan["ghosts"], shortlisted=list(range(min(cfg.k_verify, len(plan["score"])))))
            if cfg.k_verify > 1:
                sid = env.save_snapshot()
                branches = []
                for i in dec["shortlisted"]:
                    env.restore_snapshot(sid)
                    real = _exec(env, plan["actions"][i, :H], dataset)
                    branch += H
                    # Safety clauses checked on the branch window (exact). If the branch reaches the
                    # episode end, the full spec over history + branch is checked as well.
                    ok, rob, per = self.spec.verify(real, env.target, eps=cfg.eps, clauses=self.spec.safety_clauses)
                    full = None
                    if t + H >= T:
                        full = self.spec.verify(np.concatenate([np.array(history), real[1:]]), env.target, eps=cfg.eps)
                        ok = ok and full[0]
                    real_J = float(plan_score(self.spec, real, env.target, cfg.cem)["J"])
                    reach_rob = float(self.spec.batch_clause_robustness(real, env.target)["reach_target"])
                    violated = [k for k, v in per.items() if v <= cfg.eps]
                    if full is not None and not full[0]:
                        violated += [k for k, v in full[2].items() if v <= cfg.eps and k not in violated]
                    branches.append(dict(idx=i, real=real[:, [0, 1, 4, 5]], passed=bool(ok), safety_rob=rob,
                                         reach_rob=reach_rob, real_J=real_J, violated=violated,
                                         per_clause=per))
                env.restore_snapshot(sid)
                env.drop_snapshot(sid)
                passing = [b for b in branches if b["passed"]]
                pool = passing if passing else branches
                best = max(pool, key=lambda b: b["real_J"])
                chosen = best["idx"]
                self.stats["no_pass"] += int(not passing)
                self.stats["overturned"] += int(chosen != 0)
                self.stats["top1_failed"] += int(not branches[0]["passed"])
                dec.update(branches=branches, any_passed=bool(passing))
            self.stats["decisions"] += 1
            seg = _exec(env, plan["actions"][chosen, :n_act], dataset)
            committed += n_act
            history.extend(seg[1:])
            dec.update(chosen=chosen, committed=seg[:, [0, 1, 4, 5]])
            if trace:
                decisions.append(dec)
            # Warm-start the next plan from the chosen sequence, shifted by n_act.
            if self.planner.means is not None and self.planner.means.shape[1] >= H:
                self.planner.means[0, :H] = plan["actions"][chosen]
            shift = n_act
        rec = episode_record(self.spec, np.array(history), env.target, seed, committed, branch,
                             self.planner.model_steps - model0)
        if trace:
            rec["trace"] = dict(decisions=decisions, history=np.array(history), target=env.target.copy(),
                                spec=self.spec.name, k=cfg.k_verify)
        return rec


def _train_cfg(raw: dict) -> AgentConfig:
    cem = CEMConfig(**raw.get("cem", {}))
    keys = {k: v for k, v in raw.items() if k in AgentConfig.__dataclass_fields__ and k != "cem"}
    return AgentConfig(cem=cem, **keys)


def retrain(model: DynamicsEnsemble, data: Dataset, cfg: AgentConfig, seed: int):
    s, a, ns = data.arrays()
    w = None
    if cfg.error_weighted:
        err = model.one_step_error(s, a, ns)
        w = err + err.mean()  # floor keeps every transition in play
    model.fit(s, a, ns, epochs_steps=cfg.retrain_iters, weights=w, seed=seed)


def run(cfg: dict, seed: int, out_dir: Path):
    """Train online within a real-step budget, then evaluate on the fixed eval seeds."""
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("torch_threads", 4))
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = Spec.load(cfg["task"])
    acfg = _train_cfg(cfg.get("agent", {}))
    dump_json(dict(config=cfg, agent=asdict(acfg), seed=seed, git=git_hash(),
                   started=time.strftime("%Y-%m-%d %H:%M:%S")), out_dir / "meta.json")
    env = PushEnv()
    rng = np.random.default_rng(seed)
    t_start = time.time()
    data = Dataset()
    pre = collect(env, OUNoisePolicy(rng), cfg["pretrain_steps"], seed0=1_000_000 + seed * 10_000)
    data.add_batch(pre)
    model = DynamicsEnsemble()
    model.fit(*data.arrays(), epochs_steps=cfg.get("pretrain_iters", 4000), seed=seed)
    agent = IVAAgent(model, spec, acfg, seed=seed)
    budget = cfg["train_budget"]
    curve, wm_err = [], []
    since_retrain = 0
    ep = 0
    trace_dir = out_dir / "traces"
    while env.real_steps < budget:
        before = env.real_steps
        rec = agent.run_episode(env, seed=2_000_000 + seed * 10_000 + ep, dataset=data,
                                trace=(ep % cfg.get("trace_every", 10**9) == 0))
        if "trace" in rec:
            trace_dir.mkdir(exist_ok=True)
            with open(trace_dir / f"train_ep{ep:04d}.pkl", "wb") as f:
                pickle.dump(rec, f)
            rec.pop("trace")
        since_retrain += env.real_steps - before
        curve.append(dict(episode=ep, real_steps=int(env.real_steps), success=rec["success"],
                          safety_violation=rec["safety_violation"], robustness=rec["robustness"],
                          committed=rec["committed_steps"], branch=rec["branch_steps"],
                          wall=time.time() - t_start))
        if since_retrain >= acfg.retrain_every:
            s, a, ns = data.arrays()
            recent = slice(max(0, len(s) - since_retrain), len(s))
            pre_err = float(model.one_step_error(s[recent], a[recent], ns[recent]).mean())
            retrain(model, data, acfg, seed + ep)
            post_err = float(model.one_step_error(s[recent], a[recent], ns[recent]).mean())
            wm_err.append(dict(real_steps=int(env.real_steps), n_data=len(s), err_new_data_before=pre_err,
                               err_new_data_after=post_err))
            since_retrain = 0
        ep += 1
        if ep % 5 == 0:
            sr = np.mean([c["success"] for c in curve[-20:]])
            print(f"ep {ep} real {env.real_steps} succ(last20) {sr:.2f} stats {agent.stats} "
                  f"wall {time.time() - t_start:.0f}s", flush=True)
            dump_json(dict(curve=curve, wm_err=wm_err, stats=agent.stats), out_dir / "train_log.json")
    train_steps = int(env.real_steps)
    train_stats = dict(agent.stats)
    model.save(out_dir / "ensemble.pt")
    dump_json(dict(curve=curve, wm_err=wm_err, stats=train_stats), out_dir / "train_log.json")
    # ---------------------------------------------------------------- evaluation (frozen model)
    agent.stats = dict(decisions=0, overturned=0, no_pass=0, top1_failed=0)
    env_eval = PushEnv()
    t_eval = time.time()
    recs = []
    n_trace = cfg.get("eval_traces", 3)
    for i, sd in enumerate(EVAL_SEEDS[:cfg.get("eval_episodes", len(EVAL_SEEDS))]):
        rec = agent.run_episode(env_eval, sd, dataset=None, trace=i < n_trace)
        if "trace" in rec:
            trace_dir.mkdir(exist_ok=True)
            with open(trace_dir / f"eval_seed{sd}.pkl", "wb") as f:
                pickle.dump(rec, f)
            rec.pop("trace")
        recs.append(rec)
    summ = summarize(recs)
    summ.update(train_real_steps=train_steps, eval_real_steps=int(env_eval.real_steps),
                train_wall_s=t_eval - t_start, eval_wall_s=time.time() - t_eval,
                train_episodes=len(curve), train_stats=train_stats, eval_stats=agent.stats)
    dump_json(dict(summary=summ, episodes=recs), out_dir / "final_eval.json")
    print("FINAL", json.dumps(summ), flush=True)
    return summ
