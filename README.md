# Imagine → Verify → Act (MVP)

World-model search plus STL-verified branching on a MuJoCo planar push task.

1. **Imagine.** A 5-member MLP dynamics ensemble is learned online. K = 4 batched CEM
   searches plan action sequences inside the model, and each search returns one candidate.
2. **Verify.** The sim is snapshotted. The top-k candidates are executed for real from the
   snapshot, and each one is checked exactly against a Signal Temporal Logic spec
   (pass/fail, robustness, per-clause robustness). Then the snapshot is restored.
3. **Act.** The agent commits the first N steps of the best verified candidate and then
   re-plans (receding horizon). Every real transition, branches included, goes into the
   dataset, and the model is retrained with error-weighted sampling.

Every real sim step is counted, including verification branches.

Start with `MORNING_REPORT.md` for status and results, `RESULTS.md` for the analysis, and
`LAB_NOTEBOOK.md` for the timeline, errors and decisions.

## Layout

```
envs/          push_env.py (backend-agnostic API, MuJoCo backend), gym_wrapper.py (for SB3)
specs/         push.yaml, push_safely.yaml (STL specs)
verifier/      stl.py (vectorised STL robustness), verifier.py (Spec.verify)
world_model/   ensemble.py (dynamics ensemble), data.py (collection, .npz shards)
planners/      cem.py (K parallel CEM searches), agent.py (Imagine→Verify→Act loop),
               oracle_model.py (diagnostic perfect model)
baselines/     ppo.py, sac.py (Stable-Baselines3)
experiments/   run.py (single entry point), configs/*.yaml, analyze.py, common.py
scripts/       sanity_check.py, train_world_model.py, record_demo_traces.py, make_video.py
tests/         determinism, verifier
results/       runs/<config>/seed_<s>/..., plots/, table.md, demo.mp4, demo.gif, demo_traces/
```

## Reproduce

```bash
uv venv --python python3.13 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

python scripts/sanity_check.py                 # env + render sanity check
python -m pytest -q tests                      # determinism + verifier tests
python scripts/train_world_model.py --ppo_steps 0   # Phase 4 world-model study

# Phase 6 comparison (each config is one command; seeds are listed in the config)
python -m experiments.run --config experiments/configs/A_ppo_push_safely.yaml
python -m experiments.run --config experiments/configs/A2_sac_push_safely.yaml
python -m experiments.run --config experiments/configs/B_plan_k1.yaml
python -m experiments.run --config experiments/configs/C_verify_k4.yaml
python -m experiments.run --config experiments/configs/D_verify_k4_lam0.yaml
python -m experiments.analyze --extra_evals    # tables + plots + results/summary.json

# Demo video
python scripts/record_demo_traces.py           # traces from trained C and B runs
python scripts/make_video.py                   # results/demo.mp4 + results/demo.gif
```
