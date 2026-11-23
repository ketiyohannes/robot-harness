# Morning report — Imagine → Verify → Act MVP

## Summary

The whole loop works end to end on a MuJoCo planar push task:
- learned ensemble world model,
- CEM planning in imagination,
- exact STL verification of the top-k plans by snapshot → real branch → restore,
- committing the best verified plan with a receding horizon,
- online retraining with error-weighted sampling.

I used **MuJoCo, not Unity**: the Unity editor is installed but has no license, and batchmode
exits with code 198. Snapshot/restore is **bit-exact**, so the verifier uses ε = 0.

On push-safely, at an equal budget of 60k real steps (7 seeds):
- Verified branching (C, k = 4) gets **84% ± 5 success with 0% safety violations**.
- Planning without verification (B) gets 83% ± 3 success with 6% ± 4 violations.
- During training, C committed **~24× fewer violating episodes** than B (6.6 vs 158 per seed).
- The price: every eval episode costs **8.6× more real steps** (860 vs 100), branches counted.
  Verification also does **not** reduce the real steps needed to learn. B reaches 50% training
  success slightly sooner.
- PPO (1M steps) and SAC (300k steps) never solved the task.

Video: [`results/demo.mp4`](results/demo.mp4), 83 s, 1280×720, 30 fps. GIF backup:
`results/demo.gif`.

At your request I added a **bigger experiment**:
- 120k-step budgets for B and C,
- a 300k-step SAC run,
- 4 extra seeds for B and C,
- k ∈ {2, 8},
- Task 1 (push).

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 Setup + sanity | done | Unity blocked by licensing → MuJoCo fallback. `scripts/sanity_check.py` |
| 1 Env + snapshot + determinism | done | `envs/push_env.py`. Divergence 0.0 for replay, save/restore and contact. `tests/test_determinism.py` |
| 2 STL specs + verifier | done | Own vectorised STL subset (rtamt too slow for batched planning). `specs/*.yaml`, 8 unit tests |
| 3 PPO baseline | done (fails) | 4 reward variants, 1M steps, 3 seeds: 0% success. Added SAC (60k and 300k): also 0% |
| 4 World model | done | 5-member ensemble, open-loop error + calibration plots (r = 0.62) |
| 5 Imagine→Verify→Act | done | k = 1/2/4/8, λ ablation, traces logged, oracle-model check |
| 6 Comparison | done, extended | A, A2, B, C, D, E, F, 120k runs, Task 1. `RESULTS.md` |
| 7 Demo video | done | Frames rendered programmatically, sampled frames checked (`results/video_frames_check/`) |
| 8 Report + cleanup | done | This file. All processes stopped, caffeinate killed |

## Key numbers (held-out eval: 30 episodes × seeds, push-safely)

| Method | seeds | budget | success | safety-violation episodes | real steps / episode |
|---|---|---|---|---|---|
| A: PPO | 3 | 1M | 0% | 0% | 100 |
| A2: SAC | 3 / 2 | 60k / 300k | 0% / 0% | 9% / 3% | 100 |
| B: WM + CEM, no verify | 7 | 60k | 83% ± 3 | 6% ± 4 | 100 |
| **C: WM + CEM + verify k=4** | 7 | 60k | **84% ± 5** | **0%** | 860 |
| C's model acting without verify | 7 | 60k | 80% ± 5 | 6% ± 3 | 100 |
| D: C, λ = 0 | 3 | 60k | 91% ± 2 | 0% | 860 |
| E: verify k=2 | 3 | 60k | 92% ± 2 | 1% ± 2 | 480 |
| F: verify k=8 | 3 | 60k | 93% ± 0 | 0% | 1620 |
| B / C at 120k | 2 / 2 | 120k | 90% / 97% | 3% / 0% | 100 / 860 |
| Task 1 push: B / C | 2 / 2 | 60k | 100% / 100% | 0% / 0% | 100 / 860 |

Verification overturned the model's #1 plan in 39% of decisions. Only about 6% were real spec
failures (in the traced decisions, 103/131 of those broke block_slow; the rest missed the
target on branches that reached the episode deadline). The rest were cases where another passing candidate scored better
in reality.

## Decisions I made without you

1. **MuJoCo instead of Unity.** Batchmode reports "No valid Unity Editor license" and there is
   no Hub login.
2. **Project root.** I used the existing folder (`robot-harness/`) as the project root instead
   of a nested `imagine-verify-act/`.
3. **Own STL implementation.** Vectorised numpy instead of rtamt, because thousands of imagined
   trajectories are scored per decision.
4. **Task tuning.** Episodes are 5 s (100 steps) and targets are 0.15–0.30 m away. The first
   version, with 3 s episodes, was infeasible even for a scripted controller.
5. **Puck instead of box.** The block is a cylinder puck, so yaw does not matter. The pusher is
   a velocity-controlled cylinder on x/y slide joints.
6. **PPO reward.** Potential-based shaping plus a bounded safety penalty plus terminal STL
   robustness, with relative-position observations. The first two versions taught PPO to avoid
   the block.
7. **Added SAC.** PPO failed, so I added SAC as a stronger model-free baseline.
8. **Branch pass/fail.** A branch passes if the safety clauses hold exactly on its 20-step
   window, and the full spec holds if the branch reaches T. Among passing branches, the best
   real planning score is committed. If none passes, the best real score is committed and
   logged as `no_pass`.
9. **Planner score shaping.** The score adds two search heuristics (terminal distance and an
   approach term) and a 0.05 safety margin. An oracle-model check confirmed this objective
   solves the task when the model is perfect.
10. **B uses the same planner compute as C.** It runs 4 CEM searches and takes the model's #1,
    so B vs C isolates verification.
11. **CPU only.** Many parallel single-thread CPU processes instead of MPS. Around 02:30 I
    pinned `OMP/VECLIB_MAXIMUM_THREADS=1` because the load average had reached about 50.
12. **Bigger experiment (your request).** Long budgets and extra seeds as listed above. Demo
    traces come from C seed 2 and B seed 2 at 60k.

## Known problems / next steps (top 5)

1. **The headline cost is real steps.** Verification costs 8.6× per episode, and k = 2 is
   nearly as good at 4.8×. Next: verify only when ensemble disagreement or the predicted
   margin is low (adaptive k), or verify with a cheaper and slightly inaccurate simulator to
   mimic sim-to-real.
2. **Determinism is perfect here, but won't be in reality.** Add action and observation noise
   or physics perturbations between "branch" and "commit" to test the ε-margin story.
3. **Model-free baselines failed.** Try a curriculum (targets close to the block first), HER,
   or a longer SAC run with tuned entropy, so that A is not a trivial 0%.
4. **More seeds for D/E/F and the 120k runs.** They have only 2–3 seeds; the λ = 0 result
   especially needs confirmation.
5. **Liveness clause.** Use it as a verified filter near the deadline (it is currently ranking
   only), and test harder specs, e.g. "reach then stay" or obstacles.

## Reproduce

```bash
cd robot-harness
uv venv --python python3.13 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python scripts/sanity_check.py
python -m pytest -q tests                       # 14 tests
python scripts/train_world_model.py --ppo_steps 0

# Training + eval (one command per config; seeds are in the config or passed with --seeds)
for c in A_ppo_push_safely A2_sac_push_safely B_plan_k1 C_verify_k4 D_verify_k4_lam0 \
         E_verify_k2 F_verify_k8 B_plan_k1_long C_verify_k4_long A2_sac_long \
         B_plan_k1_push C_verify_k4_push; do
  python -m experiments.run --config experiments/configs/$c.yaml
done
# Main tables used seeds 0-6 for B and C:
python -m experiments.run --config experiments/configs/B_plan_k1.yaml --seeds 3 4 5 6
python -m experiments.run --config experiments/configs/C_verify_k4.yaml --seeds 3 4 5 6

python -m experiments.analyze --extra_evals     # results/table.md, plots, summary.json

# Video
python scripts/record_demo_traces.py --c_run results/runs/C_verify_k4/seed_2 --b_run results/runs/B_plan_k1/seed_2
python scripts/make_video.py                    # results/demo.mp4, results/demo.gif
```

The full timeline, errors and every intermediate result are in `LAB_NOTEBOOK.md`.
