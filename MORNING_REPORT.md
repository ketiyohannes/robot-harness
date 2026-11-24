# Morning report — night 2: robot arm (Franka Panda)

## Summary

The Imagine → Verify → Act loop now runs on a 7-joint **Franka Panda arm** in MuJoCo, end to
end. The task: push a puck into a target zone within 6 s without hitting it too hard, tipping or
moving the cup next to it, or leaving the workspace. All of these are STL clauses, checked
exactly.

At an equal budget of **100k real sim steps** (5 seeds each, branch steps counted):

| | success | test runs breaking a safety rule | real steps per test run |
|---|---|---|---|
| B: plan in the learned model, act without checking | 33% ± 4 | **51% ± 6** | 120 |
| **C: check the top 4 plans for real, then act** | **70% ± 2** | **1% ± 2** | 1040 |
| C's own model, acting without checking | 28% ± 7 | 49% ± 5 | 120 |
| G: C, but hold still when no plan passes (3 seeds) | 83% ± 5 | 0% | 1040 |
| SAC (100k) / PPO (1M) | 0% / 0% | 7% / 0% | 120 |

- **Safety.** Checking cuts rule-breaking from 51% to 1% in testing, and from 400 to 1.6 bad
  training episodes per seed. B's violations are mostly hitting the puck too hard: its model
  under-predicts how fast the puck flies off.
- **Success.** Unlike the flat task, checking also **doubles success** on the arm (70% vs 33%;
  83% vs 43% at 200k).
- **Cost.** It is the same as before: **8.7× more real steps per episode**, and no faster
  learning in real steps (both reach 50% training success after about 32–66k steps).

Video: [`results/arm_demo.mp4`](results/arm_demo.mp4), 81 s, 1280×720, 30 fps. GIF:
`results/arm_demo.gif`. Full analysis: [`RESULTS_ARM.md`](RESULTS_ARM.md).

## What went wrong overnight (and was fixed)

1. **A tall cup was unavoidable.** The Panda hand is 20 cm wide and swept an 11 cm cup in most
   scripted pushes. I made the cup 6 cm tall.
2. **Snapshot/restore was not exact at first.** The controller read stale kinematics after
   `mj_step`. Fixed, and tested bit-exact.
3. **The first pilot learned nothing (0%).** Random 3-D exploration touched the puck in only 0.2%
   of steps, so the model never learned that pushing moves it. Added a waypoint explorer
   (16% contact), shared by every method including SAC.
4. **The first full run (32 jobs) was invalid.** The reset IK diverged on about 10% of scenes and
   the arm flailed at t = 0. I found it because every method failed the same 3 eval scenes.
   Fixed, checked on 430 scenes, archived the bad results (`results/archive/arm_v1_ik_bug/`,
   not reported) and reran everything.

## Decisions I made without you

- Robot: **Franka Panda from MuJoCo Menagerie** (Apache-2.0). It is fetched by
  `scripts/fetch_assets.sh` into `third_party/`, which is gitignored.
- Control: **3-D fingertip velocity with a differential-IK controller**, gripper closed and
  used as the pusher. I did not try grasping; pick-and-place is a possible next step.
- The cup is short (6 cm), and the task is 6 s long.
- Exploration prior (`ArmWaypointExplorer`) for every arm method; see the limitations in
  `RESULTS_ARM.md`.
- Added the **"hold still if no branch passes"** variant (G) after seeing that C's remaining
  violations came from committing a failing plan when all 4 candidates failed.
- Added **B with λ = 0**. The uncertainty penalty was not why B underperforms (32% vs 33%).

## Next steps

1. Make G (hold still when nothing passes) the default, and run more seeds.
2. Check the branches in a *perturbed* simulator (different friction or masses) to mimic
   sim-to-real. This is the key open question.
3. Adaptive k: verify only when the model is unsure. k = 2 already gives 68% success with 6%
   violations at 580 steps.
4. Pick-and-place with the gripper (4-D action).

## Reproduce (arm)

```bash
scripts/fetch_assets.sh                                   # Panda model -> third_party/
python -m pytest -q tests                                 # 18 tests incl. arm determinism
python scripts/arm_scripted.py --episodes 30              # feasibility check (87%)
scripts/run_jobs.sh results/arm_wave_final.txt 10         # all 32 arm runs
python -m experiments.analyze_arm --extra_evals           # results/arm_table.md, plots, arm_summary.json
python scripts/record_arm_traces.py --c_run results/runs/arm_C_verify_k4/seed_0 --b_run results/runs/arm_B_plan_k1/seed_0
python scripts/make_arm_video.py                          # results/arm_demo.mp4 / .gif
```

---

# Night 1 report — planar push (Imagine → Verify → Act MVP)

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
