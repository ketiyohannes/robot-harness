# Lab Notebook — Imagine → Verify → Act MVP

All times are local (EAT, UTC+3). Started **2026-11-23 00:50**.

## Phase 0 — Setup + sanity (00:50 – )

- 00:50 `caffeinate -dimsu` running (PID 52054, recorded in `.caffeinate.pid`). A second
  instance I started by accident (PID 53202) was killed immediately.
- Machine: macOS 15.7.4 (24G517), arm64 (Apple M4), 16 GB RAM, 10 cores, 133 GiB free disk.
- Xcode CLT: present (`/Library/Developer/CommandLineTools`).
- Python: system `python3` 3.13.5 (python.org framework build). No conda/miniforge.
  `uv` 0.9.28 present (Homebrew) → project-local venv `.venv/` created with `uv venv`.
- Unity: **no Unity Hub installed**. Editor found at
  `~/Documents/Unity/Editors/6000.3.28f1/Unity.app` (with MacStandaloneSupport).
- Unity batchmode test (`Unity -batchmode -quit -nographics -logFile`): **exit code 198**,
  log says:
  ```
  [Licensing::Module] Error: Access token is unavailable; failed to update
  [Licensing::Client] Error: Code 404 ... Found 0 entitlement groups and 0 free entitlements
  [Licensing::Module] Error: 'com.unity.editor.headless' was not found.
  No valid Unity Editor license found. Please activate your license.
  ```
  Activating requires a Unity ID login (Hub or `-username/-password`), which nobody can do
  tonight.
- **Backend decision: MuJoCo** (fallback rule, condition (a) failed: editor cannot run in
  batchmode without a license/login). ML-Agents version pairing therefore not needed.
- Python packages (pinned in `requirements.txt`): torch 2.15.0, mujoco 4.0.1,
  gymnasium 1.4.0, stable-baselines3 2.9.2, numpy 2.5.5, imageio 2.38.2,
  imageio-ffmpeg 0.8.1 (bundled ffmpeg).
- MuJoCo offscreen rendering (`mujoco.Renderer`) works on macOS without extra setup.
- Raw MuJoCo step rate on a trivial model: ~368k steps/s.

## Decisions I made

1. **MuJoCo instead of Unity** — Unity licensing blocks batchmode (see above).
2. Project root is this folder (`robot-harness/`) rather than a nested
   `imagine-verify-act/` folder; the layout below the root follows the brief.
3. `rtamt` installed cleanly, but I implement a small STL subset myself, vectorised in
   numpy, because the planner must score thousands of imagined trajectories per decision
   (rtamt evaluates one trace at a time in pure Python). rtamt removed again.
4. Python env via `uv venv` on the python.org CPython 3.13.5 (no conda present). arm64 wheels
   for torch / mujoco installed without problems.

## Phase 1 — Task env + snapshot API + determinism (00:56 – 00:58)

- `envs/push_env.py`: MuJoCo MJCF built in code. Table 1.0 × 1.0 m (top at z = 0, no walls, so
  the block can fall off). Pusher = cylinder on two slide joints (x, y) driven by velocity
  actuators (kv = 60), so it is effectively a kinematic paddle that cannot touch the table.
  Block = cylinder "puck" (r = 4 cm, 0.2 kg) on a free joint; a puck instead of a box so
  yaw does not matter for the contact geometry. Target zone = r = 6 cm disc (visual site).
- Control: action ∈ [-1, 1]² → pusher velocity command × 0.5 m/s. dt = 0.05 s
  (physics 0.005 s × frame_skip 10), implicitfast integrator, 50 solver iterations,
  warmstart on.
- Snapshots: `mj_getState(mjSTATE_INTEGRATION)` (+ episode step counter + target).
  `restore_snapshot` → `mj_setState` + `mj_forward`.
- **Determinism (tests/test_determinism.py):**
  - replay 50 random actions × 10 from one snapshot: max = 0.0, mean = 0.0
  - save → 20 steps → restore → same 20 steps: max = 0.0, mean = 0.0
  - scripted push with contact (block moves 0.85 m) × 10: max = 0.0, mean = 0.0
  → bit-exact; **verifier margin ε = 0**.
- Sim speed: ~13.6k env steps/s single-threaded (raw MuJoCo ~368k physics steps/s).
- First render was overexposed (headlight + key light too strong) → reduced. Orthographic
  top-down camera uses `projection="orthographic"` (the `orthographic="true"` attribute from
  older docs is rejected by MuJoCo 4.0.1).

## Phase 2 — STL specs + verifier (00:58 – 01:00)

- `verifier/stl.py`: vectorised STL subset (predicates, not/and/or, G[a,b], F[a,b]) with
  standard finite-trace robustness; every node returns a robustness signal of shape
  (..., T), so one call scores thousands of imagined trajectories.
- `specs/push.yaml`, `specs/push_safely.yaml` (readable, with STL text in comments).
  Clauses: reach_target = F[0,T] (0.06 − ‖block − target‖ > 0); pusher_in_region =
  G (0.45 − max|p| > 0); block_slow = G (0.30 − ‖v_block‖ > 0); block_on_table =
  G min(0.5 − max|b_xy|, 10·(b_z − 0.01)) > 0.
- `verifier/verifier.py`: `Spec.verify(traj, target) -> (passed, robustness, per_clause)`.
- 8 verifier tests (hand-made passing/failing trajectories + batched-vs-single) all pass.

## Task tuning (01:00 – 01:02)

- With 3 s episodes, even a hand-written pushing controller rarely succeeded (getting behind
  the block + pushing under the 0.30 m/s limit does not fit in 3 s). Changed to **T = 100
  steps (5 s)**, block start in [−0.15, 0.15]², target 0.15–0.30 m from the block (within
  ±0.38). A naive scripted controller now gets 73 % (k = 0.5) / 80 % (k = 0.6) success, and
  0 % at full speed (always violates block_slow) → the safety clauses actually bind.
  (Scripted controllers are used only for this sanity check and as contact-rich *held-out*
  world-model evaluation data; never for training any method.)

## Phase 3 — PPO baseline (01:00 – )

- SB3 PPO, MlpPolicy [64, 64], 4 envs, n_steps 512, batch 256, lr 3e-4, γ 0.99.
  CPU (1 thread). Speed ~5.7k env steps/s → 1M steps ≈ 3–4 min.
- Reward v1: per-step ρ_target − 0.3‖p − b‖ + 5·Σ min(0, ρ_safety) + terminal 10·clip(ρ_spec).
  Result: 0 % success, policy parks the pusher.
- Reward v2: potential-based progress shaping Φ = ρ_target − 0.5‖p − b‖, r = 10ΔΦ + 0.5·[in
  target] + 5·Σ min(0, ρ_safety) + terminal 10·clip(ρ_spec, ±1), plus relative-position
  observation features. Result after 1M steps: 0 % success; deterministic policy jitters in
  place and never touches the block. Cause: when the block falls off the table the
  per-step penalty 5·ρ_on_table is unbounded (≈ −20/step), so PPO learns to avoid the block.
- Reward v3 (final): safety penalty bounded to −p per violated clause per step (p =
  `safety_penalty`). Variants tried for 300–400k steps: p = 1; p = 0.1; p = 0.1 + ent_coef 0.01;
  p = 0 + ent_coef 0.01 → **all 0 % success**. Sanity check: the same PPO setup learns a pure
  "reach the block" reward fine (episode return converges), so the wiring is correct; the
  failure is exploration: random contacts mostly push the block the wrong way, so "don't
  touch the block" is a strong local optimum.
- Decision: keep PPO (as required) with v3 reward at 1M steps (log curve evaluated every
  10k, so the equal-budget 60k point is read off the curve), **and add SAC** (off-policy,
  far more sample-efficient) with the same reward/obs at the shared 60k budget as a
  stronger model-free baseline (A2).

## Phase 4 — World model (01:02 – 01:05)

- `world_model/ensemble.py`: 5-member ensemble, 3×200 SiLU MLPs, batched via einsum-style
  `baddbmm` weights (E, in, out). Predicts normalised Δstate; inputs normalised.
  Disagreement = std across members.
- `scripts/train_world_model.py`: 20k OU-noise random transitions (PPO checkpoint data
  skipped: PPO never learned to touch the block, so its rollouts add nothing). 4000 Adam
  steps, 11 s on CPU. Held-out: 20 OU episodes + 20 *scripted-push* episodes (contact-rich).
  Results (`results/world_model/metrics.json`, `uncertainty_vs_error.png`):

  | held-out | 1-step block err | 10-step block err | 25-step block err |
  |---|---|---|---|
  | random OU | 0.13 mm | 2.7 mm | 9.2 mm |
  | scripted push (contacts) | 1.3 mm | 26.5 mm (p90 68 mm) | 79 mm (p90 151 mm) |

  log-uncertainty vs log-error correlation on contact data: **0.62**.
  Random exploration touches the block in only 1.6 % of transitions (uniform: 0.3 %), so the
  pretrained model is poor at contacts; online data fixes that.
- CPU vs MPS: not benchmarked separately — I ran many single-threaded CPU processes in
  parallel (9–15 at a time), which uses the 10 cores better than sharing one GPU. Logged as
  a decision.

## Phase 5 — Imagine → Verify → Act (01:05 – 01:08)

- `planners/cem.py`: K = 4 independent CEM searches batched in one model call (pop 128,
  12 elites, 4 iters, horizon 20). Each returns its best sequence → 4 structurally different
  candidates, ranked by imagined score. Score = ρ_reach(window) − ‖b_H − target‖ −
  0.3·min_t‖p − b‖ + 10·Σ_c min(0, ρ_c − 0.05) − λ·ensemble std (λ = 2). The progress and
  approach terms are search heuristics; pass/fail is always the exact verifier.
- `planners/agent.py`: receding horizon, N = 10 committed steps per decision (10 decisions
  per episode). k > 1: snapshot → run each of the top-k candidates for its full horizon in
  the real sim → verify safety clauses exactly on the branch (+ the full spec over history +
  branch when the branch reaches T) → restore → commit the passing branch with the best
  real score; if none passes, commit the best real score (logged as `no_pass`). Re-execution
  of the committed segment matches the branch bit-exactly (determinism). All transitions
  (incl. branches) go to the dataset; world model retrained every 4000 real steps
  (1500 Adam steps, warm-started) with error-weighted sampling (w ∝ err + mean err).
- Oracle diagnostic (`planners/oracle_model.py`, perfect model = private sim copy): the
  planning objective solves 4/4 eval seeds with all safety clauses satisfied → the objective
  is sound; remaining failures are world-model error.
- Smoke test B (k = 1), 20k budget: rolling success 15–30 % by ~14k real steps, 1-step error
  on new data 0.11 → 0.02 after retrains.

## Phase 6 — runs launched 01:08

All on push_safely, 60k real-step training budget (incl. 5k OU pretraining), 3 seeds,
eval on 30 fixed held-out seeds (10000–10029) with frozen world model:
`B_plan_k1`, `C_verify_k4`, `D_verify_k4_lam0` (configs in `experiments/configs/`).
Compute: torch 1 thread per process, all 9 in parallel. B is ~8 s/episode under load
(CEM = 4 × 128 × 5 members × 20 steps × 4 iters per decision), so B takes ~1 h.

## Bigger experiment (requested by the user at ~01:28)

The user asked for a bigger experiment within the resource and time limits, plus a video.
Plan (all push_safely, configs unchanged except budget/seeds):
- **Longer budgets** (launched 01:29): `B_plan_k1_long` and `C_verify_k4_long` at 120k real
  steps (2 seeds each); `A2_sac_long` at 300k steps (2 seeds), to see where each method
  levels off and whether model-free / unverified planning catch up with more data.
- **More seeds** (launched when wave 1 frees cores): seeds 3–6 for B and C at 60k → 7 seeds
  each for the main comparison.
Cost estimate: B at 120k ≈ 1,150 episodes × ~6–8 s ≈ 2–2.5 h under load. This is the
slowest run and should finish around 05:00, leaving time for analysis, video and report.

## Results log (01:23 – 03:33)

- 01:23 C/D finished (60k): C 77/80/93%, D 93/90/90% success, 0% safety violations.
- ~01:50 PPO 1M (3 seeds): 0%. SAC 60k (3 seeds): 0%. E (k=2): 93/93/90%. F (k=8): 93/93/93%.
  Task 1 C: 100/100%.
- 02:02 Load average about 50: torch was pinned to 1 thread, but numpy/Accelerate threads
  were not. All later launches use `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.
- 02:31 B finished (60k): 83/80/87% success, 7/0/7% safety violations (all block_slow).
  C at 120k: 100/93%. SAC at 300k: 0/0%.
- 02:31 Launched seeds 3–6 for B and C at 60k.
- 02:35 Demo traces recorded from C seed 2 and B seed 2. Draft video rendered. Fixes:
  - the legend overlapped the table, so it moved to the HUD panel;
  - an edit left a stray indent, so `make_video.py` failed to parse — fixed;
  - the action was too small in the full-table view, so walkthroughs and the side-by-side
    now use a zoomed orthographic camera.
- 03:30 All runs finished. Analysis run with `--extra_evals` (C's model evaluated with k=1).
  Final: B 83 ± 3% / 6 ± 4% violations (7 seeds); C 84 ± 5% / 0% (7 seeds); C's model without
  verification 80 ± 5% / 6 ± 3%. Training episodes with a violation in executed steps:
  B 158 ± 7, C 6.6 ± 1.6 per seed. Real steps to 50% rolling training success:
  B 13.8–22.7k, C 20.5–29.9k.
- 03:33 Final video: 83.1 s, 1280×720, 30 fps, 2,493 frames. Five sampled frames checked
  (`results/video_frames_check/`). None are black.
- In traced decisions, the model's rejected #1 plans failed block_slow in 103/131 cases and
  reach_target (on branches reaching T) in 46/131. A branch can fail both.

---

# Night 2: robot-arm version (Franka Panda)

User request (23:10): "go with level one, implement it end to end". Level 1 = a robot arm with a
two-finger gripper, same Imagine → Verify → Act loop, overnight on this Mac.

## Setup (23:12 – 23:20)

- Robot model: Franka Emika Panda from MuJoCo Menagerie (Apache-2.0), sparse-cloned into
  `third_party/` (gitignored) by `scripts/fetch_assets.sh`, pinned to commit 370c439.
- Scene built with `MjSpec`: the Menagerie arm is attached to a table scene. Arm bodies use
  gravity compensation, and an `ee` site sits at the fingertips.
- Control: the action is a 3-D fingertip velocity (±0.25 m/s per axis). A damped-least-squares
  differential-IK controller runs every 10 ms; it holds the gripper pointing down and pulls the
  null space toward home. It writes joint targets `q + 0.1·q̇_des` for the Panda position servos
  (kp/kv = 0.1 s, so the servos track q̇_des). The gripper stays closed.
- Task: push a puck (r 3 cm) into a target (r 5 cm) within 6 s (120 steps). A short cup stands
  beside the puck→target line, 7–13 cm off it. Spec `specs/arm_push_safely.yaml`:
  F puck_in_target ∧ G cup_upright (tilt < 0.2 rad) ∧ G cup_still (moved < 3 cm) ∧ G puck_slow
  (< 0.25 m/s) ∧ G puck_on_table ∧ G ee_in_workspace.
- World-model state: 21-D (ee pos/vel, puck pos/vel, cup xy/z/up-axis xy/lin vel xy/ang vel xy).
- The code was generalised through `envs/registry.py`: ensemble dims, CEM action dim, planner
  heuristic indices, uncertainty indices, per-clause plan margins in the spec yaml, a gym
  wrapper for both envs, and a generic oracle model. The push results are unaffected and the
  old tests pass.

## Problems hit

- **A tall cup was unavoidable.** The Panda hand is 20 cm wide and its bottom sits 3.9 cm above
  the fingertip, so while pushing it swept a 11 cm cup in 23/40 scripted episodes. Fix: shorter
  cup (6 cm) and push height 3 cm. The hand can now pass over the cup when the fingertip is high
  enough, which a careless low sweep still violates.
  Scripted controller after the fix: 75–80% success (`scripts/arm_scripted.py`).
- **Snapshot/restore was not bit-exact at first** (84% of elements differed). After `mj_step`,
  site poses describe the state before integration, while a restore is followed by `mj_forward`.
  The controller read stale poses on one path and fresh poses on the other. Fix: recompute
  kinematics before each control update, and run `mj_forward` at the end of each env step.
  Now bit-exact (`tests/test_arm.py`).
- **`ps` in this sandbox only shows a few processes.** I wrongly concluded that background jobs
  had died and launched duplicates. `pgrep -fl` shows them; I killed the duplicates. All jobs now
  run through `scripts/run_jobs.sh` (parallel queue, one log per job).
- **Pilot 1 (OU-noise pretraining): 0% success at 31k steps.** The arm hovered 6–7 cm from the
  puck and never pushed. Random 3-D OU noise moves the puck in only 0.2% of transitions, so the
  model never learns contact, and in imagination pushing does nothing. A perfect-model (oracle)
  planner solved 4/4 seeds with the same objective, so the objective is fine. Fix:
  `ArmWaypointExplorer`, which moves toward random waypoints (55% near the puck, 15% near the
  cup, 30% anywhere; mostly low) with OU noise on top. The puck now moves in 16% of transitions
  and the cup in 11%. It is observation-only and shared by every arm method; SAC gets the same
  5k transitions prefilled into its replay buffer, counted in its budget.

## Pilot 2 and launch (23:21 – 23:47)

- Pilot 2 (waypoint exploration), seed 0. C reached 35% rolling training success at 46k real
  steps with 2 violating training episodes so far. B at 14k: 0% success, and 13 of its last 20
  training episodes broke a safety clause.
- Oracle check (perfect model, k = 1, λ = 0) solved 7/8 eval seeds, so the planning objective
  works on the arm.
- 23:31: wave 1 launched (`results/arm_wave1.txt`, 9 parallel):
  - B and C at 100k, seeds 1–4 (seed 0 = pilot);
  - B and C at 200k, seeds 0–1;
  - E (k = 2) at 100k, seeds 0–2;
  - SAC at 100k (5k waypoint prefill), seeds 0–2;
  - PPO at 1M, seeds 0–2.
- 23:46: C seed 0 final eval: **27% success, 13% episodes with a safety violation**. Verification
  did not remove violations here, unlike on the planar task. The traces show why:
  - in 19% of eval decisions **no branch passed** (`no_pass`), and the agent then commits the
    best-scoring failing branch;
  - in seed 10003 all 4 candidates hit the puck too hard (puck_slow) at t = 20, so the committed
    one did too;
  - most other failures are reach_target: the puck moves toward the target but too slowly for
    the 6 s deadline.
  The learned model under-predicts the puck's speed on impact.
- Added `no_pass_fallback: stop`: if every verified branch broke a safety clause, hold still for
  N steps instead of committing a failing plan. Liveness-only failures still commit the best
  score. Wave 2 (2 parallel): G = C + stop fallback (seeds 0–2) and D = C with λ = 0 (seeds
  0–2), both at 100k.

## Run v1 invalidated: reset IK bug (01:20)

Wave 1 results looked as follows:
- C (100k, 5 seeds): 27/57/7/0/33% success, 13–20% safety violations.
- B: 0% everywhere.
- PPO: 0%.
- G (stop fallback): 20/70/3% success, 10% violations.
- D (λ = 0): 30/63/7% success.

One pattern did not make sense: **every** method, every seed, broke ee_in_workspace in exactly
the same 3 of 30 eval episodes (seeds 10013, 10027, 10028), and broke the cup clauses on 10027.
Cause: the reset-time static IK took undamped full Gauss-Newton steps and diverged on about 10%
of seeds to joint angles far outside the limits (e.g. −35 rad). The servos, clipped to their
ctrl ranges, then swung the arm wildly at t = 0. This is an environment bug, not a policy
effect.

Fix: damped, step-limited IK from home, with joint limits enforced every iteration and a
convergence check that redraws the start pose if needed. Checked on 430 seeds (eval, training
and pretraining ranges): no drift and no cup motion under zero action. Scripted controller on
the eval seeds: 87% success, 0 safety violations.

All arm runs were killed and moved to `results/archive/arm_v1_ik_bug/` (kept for the record,
not reported). Everything is relaunched on the fixed env (`results/arm_wave_final.txt`, 32
jobs, 10 parallel), plus **B with λ = 0** (3 seeds). In v1, B never descended: it hovered
about 20 cm above the table for 100k steps. B with λ = 0 tests whether the uncertainty penalty
is what kept it away from the objects.

## Final arm results (01:24 – 04:50)

- 01:24: final wave launched (32 jobs). 03:12: B at 100k finished:
  33/30/40/33/30% success, 60/53/43/47/50% safety violations, mostly puck_slow.
- 04:34: all jobs done, no errors. Extra evals (each verified run's model acting with k = 1)
  ran in parallel.
- Final (5 seeds):
  - B 33 ± 4% success / 51 ± 6% violations;
  - C 70 ± 2% / 1 ± 2%;
  - C's model without verify 28 ± 7% / 49 ± 5%.
- Variants:
  - G (stop fallback) 83 ± 5% / 0%;
  - D (λ = 0) 72 ± 6% / 1%;
  - E (k = 2) 68 ± 12% / 6%;
  - B (λ = 0) 32 ± 3% / 48%;
  - 200k budget: B 43% / 43%, C 83% / 2%.
- Model-free: SAC 0% (7% violations), PPO 1M 0%.
- Training violations per seed: B 400 ± 31 of 792 episodes; C 1.6 ± 0.8 of 92.
- Real steps to 50% rolling training success: B 35.6–62.0k, C 32.0–66.4k. Same speed.
- Demo traces: C seed 0 and B seed 0 re-run on all 30 eval scenes (C 25/30, B 16/30; planner
  RNG differs from the official eval). Picked scenes:
  - success: 10000;
  - overturn: 10007, decision 2 (model's #1 tips the cup);
  - contrast: 10003 (B breaks puck_slow, C succeeds).
- Video: 80.8 s, 1280×720, 30 fps. Nine frames checked (`results/arm_video_frames_check/`).
  Learning plots were redrawn as mean ± std over seeds, because per-seed lines were unreadable.
