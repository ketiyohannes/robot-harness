# Results: robot arm (Franka Panda), Imagine → Verify → Act

This is the second night's experiment: the same method as `RESULTS.md`, applied to a 7-joint
robot arm instead of the planar pusher.

## Task

A Franka Emika Panda (MuJoCo Menagerie model) with its gripper closed pushes a puck into a
target zone within 6 s (120 control steps of 50 ms). A cup stands next to the puck→target path.

- **Action**: 3-D fingertip velocity, ±0.25 m/s per axis. A differential-IK controller turns it
  into joint targets.
- **World-model state**: 21-D (fingertip, puck and cup states).
- **Spec** (`specs/arm_push_safely.yaml`, STL over T = 120):

| clause | kind | meaning |
|---|---|---|
| F puck_in_target | liveness | puck centre within 5 cm of the target at some point |
| G cup_upright | safety | cup tilt < 0.2 rad, always |
| G cup_still | safety | cup never moves more than 3 cm |
| G puck_slow | safety | puck speed < 0.25 m/s, always (don't hit it hard) |
| G puck_on_table | safety | puck stays on the table |
| G ee_in_workspace | safety | fingertip stays in a 70 × 70 cm box, below 40 cm |

**Protocol (same as the planar task):**
- Fixed budgets of **real** sim steps. Pretraining, committed and verification-branch steps
  all count.
- 30 held-out eval scenes (seeds 10000–10029), the same for every method, with the world model
  frozen.
- Mean ± std over training seeds.
- Every run is one config in `experiments/configs/arm_*.yaml`.
- Tables come from `python -m experiments.analyze_arm --extra_evals` (`results/arm_table.md`).

A hand-written controller (`scripts/arm_scripted.py`, not a baseline) solves 87% of the eval
scenes with no safety violations, so the task is feasible. A perfect-model planner (oracle,
diagnostic only) solved 7/8.

## Main table (held-out evaluation, 100k real-step budget unless noted)

| Method | seeds | success | episodes with a safety violation | real steps / eval episode | model's #1 overturned | model's #1 failed verification |
|---|---|---|---|---|---|---|
| A: PPO (1M steps, 10× budget) | 3 | 0% ± 0 | 0% ± 0 | 120 | – | – |
| A2: SAC | 3 | 0% ± 0 | 7% ± 3 | 120 | – | – |
| **B: world model + CEM, no verification** | 5 | **33% ± 4** | **51% ± 6** | **120** | – | – |
| B with λ = 0 (no uncertainty penalty) | 3 | 32% ± 3 | 48% ± 4 | 120 | – | – |
| **C: world model + CEM + verified branching (k = 4)** | 5 | **70% ± 2** | **1% ± 2** | **1040** (120 + 920) | 59% ± 4 | 15% ± 3 |
| C's trained model, acting **without** verification | 5 | 28% ± 7 | 49% ± 5 | 120 | – | – |
| D: C with λ = 0 | 3 | 72% ± 6 | 1% ± 2 | 1040 | 57% ± 0 | 13% ± 2 |
| E: verified, k = 2 | 3 | 68% ± 12 | 6% ± 3 | 580 | 41% ± 1 | 13% ± 3 |
| G: C + "hold still if no branch passes" | 3 | **83% ± 5** | **0% ± 0** | 1040 | 59% ± 1 | 10% ± 1 |
| B, 2× budget (200k) | 2 | 43% ± 3 | 43% ± 7 | 120 | – | – |
| C, 2× budget (200k) | 2 | 83% ± 3 | 2% ± 2 | 1040 | 57% ± 0 | 12% ± 0 |

**Which clauses broke.** Per-clause rates are in `results/arm_table.md`. Almost all of B's
violations are **puck_slow**, i.e. hitting the puck too hard: 49% ± 6 of episodes. Cup
disturbances are rare: 1–4%. C's remaining violations (2 of 150 episodes) are also
puck_slow. E's 5 of 90 are 4 puck_slow and 1 cup_still. Nobody pushed the puck off the table or
left the workspace.

## Training phase

| Method | training episodes | training episodes with a safety violation (per seed) | share of episodes | real steps to rolling-10 training success ≥ 50% | train wall-clock |
|---|---|---|---|---|---|
| B (k = 1) | 792 | 400 ± 31 | 50% | 35.6k – 62.0k (5 seeds) | 103 min |
| C (k = 4) | 92 | 1.6 ± 0.8 | 2% | 32.0k – 66.4k (5 seeds) | 19 min |
| E (k = 2) | 164 | 18 ± 7 | 11% | 36.3k – 46.8k | 26 min |
| G (stop fallback) | 92 | 0.3 ± 0.5 | 0% | 32.0k – 54.9k | 17 min |
| B, 200k | 1625 | 779 ± 17 | 48% | 36.8k, 46.4k | 188 min |
| C, 200k | 188 | 3.5 ± 0.5 | 2% | 32.0k, 66.4k | 38 min |

Wall-clock was measured with about 10 jobs sharing 10 cores. B is slow because it plans 12
times per 120 real steps; C plans 12 times per 1,040.

Plots:
- `results/plots/arm_learning_and_safety.png`: mean ± std learning curves and cumulative
  violating training episodes.
- `results/plots/arm_eval_bars.png`

## Answering the research question on the arm

**Safety: yes, more strongly than on the planar task.**
- At equal 100k budgets, verified branching cuts episodes with a safety violation from 51% to
  1% (5 seeds each).
- During training it cuts violating episodes from 400 to 1.6 per seed.
- The ablation "C's model acting without verification" (49% violations) shows where the safety
  comes from: checking plans at act time, not a better-trained model.
- The cause of B's violations is specific. The learned model **under-predicts how fast the puck
  flies off on impact**, so the model's favourite plan often looks fine in imagination and
  breaks puck_slow for real.

**Success: verification also doubled it on the arm** (70% vs 33%; 83% vs 43% at 200k). This
differs from the planar task, where B and C tied. Two effects, both visible in the logs:
1. B's "successes" often fail the spec because of puck_slow. With verification, a fast plan is
   replaced by a gentler passing one.
2. Real branching acts as a local search in the true simulator. In 59% of decisions a different
   candidate than the model's #1 was committed, while the #1 actually failed a clause in only
   15%.

**Total real steps: no.** Each verified episode costs 1,040 real steps instead of 120, so 8.7×
more. Learning speed in real steps is about the same: 50% training success after 32–66k steps
for C vs 36–62k for B. Verification spends real steps to buy safety and success; it does not
save them.

**No-pass fallback.** In about 6% of C's eval decisions (5–6% per seed) *no* candidate passed.
These include end-of-episode decisions where every branch only misses the deadline. The
original rule then commits the best-scoring failing plan, and that is where C's 2 remaining
violations came from.
Holding still instead (G) gave 0 violations in eval (90 episodes) and 1 violating training
episode in total over 3 seeds. G also had the best success (83%). The stop itself fired only
twice in G's eval, so most of G's gain comes from training. With fewer bad commits during
training, the data was better; with only 3 seeds this is a hint, not a conclusion.

**k = 2** keeps most of the benefit (68% success, 6% violations) at 580 steps per episode, as
on the planar task.

**Uncertainty penalty λ.** It doesn't matter here: B and B(λ=0) are 33% vs 32%, C and D are 70%
vs 72%.

**Model-free baselines** never solved the arm task:
- SAC (100k steps, with the same 5k exploration prefill): 0%.
- PPO (1M steps): 0%.

Same caveat as before: there was a good-faith effort but no tuning, so this does not show that
model-free RL cannot do it.

## Arm-specific limitations

- **Exploration uses task knowledge.** Pure random 3-D noise almost never touched the puck
  (0.2% of steps), so the first pilot learned nothing (see `LAB_NOTEBOOK.md`). All arm methods
  pretrain on 5k steps of `ArmWaypointExplorer`, which moves toward random points near the puck
  and cup, and SAC gets the same 5k steps in its replay buffer. This is a fixed,
  observation-only exploration prior, shared by every method, but it is task-shaped.
- **Determinism is exact** (bit-exact snapshot/restore, ε = 0), as on the planar task. On a real
  robot the branches would run in a simulator that differs from reality. This experiment does
  not test that, and it is the main open question.
- **The planner objective has two hand-made shaping terms** (puck→target distance at the end of
  the horizon, and fingertip→puck approach). They are the same for all planners. Success is
  judged only by the exact STL verifier.
- **Sample sizes.** D, E and G have 3 seeds; the 2× budget rows have 2. Differences under
  about 10 points between those rows are within noise.
- **The demo video's episodes are re-runs.** They come from re-running the trained C and B seed-0
  agents on eval scenes with a different planner random seed. On that re-scan C solved 25/30 and
  B 16/30, versus 21/30 and 10/30 in the official eval. The planner is stochastic, so a single
  episode is not a result; the table is. Re-run steps are not counted anywhere.
- **First run invalidated.** A reset-time IK bug corrupted about 10% of starting poses. Those
  runs are archived in `results/archive/arm_v1_ik_bug/` and not reported. Every number above is
  from the fixed environment.
