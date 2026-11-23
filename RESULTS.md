# Results — Imagine → Verify → Act (MVP)

Setup: the MuJoCo planar push task with the **push-safely** spec, unless a row says otherwise.
The spec is F[0,100] block_in_target ∧ G pusher_in_region ∧ G block_slow (|v| < 0.30 m/s)
∧ G block_on_table.

Protocol:
- Each training run gets a fixed budget of **real** sim steps. That budget covers world-model
  pretraining, committed steps and verification branch steps.
- Evaluation uses the same 30 held-out seeds (10000–10029) for every method, with the world
  model frozen.
- Numbers are mean ± std over training seeds. Each seed is evaluated on 30 episodes.
- Every run comes from one config in `experiments/configs/` plus `python -m experiments.run`.
- Raw outputs are in `results/runs/<config>/seed_<s>/`. Tables are regenerated with
  `python -m experiments.analyze --extra_evals`; the full output is in `results/table.md`.

## Main table (held-out evaluation)

| Method | seeds | train budget | success | episodes with a safety violation | real steps / eval episode (committed + branch) | model's #1 overturned | model's #1 failed verification |
|---|---|---|---|---|---|---|---|
| A: PPO | 3 | 60k | 0% ± 0 | 0% ± 0 | 100 | – | – |
| A: PPO | 3 | 1M | 0% ± 0 | 0% ± 0 | 100 | – | – |
| A2: SAC | 3 | 60k | 0% ± 0 | 9% ± 6 | 100 | – | – |
| A2: SAC | 2 | 300k | 0% ± 0 | 3% ± 3 | 100 | – | – |
| **B: world model + CEM, no verification (k = 1)** | 7 | 60k | **83% ± 3** | **6% ± 4** | **100** | – | – |
| **C: world model + CEM + verified branching (k = 4)** | 7 | 60k | **84% ± 5** | **0% ± 0** | **860** (100 + 760) | 39% ± 3 | 6% ± 2 |
| C's trained model, acting **without** verification (k = 1) | 7 | 60k | 80% ± 5 | 6% ± 3 | 100 | – | – |
| D: C with λ = 0 (no uncertainty penalty) | 3 | 60k | 91% ± 2 | 0% ± 0 | 860 | 39% ± 2 | 4% ± 1 |
| E: verified, k = 2 | 3 | 60k | 92% ± 2 | 1% ± 2 | 480 (100 + 380) | 30% ± 3 | 4% ± 0 |
| F: verified, k = 8 | 3 | 60k | 93% ± 0 | 0% ± 0 | 1620 (100 + 1520) | 50% ± 1 | 3% ± 0 |
| B, 2× budget | 2 | 120k | 90% ± 7 | 3% ± 3 | 100 | – | – |
| C, 2× budget | 2 | 120k | 97% ± 3 | 0% ± 0 | 860 | 35% ± 1 | 2% ± 0 |
| Task 1 (push, no speed/table clauses): B | 2 | 60k | 100% ± 0 | 0% ± 0 | 100 | – | – |
| Task 1 (push): C | 2 | 60k | 100% ± 0 | 0% ± 0 | 860 | 59% ± 1 | 0% ± 0 |

Per-clause violation rates on the eval episodes are in `results/table.md`. Every safety
violation by B and C comes from **block_slow**. No method ever pushed the block off the table
or drove the pusher out of its region during evaluation.

## Training phase

| Method | training episodes | training episodes with a safety violation in executed steps (count per seed) | real steps until rolling-10 training success ≥ 50 % | train wall-clock |
|---|---|---|---|---|
| B (k = 1), 60k | 550 | **158 ± 7** (29 %) | 13.8k – 22.7k (7 seeds) | 62 ± 16 min |
| C (k = 4), 60k | 64 | **6.6 ± 1.6** (10 %) | 20.5k – 29.9k (7 seeds) | 13 ± 1 min |
| D (λ = 0) | 64 | 5.7 | 17.9k – 26.5k | 12 min |
| E (k = 2) | 115 | 17.3 | 20.4k – 31.4k | 32 min |
| F (k = 8) | 34 | 4.0 | 26.1k – 30.9k | 22 min |
| B, 120k | 1150 | 182 | 15.1k – 21.3k | 119 min |
| C, 120k | 134 | 7.0 | 20.5k – 26.5k | 39 min |

Wall-clock was measured with 10–25 processes sharing 10 cores, so treat it as relative only.
B's wall-clock is high because it plans 10 times per 100 real steps. C plans 10 times per
860 real steps.

Plots:
- `results/plots/learning_and_safety.png`: success vs total real steps, and cumulative
  training episodes with a violation in the executed (committed) steps.
- `results/plots/wm_error_over_time.png`: 1-step error on newly collected data before each
  retrain.
- `results/plots/eval_bars.png`
- `results/world_model/uncertainty_vs_error.png`: Phase 4 calibration.

## Answering the research question

> Does spec-verified branching reduce total real env steps and safety violations, compared with
> model-free RL and with world-model planning without verification?

**Safety: yes, clearly.**
- In evaluation, verified branching (C, D, F) had **zero** episodes with a safety violation
  across 7 + 3 + 3 seeds × 30 episodes. B had 6% ± 4.
- During training, C committed about **24× fewer** violating episodes than B: 6.6 vs 158 per
  seed in the same 60k real-step budget.
- The ablation "C's model without verification" (80% success, 6% violations) shows where the
  safety comes from. It is the act-time verification, not a better world model. C's training
  data alone does not make the model safe to follow blindly.

**Total real steps: no.** Counting branch steps honestly, verification makes each episode
**8.6× more expensive** (860 vs 100 real steps at k = 4). It also does not speed up learning
measured in real steps. B reaches 50% rolling training success after 14–23k real steps, C after
20–30k. Final success at equal budget is the same (83% vs 84% at 60k). With a 2× budget C pulls
ahead (97% vs 90%), but with only 2 seeds that gap is not conclusive.

**Against model-free RL, the planners win by a wide margin.** PPO (1M steps) and SAC (300k
steps) never solved push-safely at all. I tried four reward/exploration variants for PPO
(see `LAB_NOTEBOOK.md`); none worked. A naive hand-written controller gets about 73%, so the
task is solvable. The model-free failure is an exploration problem: random contacts push the
block the wrong way, so "never touch the block" is a strong local optimum. A better-shaped
reward or a curriculum would probably fix this; I did not get there tonight.

**How often verification overturned the model's #1 choice.** It overturned 39% of decisions.
But the model's #1 actually *failed* the spec in only about 6% of decisions. In the traced
decisions, 103/131 of those failures broke block_slow; the rest missed the target on branches
that reached the episode deadline. The other ~33% were cases where another candidate that passed verification got
a better real score. Branching therefore does two jobs:
1. A safety filter (the 6%).
2. A real-sim local search over the K candidates (the remaining 33%).

The second job is why D, E and F reach 91–93%.

**Choice of k (k = 1/2/4/8).** k = 2 already gets almost all of the benefit (92%, 1%
violations) at 480 steps per episode. k = 8 gives no more success than k = 2 and costs 1,620
steps per episode.

**Uncertainty penalty λ.** λ = 0 (D) did as well as or slightly better than λ = 2 (C): 91% vs
84%, with 3 vs 7 seeds. Once verification is in the loop, the λ·uncertainty term does not help
in this task. Ensemble disagreement does correlate with real error (r = 0.62 in log-log on
contact-rich held-out data), but the verifier catches model errors directly.

**Task 1 (push, no speed or table clauses)** is easy for both planners (100%). B learns it
faster in real steps: 5–10k vs 15–17k.

## World model

- 5 × MLP(3 × 200) ensemble predicting Δstate.
- After pretraining on 5k–20k random (OU-noise) steps, contact prediction is poor: block error
  27 mm at 10 steps and 79 mm at 25 steps on scripted-push held-out data. Random exploration
  touches the block in only 1.6% of transitions.
- Online data fixes this. The 1-step error on newly collected data falls from about 0.12 to
  about 0.02 within 20k real steps, for every method (`wm_error_over_time.png`).
- Verified runs end with slightly lower error than B. Branch data is more diverse, but the
  difference is small.
- Error-weighted resampling was always on. I did not ablate it.

## Limitations (please read)

- **MVP scale.** 2–7 seeds per condition and 30 eval episodes per seed. The k, λ and 120k
  rows have only 2–3 seeds, so differences under about 5–8 points are within noise.
- **Determinism is exact here.** MuJoCo snapshot/restore replays bit-identically, so the
  verifier uses ε = 0 and committed steps reproduce the verified branch exactly (tested in
  `tests/test_agent.py`). A real robot, or a sim-to-real gap, breaks this. Branches would then
  verify against a model of reality, not reality itself.
- **The branches run in the same simulator that is being scored.** In this setup C is
  partly "planning with the true simulator", and it pays for that in real steps. That is why
  the 8.6× step cost is the headline cost.
- **Each branch is verified on a window.** A branch checks the safety clauses on its own
  20-step window, plus the full spec only when it reaches T. Liveness (reaching the target)
  is used only to rank branches, not as a pass/fail filter, until the end of the episode.
- **The planning score includes two hand-made shaping terms.** These are progress beyond the
  window and an approach term, plus a safety margin of 0.05. They shape search only; success is
  always judged by the exact verifier on real trajectories. An oracle-model check showed this
  objective solves the task when the model is perfect.
- **Unity was not used.** Licensing blocked batchmode. Everything runs on MuJoCo (see
  `LAB_NOTEBOOK.md`).
- **PPO and SAC got a logged, good-faith effort, but no hyperparameter search.** Their 0% is
  an honest result for this reward and budget, not a proof that model-free RL cannot do the
  task.
- **The demo video's episodes are re-runs.** They come from re-running trained agents on eval
  seeds, using steps that are not counted in any number above. The episodes shown are
  selected for illustration: a success, an overturn, and a seed where B fails and C succeeds.
