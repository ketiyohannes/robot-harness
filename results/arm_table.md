# Arm task results (arm_push_safely)

| Method | seeds | train budget | success | episodes with a safety violation | real steps / eval episode | model's #1 overturned | model's #1 failed verification |
|---|---|---|---|---|---|---|---|
| A: PPO | 3 | 1001k | 0% ± 0 | 0% ± 0 | 120 | – | – |
| A2: SAC | 3 | 100k | 0% ± 0 | 7% ± 3 | 120 | – | – |
| B: WM+CEM, no verify (k=1) | 5 | 100k | 33% ± 4 | 51% ± 6 | 120 | – | – |
| B with lambda=0 | 3 | 100k | 32% ± 3 | 48% ± 4 | 120 | – | – |
| C: WM+CEM+verify (k=4) | 5 | 101k | 70% ± 2 | 1% ± 2 | 1040 | 59% ± 4 | 15% ± 3 |
| C: WM+CEM+verify (k=4): its model acting WITHOUT verify | 5 | 101k | 28% ± 7 | 49% ± 5 | 120 | – | – |
| D: C with lambda=0 | 3 | 101k | 72% ± 6 | 1% ± 2 | 1040 | 57% ± 0 | 13% ± 2 |
| D: C with lambda=0: its model acting WITHOUT verify | 3 | 101k | 23% ± 5 | 49% ± 6 | 120 | – | – |
| E: verify k=2 | 3 | 100k | 68% ± 12 | 6% ± 3 | 580 | 41% ± 1 | 13% ± 3 |
| E: verify k=2: its model acting WITHOUT verify | 3 | 100k | 34% ± 8 | 48% ± 4 | 120 | – | – |
| G: C + stop if no branch passes | 3 | 101k | 83% ± 5 | 0% ± 0 | 1040 | 59% ± 1 | 10% ± 1 |
| G: C + stop if no branch passes: its model acting WITHOUT verify | 3 | 101k | 27% ± 3 | 53% ± 3 | 120 | – | – |
| B: no verify, 2x budget | 2 | 200k | 43% ± 3 | 43% ± 7 | 120 | – | – |
| C: verify k=4, 2x budget | 2 | 201k | 83% ± 3 | 2% ± 2 | 1040 | 57% ± 0 | 12% ± 0 |

## Training phase

| Method | training episodes | training episodes with a safety violation (per seed) | share | real steps to rolling-10 success ≥ 50% | train wall-clock (min) |
|---|---|---|---|---|---|
| B: WM+CEM, no verify (k=1) | 792 | 399.80 ± 31.01 | 50% | 36.8k, 46.4k, 35.6k, 62.0k, 44.6k | 103.32 ± 1.16 |
| B with lambda=0 | 792 | 430.00 ± 17.96 | 54% | 37.8k, 53.0k, 56.0k | 101.91 ± 1.05 |
| C: WM+CEM+verify (k=4) | 92 | 1.60 ± 0.80 | 2% | 32.0k, 66.4k, 39.3k, 39.3k, 59.1k | 18.94 ± 0.91 |
| D: C with lambda=0 | 92 | 1.67 ± 0.94 | 2% | 47.6k, 43.5k, 33.1k | 20.02 ± 0.26 |
| E: verify k=2 | 164 | 18.00 ± 7.48 | 11% | 36.3k, 42.7k, 46.8k | 25.97 ± 0.31 |
| G: C + stop if no branch passes | 92 | 0.33 ± 0.47 | 0% | 32.0k, 46.6k, 54.9k | 17.03 ± 0.19 |
| B: no verify, 2x budget | 1625 | 779.00 ± 17.00 | 48% | 36.8k, 46.4k | 188.47 ± 0.60 |
| C: verify k=4, 2x budget | 188 | 3.50 ± 0.50 | 2% | 32.0k, 66.4k | 37.78 ± 0.21 |

## Per-clause violation rate (eval episodes)

| Method | reach_target | cup_upright | cup_still | puck_slow | puck_on_table | ee_in_workspace |
|---|---|---|---|---|---|---|
| A: PPO | 100% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| A2: SAC | 100% ± 0 | 0% ± 0 | 2% ± 2 | 6% ± 4 | 0% ± 0 | 2% ± 2 |
| B: WM+CEM, no verify (k=1) | 35% ± 6 | 1% ± 2 | 4% ± 2 | 49% ± 6 | 0% ± 0 | 0% ± 0 |
| B with lambda=0 | 36% ± 8 | 0% ± 0 | 1% ± 2 | 48% ± 4 | 0% ± 0 | 0% ± 0 |
| C: WM+CEM+verify (k=4) | 29% ± 3 | 0% ± 0 | 0% ± 0 | 1% ± 2 | 0% ± 0 | 0% ± 0 |
| C: WM+CEM+verify (k=4): model without verify | 42% ± 8 | 0% ± 0 | 3% ± 4 | 47% ± 5 | 0% ± 0 | 0% ± 0 |
| D: C with lambda=0 | 27% ± 5 | 0% ± 0 | 0% ± 0 | 1% ± 2 | 0% ± 0 | 0% ± 0 |
| D: C with lambda=0: model without verify | 52% ± 14 | 2% ± 2 | 4% ± 2 | 48% ± 6 | 0% ± 0 | 0% ± 0 |
| E: verify k=2 | 30% ± 12 | 0% ± 0 | 1% ± 2 | 4% ± 2 | 0% ± 0 | 0% ± 0 |
| E: verify k=2: model without verify | 37% ± 9 | 0% ± 0 | 4% ± 3 | 46% ± 2 | 0% ± 0 | 0% ± 0 |
| G: C + stop if no branch passes | 17% ± 5 | 0% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| G: C + stop if no branch passes: model without verify | 39% ± 4 | 2% ± 2 | 6% ± 4 | 51% ± 4 | 0% ± 0 | 0% ± 0 |
| B: no verify, 2x budget | 27% ± 7 | 2% ± 2 | 2% ± 2 | 43% ± 7 | 0% ± 0 | 0% ± 0 |
| C: verify k=4, 2x budget | 17% ± 3 | 0% ± 0 | 0% ± 0 | 2% ± 2 | 0% ± 0 | 0% ± 0 |
