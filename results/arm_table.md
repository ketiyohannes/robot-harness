# Arm task results (arm_push_safely)

| Method | seeds | train budget | success | episodes with a safety violation | real steps / eval episode | model's #1 overturned | model's #1 failed verification |
|---|---|---|---|---|---|---|---|
| C: WM+CEM+verify (k=4) | 1 | 101k | 70% ± 0 | 3% ± 0 | 1040 | 57% ± 0 | 17% ± 0 |

## Training phase

| Method | training episodes | training episodes with a safety violation (per seed) | share | real steps to rolling-10 success ≥ 50% | train wall-clock (min) |
|---|---|---|---|---|---|
| C: WM+CEM+verify (k=4) | 92 | 2.00 ± 0.00 | 2% | 32.0k | 20.73 ± 0.00 |

## Per-clause violation rate (eval episodes)

| Method | reach_target | cup_upright | cup_still | puck_slow | puck_on_table | ee_in_workspace |
|---|---|---|---|---|---|---|
| C: WM+CEM+verify (k=4) | 27% ± 0 | 0% ± 0 | 0% ± 0 | 3% ± 0 | 0% ± 0 | 0% ± 0 |
