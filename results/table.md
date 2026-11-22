| Method | seeds | train budget (real steps) | eval success | eval safety-violation eps | real steps / eval ep (committed + branch) | #1 overturned | #1 failed verification |
|---|---|---|---|---|---|---|---|
| C: WM+CEM+verify (k=4) | 3 | 60k | 83% ± 7 | 0% ± 0 | 860 (100 + 760) | 40% ± 4 | 6% ± 2 |
| D: C with lambda=0 | 3 | 60k | 91% ± 2 | 0% ± 0 | 860 (100 + 760) | 39% ± 2 | 4% ± 1 |

Per-clause violation rate on held-out eval episodes:

| Method | reach_target | pusher_in_region | block_slow | block_on_table |
|---|---|---|---|---|
| C: WM+CEM+verify (k=4) | 17% ± 7 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| D: C with lambda=0 | 9% ± 2 | 0% ± 0 | 0% ± 0 | 0% ± 0 |

Training-phase metrics (model-based methods):

| Method | train episodes | train success | train eps w/ committed safety violation | real steps to rolling-10 success >= 50% (per seed) | train wall-clock (min) |
|---|---|---|---|---|---|
| C: WM+CEM+verify (k=4) | 64 ± 0 | 56% ± 8 | 10% ± 1 | 26500, 20480, 21340 | 11.6 ± 0.1 |
| D: C with lambda=0 | 64 ± 0 | 58% ± 8 | 9% ± 3 | 26500, 17900, 25640 | 11.8 ± 0.2 |
