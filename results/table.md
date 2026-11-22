| Method | seeds | train budget (real steps) | eval success | eval safety-violation eps | real steps / eval ep (committed + branch) | #1 overturned | #1 failed verification |
|---|---|---|---|---|---|---|---|
| A: PPO @ 60k | 3 | 60k | 0% ± 0 | 0% ± 0 | 100 (100 + 0) | - | - |
| A: PPO @ 1001k (final) | 3 | 1001k | 0% ± 0 | 0% ± 0 | 100 (100 + 0) | - | - |
| A2: SAC @ 60k | 3 | 60k | 0% ± 0 | 9% ± 6 | 100 (100 + 0) | - | - |
| B: WM+CEM, no verify (k=1) | 3 | 60k | 83% ± 3 | 4% ± 3 | 100 (100 + 0) | - | - |
| C: WM+CEM+verify (k=4) | 3 | 60k | 83% ± 7 | 0% ± 0 | 860 (100 + 760) | 40% ± 4 | 6% ± 2 |
| D: C with lambda=0 | 3 | 60k | 91% ± 2 | 0% ± 0 | 860 (100 + 760) | 39% ± 2 | 4% ± 1 |
| E: verify k=2 | 3 | 60k | 92% ± 2 | 1% ± 2 | 480 (100 + 380) | 30% ± 3 | 4% ± 0 |
| F: verify k=8 | 3 | 60k | 93% ± 0 | 0% ± 0 | 1620 (100 + 1520) | 50% ± 1 | 3% ± 0 |
| A2: SAC (long) @ 300k (final) | 2 | 300k | 0% ± 0 | 3% ± 3 | 100 (100 + 0) | - | - |
| C: verify k=4, 120k budget | 2 | 120k | 97% ± 3 | 0% ± 0 | 860 (100 + 760) | 35% ± 1 | 2% ± 0 |
| [Task 1: push] C (k=4) | 2 | 60k | 100% ± 0 | 0% ± 0 | 860 (100 + 760) | 59% ± 1 | 0% ± 0 |

Per-clause violation rate on held-out eval episodes:

| Method | reach_target | pusher_in_region | block_slow | block_on_table |
|---|---|---|---|---|
| A: PPO @ 60k | 100% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| A: PPO @ 1001k (final) | 100% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| A2: SAC @ 60k | 98% ± 2 | 3% ± 3 | 6% ± 3 | 0% ± 0 |
| B: WM+CEM, no verify (k=1) | 12% ± 6 | 0% ± 0 | 4% ± 3 | 0% ± 0 |
| C: WM+CEM+verify (k=4) | 17% ± 7 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| D: C with lambda=0 | 9% ± 2 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| E: verify k=2 | 7% ± 3 | 0% ± 0 | 1% ± 2 | 0% ± 0 |
| F: verify k=8 | 7% ± 0 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| A2: SAC (long) @ 300k (final) | 100% ± 0 | 0% ± 0 | 3% ± 3 | 0% ± 0 |
| C: verify k=4, 120k budget | 3% ± 3 | 0% ± 0 | 0% ± 0 | 0% ± 0 |
| [Task 1: push] C (k=4) | 0% ± 0 | 0% ± 0 | nan% ± nan | nan% ± nan |

Training-phase metrics (model-based methods):

| Method | train episodes | train success | train eps w/ committed safety violation | real steps to rolling-10 success >= 50% (per seed) | train wall-clock (min) |
|---|---|---|---|---|---|
| B: WM+CEM, no verify (k=1) | 550 ± 0 | 53% ± 2 | 28% ± 1 | 21300, 15100, 13800 | 79.8 ± 0.2 |
| C: WM+CEM+verify (k=4) | 64 ± 0 | 56% ± 8 | 10% ± 1 | 26500, 20480, 21340 | 11.6 ± 0.1 |
| D: C with lambda=0 | 64 ± 0 | 58% ± 8 | 9% ± 3 | 26500, 17900, 25640 | 11.8 ± 0.2 |
| E: verify k=2 | 115 ± 0 | 53% ± 4 | 15% ± 2 | 31400, 20360, 29960 | 31.5 ± 0.2 |
| F: verify k=8 | 34 ± 0 | 66% ± 4 | 12% ± 2 | 30920, 27680, 26060 | 21.6 ± 0.1 |
| C: verify k=4, 120k budget | 134 ± 0 | 74% ± 3 | 5% ± 0 | 26500, 20480 | 39.0 ± 0.2 |
| [Task 1: push] C (k=4) | 64 ± 0 | 87% ± 2 | 0% ± 0 | 17040, 15320 | 21.6 ± 0.1 |
