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
