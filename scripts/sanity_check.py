"""Phase 0 sanity check: load the MuJoCo push env, take 1,000 random steps, log FPS, render one frame."""
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.push_env import PushEnv  # noqa: E402


def main():
    print(f"python {sys.version.split()[0]} | {platform.platform()} | {platform.machine()}")
    print(f"mujoco {mujoco.__version__} | torch {torch.__version__} | mps={torch.backends.mps.is_available()}")
    env = PushEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 0
    while n < 1000:
        _, done, _ = env.step(rng.uniform(-1, 1, 2))
        n += 1
        if done:
            env.reset(seed=n)
    dt = time.perf_counter() - t0
    print(f"1000 random env steps (frame_skip=10) in {dt:.3f}s -> {1000 / dt:.0f} env steps/s")
    img = env.render()
    out = Path(__file__).resolve().parents[1] / "results" / "sanity_frame.png"
    import imageio.v2 as imageio
    imageio.imwrite(out, img)
    print(f"rendered frame {img.shape}, mean pixel {img.mean():.1f} -> {out}")
    env.close()


if __name__ == "__main__":
    main()
