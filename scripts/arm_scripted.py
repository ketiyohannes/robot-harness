"""Hand-written push controller for the arm task (feasibility check only, not a baseline).

Phases: lift to a safe height, fly above the pre-push point behind the puck, descend, push
toward the target at a moderate speed. It ignores the cup except by flying high.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.arm_push_env import EE_VMAX, PUCK_RADIUS, ArmPushEnv  # noqa: E402
from verifier.verifier import Spec  # noqa: E402


class ScriptedPusher:
    def __init__(self, safe_z=0.15, push_z=0.03, push_speed=0.4):
        self.safe_z, self.push_z, self.push_speed = safe_z, push_z, push_speed
        self.phase = 0

    def reset(self):
        self.phase = 0

    def __call__(self, state, target):
        ee, puck = state[0:3], state[6:8]
        tgt = target[:2]
        d = tgt - puck
        u = d / (np.linalg.norm(d) + 1e-9)
        pre = puck - u * (PUCK_RADIUS + 0.035)

        def go(p, gain=4.0, vmax=1.0):
            return np.clip(gain * (p - ee) / EE_VMAX, -vmax, vmax)

        if self.phase == 0:  # lift
            if ee[2] > self.safe_z - 0.01:
                self.phase = 1
            return go(np.array([ee[0], ee[1], self.safe_z]))
        if self.phase == 1:  # fly above pre-push point
            p = np.array([pre[0], pre[1], self.safe_z])
            if np.linalg.norm(p - ee) < 0.01:
                self.phase = 2
            return go(p)
        if self.phase == 2:  # descend
            p = np.array([pre[0], pre[1], self.push_z])
            if np.linalg.norm(p - ee) < 0.008:
                self.phase = 3
            return go(p)
        # push: aim the fingertip through the puck centre toward the target
        goal = np.concatenate([tgt - u * (PUCK_RADIUS + 0.01), [self.push_z]])
        lateral = (puck - ee[:2]) - np.dot(puck - ee[:2], u) * u
        a = go(goal, vmax=self.push_speed)
        a[:2] += np.clip(3.0 * lateral / EE_VMAX, -0.3, 0.3)
        return np.clip(a, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--spec", default="arm_push_safely")
    args = ap.parse_args()
    env, spec, pol = ArmPushEnv(), Spec.load(args.spec), ScriptedPusher()
    succ, viol = [], {c.name: 0 for c in spec.clauses}
    for ep in range(args.episodes):
        env.reset(10_000 + ep)
        pol.reset()
        states = [env.get_state()]
        done = False
        while not done:
            _, done, _ = env.step(pol(env.get_state(), env.target))
            states.append(env.get_state())
        ok, rob, per = spec.verify(np.array(states), env.target)
        succ.append(ok)
        for k, v in per.items():
            viol[k] += int(v <= 0)
    print(f"scripted success {np.mean(succ):.2f} over {args.episodes}; clause failures {viol}")


if __name__ == "__main__":
    main()
