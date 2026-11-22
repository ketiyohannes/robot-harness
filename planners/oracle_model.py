"""Diagnostic 'perfect world model': rolls candidate plans out in a private copy of the sim.

Used only to check whether the planning objective itself can solve the task (upper bound);
it is not one of the compared methods. Its rollouts are NOT counted as real steps of any
method because it is a diagnostic.
"""
import numpy as np

from envs.push_env import PushEnv


class OracleModel:
    n_members = 1

    def __init__(self, src_env: PushEnv):
        self.src = src_env
        self.sim = PushEnv()

    def rollout(self, state0, actions):
        import mujoco
        buf = np.empty(self.src._state_size)
        mujoco.mj_getState(self.src.model, self.src.data, buf, self.src._STATE_SPEC)
        out = np.empty((1, actions.shape[0], actions.shape[1] + 1, 10))
        for n in range(actions.shape[0]):
            mujoco.mj_setState(self.sim.model, self.sim.data, buf, self.src._STATE_SPEC)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            out[0, n, 0] = self.sim.get_state()
            for h in range(actions.shape[1]):
                self.sim.step(actions[n, h])
                out[0, n, h + 1] = self.sim.get_state()
        return out
