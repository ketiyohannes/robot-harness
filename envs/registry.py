"""Environment registry: maps a spec's ``env`` field to the env class and its state layout."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvInfo:
    name: str
    state_dim: int
    act_dim: int
    unc_idx: tuple[int, ...]       # state dims averaged for the ensemble-disagreement penalty
    vis_idx: tuple[int, ...]       # state dims stored in traces for drawing (positions)
    tool_idx: tuple[int, ...]      # pusher / end-effector position dims
    object_idx: tuple[int, ...]    # pushed object position dims (planner approach term)
    object_xy: tuple[int, int]     # pushed object xy (planner progress term)


ENVS = {
    "push": EnvInfo("push", 10, 2, (0, 1, 4, 5), (0, 1, 4, 5), (0, 1), (4, 5), (4, 5)),
    "arm_push": EnvInfo("arm_push", 21, 3, (0, 1, 6, 7, 12, 13), (0, 1, 2, 6, 7, 8, 12, 13),
                        (0, 1, 2), (6, 7, 8), (6, 7)),
}


def env_info(name: str) -> EnvInfo:
    return ENVS[name]


def make_env(name: str, **kwargs):
    if name == "push":
        from envs.push_env import PushEnv
        return PushEnv(**kwargs)
    if name == "arm_push":
        from envs.arm_push_env import ArmPushEnv
        return ArmPushEnv(**kwargs)
    raise ValueError(f"unknown env {name}")
