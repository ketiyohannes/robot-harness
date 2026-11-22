"""Agent-level checks: accounting and branch/commit consistency (uses a tiny untrained model)."""
import numpy as np
import torch

from envs.push_env import PushEnv
from planners.agent import AgentConfig, Dataset, IVAAgent
from planners.cem import CEMConfig
from verifier.verifier import Spec
from world_model.ensemble import DynamicsEnsemble


def _agent(k):
    torch.manual_seed(0)
    cem = CEMConfig(population=16, iterations=2, elites=4)
    return IVAAgent(DynamicsEnsemble(hidden=32), Spec.load("push_safely"), AgentConfig(k_verify=k, cem=cem), seed=0)


def test_committed_segment_replays_chosen_branch_exactly():
    env = PushEnv()
    rec = _agent(4).run_episode(env, 10000, Dataset(), trace=True)
    for d in rec["trace"]["decisions"]:
        br = {b["idx"]: b for b in d["branches"]}[d["chosen"]]
        n = len(d["committed"])
        np.testing.assert_array_equal(d["committed"], br["real"][:n])


def test_step_accounting_counts_branches():
    env = PushEnv()
    data = Dataset()
    rec = _agent(4).run_episode(env, 10001, data, trace=False)
    assert rec["committed_steps"] == 100
    assert rec["branch_steps"] == 4 * sum(min(20, 100 - t) for t in range(0, 100, 10))
    assert env.real_steps == rec["committed_steps"] + rec["branch_steps"]
    assert len(data) == env.real_steps  # every real transition is stored


def test_k1_has_no_branches():
    env = PushEnv()
    rec = _agent(1).run_episode(env, 10002, Dataset())
    assert rec["branch_steps"] == 0 and env.real_steps == 100
