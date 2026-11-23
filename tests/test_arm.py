"""Arm task: snapshot/restore determinism, spec sanity and agent accounting."""
import numpy as np
import torch

from envs.arm_push_env import EPISODE_STEPS, STATE_DIM, ArmPushEnv
from planners.agent import AgentConfig, Dataset, IVAAgent
from planners.cem import CEMConfig
from verifier.verifier import Spec
from world_model.ensemble import DynamicsEnsemble


def _actions(n, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, 3))


def test_replay_is_bit_exact():
    env = ArmPushEnv()
    acts = _actions(60)
    runs = []
    for _ in range(2):
        env.reset(123)
        runs.append(np.array([env.step(a)[0] for a in acts]))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_snapshot_restore_is_bit_exact_with_contacts():
    env = ArmPushEnv()
    env.reset(7)
    # Drive down onto the table and into the scene so contacts are active at the snapshot.
    for _ in range(25):
        env.step([0.3, 0.0, -1.0])
    sid = env.save_snapshot()
    acts = _actions(40, seed=3)
    a = np.array([env.step(x)[0] for x in acts])
    env.restore_snapshot(sid)
    b = np.array([env.step(x)[0] for x in acts])
    np.testing.assert_array_equal(a, b)
    assert env.get_state().shape == (STATE_DIM,)


def test_spec_untouched_scene_is_safe_but_not_successful():
    env, spec = ArmPushEnv(), Spec.load("arm_push_safely")
    env.reset(10000)
    states = [env.get_state()] + [env.step([0, 0, 0])[0][:STATE_DIM] for _ in range(EPISODE_STEPS)]
    ok, rob, per = spec.verify(np.array(states), env.target)
    assert not ok and per["reach_target"] < 0
    assert all(per[c] > 0 for c in spec.safety_clauses)


def test_agent_accounting_and_branch_replay():
    torch.manual_seed(0)
    spec = Spec.load("arm_push_safely")
    cem = CEMConfig(population=16, iterations=2, elites=4)
    agent = IVAAgent(DynamicsEnsemble(hidden=32, state_dim=21, act_dim=3), spec, AgentConfig(k_verify=3, cem=cem))
    env, data = ArmPushEnv(), Dataset()
    rec = agent.run_episode(env, 10001, data, trace=True)
    assert rec["committed_steps"] == EPISODE_STEPS
    assert env.real_steps == rec["committed_steps"] + rec["branch_steps"] == len(data)
    for d in rec["trace"]["decisions"]:
        br = {b["idx"]: b for b in d["branches"]}[d["chosen"]]
        np.testing.assert_array_equal(d["committed"], br["real"][:len(d["committed"])])
