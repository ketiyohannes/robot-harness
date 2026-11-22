"""Unit tests for the STL subset and the push specs (hand-made passing/failing trajectories)."""
import numpy as np
import pytest

from envs.push_env import STATE_DIM, TARGET_RADIUS
from verifier.stl import Always, And, Eventually, Not, Or, Predicate
from verifier.verifier import Spec

T = 101
TARGET = np.array([0.2, 0.1])


def _traj(block_path, pusher=(0.0, 0.0), block_v=(0.0, 0.0), bz=0.02):
    s = np.zeros((T, STATE_DIM))
    s[:, 0:2] = pusher
    s[:, 4:6] = block_path
    s[:, 6] = bz
    s[:, 7:9] = block_v
    return s


def _line(a, b):
    return np.linspace(a, b, T)


# ---------------------------------------------------------------- STL core
def _sig_pred(values):
    return Predicate("p", lambda ctx: np.asarray(values, dtype=float))


def test_stl_operators_on_known_signal():
    p = _sig_pred([-1.0, 2.0, 0.5, -3.0])
    ctx = {}
    assert Always(p).robustness(ctx) == -3.0
    assert Eventually(p).robustness(ctx) == 2.0
    assert Eventually(p, 2, 3).robustness(ctx) == 0.5
    assert Always(p, 1, 2).robustness(ctx) == 0.5
    assert Not(p).robustness(ctx) == 1.0
    q = _sig_pred([0.3, 0.3, 0.3, 0.3])
    assert And([p, q]).robustness(ctx) == -1.0
    assert Or([p, q]).robustness(ctx) == 0.3
    # Nested: F[0,1] G[0,1] p  -> max over t in {0,1} of min(p_t, p_{t+1}) = max(-1, 0.5) = 0.5
    assert Eventually(Always(p, 0, 1), 0, 1).robustness(ctx) == 0.5


def test_stl_batched_shapes():
    vals = np.array([[1.0, -1.0, 2.0], [0.0, 0.5, 0.25]])
    p = _sig_pred(vals)
    np.testing.assert_allclose(Always(p).robustness({}), [-1.0, 0.0])
    np.testing.assert_allclose(Eventually(p).robustness({}), [2.0, 0.5])


# ---------------------------------------------------------------- push spec
def test_push_passes_when_block_reaches_target():
    spec = Spec.load("push")
    ok, rob, per = spec.verify(_traj(_line([0.0, 0.0], TARGET)), TARGET)
    assert ok and rob > 0
    assert per["reach_target"] == pytest.approx(TARGET_RADIUS)


def test_push_fails_when_block_never_arrives():
    spec = Spec.load("push")
    ok, rob, per = spec.verify(_traj(_line([0.0, 0.0], [0.0, 0.05])), TARGET)
    assert not ok and per["reach_target"] < 0 and per["pusher_in_region"] > 0


def test_push_fails_when_pusher_leaves_region():
    spec = Spec.load("push")
    s = _traj(_line([0.0, 0.0], TARGET))
    s[30, 0] = 0.48
    ok, rob, per = spec.verify(s, TARGET)
    assert not ok and per["pusher_in_region"] == pytest.approx(0.45 - 0.48)
    assert per["reach_target"] > 0


def test_push_safely_speed_clause():
    spec = Spec.load("push_safely")
    s = _traj(_line([0.0, 0.0], TARGET), block_v=(0.1, 0.0))
    ok, _, per = spec.verify(s, TARGET)
    assert ok, per
    s[10, 7] = 0.5
    ok, _, per = spec.verify(s, TARGET)
    assert not ok and per["block_slow"] == pytest.approx(0.30 - 0.5)


def test_push_safely_block_falls_off_table():
    spec = Spec.load("push_safely")
    s = _traj(_line([0.0, 0.0], [0.6, 0.0]))
    ok, _, per = spec.verify(s, TARGET)
    assert not ok and per["block_on_table"] < 0


def test_batch_matches_single():
    spec = Spec.load("push_safely")
    rng = np.random.default_rng(0)
    batch = rng.normal(0, 0.2, size=(8, T, STATE_DIM))
    per_b = spec.batch_clause_robustness(batch, TARGET)
    for i in range(8):
        _, _, per = spec.verify(batch[i], TARGET)
        for k, v in per.items():
            assert per_b[k][i] == pytest.approx(v)
