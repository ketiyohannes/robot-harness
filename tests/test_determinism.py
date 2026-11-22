"""Phase 1 determinism tests for snapshot/restore.

Run directly (`python tests/test_determinism.py`) to print the divergence numbers that go
into the lab notebook, or via pytest for pass/fail.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.push_env import PushEnv  # noqa: E402


def _replay(env, sid, actions):
    env.restore_snapshot(sid)
    states = []
    for a in actions:
        env.step(a)
        states.append(env.get_state())
    return np.array(states)


def replay_divergence(n_replays=10, n_actions=50, seed=3):
    env = PushEnv()
    env.reset(seed)
    rng = np.random.default_rng(seed)
    # Move a few steps first so the snapshot is mid-episode with contacts possible.
    for _ in range(5):
        env.step(rng.uniform(-1, 1, 2))
    sid = env.save_snapshot()
    # Actions biased toward the block so contacts happen during the replay.
    actions = []
    for _ in range(n_actions):
        actions.append(rng.uniform(-1, 1, 2))
    trajs = np.stack([_replay(env, sid, actions) for _ in range(n_replays)])
    div = np.abs(trajs - trajs[0:1])
    return float(div.max()), float(div.mean())


def save_restore_divergence(seed=5, n=20):
    env = PushEnv()
    env.reset(seed)
    rng = np.random.default_rng(seed)
    acts = [rng.uniform(-1, 1, 2) for _ in range(n)]
    sid = env.save_snapshot()
    first = []
    for a in acts:
        env.step(a)
        first.append(env.get_state())
    env.restore_snapshot(sid)
    second = []
    for a in acts:
        env.step(a)
        second.append(env.get_state())
    d = np.abs(np.array(first) - np.array(second))
    return float(d.max()), float(d.mean())


def contact_replay_divergence(seed=7, n_replays=10):
    """Same as replay_divergence, but with a scripted push so the block is definitely hit."""
    env = PushEnv()
    env.reset(seed)
    sid = env.save_snapshot()
    s = env.get_state()
    trajs = []
    for _ in range(n_replays):
        env.restore_snapshot(sid)
        states = []
        for _ in range(50):
            st = env.get_state()
            d = st[4:6] - st[0:2]
            env.step(np.clip(d / (np.linalg.norm(d) + 1e-9), -1, 1))
            states.append(env.get_state())
        trajs.append(states)
    trajs = np.array(trajs)
    moved = float(np.linalg.norm(trajs[0, -1, 4:6] - s[4:6]))
    div = np.abs(trajs - trajs[0:1])
    return float(div.max()), float(div.mean()), moved


def test_replay_is_deterministic():
    mx, _ = replay_divergence()
    assert mx < 1e-9


def test_save_restore_is_deterministic():
    mx, _ = save_restore_divergence()
    assert mx < 1e-9


def test_contact_replay_is_deterministic():
    mx, _, moved = contact_replay_divergence()
    assert moved > 0.02, "scripted push should move the block"
    assert mx < 1e-9


if __name__ == "__main__":
    print("replay 50 random actions x10: max=%.3e mean=%.3e" % replay_divergence())
    print("save -> 20 steps -> restore -> same 20: max=%.3e mean=%.3e" % save_restore_divergence())
    print("scripted contact push x10: max=%.3e mean=%.3e block moved=%.3f m" % contact_replay_divergence())
