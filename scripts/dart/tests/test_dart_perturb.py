"""Pure (numpy-only) tests for robofactory.planner.dart_perturb.

No sapien / gym / torch / SAPIEN sim is touched. FakeRobot returns a fixed
9-vector qpos (7 arm + 2 gripper joints); FakeEnv records every step call.
"""

import copy

import numpy as np
import pytest

from robofactory.planner.dart_perturb import (
    inject_joint_disturbance,
    sample_floored_offset,
)


# ---- fakes -----------------------------------------------------------------


class FakeRobot:
    """Returns a FIXED 9-vector qpos (7 arm joints + 2 gripper joints)."""

    def __init__(self, q):
        self._q = np.asarray(q, dtype=np.float64)

    def get_qpos(self):
        return self._q.copy()


class FakeTorchTensor:
    """Mimics a torch tensor with a leading batch dim and .cpu().numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr.copy()


class FakeBatchedRobot:
    """get_qpos() returns a (1, 9) torch-like tensor (batch dim + .cpu().numpy())."""

    def __init__(self, q):
        self._q = np.asarray(q, dtype=np.float64).reshape(1, -1)

    def get_qpos(self):
        return FakeTorchTensor(self._q)


class FakeEnv:
    """Records every step(action_dict). Stores deep copies so later mutation of
    the live dict cannot retroactively change what we recorded."""

    def __init__(self):
        self.steps = []

    def step(self, action_dict):
        self.steps.append(copy.deepcopy(action_dict))


# ---- (a) sample_floored_offset: L2 >= floor, direction preserved ----------


def test_floor_enforced_when_sample_small():
    """Tiny sigma forces sub-floor draws -> must be scaled UP to ~floor.

    NOTE: the ported math scales by ``floor / (mag + 1e-9)``, so the result is
    ``mag/(mag+1e-9)*floor`` -- asymptotically the floor but slightly under it
    when ``mag`` itself is tiny (here sigma=1e-6 -> mag~1e-6, shortfall ~5e-4).
    The tolerance below reflects that intentional guard, not a bug. With any
    realistic sigma the shortfall is negligible (see test below)."""
    rng = np.random.default_rng(0)
    floor = 0.5
    for _ in range(200):
        off = sample_floored_offset(rng, sigma=1e-6, n=7, floor=floor)
        assert np.linalg.norm(off) >= floor - 1e-3


def test_floor_enforced_large_sigma():
    """Large sigma usually exceeds floor; still must never be below it."""
    rng = np.random.default_rng(1)
    floor = 0.2
    for _ in range(200):
        off = sample_floored_offset(rng, sigma=1.0, n=7, floor=floor)
        assert np.linalg.norm(off) >= floor - 1e-9


def test_direction_preserved_when_scaled():
    """When a sub-floor vector is scaled UP, its direction is unchanged and the
    resulting norm equals the floor."""
    floor = 1.0

    class _OneRng:
        # deterministic sub-floor raw vector: norm ~= 0.141 < floor
        def normal(self, loc, scale, size):
            v = np.zeros(size, dtype=np.float64)
            v[0] = 0.1
            v[1] = -0.1
            return v

    raw = np.array([0.1, -0.1] + [0.0] * 5, dtype=np.float64)
    out = sample_floored_offset(_OneRng(), sigma=123.0, n=7, floor=floor)
    # norm scaled to (essentially) the floor
    assert np.linalg.norm(out) == pytest.approx(floor, rel=1e-6)
    # direction preserved: out is a positive scalar multiple of raw
    raw_unit = raw / np.linalg.norm(raw)
    out_unit = out / np.linalg.norm(out)
    np.testing.assert_allclose(out_unit, raw_unit, atol=1e-9)


def test_no_scaling_when_above_floor():
    """A draw already above the floor is returned unchanged (no scaling)."""

    class _BigRng:
        def normal(self, loc, scale, size):
            v = np.zeros(size, dtype=np.float64)
            v[0] = 5.0
            return v

    out = sample_floored_offset(_BigRng(), sigma=1.0, n=7, floor=1.0)
    expected = np.zeros(7)
    expected[0] = 5.0
    np.testing.assert_allclose(out, expected)


def test_offset_length():
    off = sample_floored_offset(np.random.default_rng(2), sigma=0.1, n=7, floor=0.1)
    assert off.shape == (7,)


# ---- shared fixtures for inject_joint_disturbance --------------------------

Q0_A = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.91, 0.92])  # arm0 + 2 grip
Q0_B = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.81, 0.82])  # arm1 + 2 grip


def _make(robots_q=(Q0_A, Q0_B)):
    env = FakeEnv()
    robots = [FakeRobot(q) for q in robots_q]
    return env, robots


# ---- (b) exactly K env.step calls -----------------------------------------


@pytest.mark.parametrize("K", [1, 3, 10])
def test_exactly_K_steps(K):
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.1, K=K, floor=0.1,
    )
    assert len(env.steps) == K


# ---- (c) only move_ids arms differ from q0; others held at q0 -------------


def test_only_move_ids_perturbed():
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(7),
        sigma=0.3, K=1, floor=0.2,
    )
    act = env.steps[0]
    arm0 = act["panda-0"][:7]
    arm1 = act["panda-1"][:7]
    # arm 0 perturbed -> differs from its q0[:7]
    assert not np.allclose(arm0, Q0_A[:7])
    # arm 1 held -> exactly q0[:7]
    np.testing.assert_allclose(arm1, Q0_B[:7])


def test_multiple_move_ids():
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0, 1], rng=np.random.default_rng(3),
        sigma=0.3, K=1, floor=0.2,
    )
    act = env.steps[0]
    assert not np.allclose(act["panda-0"][:7], Q0_A[:7])
    assert not np.allclose(act["panda-1"][:7], Q0_B[:7])


def test_move_id_int_scalar():
    """move_ids accepts a bare int (not just a list)."""
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=1, rng=np.random.default_rng(5),
        sigma=0.3, K=1, floor=0.2,
    )
    act = env.steps[0]
    np.testing.assert_allclose(act["panda-0"][:7], Q0_A[:7])  # held
    assert not np.allclose(act["panda-1"][:7], Q0_B[:7])      # perturbed


def test_empty_move_ids_all_held():
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[], rng=np.random.default_rng(9),
        sigma=0.3, K=2, floor=0.2,
    )
    for act in env.steps:
        np.testing.assert_allclose(act["panda-0"][:7], Q0_A[:7])
        np.testing.assert_allclose(act["panda-1"][:7], Q0_B[:7])


def test_perturbed_offset_respects_floor():
    """The perturbed arm's net displacement from q0 has L2 >= floor."""
    env, robots = _make()
    grips = [0.04, -0.02]
    floor = 0.5
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(11),
        sigma=1e-6, K=1, floor=floor,  # tiny sigma -> floor must kick in
    )
    disp = env.steps[0]["panda-0"][:7] - Q0_A[:7]
    # tolerance reflects the +1e-9 floor guard (see sample_floored_offset).
    assert np.linalg.norm(disp) >= floor - 1e-3


def test_held_target_constant_across_K():
    """Same drifted/held target is commanded for every one of the K steps."""
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(13),
        sigma=0.3, K=5, floor=0.2,
    )
    first = env.steps[0]
    for act in env.steps[1:]:
        np.testing.assert_allclose(act["panda-0"], first["panda-0"])
        np.testing.assert_allclose(act["panda-1"], first["panda-1"])


# ---- (d) gripper channel == grips[i] (locks carry-grip fix) ---------------


def test_gripper_channel_carries_caller_grip_pd_joint_pos():
    env, robots = _make()
    grips = [0.037, -0.021]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=4, floor=0.2, control_mode="pd_joint_pos",
    )
    for act in env.steps:
        # pd_joint_pos action = [7 arm joints, grip] -> last entry is grip
        assert act["panda-0"].shape == (8,)
        assert act["panda-0"][-1] == grips[0]
        assert act["panda-1"][-1] == grips[1]


def test_gripper_channel_carries_caller_grip_pd_joint_pos_vel():
    env, robots = _make()
    grips = [0.05, -0.05]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=2, floor=0.2, control_mode="pd_joint_pos_vel",
    )
    for act in env.steps:
        # pd_joint_pos_vel action = [7 pos, 7 vel(=0), grip] -> 15 entries
        assert act["panda-0"].shape == (15,)
        np.testing.assert_allclose(act["panda-0"][7:14], 0.0)  # vel block zeroed
        assert act["panda-0"][-1] == grips[0]
        assert act["panda-1"][-1] == grips[1]


def test_module_does_not_read_planner_gripper_state():
    """Sanity: the only gripper source is grips[i]. We feed a sentinel grip and
    confirm it (and nothing else) appears in the gripper channel."""
    env, robots = _make()
    sentinel = [0.123456, 0.654321]
    inject_joint_disturbance(
        env, robots, sentinel, move_ids=[0, 1], rng=np.random.default_rng(0),
        sigma=0.3, K=1, floor=0.2,
    )
    act = env.steps[0]
    assert act["panda-0"][-1] == sentinel[0]
    assert act["panda-1"][-1] == sentinel[1]


# ---- (e) sink captures K action dicts -------------------------------------


def test_sink_captures_K_action_dicts():
    env, robots = _make()
    grips = [0.04, -0.02]
    sink = []
    K = 6
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=K, floor=0.2, sink=sink,
    )
    assert len(sink) == K
    # each captured entry is a dict with both arm keys
    for entry in sink:
        assert set(entry.keys()) == {"panda-0", "panda-1"}
    # deep copies: each entry equals the actually-stepped action
    for captured, stepped in zip(sink, env.steps):
        np.testing.assert_allclose(captured["panda-0"], stepped["panda-0"])
        np.testing.assert_allclose(captured["panda-1"], stepped["panda-1"])


def test_sink_entries_are_independent_deep_copies():
    """Mutating one sink entry must not change others (true deep copies)."""
    env, robots = _make()
    grips = [0.04, -0.02]
    sink = []
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=3, floor=0.2, sink=sink,
    )
    sink[0]["panda-0"][:] = -999.0
    assert not np.allclose(sink[1]["panda-0"], -999.0)


def test_no_sink_is_ok():
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=2, floor=0.2, sink=None,
    )
    assert len(env.steps) == 2


# ---- misc: action_prefix + torch-tensor / batched qpos tolerance ----------


def test_custom_action_prefix():
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=1, floor=0.2, action_prefix="agent",
    )
    assert set(env.steps[0].keys()) == {"agent-0", "agent-1"}


def test_tolerates_batched_torch_qpos():
    """Robot returning a (1,9) torch-like tensor must be sliced to first 7."""
    env = FakeEnv()
    robots = [FakeBatchedRobot(Q0_A), FakeBatchedRobot(Q0_B)]
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[], rng=np.random.default_rng(0),
        sigma=0.3, K=1, floor=0.2,
    )
    act = env.steps[0]
    np.testing.assert_allclose(act["panda-0"][:7], Q0_A[:7])
    np.testing.assert_allclose(act["panda-1"][:7], Q0_B[:7])
    assert act["panda-0"].shape == (8,)
