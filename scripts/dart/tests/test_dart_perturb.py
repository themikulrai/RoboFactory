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


# ---- (f) grasp-protection SETTLE window (pre_hold_steps) -------------------


# hold setpoints DISTINCT from the live q0 so a hold step (== hold_qpos, no offset)
# is unmistakably different from a shove step (== q0 + floored offset).
HOLD_A = np.array([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6])  # arm0 frozen setpoint
HOLD_B = np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6])  # arm1 frozen setpoint


def test_pre_hold_default_zero_is_legacy_behavior():
    """pre_hold_steps defaults to 0 -> exactly K steps, no settle phase (legacy)."""
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=4, floor=0.2,
    )
    assert len(env.steps) == 4  # no pre-hold steps prepended


@pytest.mark.parametrize("G,K", [(10, 5), (1, 1), (3, 8)])
def test_settle_then_shove_step_count_and_order(G, K):
    """With pre_hold_steps=G: env gets G hold steps (ALL arms at hold_qpos, NO offset)
    FOLLOWED BY K shove steps (move arm = q0+offset), in that order."""
    env, robots = _make()
    grips = [0.037, -0.021]
    hold_qpos = [HOLD_A.copy(), HOLD_B.copy()]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(7),
        sigma=0.3, K=K, floor=0.2, hold_qpos=hold_qpos, pre_hold_steps=G,
    )
    assert len(env.steps) == G + K, "total steps must be pre_hold_steps + K"

    # phase 1: the FIRST G steps hold BOTH arms at hold_qpos with NO offset.
    for s in env.steps[:G]:
        np.testing.assert_allclose(s["panda-0"][:7], HOLD_A)  # move arm: NO offset
        np.testing.assert_allclose(s["panda-1"][:7], HOLD_B)  # non-move arm held
        assert s["panda-0"][-1] == grips[0]  # grip carried in settle
        assert s["panda-1"][-1] == grips[1]

    # phase 2: the LAST K steps shove the move arm (q0 + offset, != hold AND != q0
    # is not required, but it must DIFFER from the settle hold) and hold arm1.
    for s in env.steps[G:]:
        assert not np.allclose(s["panda-0"][:7], HOLD_A), \
            "shove step must differ from the settle hold (offset applied)"
        # move arm shove target = live q0 (Q0_A) + floored offset -> differs from q0 too
        assert not np.allclose(s["panda-0"][:7], Q0_A[:7])
        np.testing.assert_allclose(s["panda-1"][:7], HOLD_B)  # arm1 still held
        assert s["panda-0"][-1] == grips[0]  # grip carried through the shove
        assert s["panda-1"][-1] == grips[1]


def test_settle_holds_move_arm_at_setpoint_no_offset():
    """Specifically: during the settle window the MOVE arm holds its setpoint with
    NO offset (the whole point of the grasp-protection window)."""
    env, robots = _make()
    grips = [0.04, -0.02]
    hold_qpos = [HOLD_A.copy(), HOLD_B.copy()]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(3),
        sigma=1.0, K=2, floor=0.5, hold_qpos=hold_qpos, pre_hold_steps=4,
    )
    # every one of the 4 settle steps holds the move arm exactly at HOLD_A.
    for s in env.steps[:4]:
        np.testing.assert_allclose(s["panda-0"][:7], HOLD_A)


def test_settle_uses_live_qpos_when_no_hold_qpos():
    """If hold_qpos is None, the settle phase holds each arm at its LIVE qpos."""
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[1], rng=np.random.default_rng(1),
        sigma=0.3, K=1, floor=0.2, hold_qpos=None, pre_hold_steps=3,
    )
    for s in env.steps[:3]:
        np.testing.assert_allclose(s["panda-0"][:7], Q0_A[:7])  # held at live qpos
        np.testing.assert_allclose(s["panda-1"][:7], Q0_B[:7])  # move arm: no offset yet


def test_settle_grip_carried_pd_joint_pos_vel():
    """Grip carried (and vel block zeroed) through BOTH settle and shove under
    pd_joint_pos_vel."""
    env, robots = _make()
    grips = [0.05, -0.05]
    hold_qpos = [HOLD_A.copy(), HOLD_B.copy()]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=2, floor=0.2, hold_qpos=hold_qpos, pre_hold_steps=2,
        control_mode="pd_joint_pos_vel",
    )
    assert len(env.steps) == 4
    for s in env.steps:  # all settle + shove steps
        assert s["panda-0"].shape == (15,)
        np.testing.assert_allclose(s["panda-0"][7:14], 0.0)  # vel block zeroed
        assert s["panda-0"][-1] == grips[0]
        assert s["panda-1"][-1] == grips[1]


def test_settle_sink_orders_holds_then_shoves():
    """sink captures pre_hold_steps hold dicts FIRST, then K shove dicts, in order,
    matching what was actually stepped."""
    env, robots = _make()
    grips = [0.04, -0.02]
    hold_qpos = [HOLD_A.copy(), HOLD_B.copy()]
    sink = []
    G, K = 3, 4
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.3, K=K, floor=0.2, hold_qpos=hold_qpos, pre_hold_steps=G, sink=sink,
    )
    assert len(sink) == G + K
    # sink mirrors env.steps exactly (order + content)
    for cap, st in zip(sink, env.steps):
        np.testing.assert_allclose(cap["panda-0"], st["panda-0"])
        np.testing.assert_allclose(cap["panda-1"], st["panda-1"])
    # the first G sink entries are the settle holds (move arm at HOLD_A, no offset)
    for cap in sink[:G]:
        np.testing.assert_allclose(cap["panda-0"][:7], HOLD_A)
    # the last K differ (offset applied)
    for cap in sink[G:]:
        assert not np.allclose(cap["panda-0"][:7], HOLD_A)


def test_settle_does_not_consume_extra_rng():
    """The settle phase consumes NO rng: the SHOVE offset is identical whether the
    settle window is 0 or G (callers' per-transition RNG streams stay unchanged)."""
    # run A: no settle window
    envA, robotsA = _make()
    inject_joint_disturbance(
        envA, robotsA, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(123),
        sigma=0.4, K=3, floor=0.2, hold_qpos=[HOLD_A.copy(), HOLD_B.copy()],
        pre_hold_steps=0,
    )
    # run B: settle window of 5; SAME seed
    envB, robotsB = _make()
    inject_joint_disturbance(
        envB, robotsB, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(123),
        sigma=0.4, K=3, floor=0.2, hold_qpos=[HOLD_A.copy(), HOLD_B.copy()],
        pre_hold_steps=5,
    )
    # the SHOVE target (the move arm in the post-settle steps) must be identical:
    # run A's first shove step vs run B's first POST-settle shove step.
    shove_A = envA.steps[0]["panda-0"][:7]
    shove_B = envB.steps[5]["panda-0"][:7]
    np.testing.assert_allclose(shove_A, shove_B,
                               err_msg="settle window must not perturb the rng stream")


# ---- (g) shove_joints: PROXIMAL-only offset mask --------------------------


def test_shove_joints_proximal_masks_wrist():
    """shove_joints=(0,1,2,3): the move arm's commanded target has an offset applied
    ONLY at joints 0-3; joints 4,5,6 hold q0 EXACTLY (no offset). Non-move arm held."""
    env, robots = _make()
    grips = [0.04, -0.02]
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(7),
        sigma=0.5, K=1, floor=0.3, shove_joints=(0, 1, 2, 3),
    )
    arm0 = env.steps[0]["panda-0"][:7]
    # joints 4,5,6 unchanged from q0 (wrist holds its grip geometry)
    np.testing.assert_allclose(arm0[4:7], Q0_A[4:7])
    # joints 0-3 perturbed (at least one differs from q0)
    assert not np.allclose(arm0[:4], Q0_A[:4])
    # non-move arm fully held
    np.testing.assert_allclose(env.steps[0]["panda-1"][:7], Q0_B[:7])


def test_shove_joints_offset_equals_full_offset_on_selected_indices():
    """The masked offset at the SELECTED indices is bit-identical to the unmasked
    offset there (same rng draw, just zeroed elsewhere). This pins that masking does
    NOT renormalize or perturb the rng stream -- it only zeros the unselected joints."""
    shove = (0, 1, 2, 3)
    # masked run
    envM, robotsM = _make()
    inject_joint_disturbance(
        envM, robotsM, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(99),
        sigma=0.5, K=1, floor=0.3, shove_joints=shove,
    )
    # full (None) run, SAME seed -> same underlying offset draw
    envF, robotsF = _make()
    inject_joint_disturbance(
        envF, robotsF, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(99),
        sigma=0.5, K=1, floor=0.3, shove_joints=None,
    )
    offM = envM.steps[0]["panda-0"][:7] - Q0_A[:7]
    offF = envF.steps[0]["panda-0"][:7] - Q0_A[:7]
    # selected indices identical; unselected zeroed in the masked run
    np.testing.assert_allclose(offM[list(shove)], offF[list(shove)])
    np.testing.assert_allclose(offM[4:7], 0.0)
    # the full run perturbs the wrist (so the mask genuinely changed something)
    assert not np.allclose(offF[4:7], 0.0)


def test_shove_joints_none_is_legacy_all_seven():
    """shove_joints=None (default) -> all 7 joints get the offset (legacy behavior)."""
    env, robots = _make()
    inject_joint_disturbance(
        env, robots, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(11),
        sigma=1e-6, K=1, floor=0.7,  # tiny sigma -> floor kicks in, all 7 nonzero
    )
    off = env.steps[0]["panda-0"][:7] - Q0_A[:7]
    # with shove_joints=None and the floor active, every joint carries some offset
    assert np.all(np.abs(off) > 0.0)


def test_shove_joints_single_index():
    """A single-index shove_joints perturbs exactly that joint; all others hold q0."""
    env, robots = _make()
    inject_joint_disturbance(
        env, robots, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(5),
        sigma=0.5, K=1, floor=0.3, shove_joints=(2,),
    )
    off = env.steps[0]["panda-0"][:7] - Q0_A[:7]
    assert abs(off[2]) > 0.0
    mask = np.ones(7, dtype=bool)
    mask[2] = False
    np.testing.assert_allclose(off[mask], 0.0)


def test_shove_joints_held_constant_across_K():
    """The proximal-masked target is held identical for all K shove steps."""
    env, robots = _make()
    inject_joint_disturbance(
        env, robots, [0.04, -0.02], move_ids=[0], rng=np.random.default_rng(13),
        sigma=0.4, K=5, floor=0.2, shove_joints=(0, 1, 2, 3),
    )
    first = env.steps[0]["panda-0"]
    for act in env.steps[1:]:
        np.testing.assert_allclose(act["panda-0"], first)


def test_shove_joints_with_settle_window():
    """With a settle window: the G hold steps apply NO offset anywhere; the K shove
    steps apply offset ONLY at the proximal indices of the move arm."""
    env, robots = _make()
    grips = [0.04, -0.02]
    hold_qpos = [HOLD_A.copy(), HOLD_B.copy()]
    G, K = 3, 2
    inject_joint_disturbance(
        env, robots, grips, move_ids=[0], rng=np.random.default_rng(0),
        sigma=0.5, K=K, floor=0.3, hold_qpos=hold_qpos, pre_hold_steps=G,
        shove_joints=(0, 1, 2, 3),
    )
    assert len(env.steps) == G + K
    # settle: move arm holds HOLD_A exactly (no offset at any joint)
    for s in env.steps[:G]:
        np.testing.assert_allclose(s["panda-0"][:7], HOLD_A)
    # shove: wrist (4,5,6) holds the move arm's LIVE q0 (the shove target = q0+offset,
    # offset masked to 0 on the wrist -> wrist == Q0_A[4:7]); proximal differs.
    for s in env.steps[G:]:
        np.testing.assert_allclose(s["panda-0"][4:7], Q0_A[4:7])
        assert not np.allclose(s["panda-0"][:4], Q0_A[:4])


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
