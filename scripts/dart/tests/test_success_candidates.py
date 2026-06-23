"""PURE tests for robofactory.tasks.success_candidates (no sim / no sapien).

These use tiny FAKE envs that mimic just the attributes the candidate functions
read (barrier.pose.p, agent.agents[i].robot.get_qpos() [for the gripper-closed
check], cubeA/B/C, goal_region, cube_half_size, goal_radius,
left/right/middle_agent). They verify:

  LiftBarrier (SUSTAINED GEOMETRIC criterion; is_grasping DROPPED):
    * ``lift_barrier_success_strict`` is the PER-FRAME predicate C:
        - z high but ONE gripper OPEN (finger-sum >= GRIPPER_CLOSE_MAX) -> False
            (the grasp-blind case the OLD height-only check WOULD pass)
        - both grippers CLOSED + both ends z high -> True
        - both grippers closed but ends z low -> False
        - a TIPPED bar (one end low) -> False
    * ``arm_gripper_closed`` reads the LAST 2 qpos (finger joints) and thresholds.
    * the ENV ``LiftBarrierEnv.evaluate`` SUSTAINED counter: success only after C
      holds for HOLD_FRAMES_K consecutive env.steps; any break resets the count.
    * ``subtask_interpreter.check_barrier_ends_held`` is the single-frame predicate
      (drops is_grasping; the env counter owns the sustain).
    * is_grasping is NOT called anywhere in the criterion path.

  TSC fixed:
    * the cube-C grasp is read from MIDDLE arm (arm 2), NOT left arm (arm 0)
      -- confirmed by tracking which agent's is_grasping(cubeC) was called.
    * a fully-stacked, placed, all-released state -> True
    * if the MIDDLE arm is still holding cube C    -> False (gated by ~grasped)
    * the BUGGY behavior (left arm holds C, middle arm released) -> the OLD
      check would mis-read left arm and wrongly fail; the FIXED candidate
      correctly returns True (it ignores the left arm's cube-C grasp).

Run PURE:
    cd /iris/u/mikulrai/RoboFactory-subtask-wt && PYTHONPATH=. \\
      /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \\
      scripts/dart/tests/test_success_candidates.py -x -q
"""
from __future__ import annotations

import os.path as osp
import sys

import numpy as np
import torch

# make the repo root importable when pytest runs from elsewhere
_THIS = osp.dirname(osp.abspath(__file__))
_ROOT = osp.dirname(osp.dirname(osp.dirname(_THIS)))  # .../RoboFactory-subtask-wt
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from robofactory.tasks.success_candidates import (  # noqa: E402
    lift_barrier_success_strict,
    arm_gripper_closed,
    tsc_success_fixed,
    barrier_grasp_ends_world,
    BARRIER_END_OFFSETS,
    BARRIER_LIFT_DZ,
    GRIPPER_CLOSE_MAX,
    HOLD_FRAMES_K,
)

# finger-joint sums (sum over the 2 finger joints) for an OPEN vs CLOSED gripper.
# open ~0.08 EACH -> sum ~0.16 (>> GRIPPER_CLOSE_MAX); closed ~0.00 -> sum ~0.0.
_GRIP_OPEN_SUM = 0.16
_GRIP_CLOSED_SUM = 0.0


# ===========================================================================
# Fakes
# ===========================================================================
class FakePose:
    """Holds a (1,3) position tensor at .p and a (1,4) wxyz quaternion at .q
    (matches actor.pose.p / .q batched layout). Default quat is identity."""
    def __init__(self, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        self.p = torch.tensor([list(xyz)], dtype=torch.float32)   # shape (1,3)
        self.q = torch.tensor([list(quat)], dtype=torch.float32)  # shape (1,4) wxyz


class FakeActor:
    def __init__(self, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
        self.pose = FakePose(xyz, quat)


class FakeRobot:
    """A robot exposing pose.p (for base z) and get_qpos() (for the gripper check).

    ``finger_sum`` is the sum of the 2 finger joints; the 9-dim qpos is
    [7 zeros (arm), finger_sum/2, finger_sum/2] so the LAST 2 entries sum to
    ``finger_sum`` (matches the Panda layout 7 arm + 2 fingers).
    """
    def __init__(self, base_z, finger_sum=_GRIP_OPEN_SUM):
        # pose.p[0, 2] is read for the base z
        self.pose = FakePose((0.0, 0.0, base_z))
        half = finger_sum / 2.0
        self._qpos = torch.tensor(
            [[0.0] * 7 + [half, half]], dtype=torch.float32
        )  # shape (1, 9)

    def get_qpos(self):
        return self._qpos


class FakeAgent:
    """An arm. ``grasping`` maps object-identity -> bool returned by is_grasping
    (kept only for the TSC tests, which still use is_grasping). ``finger_sum`` sets
    the gripper-closed reading via robot.get_qpos().

    Records every object it was asked about in ``self.queried`` so a test can
    confirm WHICH arm was used for a given cube AND assert is_grasping is NOT used in
    the LiftBarrier criterion path.
    """
    def __init__(self, base_z=0.0, grasping=None, name="arm",
                 finger_sum=_GRIP_OPEN_SUM):
        self.robot = FakeRobot(base_z, finger_sum=finger_sum)
        self._grasping = grasping or {}
        self.name = name
        self.queried = []

    def is_grasping(self, obj):
        self.queried.append(id(obj))
        val = self._grasping.get(id(obj), False)
        return torch.tensor([bool(val)])


class FakeMultiAgent:
    def __init__(self, agents):
        self.agents = agents


# --- LiftBarrier fake env ---------------------------------------------------
class FakeLiftBarrierEnv:
    """Mimics LiftBarrierEnv just enough for lift_barrier_success_strict.

    ``barrier_z`` is the barrier CENTRE z; with the default identity quat the two
    grasp ENDS sit at centre_z + 0.074 (the +z component of [+/-0.222, 0, 0.074]).
    Pass ``barrier_p`` / ``barrier_q`` to place / rotate the barrier exactly.
    ``closed0`` / ``closed1`` set each arm's gripper-closed state via its finger-sum
    (closed -> finger-sum ~0.0 < GRIPPER_CLOSE_MAX; open -> ~0.16 >= it).
    """
    def __init__(self, barrier_z, base_z, closed0, closed1,
                 barrier_p=None, barrier_q=(1.0, 0.0, 0.0, 0.0)):
        p = (0.0, 0.0, barrier_z) if barrier_p is None else tuple(barrier_p)
        self.barrier = FakeActor(p, barrier_q)
        bid = id(self.barrier)
        fs0 = _GRIP_CLOSED_SUM if closed0 else _GRIP_OPEN_SUM
        fs1 = _GRIP_CLOSED_SUM if closed1 else _GRIP_OPEN_SUM
        # grasping={} so any stray is_grasping call returns False AND is recorded in
        # .queried (the test asserts the criterion path never queries is_grasping).
        self.agent = FakeMultiAgent([
            FakeAgent(base_z=base_z, grasping={bid: False}, name="arm0", finger_sum=fs0),
            FakeAgent(base_z=base_z, grasping={bid: False}, name="arm1", finger_sum=fs1),
        ])
    # candidate reads env.unwrapped or env -> make .unwrapped return self
    @property
    def unwrapped(self):
        return self


# --- TSC fake env -----------------------------------------------------------
class FakeTSCEnv:
    """Mimics ThreeRobotsStackCubeEnv for tsc_success_fixed.

    Stacked-success geometry: cubes are 0.04 apart in z (= 2*half_size), aligned
    in xy, B & C both inside goal_radius of the goal. Grasp flags per arm are set
    so the candidate's ~grasped gating can be exercised.
    """
    def __init__(self,
                 a_grasped=False, b_grasped=False,
                 left_holds_C=False, middle_holds_C=False,
                 stacked_and_placed=True):
        half = 0.02
        if stacked_and_placed:
            self.cubeA = FakeActor((0.0, 0.0, 0.02))
            self.cubeB = FakeActor((0.0, 0.0, 0.06))   # +0.04 over A
            self.cubeC = FakeActor((0.0, 0.0, 0.10))   # +0.04 over B
            self.goal_region = FakeActor((0.0, 0.0, 0.0))
        else:
            # knock C off the stack (z gap wrong) so on-cube fails
            self.cubeA = FakeActor((0.0, 0.0, 0.02))
            self.cubeB = FakeActor((0.0, 0.0, 0.06))
            self.cubeC = FakeActor((0.5, 0.5, 0.10))   # far in xy -> off-stack
            self.goal_region = FakeActor((0.0, 0.0, 0.0))

        self.cube_half_size = torch.tensor([half, half, half], dtype=torch.float32)
        self.goal_radius = 0.12

        cA, cB, cC = id(self.cubeA), id(self.cubeB), id(self.cubeC)
        # arm 0 (left): cube A, and (buggy) is queried for C by the OLD check
        self._left = FakeAgent(grasping={cA: a_grasped, cC: left_holds_C}, name="left")
        # arm 1 (right): cube B
        self._right = FakeAgent(grasping={cB: b_grasped}, name="right")
        # arm 2 (middle): cube C
        self._middle = FakeAgent(grasping={cC: middle_holds_C}, name="middle")
        self.agent = FakeMultiAgent([self._left, self._right, self._middle])

    @property
    def left_agent(self):
        return self.agent.agents[0]

    @property
    def right_agent(self):
        return self.agent.agents[1]

    @property
    def middle_agent(self):
        return self.agent.agents[2]

    @property
    def unwrapped(self):
        return self


# ===========================================================================
# LiftBarrier tests
# ===========================================================================
def _b(t):
    return bool(t.reshape(-1)[0])


def test_lb_strict_high_but_one_gripper_open_is_false():
    """ends high (OLD centre check passes) but arm1 gripper OPEN -> per-frame False."""
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=True, closed1=False)
    # sanity: the OLD grasp-blind CENTRE-height check WOULD pass here
    old_height = _b(env.barrier.pose.p[..., 2] > env.agent.agents[0].robot.pose.p[0, 2] + 0.15)
    assert old_height is True
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_high_but_other_gripper_open_is_false():
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=False, closed1=True)
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_both_closed_and_high_is_true():
    # centre 0.30 -> ends 0.374 > base+0.25=0.25, both grippers closed -> True
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=True, closed1=True)
    assert _b(lift_barrier_success_strict(env)) is True


def test_lb_strict_both_closed_but_low_is_false():
    """Both grippers closed but ends not lifted past base+0.25 -> False."""
    # centre 0.10 -> ends 0.174, < 0.25 -> False
    env = FakeLiftBarrierEnv(barrier_z=0.10, base_z=0.0, closed0=True, closed1=True)
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_respects_base_offset():
    """Threshold is base_z + 0.25, not absolute 0.25 (ends = centre + 0.074)."""
    # centre 0.20, base 0.10 -> ends 0.274; 0.274 > 0.10+0.25=0.35 is False
    env = FakeLiftBarrierEnv(barrier_z=0.20, base_z=0.10, closed0=True, closed1=True)
    assert _b(lift_barrier_success_strict(env)) is False
    # centre 0.30, base 0.10 -> ends 0.374 > 0.35 True
    env2 = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.10, closed0=True, closed1=True)
    assert _b(lift_barrier_success_strict(env2)) is True


def test_lb_strict_tipped_bar_one_end_low_is_false():
    """A TIPPED bar (one end high, other end low) must FAIL even though both
    grippers are closed and the CENTRE is above threshold -- the flung/tipped false-
    positive the both-ends criterion closes.

    Barrier rotated ~80deg about local-y (pitch) so the long axis (local-X) points
    mostly along world-z: +x end goes up, -x end goes down. Centre placed high.
    """
    import transforms3d as t3d
    # 80deg pitch about y: +x end -> +z (up), -x end -> -z (down).
    q = t3d.quaternions.mat2quat(t3d.euler.euler2mat(0.0, np.deg2rad(80.0), 0.0))
    env = FakeLiftBarrierEnv(
        barrier_z=0.0, base_z=0.0, closed0=True, closed1=True,
        barrier_p=(0.0, 0.0, 0.30), barrier_q=tuple(q.tolist()),
    )
    # one end is well below base+0.25 -> .all() over ends is False
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_accepts_wrapped_env():
    """Candidate unwraps env.unwrapped; pass a wrapper that holds the real env."""
    class Wrapper:
        def __init__(self, inner):
            self.unwrapped = inner
    inner = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=True, closed1=True)
    assert _b(lift_barrier_success_strict(Wrapper(inner))) is True


def test_lb_strict_does_not_call_is_grasping():
    """The criterion path must NOT consult is_grasping (it was DROPPED as unreliable
    on the thin bar ends). Drive a clean held lift and assert neither arm's
    is_grasping was queried about the barrier."""
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=True, closed1=True)
    _ = lift_barrier_success_strict(env)
    bid = id(env.barrier)
    assert bid not in env.agent.agents[0].queried, "arm0 is_grasping was wrongly used"
    assert bid not in env.agent.agents[1].queried, "arm1 is_grasping was wrongly used"
    # and nothing at all was queried (no is_grasping call of any kind)
    assert env.agent.agents[0].queried == []
    assert env.agent.agents[1].queried == []


# ===========================================================================
# arm_gripper_closed: the LAST 2 qpos (finger joints) summed vs GRIPPER_CLOSE_MAX
# ===========================================================================
def test_constants_are_as_specified():
    assert HOLD_FRAMES_K == 8
    assert GRIPPER_CLOSE_MAX == 0.06
    assert BARRIER_LIFT_DZ == 0.25


def test_arm_gripper_closed_reads_last_two_qpos():
    """finger-sum below GRIPPER_CLOSE_MAX -> closed; above -> open. The 7 arm joints
    (whatever their values) must NOT affect the verdict."""
    # closed: fingers sum 0.0
    closed_arm = FakeAgent(finger_sum=_GRIP_CLOSED_SUM)
    assert _b(arm_gripper_closed(closed_arm)) is True
    # open: fingers sum 0.16
    open_arm = FakeAgent(finger_sum=_GRIP_OPEN_SUM)
    assert _b(arm_gripper_closed(open_arm)) is False
    # exactly at threshold sum 0.06 is NOT < 0.06 -> open (strict <)
    at_thresh = FakeAgent(finger_sum=0.06)
    assert _b(arm_gripper_closed(at_thresh)) is False
    # just under threshold -> closed
    under = FakeAgent(finger_sum=0.0599)
    assert _b(arm_gripper_closed(under)) is True


def test_arm_gripper_closed_ignores_arm_joints():
    """Large arm-joint values must not be mistaken for finger joints."""
    arm = FakeAgent(finger_sum=_GRIP_CLOSED_SUM)
    # overwrite the 7 arm joints with big numbers; fingers (last 2) still ~0.0
    arm.robot._qpos = torch.tensor([[1.0] * 7 + [0.0, 0.0]], dtype=torch.float32)
    assert _b(arm_gripper_closed(arm)) is True


# ===========================================================================
# ENV-LEVEL SUSTAINED COUNTER: LiftBarrierEnv.evaluate (logic only, no sim).
# We drive a lightweight stand-in that reuses the REAL evaluate() bound method so
# the counter maintenance / reset is exercised exactly as in the task, without
# gym.make. The counter logic is pure torch (no sapien), so it runs on the login node.
# ===========================================================================
class _EvalEnv(FakeLiftBarrierEnv):
    """FakeLiftBarrierEnv + the REAL LiftBarrierEnv.evaluate bound to it, plus the
    minimal state evaluate touches (num_envs, _lift_hold). Mutators let a test march
    the per-frame condition C across env.steps."""

    def __init__(self):
        super().__init__(barrier_z=0.30, base_z=0.0, closed0=True, closed1=True)
        self.num_envs = 1
        self._lift_hold = torch.zeros(self.num_envs, dtype=torch.long)
        from robofactory.tasks.lift_barrier import LiftBarrierEnv
        self._evaluate_impl = LiftBarrierEnv.evaluate

    def set_condition(self, c: bool):
        """Make the per-frame C True/False: True = both ends high + both closed."""
        if c:
            self.barrier.pose.p = torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32)
            for a in self.agent.agents:
                a.robot._qpos = torch.tensor(
                    [[0.0] * 7 + [_GRIP_CLOSED_SUM / 2] * 2], dtype=torch.float32)
        else:
            # drop the bar low (ends below threshold) -> C False
            self.barrier.pose.p = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    def evaluate(self):
        return self._evaluate_impl(self)


def _succ(env):
    out = env.evaluate()["success"]
    return bool(out.reshape(-1)[0])


def test_env_sustained_success_only_after_K_consecutive_frames():
    """The env declares success ONLY once C has held for HOLD_FRAMES_K consecutive
    env.steps -- not on the first frame C is True."""
    env = _EvalEnv()
    env.set_condition(True)
    for frame in range(1, HOLD_FRAMES_K):
        assert _succ(env) is False, f"frame {frame}: success before K frames"
    # the K-th consecutive True frame flips success on
    assert _succ(env) is True
    # and it stays True while C persists
    assert _succ(env) is True


def test_env_sustained_resets_on_gap():
    """A single False frame (C breaks) RESETS the counter; success requires another
    full K-consecutive run afterwards (kills transient single-frame flings)."""
    env = _EvalEnv()
    env.set_condition(True)
    for _ in range(HOLD_FRAMES_K - 1):     # K-1 True frames (not yet success)
        assert _succ(env) is False
    # one False frame breaks the run -> counter resets to 0
    env.set_condition(False)
    assert _succ(env) is False
    # back to True: must accumulate K MORE consecutive frames from scratch
    env.set_condition(True)
    for frame in range(1, HOLD_FRAMES_K):
        assert _succ(env) is False, f"post-gap frame {frame} succeeded too early"
    assert _succ(env) is True


def test_env_sustained_grippers_open_never_succeeds():
    """Ends high for many frames but a gripper OPEN the whole time -> never success
    (the counter never increments because C is always False)."""
    env = _EvalEnv()
    # ends high but arm1 gripper OPEN every frame
    env.barrier.pose.p = torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32)
    env.agent.agents[0].robot._qpos = torch.tensor(
        [[0.0] * 7 + [_GRIP_CLOSED_SUM / 2] * 2], dtype=torch.float32)
    env.agent.agents[1].robot._qpos = torch.tensor(
        [[0.0] * 7 + [_GRIP_OPEN_SUM / 2] * 2], dtype=torch.float32)
    for _ in range(HOLD_FRAMES_K + 5):
        assert _succ(env) is False


def test_env_sustained_ends_low_never_succeeds():
    """Both grippers closed but ends never clear base+0.25 -> never success."""
    env = _EvalEnv()
    # centre 0.10 -> ends 0.174 < 0.25; both grippers closed
    env.barrier.pose.p = torch.tensor([[0.0, 0.0, 0.10]], dtype=torch.float32)
    for a in env.agent.agents:
        a.robot._qpos = torch.tensor([[0.0] * 7 + [0.0, 0.0]], dtype=torch.float32)
    for _ in range(HOLD_FRAMES_K + 5):
        assert _succ(env) is False


def test_env_evaluate_returns_batch_shaped_bool():
    """success shape matches the per-env batch (num_envs,)."""
    env = _EvalEnv()
    env.set_condition(True)
    out = env.evaluate()["success"]
    assert tuple(out.shape) == (1,)
    assert out.dtype == torch.bool


# ===========================================================================
# INTERPRETER single-frame check_barrier_ends_held (drops is_grasping)
# ===========================================================================
def test_interpreter_check_is_single_frame_geometric():
    """subtask_interpreter.check_barrier_ends_held is the SINGLE-FRAME predicate:
    True iff both ends high AND both grippers closed AT the call. It must NOT need K
    calls (the env counter owns the sustain) and must NOT call is_grasping."""
    from robofactory.planner.subtask_interpreter import check_barrier_ends_held
    chk = check_barrier_ends_held(dz=0.25)
    env_ok = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=True, closed1=True)
    # passes on a SINGLE call (not after K) -> stateless single-frame predicate
    assert chk(env_ok, 0, None) is True
    # is_grasping was never consulted
    assert env_ok.agent.agents[0].queried == []
    assert env_ok.agent.agents[1].queried == []
    # gripper open -> False
    env_open = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, closed0=True, closed1=False)
    assert check_barrier_ends_held(dz=0.25)(env_open, 0, None) is False
    # ends low -> False
    env_low = FakeLiftBarrierEnv(barrier_z=0.10, base_z=0.0, closed0=True, closed1=True)
    assert check_barrier_ends_held(dz=0.25)(env_low, 0, None) is False


def test_interpreter_check_matches_candidate_predicate():
    """The interpreter single-frame check and the candidate per-frame predicate must
    AGREE on the same hand-computed cases (lock-step across the two code paths)."""
    from robofactory.planner.subtask_interpreter import check_barrier_ends_held
    cases = [
        # (barrier_z, closed0, closed1, expected)
        (0.30, True, True, True),
        (0.30, True, False, False),
        (0.30, False, True, False),
        (0.10, True, True, False),
        (0.10, False, False, False),
    ]
    for bz, c0, c1, exp in cases:
        env = FakeLiftBarrierEnv(barrier_z=bz, base_z=0.0, closed0=c0, closed1=c1)
        cand = _b(lift_barrier_success_strict(env))
        env2 = FakeLiftBarrierEnv(barrier_z=bz, base_z=0.0, closed0=c0, closed1=c1)
        interp = check_barrier_ends_held(dz=0.25)(env2, 0, None)
        assert cand is exp, (bz, c0, c1, "candidate")
        assert interp is exp, (bz, c0, c1, "interpreter")
        assert cand == interp, (bz, c0, c1, "candidate vs interpreter disagree")


# ===========================================================================
# Both-ends GEOMETRY: world-z of the two grasp ends for known poses, hand-checked.
# ===========================================================================
class _Barrier:
    """A bare barrier with a pose carrying .p (1,3) and .q (1,4 wxyz)."""
    def __init__(self, p, q):
        self.pose = FakePose(p, q)


def _ends_z(p, q):
    ends = barrier_grasp_ends_world(_Barrier(p, q)).reshape(2, 3).numpy()
    return ends


def test_geom_offsets_are_pm_x_with_small_z():
    off = BARRIER_END_OFFSETS.numpy()
    assert np.allclose(off[0], [0.222, 0.0, 0.074])
    assert np.allclose(off[1], [-0.222, 0.0, 0.074])
    assert BARRIER_LIFT_DZ == 0.25


def test_geom_identity_quat_ends_are_p_plus_offset():
    """Identity quat: ends = p + [+/-0.222, 0, 0.074] exactly."""
    p = (0.10, -0.20, 0.30)
    ends = _ends_z(p, (1.0, 0.0, 0.0, 0.0))
    assert np.allclose(ends[0], [0.10 + 0.222, -0.20, 0.30 + 0.074], atol=1e-5)
    assert np.allclose(ends[1], [0.10 - 0.222, -0.20, 0.30 + 0.074], atol=1e-5)


def test_geom_90deg_about_z():
    """90deg yaw about z: local +x -> world +y; both ends keep z = p_z + 0.074."""
    import transforms3d as t3d
    q = t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, np.pi / 2))
    p = (0.0, 0.0, 0.30)
    ends = _ends_z(p, tuple(q.tolist()))
    # +x end -> +y; -x end -> -y; z unchanged (= 0.30 + 0.074)
    assert np.allclose(ends[0], [0.0, 0.222, 0.374], atol=1e-4)
    assert np.allclose(ends[1], [0.0, -0.222, 0.374], atol=1e-4)


def test_geom_90deg_about_y_sends_ends_to_z():
    """90deg pitch about y: local +x -> world -z, local -x -> world +z.

    Hand: R_y(90) maps [x,0,z] -> [z, 0, -x]. So +x end offset [0.222,0,0.074]
    -> [0.074, 0, -0.222]; -x end [-0.222,0,0.074] -> [0.074, 0, +0.222].
    With p=(0,0,0.30): +x end z = 0.30-0.222=0.078 ; -x end z = 0.30+0.222=0.522.
    """
    import transforms3d as t3d
    q = t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, np.pi / 2, 0))
    p = (0.0, 0.0, 0.30)
    ends = _ends_z(p, tuple(q.tolist()))
    assert np.allclose(ends[0], [0.074, 0.0, 0.30 - 0.222], atol=1e-4)
    assert np.allclose(ends[1], [0.074, 0.0, 0.30 + 0.222], atol=1e-4)


def test_geom_45deg_about_z_matches_handcalc():
    """45deg yaw about z: local +x -> (cos45, sin45, 0)*0.222 in xy; z unchanged."""
    import transforms3d as t3d
    q = t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, np.pi / 4))
    p = (0.0, 0.0, 0.30)
    ends = _ends_z(p, tuple(q.tolist()))
    c = 0.222 / np.sqrt(2.0)
    assert np.allclose(ends[0], [c, c, 0.374], atol=1e-4)
    assert np.allclose(ends[1], [-c, -c, 0.374], atol=1e-4)


def test_geom_batched_two_envs():
    """barrier_grasp_ends_world keeps the leading batch dim: (B,2,3)."""
    class _BatchBarrier:
        def __init__(self):
            class _P:
                p = torch.tensor([[0.0, 0.0, 0.30], [1.0, 2.0, 0.50]])
                q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
            self.pose = _P()
    ends = barrier_grasp_ends_world(_BatchBarrier())
    assert tuple(ends.shape) == (2, 2, 3)
    # env0 +x end
    assert np.allclose(ends[0, 0].numpy(), [0.222, 0.0, 0.374], atol=1e-5)
    # env1 -x end
    assert np.allclose(ends[1, 1].numpy(), [1.0 - 0.222, 2.0, 0.574], atol=1e-5)


# ===========================================================================
# TSC tests
# ===========================================================================
def test_tsc_fixed_queries_middle_agent_for_cubeC_not_left():
    """The fixed candidate must ask the MIDDLE arm (arm 2) about cube C, and
    must NOT use the left arm (arm 0) for the cube-C grasp."""
    env = FakeTSCEnv(stacked_and_placed=True)
    cC = id(env.cubeC)
    _ = tsc_success_fixed(env)
    # middle arm WAS queried about cube C
    assert cC in env.middle_agent.queried, "middle_agent (arm 2) was not asked about cubeC"
    # left arm was NOT asked about cube C (the bug would have queried it)
    assert cC not in env.left_agent.queried, "left_agent (arm 0) was wrongly asked about cubeC"


def test_tsc_fixed_full_success_true():
    """Stacked + placed + all released -> True."""
    env = FakeTSCEnv(stacked_and_placed=True,
                     a_grasped=False, b_grasped=False,
                     left_holds_C=False, middle_holds_C=False)
    assert _b(tsc_success_fixed(env)) is True


def test_tsc_fixed_middle_still_holding_C_is_false():
    """Geometry good but the MIDDLE arm still grasps C -> gated to False."""
    env = FakeTSCEnv(stacked_and_placed=True, middle_holds_C=True)
    assert _b(tsc_success_fixed(env)) is False


def test_tsc_fixed_ignores_left_arm_phantom_C_grasp():
    """The CORE bug fix: if the LEFT arm reports grasping C (the OLD check would
    read this and wrongly fail) but the MIDDLE arm released C, the FIXED
    candidate ignores the left arm and returns True."""
    env = FakeTSCEnv(stacked_and_placed=True,
                     left_holds_C=True,    # OLD check (arm 0) would see this -> fail
                     middle_holds_C=False)  # FIXED check (arm 2) sees released
    assert _b(tsc_success_fixed(env)) is True


def test_tsc_fixed_off_stack_is_false():
    """C knocked off the stack -> on-cube geometry fails -> False."""
    env = FakeTSCEnv(stacked_and_placed=False)
    assert _b(tsc_success_fixed(env)) is False


def test_tsc_fixed_other_arm_grasps_gate():
    """A still-grasping A or B also gates success to False."""
    env_a = FakeTSCEnv(stacked_and_placed=True, a_grasped=True)
    env_b = FakeTSCEnv(stacked_and_placed=True, b_grasped=True)
    assert _b(tsc_success_fixed(env_a)) is False
    assert _b(tsc_success_fixed(env_b)) is False
