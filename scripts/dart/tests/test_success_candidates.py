"""PURE tests for robofactory.tasks.success_candidates (no sim / no sapien).

These use tiny FAKE envs that mimic just the attributes the candidate functions
read (barrier.pose.p, agent.agents[i].is_grasping, cubeA/B/C, goal_region,
cube_half_size, goal_radius, left/right/middle_agent). They verify:

  LiftBarrier strict:
    * z high but ONE gripper open  -> False
        (this is the grasp-blind case the OLD height-only check WOULD pass)
    * both grasping + z high       -> True
    * both grasping but z low      -> False

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
    tsc_success_fixed,
    barrier_grasp_ends_world,
    BARRIER_END_OFFSETS,
    BARRIER_LIFT_DZ,
)


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
    def __init__(self, base_z):
        # pose.p[0, 2] is read for the base z
        self.pose = FakePose((0.0, 0.0, base_z))


class FakeAgent:
    """An arm. ``grasping`` maps object-identity -> bool returned by is_grasping.

    Records every object it was asked about in ``self.queried`` so a test can
    confirm WHICH arm was used for a given cube.
    """
    def __init__(self, base_z=0.0, grasping=None, name="arm"):
        self.robot = FakeRobot(base_z)
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
    """
    def __init__(self, barrier_z, base_z, grasp0, grasp1,
                 barrier_p=None, barrier_q=(1.0, 0.0, 0.0, 0.0)):
        p = (0.0, 0.0, barrier_z) if barrier_p is None else tuple(barrier_p)
        self.barrier = FakeActor(p, barrier_q)
        bid = id(self.barrier)
        self.agent = FakeMultiAgent([
            FakeAgent(base_z=base_z, grasping={bid: grasp0}, name="arm0"),
            FakeAgent(base_z=base_z, grasping={bid: grasp1}, name="arm1"),
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
    """ends high (OLD centre check passes) but arm1 NOT grasping -> strict False."""
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=True, grasp1=False)
    # sanity: the OLD grasp-blind CENTRE-height check WOULD pass here
    old_height = _b(env.barrier.pose.p[..., 2] > env.agent.agents[0].robot.pose.p[0, 2] + 0.15)
    assert old_height is True
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_high_but_other_gripper_open_is_false():
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=False, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_both_grasping_and_high_is_true():
    # centre 0.30 -> ends 0.374 > base+0.25=0.25, both grasping -> True
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is True


def test_lb_strict_both_grasping_but_low_is_false():
    """Both grasping but ends not lifted past base+0.25 -> False."""
    # centre 0.10 -> ends 0.174, < 0.25 -> False
    env = FakeLiftBarrierEnv(barrier_z=0.10, base_z=0.0, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_respects_base_offset():
    """Threshold is base_z + 0.25, not absolute 0.25 (ends = centre + 0.074)."""
    # centre 0.20, base 0.10 -> ends 0.274; 0.274 > 0.10+0.25=0.35 is False
    env = FakeLiftBarrierEnv(barrier_z=0.20, base_z=0.10, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is False
    # centre 0.30, base 0.10 -> ends 0.374 > 0.35 True
    env2 = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.10, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env2)) is True


def test_lb_strict_tipped_bar_one_end_low_is_false():
    """A TIPPED bar (one end high, other end low) must FAIL even though both arms
    grasp and the CENTRE is above threshold -- this is the flung/tipped false-
    positive the both-ends criterion closes.

    Barrier rotated ~90deg about local-y (pitch) so the long axis (local-X) points
    mostly along world-z: +x end goes up, -x end goes down. Centre placed high.
    """
    import transforms3d as t3d
    # 80deg pitch about y: +x end -> +z (up), -x end -> -z (down).
    q = t3d.quaternions.mat2quat(t3d.euler.euler2mat(0.0, np.deg2rad(80.0), 0.0))
    env = FakeLiftBarrierEnv(
        barrier_z=0.0, base_z=0.0, grasp0=True, grasp1=True,
        barrier_p=(0.0, 0.0, 0.30), barrier_q=tuple(q.tolist()),
    )
    # one end is well below base+0.25 -> .all() over ends is False
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_accepts_wrapped_env():
    """Candidate unwraps env.unwrapped; pass a wrapper that holds the real env."""
    class Wrapper:
        def __init__(self, inner):
            self.unwrapped = inner
    inner = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(Wrapper(inner))) is True


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
