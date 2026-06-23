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

import torch

# make the repo root importable when pytest runs from elsewhere
_THIS = osp.dirname(osp.abspath(__file__))
_ROOT = osp.dirname(osp.dirname(osp.dirname(_THIS)))  # .../RoboFactory-subtask-wt
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from robofactory.tasks.success_candidates import (  # noqa: E402
    lift_barrier_success_strict,
    tsc_success_fixed,
)


# ===========================================================================
# Fakes
# ===========================================================================
class FakePose:
    """Holds a (1,3) position tensor at .p (matches actor.pose.p batched layout)."""
    def __init__(self, xyz):
        self.p = torch.tensor([list(xyz)], dtype=torch.float32)  # shape (1,3)


class FakeActor:
    def __init__(self, xyz):
        self.pose = FakePose(xyz)


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
    """Mimics LiftBarrierEnv just enough for lift_barrier_success_strict."""
    def __init__(self, barrier_z, base_z, grasp0, grasp1):
        self.barrier = FakeActor((0.0, 0.0, barrier_z))
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
    """z high (OLD check passes) but arm1 NOT grasping -> strict says False."""
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=True, grasp1=False)
    # sanity: the OLD grasp-blind height check WOULD pass here
    old_height = _b(env.barrier.pose.p[..., 2] > env.agent.agents[0].robot.pose.p[0, 2] + 0.15)
    assert old_height is True
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_high_but_other_gripper_open_is_false():
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=False, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_both_grasping_and_high_is_true():
    env = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is True


def test_lb_strict_both_grasping_but_low_is_false():
    """Both grasping but barrier not lifted past threshold -> False."""
    env = FakeLiftBarrierEnv(barrier_z=0.10, base_z=0.0, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is False


def test_lb_strict_respects_base_offset():
    """Threshold is base_z + 0.15, not absolute 0.15."""
    # barrier z 0.20, base z 0.10 -> 0.20 > 0.10+0.15=0.25 is False
    env = FakeLiftBarrierEnv(barrier_z=0.20, base_z=0.10, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env)) is False
    # barrier z 0.30, base z 0.10 -> 0.30 > 0.25 True
    env2 = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.10, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(env2)) is True


def test_lb_strict_accepts_wrapped_env():
    """Candidate unwraps env.unwrapped; pass a wrapper that holds the real env."""
    class Wrapper:
        def __init__(self, inner):
            self.unwrapped = inner
    inner = FakeLiftBarrierEnv(barrier_z=0.30, base_z=0.0, grasp0=True, grasp1=True)
    assert _b(lift_barrier_success_strict(Wrapper(inner))) is True


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
