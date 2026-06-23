"""PURE candidate success-check functions for the success-check AUDIT.

These mirror the PROPOSED fixes to the two task ``evaluate()`` methods WITHOUT
touching the tasks themselves (that change is gated on human review). Each
function takes an *unwrapped* env and returns a boolean tensor (the same shape /
dtype the env's own ``evaluate()['success']`` produces), so the audit can compare
OLD env success vs. the candidate per-episode and surface disagreements.

Why this lives apart from the task files:
  * ``robofactory/tasks/lift_barrier.py:117-122`` -- the LiftBarrier success is
    purely a height check (barrier.z > base.z + 0.15); it is GRASP-BLIND, so a
    barrier that was flung / tipped past the threshold (no held grasp) counts as
    a success (a known false-positive source). The strict candidate ANDs the
    height check with BOTH arms actually grasping the barrier.
  * ``robofactory/tasks/three_robots_stack_cube.py:125-165`` -- the TSC
    ``evaluate()`` computes ``is_cubeC_grasped`` from ``self.left_agent`` (arm 0)
    at line 153, but cube C is handled by the MIDDLE arm (arm 2). The fixed
    candidate recomputes the *exact same* success expression but reads
    ``is_cubeC_grasped`` from ``self.middle_agent``.

Both functions REUSE the env's own attributes / predicates (self.barrier,
self.agent.agents, self.left/right/middle_agent, self.cubeA/B/C,
self.goal_region, self.cube_half_size, self.goal_radius) so they stay in lock-step
with the task definitions. They do NOT import sapien / torch at module top beyond
what is needed for tensor logic; ``torch`` is required because is_grasping and the
pose comparisons return torch tensors.

These are PURE (no gym.make, no sim): they only read attributes off a live env
object you already have. The audit driver (scripts/dart/audit_success.py) builds
that env on a GPU compute node.
"""
from __future__ import annotations

import torch


def _unwrap(env):
    """Return the underlying task env (handles gym / RecordEpisodeMA wrappers)."""
    return getattr(env, "unwrapped", env)


# ----------------------------------------------------------------------------
# LiftBarrier: grasp-blind height check  ->  height AND both arms grasping
# ----------------------------------------------------------------------------
def lift_barrier_success_strict(env):
    """Strict LiftBarrier success: barrier lifted AND held by BOTH arms.

    Mirrors ``LiftBarrierEnv.evaluate`` (lift_barrier.py:117-122):
        success = barrier.z > base.z + 0.15
    and ANDs it with both grippers grasping the barrier:
        AND env.agent.agents[0].is_grasping(env.barrier)
        AND env.agent.agents[1].is_grasping(env.barrier)

    Args:
        env: a live (possibly wrapped) LiftBarrier env.

    Returns:
        torch.BoolTensor, same batch shape as the env's own success flag.
    """
    e = _unwrap(env)
    # height check -- IDENTICAL to the task's evaluate()
    height_ok = e.barrier.pose.p[..., 2] > e.agent.agents[0].robot.pose.p[0, 2] + 0.15
    grasp0 = e.agent.agents[0].is_grasping(e.barrier)
    grasp1 = e.agent.agents[1].is_grasping(e.barrier)
    height_ok = torch.as_tensor(height_ok)
    grasp0 = torch.as_tensor(grasp0)
    grasp1 = torch.as_tensor(grasp1)
    return (height_ok & grasp0 & grasp1).bool()


# ----------------------------------------------------------------------------
# ThreeRobotsStackCube: is_cubeC_grasped from middle_agent (arm 2), not left_agent
# ----------------------------------------------------------------------------
def tsc_success_fixed(env):
    """TSC success with the cube-C grasp read from the MIDDLE arm (arm 2).

    This is a faithful copy of ``ThreeRobotsStackCubeEnv.evaluate``
    (three_robots_stack_cube.py:125-165), with the SINGLE fix at line 153:
        is_cubeC_grasped = self.left_agent.is_grasping(self.cubeC)   # BUG (arm 0)
    becomes
        is_cubeC_grasped = self.middle_agent.is_grasping(self.cubeC)  # arm 2

    Everything else (the on-cube geometry, placed-in-goal checks, the
    not-grasped gating, the final product) is reproduced verbatim from the env so
    the candidate stays in lock-step with the task definition.

    Args:
        env: a live (possibly wrapped) ThreeRobotsStackCube env.

    Returns:
        torch.BoolTensor, same batch shape as the env's own success flag.
    """
    e = _unwrap(env)
    pos_A = e.cubeA.pose.p
    pos_B = e.cubeB.pose.p
    pos_C = e.cubeC.pose.p

    offset = pos_B - pos_A
    xy_flag = (
        torch.linalg.norm(offset[..., :2], axis=1)
        <= torch.linalg.norm(e.cube_half_size[:2]) + 0.005
    )
    z_flag = torch.abs(offset[..., 2] - e.cube_half_size[..., 2] * 2) <= 0.005
    is_cubeB_on_cubeA = torch.logical_and(xy_flag, z_flag)

    offset = pos_C - pos_B
    xy_flag = (
        torch.linalg.norm(offset[..., :2], axis=1)
        <= torch.linalg.norm(e.cube_half_size[:2]) + 0.005
    )
    z_flag = torch.abs(offset[..., 2] - e.cube_half_size[..., 2] * 2) <= 0.005
    is_cubeC_on_cubeB = torch.logical_and(xy_flag, z_flag)

    cubeB_to_goal_dist = torch.linalg.norm(
        e.cubeB.pose.p[:, :2] - e.goal_region.pose.p[..., :2], axis=1
    )
    cubeB_placed = cubeB_to_goal_dist < e.goal_radius
    cubeC_to_goal_dist = torch.linalg.norm(
        e.cubeC.pose.p[:, :2] - e.goal_region.pose.p[..., :2], axis=1
    )
    cubeC_placed = cubeC_to_goal_dist < e.goal_radius

    is_cubeA_grasped = e.left_agent.is_grasping(e.cubeA)
    is_cubeB_grasped = e.right_agent.is_grasping(e.cubeB)
    # THE FIX: cube C is the MIDDLE arm's cube (arm 2), not left_agent (arm 0).
    is_cubeC_grasped = e.middle_agent.is_grasping(e.cubeC)

    success = (
        is_cubeC_on_cubeB
        * is_cubeB_on_cubeA
        * cubeB_placed
        * cubeC_placed
        * (~is_cubeA_grasped)
        * (~is_cubeB_grasped)
        * (~is_cubeC_grasped)
    )
    return success.bool()


# Registry so the audit driver can look a candidate up by task name.
CANDIDATES = {
    "LiftBarrier": lift_barrier_success_strict,
    "ThreeRobotsStackCube": tsc_success_fixed,
}
