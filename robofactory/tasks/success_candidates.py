"""PURE candidate success-check functions for the success-check AUDIT.

These mirror the PROPOSED fixes to the two task ``evaluate()`` methods WITHOUT
touching the tasks themselves (that change is gated on human review). Each
function takes an *unwrapped* env and returns a boolean tensor (the same shape /
dtype the env's own ``evaluate()['success']`` produces), so the audit can compare
OLD env success vs. the candidate per-episode and surface disagreements.

Why this lives apart from the task files:
  * ``robofactory/tasks/lift_barrier.py`` -- the OLD LiftBarrier success was
    purely a CENTRE-height check (barrier.z > base.z + 0.15); it was GRASP-BLIND, so
    a barrier flung / tipped past the threshold (no held grasp) counted as a success
    (a known false-positive source). The criterion is now a SUSTAINED GEOMETRIC check:
    a PER-FRAME condition C = (BOTH grasp ENDS clear base_z + 0.25 AND BOTH arms'
    grippers closed) that must hold for HOLD_FRAMES_K CONSECUTIVE frames (the env owns
    the counter). is_grasping is DROPPED: the contact-force probe UNRELIABLY registers
    a real load-bearing grasp on the thin (~0.09 m) bar ends, wrongly rejecting genuine
    held lifts. ``lift_barrier_success_strict`` here is the PER-FRAME predicate C; the
    consecutive-frame count lives in ``LiftBarrierEnv.evaluate``. See
    ``barrier_grasp_ends_world`` for the end geometry and ``arm_gripper_closed`` for the
    gripper check.
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
# Barrier geometry (the strict-success crux). Verified 2026-06-23 against
# robofactory/assets/objects/steel_barrier_annotated/model_data.json + the env
# config + motionplanner.get_grasp_pose_w_labeled_direction:
#
#   * The barrier's LONG AXIS is local-X. The two grasp ends are the annotation's
#     contact points at local-X = +/-0.37 in the UNSCALED mesh frame.
#   * The actor is built with scale [0.6, 0.6, 0.2]; the grasp pipeline scales the
#     local contact translation by ``scale`` BEFORE applying the actor world pose:
#         local_end *= scale  ->  [0.37*0.6, 0*0.6, 0.37*0.2] = [0.222, 0, 0.074].
#     (The grasp-pose ``convert_matrix`` is a right-multiply that only rotates the
#     gripper ORIENTATION; it does NOT move the end POSITION — confirmed
#     numerically: world_end == barrier.pose.p + R(barrier.pose.q) @ local_end.)
#   * So in the barrier's LOCAL frame the two grasp ends sit at
#         [+0.222, 0, +0.074]  and  [-0.222, 0, +0.074].
# These are rotated by the barrier quaternion (wxyz) and offset by its world
# position to get the world grasp ends.
# ----------------------------------------------------------------------------
BARRIER_END_OFFSETS = torch.tensor(
    [[0.222, 0.0, 0.074], [-0.222, 0.0, 0.074]], dtype=torch.float32
)
# strict-success lift threshold above the arm-0 robot base z.
BARRIER_LIFT_DZ = 0.25
# Number of CONSECUTIVE frames the per-frame geometric condition must hold for the
# env's sustained counter to declare success (the env owns the counter; this is the
# REQUIRED run length). Kills transient flings (single-frame z-teleports) without the
# fragile contact check.
HOLD_FRAMES_K = 8
# A Panda arm's gripper is "closed" when its two finger joints (the LAST 2 entries of
# that arm's 9-dim qpos = 7 arm + 2 fingers) sum below this. Open ~0.08, closed ~0.00,
# so a sum < 0.06 means BOTH fingers are well off the open stop (clamping the bar).
GRIPPER_CLOSE_MAX = 0.06
# Max distance (m) from an arm's TCP to a barrier grasp-end for that arm to count as
# HOLDING that end. Closing the gripper is NOT enough — a closed EMPTY gripper parked
# in free space has finger-sum < GRIPPER_CLOSE_MAX yet grasps nothing. Requiring the
# TCP to be AT the end closes that false-positive loophole (one arm lifts/balances the
# bar so both ends clear the height while the other arm just closes on air). The grasp
# point sits ~0.074 m above the bar's long axis; a real wrap-grasp puts the panda TCP
# within a few cm of it, so 0.08 m is generous for genuine grasps but excludes an
# empty gripper parked elsewhere.
TCP_NEAR_END_MAX = 0.08


def _quat_rotate_wxyz(q, v):
    """Rotate vector(s) ``v`` by quaternion(s) ``q`` (wxyz), batched.

    q: (..., 4) wxyz ; v: (..., 3) -> (..., 3). Uses the standard
    v' = v + 2*w*(u x v) + 2*(u x (u x v)) form (u = q[..., 1:4]). Pure torch, so
    it runs on the env's batched GPU tensors and keeps the batch dim.
    """
    q = torch.as_tensor(q)
    v = torch.as_tensor(v).to(q.dtype)
    w = q[..., 0:1]
    u = q[..., 1:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def barrier_grasp_ends_world(barrier):
    """World xyz of the barrier's TWO grasp ends, batched.

    ``barrier.pose.p`` is (..., 3) world position; ``barrier.pose.q`` is (..., 4)
    wxyz world orientation. Returns a (..., 2, 3) tensor: index 0 = +x end,
    index 1 = -x end. Keeps any leading batch dims.
    """
    p = torch.as_tensor(barrier.pose.p)          # (..., 3)
    q = torch.as_tensor(barrier.pose.q)          # (..., 4) wxyz
    offsets = BARRIER_END_OFFSETS.to(p.dtype)    # (2, 3)
    # broadcast: rotate each of the 2 ends by the (batched) quaternion.
    q_b = q.unsqueeze(-2)                         # (..., 1, 4)
    off_b = offsets.reshape((1,) * (p.ndim - 1) + (2, 3))  # (..., 2, 3)
    ends_local_rot = _quat_rotate_wxyz(q_b, off_b)         # (..., 2, 3)
    return p.unsqueeze(-2) + ends_local_rot                # (..., 2, 3)


def arm_gripper_closed(agent):
    """Batched bool: is this arm's gripper CLOSED (finger-sum < GRIPPER_CLOSE_MAX)?

    The Panda arm's qpos is 9-dim (7 arm + 2 fingers); the gripper is "closed" when
    the LAST 2 entries (the finger joints) sum below ``GRIPPER_CLOSE_MAX`` (open
    ~0.08 each, closed ~0.00). ``agent.robot.get_qpos()`` is the batched (num_envs,
    9) tensor (GPU) -- we take the last 2 columns and sum. Returns a (num_envs,) bool
    tensor (kept batch-safe). This is the GEOMETRIC stand-in for is_grasping, which
    UNRELIABLY registers a real load-bearing grasp on the thin (~0.09 m) bar ends.
    """
    q = torch.as_tensor(agent.robot.get_qpos())     # (..., 9)  (num_envs, 9) on GPU
    finger_sum = q[..., -2:].sum(dim=-1)             # (...,)  last 2 = finger joints
    return finger_sum < GRIPPER_CLOSE_MAX


def arm_tcp_world(agent):
    """Batched world xyz (..., 3) of an arm's TCP (tool-center-point).

    ManiSkill's Panda sets ``agent.tcp`` (the ee link) at init; fall back to the named
    ee link if needed. Used to verify the gripper is AT a bar end (holding it), not
    just closed."""
    tcp = getattr(agent, "tcp", None)
    if tcp is None:
        from mani_skill.utils import sapien_utils
        tcp = sapien_utils.get_obj_by_name(agent.robot.links, agent.ee_link_name)
    return torch.as_tensor(tcp.pose.p)               # (..., 3)


def both_ends_held(agents, ends):
    """Batched bool (...,): is EACH grasp end held by a DISTINCT arm's TCP (within
    TCP_NEAR_END_MAX)? Assignment-agnostic — tries both arm<->end pairings and takes
    the better one — so it doesn't depend on the left/right convention. This is the
    gate that distinguishes a real two-arm grasp from one arm lifting while the other
    closes on air."""
    t0 = arm_tcp_world(agents[0])                    # (..., 3)
    t1 = arm_tcp_world(agents[1])
    e0 = ends[..., 0, :]                             # (..., 3)
    e1 = ends[..., 1, :]
    d00 = torch.linalg.norm(t0 - e0, dim=-1)
    d01 = torch.linalg.norm(t0 - e1, dim=-1)
    d10 = torch.linalg.norm(t1 - e0, dim=-1)
    d11 = torch.linalg.norm(t1 - e1, dim=-1)
    D = TCP_NEAR_END_MAX
    assign_a = (d00 < D) & (d11 < D)                 # arm0->end0, arm1->end1
    assign_b = (d01 < D) & (d10 < D)                 # arm0->end1, arm1->end0
    return (assign_a | assign_b).bool()


def lift_barrier_success_strict(env):
    """Strict LiftBarrier PER-FRAME geometric predicate (drops is_grasping).

    This is the SINGLE-FRAME condition that the env's SUSTAINED counter
    (``LiftBarrierEnv.evaluate`` / ``LiftBarrierEnv._lift_hold``) requires to hold for
    ``HOLD_FRAMES_K`` consecutive frames before declaring success. The single-frame
    predicate is:

        C = (BOTH grasp-end world-z > base_z + 0.25)
            AND arm0 gripper closed (finger-sum < GRIPPER_CLOSE_MAX)
            AND arm1 gripper closed (finger-sum < GRIPPER_CLOSE_MAX)

    where ``base_z = env.agent.agents[0].robot.pose.p[0, 2]`` and the two grasp
    ends are ``barrier.pose.p + R(barrier.pose.q) @ [+/-0.222, 0, 0.074]`` (see
    BARRIER_END_OFFSETS). Both ends must clear the threshold (not the centre), so a
    tipped / one-end-up bar fails this frame; the env's HOLD_FRAMES_K counter kills
    transient flings (single-frame z-teleports) on top.

    WHY no is_grasping: the contact-force ``is_grasping`` UNRELIABLY registers a real
    load-bearing grasp on the thin (~0.09 m) bar ends (one arm read False for 25
    consecutive frames on a genuine clean held lift). The gripper-closed geometric
    check is the robust stand-in.

    Args:
        env: a live (possibly wrapped) LiftBarrier env.

    Returns:
        torch.BoolTensor, same batch shape as the env's own success flag (this is the
        PER-FRAME C, NOT the sustained verdict).
    """
    e = _unwrap(env)
    ends = barrier_grasp_ends_world(e.barrier)               # (..., 2, 3)
    base_z = torch.as_tensor(e.agent.agents[0].robot.pose.p[0, 2])
    ends_z = ends[..., 2]                                    # (..., 2)
    height_ok = (ends_z > base_z + BARRIER_LIFT_DZ).all(dim=-1)  # (...,) both ends
    closed0 = arm_gripper_closed(e.agent.agents[0])
    closed1 = arm_gripper_closed(e.agent.agents[1])
    # NEW: each end must actually be HELD (an arm's TCP at it), not just have a closed
    # gripper somewhere. Closes the empty-closed-gripper false positive.
    held = both_ends_held(e.agent.agents, ends)
    return (height_ok & closed0 & closed1 & held).bool()


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
