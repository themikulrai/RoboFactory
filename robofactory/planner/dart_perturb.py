"""Shared, numpy-only DART joint-disturbance math.

Ported (MATH ONLY) from ``scripts/dart/run_dart_rollouts.py`` so that BOTH
``run_dart_rollouts.py`` and ``run_subtask_rollouts.py`` drive disturbances
through one implementation. There is intentionally NO sapien / torch / gym
import at module top: this file depends on numpy alone. Robot handles may
*return* torch tensors (we tolerate that via ``.cpu().numpy()``), but we never
import torch ourselves.

Key contract (DO NOT violate -- this locks the carry-grip fix):
  * The gripper command is supplied by the CALLER (e.g. a frozen grip captured
    before the disturbance). This module NEVER reads ``planner.gripper_state``
    or any other implicit gripper source. Whatever the caller passes in
    ``grips[i]`` is the gripper channel that gets emitted for arm ``i``.

The disturbance steps the UNWRAPPED env (``base_env``) so that the noisy
actions are NOT captured by a RecordEpisode wrapper -- the noise drives the
simulation off-path but never becomes a training label.
"""

import copy

import numpy as np

__all__ = ["sample_floored_offset", "inject_joint_disturbance"]


def sample_floored_offset(rng, sigma, n, floor):
    """Sample a random ``n``-vector offset whose L2 norm is never below ``floor``.

    Draw ``off ~ N(0, sigma, n)``. If ``||off|| < floor`` scale the vector UP to
    exactly ``floor`` while preserving its (random) direction, so the net joint
    displacement is never trivial. A small epsilon guards against div-by-zero
    when the sampled vector is (near-)zero.

    Args:
        rng: a ``numpy.random.Generator`` (or anything with ``.normal``).
        sigma: stddev of the per-component normal draw.
        n: number of components (e.g. 7 arm joints).
        floor: minimum allowed L2 norm of the returned offset.

    Returns:
        np.ndarray of shape (n,), dtype float64, with ``||off|| >= floor``.
    """
    off = rng.normal(0.0, sigma, size=n).astype(np.float64)
    mag = float(np.linalg.norm(off))
    if mag < floor:
        # scale to the floor, keep direction; +1e-9 guards div-by-zero.
        off = off / (mag + 1e-9) * floor
    return off


def _arm_qpos7(robot):
    """Return the first 7 (arm) joints of ``robot.get_qpos()`` as a 1-D numpy
    array, tolerating torch tensors and a leading batch dim.

    ``get_qpos()`` may return:
      * a plain array-like of shape (>=7,), or
      * a torch tensor (has ``.cpu().numpy()``), possibly shape (1, >=7).
    We convert to numpy, squeeze a leading batch axis if present, then slice
    the first 7 entries (the arm joints; trailing entries are gripper joints).
    """
    q = robot.get_qpos()
    # Tolerate torch tensors without importing torch.
    if hasattr(q, "cpu") and hasattr(q, "numpy"):
        q = q.cpu().numpy()
    q = np.asarray(q, dtype=np.float64)
    if q.ndim > 1:
        # collapse a leading batch dim, e.g. (1, 9) -> (9,)
        q = q.reshape(-1)
    return q[:7].copy()


def _assemble_action(target, grip, control_mode):
    """Pack one arm's 7-joint target + grip into an action vector for control_mode."""
    if control_mode == "pd_joint_pos_vel":
        return np.hstack([target, target * 0, grip])
    return np.hstack([target, grip])  # pd_joint_pos (default)


def inject_joint_disturbance(
    base_env,
    robots,
    grips,
    move_ids,
    rng,
    sigma,
    K,
    floor,
    hold_qpos=None,
    control_mode="pd_joint_pos",
    action_prefix="panda",
    sink=None,
    pre_hold_steps=0,
):
    """Step the UNWRAPPED env with a noisy joint target so selected arms drift
    off-path. NOT recorded (the unwrapped env bypasses any RecordEpisode wrapper),
    so the noise never becomes a training label.

    Two phases (total unrecorded steps = ``pre_hold_steps + K``):

      1. SETTLE / pre-hold (``pre_hold_steps`` steps): EVERY arm -- including the
         move arms -- holds its setpoint with NO offset applied. This is the
         grasp-protection window: after a just-completed grasp it lets the grasp
         firmly seat and the lift begin BEFORE any disturbance, so the subsequent
         shove lands clear of the grasp moment (a shove at/near the grasp would
         break it and fail the episode for certain). The hold target for arm ``i``
         is ``hold_qpos[i][:7]`` if supplied, else its live qpos.
      2. SHOVE (``K`` steps): arms in ``move_ids`` get a single floored random
         offset added to their captured ``q0`` (see :func:`sample_floored_offset`);
         all other arms hold their setpoint. That same drifted/held target is
         commanded for all ``K`` steps.

    The grips are carried verbatim from ``grips[i]`` through BOTH phases -- the
    caller MUST pass the live/frozen grip (e.g. ``frozen_grip``). This function
    never reads ``planner.gripper_state``, so the disturbance (and the settle hold)
    never changes a grasp's open/closed state.

    The random offset is drawn ONCE, AFTER the pre-hold phase is set up but using
    the same ``q0`` capture; the pre-hold phase consumes NO rng, so callers' per-
    transition RNG streams are unchanged whether or not a settle window is used.

    Args:
        base_env: the UNWRAPPED env; ``base_env.step(action_dict)`` is called.
        robots: list of robot handles, each exposing ``get_qpos()`` (array-like
            or torch tensor; first 7 entries are arm joints).
        grips: list of current gripper commands, one float per arm. Indexed by
            arm id and emitted verbatim as the gripper channel in BOTH phases.
        move_ids: int or list[int] of arm indices to perturb. Arms not listed
            hold ``hold_qpos[i]`` (or their live qpos if ``hold_qpos`` is None).
        rng: a ``numpy.random.Generator``.
        hold_qpos: optional list of per-arm hold targets (e.g. the interpreter's
            ``frozen_qpos`` setpoints). Used for the pre-hold of ALL arms and the
            shove-phase hold of non-move arms. None -> live qpos.
        sigma: stddev of the per-joint normal offset.
        K: number of (identical) unwrapped SHOVE steps to hold the drifted target.
        floor: minimum L2 norm of each perturbing offset.
        control_mode: "pd_joint_pos" -> action = [target(7), grip];
            "pd_joint_pos_vel" -> action = [target(7), zeros(7), grip].
        action_prefix: action-dict key prefix; key is f"{action_prefix}-{i}".
        sink: optional list; if not None, a DEEP COPY of the action dict stepped is
            appended once per step (``pre_hold_steps + K`` copies total, in order:
            the pre-hold dicts first, then the shove dicts) for introspection/testing.
        pre_hold_steps: number of SETTLE steps (all arms hold, no offset) to run
            BEFORE the K shove steps. Default 0 (no settle window -> exact legacy
            behavior). The grasp-protection settle guard passes a positive value.

    Returns:
        None. (Side effect: ``pre_hold_steps + K`` calls to ``base_env.step``.)
    """
    move_ids = move_ids if isinstance(move_ids, list) else [move_ids]

    def _hold_target(i):
        """The no-offset setpoint arm i holds (frozen_qpos if given, else live)."""
        if hold_qpos is not None:
            return np.asarray(hold_qpos[i], dtype=np.float64).reshape(-1)[:7]
        return _arm_qpos7(robots[i])  # fallback: live qpos

    # capture each arm's hold target ONCE (same target reused in both phases for
    # non-move arms; the move arms hold here in the settle phase, then drift).
    holds = [_hold_target(i) for i in range(len(robots))]

    # --- phase 1: SETTLE -- every arm holds, NO offset (grasp-protection window) ---
    if pre_hold_steps > 0:
        hold_dict = {
            f"{action_prefix}-{i}": _assemble_action(holds[i], grips[i], control_mode)
            for i in range(len(robots))
        }
        for _ in range(int(pre_hold_steps)):
            if sink is not None:
                sink.append(copy.deepcopy(hold_dict))
            base_env.step(hold_dict)  # UNWRAPPED -> unrecorded

    # --- phase 2: SHOVE -- move arms drift, others hold (same target across K) ---
    action_dict = {}
    for i in range(len(robots)):
        if i in move_ids:
            q0 = _arm_qpos7(robots[i])  # captured ONCE (live pose of the shoved arm)
            target = q0 + sample_floored_offset(rng, sigma, q0.shape[0], floor)
        else:
            # HOLD the interpreter's last-commanded setpoint (frozen_qpos) if the
            # caller supplied one -- NOT the lagging live qpos. A concurrently moving
            # non-move arm is mid-trajectory (PD controller behind its target); pinning
            # it at live qpos for K steps STALLS it (a corruption path that fails the
            # arm's success check). Holding its commanded setpoint keeps it tracking.
            target = holds[i]
        grip = grips[i]  # caller-supplied (frozen) grip -- NOT planner state
        action_dict[f"{action_prefix}-{i}"] = _assemble_action(target, grip, control_mode)

    for _ in range(K):  # HOLD the same target for all K steps
        if sink is not None:
            sink.append(copy.deepcopy(action_dict))
        base_env.step(action_dict)  # UNWRAPPED -> unrecorded
