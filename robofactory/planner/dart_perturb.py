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


def inject_joint_disturbance(
    base_env,
    robots,
    grips,
    move_ids,
    rng,
    sigma,
    K,
    floor,
    control_mode="pd_joint_pos",
    action_prefix="panda",
    sink=None,
):
    """Step the UNWRAPPED env ``K`` times with a noisy joint target so selected
    arms drift off-path. NOT recorded (the unwrapped env bypasses any
    RecordEpisode wrapper), so the noise never becomes a training label.

    For each arm ``i`` in ``range(len(robots))`` we capture ``q0`` = first 7
    joints of ``robots[i].get_qpos()`` ONCE. Arms whose index is in
    ``move_ids`` get a single floored random offset added to ``q0`` (see
    :func:`sample_floored_offset`); all other arms HOLD ``q0``. That same
    drifted/held target is then commanded for all ``K`` steps.

    The gripper channel for arm ``i`` is taken verbatim from ``grips[i]`` --
    the caller MUST pass the live/frozen grip (e.g. ``frozen_grip``). This
    function never reads ``planner.gripper_state``.

    Args:
        base_env: the UNWRAPPED env; ``base_env.step(action_dict)`` is called.
        robots: list of robot handles, each exposing ``get_qpos()`` (array-like
            or torch tensor; first 7 entries are arm joints).
        grips: list of current gripper commands, one float per arm. Indexed by
            arm id and emitted verbatim as the gripper channel.
        move_ids: int or list[int] of arm indices to perturb. Arms not listed
            hold their current qpos.
        rng: a ``numpy.random.Generator``.
        sigma: stddev of the per-joint normal offset.
        K: number of (identical) unwrapped steps to hold the drifted target.
        floor: minimum L2 norm of each perturbing offset.
        control_mode: "pd_joint_pos" -> action = [target(7), grip];
            "pd_joint_pos_vel" -> action = [target(7), zeros(7), grip].
        action_prefix: action-dict key prefix; key is f"{action_prefix}-{i}".
        sink: optional list; if not None, a DEEP COPY of the action dict is
            appended once per step (K copies total) for introspection/testing.

    Returns:
        None. (Side effect: ``K`` calls to ``base_env.step``.)
    """
    move_ids = move_ids if isinstance(move_ids, list) else [move_ids]

    action_dict = {}
    for i in range(len(robots)):
        q0 = _arm_qpos7(robots[i])  # captured ONCE
        if i in move_ids:
            target = q0 + sample_floored_offset(rng, sigma, q0.shape[0], floor)
        else:
            target = q0  # HOLD current qpos
        grip = grips[i]  # caller-supplied (frozen) grip -- NOT planner state
        if control_mode == "pd_joint_pos_vel":
            action = np.hstack([target, target * 0, grip])
        else:  # pd_joint_pos (default)
            action = np.hstack([target, grip])
        action_dict[f"{action_prefix}-{i}"] = action

    for _ in range(K):  # HOLD the same target for all K steps
        if sink is not None:
            sink.append(copy.deepcopy(action_dict))
        base_env.step(action_dict)  # UNWRAPPED -> unrecorded
