"""Per-arm subtask primitives for subtask-conditioned data generation.

A :class:`Primitive` is a *planned but un-stepped* unit of arm behavior: a verb,
its target, the rendered NL prompt, and a list of ``ticks`` where each tick is a
``(qpos7, grip_cmd)`` pair. The interpreter (see ``subtask_interpreter.py``) owns
the env step loop and consumes these ticks one per ``env.step``.

CRITICAL invariants (see plan ``1-this-is-not-witty-quokka.md`` top risks):

* **Per-arm dry_run ONLY.** Planning calls
  ``planner.move_to_pose_with_screw(pose, move_id=arm, dry_run=True)`` with a
  SINGLE pose. A pose-LIST under dry_run silently returns only the first arm's
  plan (motionplanner.py:185-186), so we NEVER pass a list here.
* **Gripper ramp reproduced.** ``close_gripper`` / ``open_gripper`` reproduce the
  ±0.1 ramp (20 ticks) from motionplanner.{open,close}_gripper, holding the arm's
  qpos frozen across the ramp.
* **approach-stop = approach with NO following close_gripper** — i.e. an
  ``approach`` primitive by itself never closes the gripper; closing is a separate
  ``close_gripper`` primitive. (Tested: approach-stop has zero gripper-close ticks.)
* **hold/wait repeats a FROZEN qpos byte-for-byte**, keeping the current grip.

This module imports numpy at top (cheap, no GPU). It does NOT import sapien or
robofactory.planner.motionplanner at module top; it only *uses* a passed-in
``planner`` object at call time, so off-GPU unit tests can pass a fake planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import subtask_vocab as vocab

# Gripper command extremes, matching motionplanner.OPEN / CLOSED.
OPEN = 1.0
CLOSED = -1.0
# Number of ramp ticks for the gripper, matching motionplanner.{open,close}_gripper.
GRIPPER_RAMP_TICKS = 20
GRIPPER_RAMP_STEP = 0.1

# A single control tick: (qpos7, grip_cmd). qpos7 is a length-7 float32 array.
Tick = Tuple[np.ndarray, float]


@dataclass
class Primitive:
    """A planned-but-un-stepped per-arm subtask.

    Attributes
    ----------
    verb_id, target_id : small ints from subtask_vocab.
    text : rendered NL prompt (vocab.render).
    ticks : list of (qpos7, grip_cmd). One tick == one env.step for this arm.
    task : task name (for re-rendering / debugging).
    success_check : optional callable(env, arm, primitive) -> bool, run by the
        interpreter AFTER the primitive's ticks to drop liar-labels (the crux:
        a labelled ``close_gripper`` that did not actually grasp is dropped).
    name : human label for debugging (e.g. "approach(right_end)").
    """

    verb_id: int
    target_id: int
    text: str
    ticks: List[Tick] = field(default_factory=list)
    task: str = "LiftBarrier"
    success_check: Optional[Callable] = None
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{vocab.VERBS[self.verb_id]}({self.target_id})"

    @property
    def n_ticks(self) -> int:
        return len(self.ticks)

    @property
    def last_qpos(self) -> Optional[np.ndarray]:
        """The qpos of the final tick (the commanded target to FREEZE on)."""
        if not self.ticks:
            return None
        return np.asarray(self.ticks[-1][0], dtype=np.float32)

    @property
    def last_grip(self) -> Optional[float]:
        if not self.ticks:
            return None
        return float(self.ticks[-1][1])


# --------------------------------------------------------------------------- helpers


def _waypoints_from_plan(result) -> np.ndarray:
    """Extract the (n, 7) joint waypoint array from a dry_run plan result.

    ``move_to_pose_with_screw(..., dry_run=True)`` returns the planner result dict
    (NOT a tuple) with ``result["position"]`` shape (n_steps, 7). A failed plan
    returns the sentinel -1 (an int), which we surface loudly.
    """
    if isinstance(result, int) and result == -1:
        raise RuntimeError("motion plan failed (move_to_pose_with_screw returned -1)")
    if not isinstance(result, dict) or "position" not in result:
        raise RuntimeError(
            f"unexpected dry_run plan result type {type(result)}; expected dict with "
            f"'position'. Did you pass a pose LIST under dry_run (silent single-arm drop)?"
        )
    pos = np.asarray(result["position"], dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 7:
        raise RuntimeError(f"plan position has unexpected shape {pos.shape}; expected (n,7)")
    if pos.shape[0] == 0:
        raise RuntimeError("plan returned zero waypoints")
    return pos


def _plan_to_pose(planner, arm: int, pose) -> np.ndarray:
    """Per-arm dry_run plan to a SINGLE pose -> (n, 7) waypoints.

    HARD guard: refuse a list of poses (the silent single-arm-drop trap).
    """
    if isinstance(pose, list):
        raise ValueError(
            "subtask_primitives plans PER-ARM with a single pose; a pose LIST under "
            "dry_run silently returns only the first arm's plan (motionplanner.py:185-186)."
        )
    result = planner.move_to_pose_with_screw(pose, move_id=arm, dry_run=True)
    return _waypoints_from_plan(result)


def _ticks_from_waypoints(waypoints: np.ndarray, grip: float) -> List[Tick]:
    """Turn a (n,7) waypoint array into ticks, all carrying the SAME grip cmd.

    Matches follow_path: each control tick is hstack([qpos7, grip]); the gripper
    is held constant across a motion (it only changes in the gripper primitives).
    """
    return [(np.asarray(wp, dtype=np.float32).copy(), float(grip)) for wp in waypoints]


def _current_qpos(planner, arm: int) -> np.ndarray:
    """Current 7-dof arm qpos (drops the 2 gripper joints), matching follow_path:122."""
    q = planner.robot[arm].get_qpos()
    # torch tensor on GPU -> numpy; shape (1, 9) -> first 7
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    q = np.asarray(q)
    if q.ndim == 2:
        q = q[0]
    return q[:7].astype(np.float32)


# --------------------------------------------------------------------------- grasp-pose resolvers
# These read the LIVE actor pose at plan time, so object perturbation (DART) at a
# primitive boundary is absorbed: the next primitive re-resolves the pose.


def resolve_cube_grasp_pose(planner, env, arm: int, color: str):
    """Grasp pose for a cube (by color) via get_grasp_pose_from_obb (live pose)."""
    cube_name = vocab.TSC_COLOR_CUBE[color]
    actor = getattr(env, cube_name)
    return planner.get_grasp_pose_from_obb(actor, arm)


def resolve_lb_end_grasp_pose(planner, env, target_id: int, pre_dis: float = 0.0):
    """Grasp pose for a LiftBarrier end via labeled-direction contact point (live pose)."""
    contact_id = vocab.LB_TARGET_CONTACT_ID[target_id]
    return planner.get_grasp_pose_w_labeled_direction(
        actor=env.barrier, actor_data=env.annotation_data["barrier"],
        pre_dis=pre_dis, id=contact_id,
    )


# --------------------------------------------------------------------------- builder functions
# All builders PLAN (per-arm dry_run) and return a Primitive with ticks. They do
# NOT step the env. ``grip`` defaults preserve the current gripper command so the
# motion primitives never accidentally open/close.


def approach(
    planner, env, arm: int, target_id: int, task: str,
    grip: float = OPEN, pre_dis: float = 0.0,
    success_check: Optional[Callable] = None,
) -> Primitive:
    """approach-stop: plan a reach to the target's grasp pose. NO gripper close.

    LiftBarrier: target is an end (left_end/right_end) -> labeled-direction grasp.
    ThreeRobotsStackCube: target is a color -> OBB grasp.
    The returned primitive's ticks carry the *unchanged* ``grip`` (open by default);
    closing is a SEPARATE close_gripper primitive (approach-stop == no close ticks).
    """
    if task == "LiftBarrier":
        pose = resolve_lb_end_grasp_pose(planner, env, target_id, pre_dis=pre_dis)
    elif task == "ThreeRobotsStackCube":
        color = vocab.target_name(task, target_id)
        pose = resolve_cube_grasp_pose(planner, env, arm, color)
    else:
        raise KeyError(f"approach not implemented for task {task!r}")
    waypoints = _plan_to_pose(planner, arm, pose)
    ticks = _ticks_from_waypoints(waypoints, grip)
    text = vocab.render(vocab.APPROACH, target_id, task)
    return Primitive(
        verb_id=vocab.APPROACH, target_id=target_id, text=text, ticks=ticks,
        task=task, success_check=success_check,
        name=f"approach({vocab.target_name(task, target_id)})",
    )


def _gripper_ramp(
    planner, env, arm: int, verb_id: int, task: str,
    start_grip: float, target_id: int = 0,
    success_check: Optional[Callable] = None,
) -> Primitive:
    """Shared gripper-ramp primitive (open or close).

    Reproduces motionplanner.{open,close}_gripper: 20 ticks, ±0.1 per tick clamped
    at OPEN(+1)/CLOSED(-1), with the arm qpos held FROZEN at its current value.
    """
    qpos = _current_qpos(planner, arm)  # frozen across the whole ramp
    sign = +1.0 if verb_id == vocab.OPEN_GRIPPER else -1.0
    limit = OPEN if verb_id == vocab.OPEN_GRIPPER else CLOSED
    g = float(start_grip)
    ticks: List[Tick] = []
    for _ in range(GRIPPER_RAMP_TICKS):
        if sign > 0:
            g = min(g + GRIPPER_RAMP_STEP, limit)
        else:
            g = max(g - GRIPPER_RAMP_STEP, limit)
        ticks.append((qpos.copy(), float(g)))
    text = vocab.render(verb_id, target_id, task)
    return Primitive(
        verb_id=verb_id, target_id=target_id, text=text, ticks=ticks,
        task=task, success_check=success_check,
        name=vocab.VERBS[verb_id],
    )


def close_gripper(
    planner, env, arm: int, task: str, start_grip: float = OPEN,
    target_id: int = 0, success_check: Optional[Callable] = None,
) -> Primitive:
    """Close-gripper ramp (OPEN -> CLOSED over 20 ticks), qpos frozen."""
    return _gripper_ramp(
        planner, env, arm, vocab.CLOSE_GRIPPER, task, start_grip=start_grip,
        target_id=target_id, success_check=success_check,
    )


def open_gripper(
    planner, env, arm: int, task: str, start_grip: float = CLOSED,
    target_id: int = 0, success_check: Optional[Callable] = None,
) -> Primitive:
    """Open-gripper ramp (CLOSED -> OPEN over 20 ticks), qpos frozen."""
    return _gripper_ramp(
        planner, env, arm, vocab.OPEN_GRIPPER, task, start_grip=start_grip,
        target_id=target_id, success_check=success_check,
    )


def lift(
    planner, env, arm: int, target_id: int, task: str, dz: float = 0.2,
    grip: float = CLOSED, success_check: Optional[Callable] = None,
) -> Primitive:
    """Lift: re-plan from the current tcp grasp pose raised by ``dz`` in z.

    Plans from the CURRENT arm grasp pose (read live from the planner's grasp
    resolver for the target) lifted by dz. The barrier/cube is carried (grip held
    closed by default). Matches the canonical solver's ``pose[2] += 0.2`` lift.
    """
    if task == "LiftBarrier":
        pose = resolve_lb_end_grasp_pose(planner, env, target_id)
    elif task == "ThreeRobotsStackCube":
        color = vocab.target_name(task, target_id)
        pose = resolve_cube_grasp_pose(planner, env, arm, color)
    else:
        raise KeyError(f"lift not implemented for task {task!r}")
    pose = np.asarray(pose, dtype=np.float32).copy()
    pose[2] += dz
    waypoints = _plan_to_pose(planner, arm, pose)
    ticks = _ticks_from_waypoints(waypoints, grip)
    text = vocab.render(vocab.LIFT, target_id, task)
    return Primitive(
        verb_id=vocab.LIFT, target_id=target_id, text=text, ticks=ticks,
        task=task, success_check=success_check,
        name=f"lift({vocab.target_name(task, target_id)})",
    )


def place(
    planner, env, arm: int, target_id: int, task: str,
    now_pose, dest_actor_name: str, height_offset: float = 0.05,
    grip: float = CLOSED, success_check: Optional[Callable] = None,
) -> Primitive:
    """Place: plan to a stack destination above ``dest_actor_name``.

    Uses get_grasp_pose_for_stack(now_pose, dest_actor) to align the held object
    above the destination actor (goal_region or another cube), reading the dest's
    LIVE pose. ``now_pose`` is the carried object's current grasp pose (7-vec); the
    caller supplies it (from the prior approach/lift). Grip held closed (carrying).
    """
    dest_actor = getattr(env, dest_actor_name)
    target_pose = planner.get_grasp_pose_for_stack(
        np.asarray(now_pose, dtype=np.float32), dest_actor, height_offset=height_offset,
    )
    waypoints = _plan_to_pose(planner, arm, target_pose)
    ticks = _ticks_from_waypoints(waypoints, grip)
    text = vocab.render(vocab.PLACE, target_id, task)
    return Primitive(
        verb_id=vocab.PLACE, target_id=target_id, text=text, ticks=ticks,
        task=task, success_check=success_check,
        name=f"place({vocab.target_name(task, target_id)})",
    )


def hold(
    planner, env, arm: int, n: int, task: str,
    qpos: Optional[np.ndarray] = None, grip: float = OPEN,
    verb_id: int = vocab.WAIT, target_id: int = 0,
) -> Primitive:
    """Hold/wait: repeat a FROZEN qpos for ``n`` ticks, keeping the grip command.

    The interpreter uses this for idle/blocked arms and for short seam-holds. The
    qpos is frozen byte-for-byte (NOT re-read live), so the arm does not sag. If
    ``qpos`` is None it is sampled ONCE from the current arm qpos and then frozen.
    Labelled ``wait`` by default.
    """
    if qpos is None:
        qpos = _current_qpos(planner, arm)
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    ticks: List[Tick] = [(qpos.copy(), float(grip)) for _ in range(int(n))]
    text = vocab.render(verb_id, target_id, task)
    return Primitive(
        verb_id=verb_id, target_id=target_id, text=text, ticks=ticks,
        task=task, name=f"hold({n})",
    )


# alias: a bare "wait" primitive is a hold labelled wait
def wait(planner, env, arm: int, n: int, task: str,
         qpos: Optional[np.ndarray] = None, grip: float = OPEN) -> Primitive:
    """Wait: alias for hold labelled with verb wait (the idle behavior)."""
    return hold(planner, env, arm, n, task, qpos=qpos, grip=grip,
                verb_id=vocab.WAIT, target_id=0)
