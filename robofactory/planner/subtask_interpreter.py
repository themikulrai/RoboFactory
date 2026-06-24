"""Subtask interpreter: owns the env step loop, runs per-arm primitive queues,
labels every action tick, and records the per-arm subtask stream live.

Design (see plan ``1-this-is-not-witty-quokka.md`` "Architecture: invert control"):

* Each arm has a QUEUE of :class:`QueuedRecipe`. A ``QueuedRecipe`` wraps a
  :class:`PrimitiveRecipe` — a callable ``recipe(planner, env, arm) -> Primitive``
  that PLANS one primitive against the LIVE env/qpos. The interpreter calls the
  recipe WHEN THE ARM ENTERS THAT PRIMITIVE (ti==0), not all upfront, so every
  primitive plans from the pose the arm is actually in at execution time (the
  JIT / correct-start-state fix). One outer loop advances every arm by one tick,
  assembles a single ``action_dict`` keyed ``f"{action_prefix}-{i}"`` (format per
  follow_path:118-138), and does ONE ``env.step(action_dict)`` per tick (wrapped
  env -> RecordEpisodeMA records it).
* **Idle / blocked arm HOLDS the FROZEN last commanded qpos** (NOT live qpos) and
  is labelled ``wait``. An arm is blocked when its next primitive has an unmet
  ``wait_for=(other_arm, predicate)`` gate.
* **Labels align to the ACTION index** (length T; obs is T+1). The label for
  action[t] is the subtask that DROVE action[t] (matches the eval's "subtask
  active for THIS step" convention). Guaranteed by construction: we append the
  label at the same loop iteration we build and step the action.
* Between primitives the interpreter calls an optional
  ``boundary_hook(env, state)`` where the user's DART arm-noise / object-pose
  perturbation runs on the UNWRAPPED env (unrecorded).

  JIT RE-PLANNING (the key contract): primitives are now built LAZILY, by calling
  ``recipe(planner, env, arm)`` at the moment the arm ENTERS the primitive — i.e.
  AFTER the boundary_hook for that transition has fired. So the hook may perturb
  BOTH the arm joints AND the OBJECT poses: the very next recipe re-plans (a fresh
  per-arm ``dry_run`` screw plan) from the perturbed state, so its waypoints and
  its stored ``target_pose`` reflect the moved object. Both modes are now first-class:
    1. **Arm-joint noise**: the recipe re-plans from the nudged qpos.
    2. **Object-pose perturbation**: the recipe re-resolves the moved actor's grasp
       pose (resolvers read the LIVE actor pose at call time). The label still tracks
       the ACTOR (e.g. "the green cube"), so it stays correct.
  The FIRST primitive of each arm is also built at its entry (arm still at reset =
  the same state the builders previously planned against), and the hook does NOT
  fire before the first primitive.
* Per-primitive **success checks** (run after the primitive's last tick) drop
  trajectories whose labelled subtask did not actually happen (the liar-label
  guard / the crux).

This module imports numpy at top; it does NOT import sapien. It only *uses* the
passed-in wrapped ``env`` / ``planner`` at call time. ``dry_run`` planning inside a
recipe only PLANS (``move_to_pose_with_screw(dry_run=True)`` returns the joint
waypoint dict; it does NOT ``env.step`` / advance physics). It may set the
planner's INTERNAL planning-model qpos, but that is a throwaway planning model,
not the simulated env — confirmed safe for mid-rollout re-planning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import subtask_vocab as vocab
from . import dart_perturb
from .subtask_primitives import Primitive, OPEN

# A PrimitiveRecipe plans ONE primitive against the LIVE env/qpos at entry time:
#   recipe(planner, env, arm) -> Primitive
# (the interpreter calls it when the arm enters the primitive, NOT upfront).
PrimitiveRecipe = Callable[[object, object, int], Primitive]

# Verbs whose label is a CLAIM about a spatial/grasp outcome and therefore MUST
# carry a success_check, or a failed primitive silently mislabels the data (the
# liar-label loophole). CLOSE_GRIPPER is NOT in this set: is_grasping does not
# register immediately after the close ramp (the grasp seats during the subsequent
# lift; even the canonical solveLiftBarrier shows is_grasping=False right after its
# own close), so checking it right after the close is PREMATURE and false-fails on
# success. A close that did not grasp is caught DOWNSTREAM (no grasp -> the lift
# never reaches env success -> the group is dropped), so close needs no own check.
# wait/open_gripper have no spatial claim to verify (open just releases), so they
# are exempt too. ``run_program`` lints the program against this set (warning by
# default; raise with check_success_coverage="error").
VERBS_REQUIRING_SUCCESS_CHECK = frozenset(
    {vocab.APPROACH, vocab.LIFT, vocab.PLACE}
)

# A per-primitive gate: advance this primitive only once predicate(env, state)
# is True for the named other arm's progress. predicate signature:
#   predicate(env, state) -> bool
WaitFor = Tuple[int, Callable]


@dataclass
class QueuedRecipe:
    """A primitive RECIPE plus its optional cross-arm gate.

    ``recipe`` is a :data:`PrimitiveRecipe` — ``recipe(planner, env, arm) ->
    Primitive`` — called by the interpreter WHEN THE ARM ENTERS this primitive, so
    the primitive is planned from the LIVE env/qpos at that moment (JIT building).

    ``wait_for`` is (other_arm, predicate); the primitive is not built/ticked until
    predicate(env, state) is True. ``predicate`` is typically a success check on the
    other arm (e.g. "arm A is grasping"), enabling sequential or hold-while-other-
    works choreography. (The recipe is built only AFTER the gate opens, so it plans
    from the state at the moment the gate releases.)
    """

    recipe: PrimitiveRecipe
    wait_for: Optional[WaitFor] = None


# --------------------------------------------------------------------------- recorder


class SubtaskRecorder:
    """Accumulates per-tick per-arm ``(verb_id, target_id, text)`` aligned to the
    action index, then flushes a ``subtask_stream`` keyed ``traj_{id}``.

    One ``append(labels)`` call per env.step, where ``labels`` is
    ``{arm: (verb_id, target_id, text)}`` for EVERY arm. After T steps the per-arm
    arrays have length T == len(actions). ``flush(episode_id, ...)`` writes an npz
    (default) or appends to an open h5 group.
    """

    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self._verb: List[List[int]] = [[] for _ in range(num_arms)]
        self._target: List[List[int]] = [[] for _ in range(num_arms)]
        self._text: List[List[str]] = [[] for _ in range(num_arms)]

    def reset(self):
        self._verb = [[] for _ in range(self.num_arms)]
        self._target = [[] for _ in range(self.num_arms)]
        self._text = [[] for _ in range(self.num_arms)]

    def append(self, labels: Dict[int, Tuple[int, int, str]]):
        for arm in range(self.num_arms):
            if arm not in labels:
                raise ValueError(
                    f"SubtaskRecorder.append: missing label for arm {arm} "
                    f"(got arms {sorted(labels)}); EVERY arm must be labelled each tick"
                )
            v, t, txt = labels[arm]
            self._verb[arm].append(int(v))
            self._target[arm].append(int(t))
            self._text[arm].append(str(txt))

    @property
    def length(self) -> int:
        """Number of recorded ticks (== T == len(actions))."""
        return len(self._verb[0]) if self.num_arms else 0

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Return per-arm arrays: subtask_arm{i}_{verb,target,text}, plus length."""
        out: Dict[str, np.ndarray] = {}
        for arm in range(self.num_arms):
            out[f"subtask_arm{arm}_verb"] = np.asarray(self._verb[arm], dtype=np.int64)
            out[f"subtask_arm{arm}_target"] = np.asarray(self._target[arm], dtype=np.int64)
            out[f"subtask_arm{arm}_text"] = np.asarray(self._text[arm], dtype=object)
        out["length"] = np.asarray(self.length, dtype=np.int64)
        return out

    def flush(
        self,
        episode_id: int,
        out_dir: Optional[str] = None,
        h5_group=None,
        expected_T: Optional[int] = None,
    ) -> Optional[str]:
        """Write the subtask stream for this episode keyed ``traj_{episode_id}``.

        * If ``h5_group`` is given (an open h5py.File / Group), create a subgroup
          ``traj_{episode_id}`` with the per-arm datasets (text stored as a
          variable-length-utf8 dataset). Returns None.
        * Else write ``{out_dir}/subtask_stream_traj_{episode_id}.npz`` and return
          its path.

        ``expected_T`` (the len(actions)) is asserted == self.length if given —
        the off-by-one / length guard.
        """
        if expected_T is not None and self.length != expected_T:
            raise AssertionError(
                f"subtask stream length {self.length} != expected actions length "
                f"{expected_T} for episode {episode_id} (off-by-one / alignment bug)"
            )
        arrays = self.to_arrays()
        key = f"traj_{episode_id}"

        if h5_group is not None:
            import h5py  # lazy
            grp = h5_group.create_group(key, track_order=True)
            for arm in range(self.num_arms):
                grp.create_dataset(f"subtask_arm{arm}_verb",
                                   data=arrays[f"subtask_arm{arm}_verb"])
                grp.create_dataset(f"subtask_arm{arm}_target",
                                   data=arrays[f"subtask_arm{arm}_target"])
                str_dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset(
                    f"subtask_arm{arm}_text",
                    data=[s for s in arrays[f"subtask_arm{arm}_text"]],
                    dtype=str_dt,
                )
            grp.attrs["length"] = int(self.length)
            grp.attrs["num_arms"] = int(self.num_arms)
            return None

        if out_dir is None:
            raise ValueError("flush needs either out_dir or h5_group")
        import os
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"subtask_stream_{key}.npz")
        np.savez(path, **{k: v for k, v in arrays.items()})
        return path


# --------------------------------------------------------------------------- success checks
# Factory functions returning predicate(env, arm, primitive)->bool (for a
# primitive's own success_check) OR predicate(env, state)->bool (for wait_for
# gates). The interpreter normalizes the call (see _eval_success).


def _unwrap(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _scalar(x) -> float:
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return float(np.asarray(a).reshape(-1)[0])


def check_is_grasping(target_actor_name: str) -> Callable:
    """close_gripper success: arm i is grasping the named actor."""
    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        actor = getattr(u, target_actor_name)
        agent = u.agent.agents[arm]
        return bool(np.asarray(_grasp_bool(agent, actor)).reshape(-1)[0])
    return _check


def _grasp_bool(agent, actor):
    g = agent.is_grasping(actor)
    return g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)


def check_tcp_near(target_pose_fn: Optional[Callable] = None, tol: float = 0.05) -> Callable:
    """approach/lift success: arm tcp within ``tol`` (m) of the target xyz.

    The target xyz is resolved one of two ways:
      * if ``target_pose_fn`` is given: ``target_pose_fn(env, arm)`` (length-3 xyz
        or 7-vec, first 3 used).
      * if ``target_pose_fn`` is None (the default, and what the LB scenarios use):
        the PLANNED ``primitive.target_pose`` stored by the spatial builder at plan
        time. The TCP converges to the planned pose, so "TCP within tol of the
        planned target" is the correct check. (The old barrier-CENTRE fn was wrong:
        each arm reaches an END ~0.37m off centre, so it could never pass.)

    Raises if neither a fn nor a stored target_pose is available (a spatial verb
    with no target would silently never pass otherwise).
    """
    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        tcp_p = u.agent.agents[arm].tcp.pose.p
        tcp_p = tcp_p.detach().cpu().numpy() if hasattr(tcp_p, "detach") else np.asarray(tcp_p)
        tcp_p = np.asarray(tcp_p).reshape(-1)[:3]
        if target_pose_fn is not None:
            tgt = np.asarray(target_pose_fn(env, arm)).reshape(-1)[:3]
        else:
            if primitive.target_pose is None:
                raise ValueError(
                    "check_tcp_near: no target_pose_fn given and primitive "
                    f"{primitive.name!r} has no stored target_pose (the spatial "
                    "builder must set it). Cannot verify approach/lift."
                )
            tgt = np.asarray(primitive.target_pose).reshape(-1)[:3]
        return float(np.linalg.norm(tcp_p - tgt)) <= tol
    return _check


def check_barrier_lifted(dz: float = 0.15) -> Callable:
    """lift success (LiftBarrier): barrier CENTRE z > arm0 base z + dz.

    LEGACY grasp-blind centre check (kept for back-compat / reference). The strict
    success now used by the scenarios is :func:`check_barrier_ends_held`.
    """
    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        bz = _scalar(u.barrier.pose.p[..., 2])
        base_z = _scalar(u.agent.agents[0].robot.pose.p[0, 2])
        return bz > base_z + dz
    return _check


# Barrier grasp-end offsets in the barrier's LOCAL frame (long axis = local-X).
# [+/-0.222, 0, +0.074] -- the annotation contact points (local-X +/-0.37) scaled by
# the actor scale [0.6, 0.6, 0.2]. See robofactory.tasks.success_candidates
# (BARRIER_END_OFFSETS) for the full derivation; kept duplicated here so the
# numpy-only interpreter does not import torch / the tasks package.
_BARRIER_END_OFFSETS_NP = np.array(
    [[0.222, 0.0, 0.074], [-0.222, 0.0, 0.074]], dtype=np.float64
)
# Sustained-criterion constants, kept in lock-step with
# robofactory.tasks.success_candidates (HOLD_FRAMES_K, GRIPPER_CLOSE_MAX). Duplicated
# here so the numpy-only interpreter does not import torch / the tasks package.
_HOLD_FRAMES_K = 8
_GRIPPER_CLOSE_MAX = 0.06


def _arm_gripper_closed_np(agent) -> bool:
    """Single-env: is this arm's gripper CLOSED (finger-sum < _GRIPPER_CLOSE_MAX)?

    The Panda arm's qpos is 9-dim (7 arm + 2 fingers); "closed" means the LAST 2
    entries (finger joints) sum below the threshold (open ~0.08 each, closed ~0.00).
    Reads ``agent.robot.get_qpos()`` (obs-mode-agnostic), tolerating a torch tensor
    and a leading batch dim, and takes the first env's last 2 qpos entries.
    """
    q = agent.robot.get_qpos()
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float64)
    if q.ndim > 1:
        q = q[0]  # first env (single-env generation path)
    return float(q[-2:].sum()) < _GRIPPER_CLOSE_MAX


def _quat_rotate_wxyz_np(q, v):
    """Rotate vector ``v`` by quaternion ``q`` (wxyz), numpy single-env.

    q: (4,) wxyz ; v: (3,) or (N, 3) -> same shape as v. Standard
    v' = v + 2*w*(u x v) + 2*(u x (u x v)) form.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    w, u = q[0], q[1:4]
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def check_barrier_ends_held(dz: float = 0.25, k: int = _HOLD_FRAMES_K) -> Callable:
    """GEOMETRIC lift success PER-FRAME predicate C (LiftBarrier), single-env / numpy.

    Mirrors ``lift_barrier_success_strict`` (the per-frame predicate) and the C used by
    ``LiftBarrierEnv.evaluate``'s sustained counter. Returns True iff, AT THE FRAME IT
    IS CALLED:
      * BOTH grasp ENDS of the barrier have world-z > base_z + ``dz``  AND
      * BOTH arms' grippers are CLOSED (each arm's last-2 finger-joint qpos sum <
        ``_GRIPPER_CLOSE_MAX``).
    base_z = arm-0 robot base z. The two grasp ends are
    ``barrier.pose.p + R(barrier.pose.q) @ [+/-0.222, 0, 0.074]`` (barrier.pose.q is
    wxyz). ``is_grasping`` is DROPPED: the contact-force probe UNRELIABLY registers a
    real load-bearing grasp on the thin (~0.09 m) bar ends (one arm read False for 25
    consecutive frames on a genuine clean held lift), so it wrongly rejects genuine
    held lifts. The gripper-closed check is the robust geometric stand-in.
    (``arm``/``primitive`` are unused -- this is a whole-bar predicate.)

    WHY SINGLE-FRAME HERE (not a K-consecutive counter): the interpreter calls a
    primitive's ``success_check`` EXACTLY ONCE, at end-of-primitive (after the lift's
    last tick -- see ``run_program``), NOT once per frame. A stateful K-consecutive
    closure called once could only reach count 1 and would FALSE-DROP every genuine
    lift (poisoning ``all_success`` -> the keep-predicate drops the trajectory). The
    SUSTAINED guarantee (kills transient single-frame flings) is enforced ONCE, in the
    AUTHORITATIVE place: the env's batched ``LiftBarrierEnv._lift_hold`` counter, which
    requires ``HOLD_FRAMES_K`` consecutive frames before ``info['success']`` flips. The
    keep-predicate already requires env-success, so the sustain is honored regardless
    of this check; this check is the per-arm liar-label guard ("did the lift actually
    raise its end while gripping?"), for which the representative end-of-lift frame is
    the right thing to test. ``k`` is accepted for signature/lock-step parity but is
    NOT used here (the env owns the counter); it is documented to avoid implying this
    function sustains on its own.
    """
    del k  # the env's _lift_hold owns the K-consecutive sustain; see docstring.

    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        p = u.barrier.pose.p
        p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
        p = np.asarray(p).reshape(-1)[:3]
        q = u.barrier.pose.q
        q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        q = np.asarray(q).reshape(-1)[:4]
        ends = p[None, :] + _quat_rotate_wxyz_np(q, _BARRIER_END_OFFSETS_NP)  # (2,3)
        base_z = _scalar(u.agent.agents[0].robot.pose.p[0, 2])
        ends_high = bool(np.all(ends[:, 2] > base_z + dz))
        closed0 = _arm_gripper_closed_np(u.agent.agents[0])
        closed1 = _arm_gripper_closed_np(u.agent.agents[1])
        return ends_high and closed0 and closed1

    return _check


def check_actor_z_above(actor_name: str, base_z: float, dz: float = 0.15) -> Callable:
    """generic lift success: named actor z above base_z + dz."""
    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        z = _scalar(getattr(u, actor_name).pose.p[..., 2])
        return z > base_z + dz
    return _check


# --------------------------------------------------------------------------- interpreter


@dataclass
class _ArmState:
    """Per-arm runtime state inside the loop."""

    queue: List[QueuedRecipe]
    qi: int = 0                       # index of the current recipe in queue
    ti: int = 0                       # tick index within the current primitive
    frozen_qpos: Optional[np.ndarray] = None  # last commanded qpos (for holds)
    frozen_grip: float = OPEN         # last commanded grip
    done: bool = False
    # the Primitive built from the recipe for the CURRENT qi (cached for its
    # duration); None until built at entry, reset to None when qi advances.
    built: Optional[Primitive] = None
    # success flags per primitive index (None=not yet checked)
    success: Dict[int, Optional[bool]] = field(default_factory=dict)

    @property
    def current(self) -> Optional[QueuedRecipe]:
        if self.qi >= len(self.queue):
            return None
        return self.queue[self.qi]


def _action_for(control_mode: str, qpos7: np.ndarray, grip: float) -> np.ndarray:
    """Assemble one arm's action vector (format per follow_path:114-136)."""
    qpos7 = np.asarray(qpos7, dtype=np.float32)
    if control_mode == "pd_joint_pos_vel":
        return np.hstack([qpos7, qpos7 * 0.0, np.float32(grip)]).astype(np.float32)
    return np.hstack([qpos7, np.float32(grip)]).astype(np.float32)


def run_program(
    env,
    planner,
    programs: Dict[int, List[QueuedRecipe]],
    recorder: SubtaskRecorder,
    max_steps: int,
    boundary_hook: Optional[Callable] = None,
    action_prefix: str = "panda",
    control_mode: Optional[str] = None,
    verbose: bool = False,
    check_success_coverage: str = "warn",
    jitter_frac: float = 0.0,
    jitter_sigma: float = 0.0,
    jitter_rng: Optional[np.random.Generator] = None,
    jitter_freeze_qi: Optional[int] = None,
    settle_steps: int = 0,
) -> Dict:
    """Run per-arm primitive RECIPE programs through the wrapped env, labelling live.

    Primitives are built JUST-IN-TIME: when an arm enters a primitive (ti==0), its
    ``QueuedRecipe.recipe(planner, env, arm)`` is called NOW to plan against the
    LIVE env/qpos, then its ticks are consumed. This is the correct-start-state fix
    (a primitive after the first no longer plans from the HOME pose).

    Parameters
    ----------
    env : the WRAPPED env (RecordEpisodeMA). env.step(action_dict) is recorded.
    planner : PandaArmMotionPlanningSolver (for current qpos of idle arms / planning).
    programs : {arm: [QueuedRecipe, ...]}. Each arm runs its recipe queue in order.
    recorder : SubtaskRecorder (already reset). One append per env.step.
    max_steps : hard cap on env.steps (safety).
    boundary_hook : optional callable(env, state) invoked BETWEEN primitives (after
        one completes, BEFORE the next is built; NOT before the first primitive).
        Runs on the UNWRAPPED env, unrecorded. Because the next primitive is built
        AFTER the hook (JIT), the hook MAY perturb arm joints AND object poses: the
        next recipe re-plans (fresh per-arm dry_run screw plan) from the perturbed
        state, re-resolving moved actors' grasp poses. See module docstring
        "JIT RE-PLANNING".
    action_prefix : agent key prefix; action_dict keys are f"{action_prefix}-{i}".
    control_mode : env control mode; defaults to planner.control_mode.
    check_success_coverage : how to handle primitives whose verb is in
        VERBS_REQUIRING_SUCCESS_CHECK but that carry no success_check (the
        liar-label loophole). "warn" (default) emits a UserWarning listing the
        offenders; "error" raises ValueError; "off" skips the lint. NOTE: because
        recipes are built lazily, the lint builds a PROBE primitive per recipe to
        inspect its verb/success_check; the probe is built against the CURRENT state
        (pre-run) and discarded (it does not advance physics — dry_run only plans).
    jitter_frac : DART DENSE jitter probability. Before each RECORDED step, with
        probability ``jitter_frac`` ONE moving arm (one whose primitive produced a
        real tick this step — NOT an idle/blocked arm) is nudged by ONE UNRECORDED
        unwrapped-env step (``dart_perturb.jitter_nudge``) commanding that arm's
        CURRENT CLEAN WAYPOINT (the same qpos7 the recorded step is about to send)
        + N(0, ``jitter_sigma``, 7); all other arms hold their live qpos. The
        recorded step is UNCHANGED (clean waypoint), so the recorded ACTION stays
        clean while the next recorded OBS shows a small bounded drift. Default 0.0
        (with ``jitter_rng=None``) -> NO jitter, byte-identical to the no-jitter path.
    jitter_sigma : stddev of the per-joint jitter nudge (see ``jitter_frac``).
    jitter_rng : a ``numpy.random.Generator`` (seeded per episode/variant by the
        caller, a DISTINCT stream from the boundary_hook's shove RNG and the env
        reset RNG). MUST be supplied for jitter to fire: None -> NO jitter regardless
        of ``jitter_frac``. The same draw sequence is used for both the per-step
        inject decision and the arm choice, so a given (rng) replays identically.

    Returns
    -------
    dict with keys:
        steps : number of env.steps taken (== recorder.length)
        success : {arm: {primitive_index: bool}} per-primitive success results
        all_success : bool, True iff every primitive with a success_check passed
        terminated, truncated, info : last env.step outputs
        completed : bool, True iff all arms emptied their queues before max_steps
        n_jitter_steps : number of UNRECORDED jitter nudges fired (0 if jitter off)
    """
    if control_mode is None:
        control_mode = getattr(planner, "control_mode", "pd_joint_pos")

    if check_success_coverage != "off":
        _lint_success_coverage(programs, planner, env, mode=check_success_coverage)

    num_arms = recorder.num_arms
    arms: Dict[int, _ArmState] = {}
    for i in range(num_arms):
        arms[i] = _ArmState(queue=programs.get(i, []))
        # seed frozen qpos with current live qpos so a leading wait holds the start
        arms[i].frozen_qpos = _live_qpos(planner, i)
        arms[i].frozen_grip = OPEN

    # state passed to boundary_hook / predicates (extensible scratchpad)
    state: Dict = {"step": 0, "arms": arms, "planner": planner}

    # Task name for rendering idle "wait" labels, resolved LAZILY on the first idle
    # tick and cached (so we don't probe-build a recipe unless an arm actually
    # idles, and never more than once). render(WAIT, 0, task) is just "wait" for
    # every known task; this only guards an unknown-task KeyError.
    task_name_cache: Dict[str, Optional[str]] = {"v": None}

    def _task_name() -> str:
        if task_name_cache["v"] is None:
            # prefer a task from an already-built primitive (free, no probe)
            for a in arms.values():
                if a.built is not None:
                    task_name_cache["v"] = a.built.task
                    break
            else:
                task_name_cache["v"] = _resolve_task_name(programs, planner, env)
        return task_name_cache["v"]

    # jitter is active only when an rng is supplied AND a positive frac/sigma are set
    # (defaults frac=0/rng=None -> NO jitter, byte-identical to the no-jitter path).
    jitter_active = (
        jitter_rng is not None and jitter_frac > 0.0 and jitter_sigma > 0.0
    )

    info = {}
    terminated = truncated = False
    step = 0
    n_jitter_steps = 0

    def _all_done() -> bool:
        return all(a.current is None for a in arms.values())

    # Track the last primitive index each arm has *entered* to detect transitions.
    # The boundary_hook fires only BETWEEN primitives (after one completes, before
    # the next is BUILT), per the spec. Init to each arm's starting qi (NOT -1): on
    # the first loop iteration entered[i]==a.qi so no transition is detected and the
    # hook is NOT called before the first primitive. The first primitive is still
    # JIT-built at its entry (arm at reset = the same state the builders used).
    entered: Dict[int, int] = {i: arms[i].qi for i in range(num_arms)}

    while step < max_steps and not _all_done():
        state["step"] = step

        # --- gate check first: an arm with an unmet wait_for is BLOCKED (it holds
        # and does NOT enter/build its next primitive yet). We evaluate the gate
        # ONCE here and reuse the decision in the advance phase. ---
        blocked_this_tick: Dict[int, bool] = {}
        for i in range(num_arms):
            a = arms[i]
            cur = a.current
            blocked = False
            if cur is not None and cur.wait_for is not None and a.ti == 0 and a.built is None:
                _other_arm, predicate = cur.wait_for  # other_arm is documentation only
                if not _eval_gate(predicate, env, state):
                    blocked = True
            blocked_this_tick[i] = blocked

        # --- detect primitive transitions for arms that will ACTUALLY enter a new
        # primitive this tick (ti==0, not yet built, NOT blocked, and a different qi
        # than last entered). Fire boundary_hook ONCE, BETWEEN primitives, BEFORE
        # building so the next recipe's plan absorbs the perturbation. A blocked
        # (gated) arm does NOT count as a transition -> the hook is not re-fired
        # every waiting tick. ---
        transitioning = [
            i for i in range(num_arms)
            if arms[i].current is not None and arms[i].ti == 0 and arms[i].built is None
            and not blocked_this_tick[i] and entered[i] != arms[i].qi
        ]
        if transitioning and boundary_hook is not None:
            # expose the EXACT transitioning-arm set so the hook only perturbs an arm
            # that is genuinely entering a new primitive (never a blocked/gated arm
            # that merely matches ti==0 & built is None while waiting on a gate).
            state["transitioning"] = transitioning
            boundary_hook(_unwrap(env), state)

        # --- per-arm tick assembly ---
        action_dict: Dict[str, np.ndarray] = {}
        labels: Dict[int, Tuple[int, int, str]] = {}
        # arms that produced a REAL primitive tick this step (NOT idle/blocked), with
        # their CLEAN commanded waypoint (the 7 arm joints about to be sent). These
        # are the jitter-eligible "moving" arms; the clean waypoint is the nudge
        # center (waypoint-centered, not qpos-centered -> bounded around the path).
        moving_waypoints: Dict[int, np.ndarray] = {}
        moving_grips: Dict[int, float] = {}

        for i in range(num_arms):
            a = arms[i]
            cur = a.current
            blocked = blocked_this_tick[i]

            if cur is None or blocked:
                # idle/blocked: HOLD frozen last-commanded qpos, labelled wait
                qpos = a.frozen_qpos
                grip = a.frozen_grip
                action_dict[f"{action_prefix}-{i}"] = _action_for(control_mode, qpos, grip)
                labels[i] = (vocab.WAIT, 0, vocab.render(vocab.WAIT, 0, _task_name()))
                continue

            # entering this primitive for the first time: BUILD it now (JIT) from
            # the live env/qpos. The boundary_hook (if any) already fired above, so
            # the recipe re-plans from the perturbed state.
            if a.ti == 0 and a.built is None:
                a.built = cur.recipe(planner, env, i)
                entered[i] = a.qi

            prim = a.built
            qpos7, grip = prim.ticks[a.ti]
            qpos7 = np.asarray(qpos7, dtype=np.float32)
            action_dict[f"{action_prefix}-{i}"] = _action_for(control_mode, qpos7, grip)
            labels[i] = (prim.verb_id, prim.target_id, prim.text)
            # remember as the frozen command (for subsequent holds)
            a.frozen_qpos = qpos7.copy()
            a.frozen_grip = float(grip)
            # this arm is genuinely moving this tick -> jitter-eligible; the clean
            # waypoint it is about to command is the nudge center.
            moving_waypoints[i] = qpos7
            moving_grips[i] = float(grip)

        # --- DART DENSE jitter: BEFORE the recorded step, with prob jitter_frac,
        # nudge ONE moving arm by ONE UNRECORDED unwrapped step centered on that
        # arm's CLEAN waypoint (the same qpos7 about to be commanded) + N(0, sigma).
        # The recorded env.step below is UNCHANGED (still the clean waypoint), so the
        # recorded ACTION stays clean while the next recorded OBS drifts. The arm is
        # chosen from the genuinely-moving arms only (never an idle/blocked arm). ---
        # PHASE-GATE jitter: exclude arms at qi >= jitter_freeze_qi (e.g. an arm in
        # its place/open/retreat for TSC) so the dense nudge never wobbles a fragile
        # tower mid-stack. None -> no freeze (every moving arm eligible, LB unchanged).
        jitter_eligible = (
            moving_waypoints if jitter_freeze_qi is None
            else {i: w for i, w in moving_waypoints.items()
                  if int(arms[i].qi) < jitter_freeze_qi}
        )
        if jitter_active and jitter_eligible and jitter_rng.random() < jitter_frac:
            arm_id = int(jitter_rng.choice(np.array(sorted(jitter_eligible))))
            # carry each arm's FROZEN grip (last commanded) so the nudge never flips a
            # grasp; non-nudged arms hold their live qpos inside jitter_nudge.
            grips = [float(arms[i].frozen_grip) for i in range(num_arms)]
            dart_perturb.jitter_nudge(
                _unwrap(env),
                planner.robot,
                grips,
                arm_id,
                moving_waypoints[arm_id],
                jitter_rng,
                jitter_sigma,
                control_mode=control_mode,
                action_prefix=action_prefix,
            )
            n_jitter_steps += 1

        # --- ONE env.step per tick (wrapped -> recorded) ---
        obs, reward, terminated, truncated, info = env.step(action_dict)
        recorder.append(labels)
        step += 1

        # --- advance tick pointers; on primitive completion, run success check ---
        for i in range(num_arms):
            a = arms[i]
            cur = a.current
            if cur is None:
                continue
            # was this arm blocked this iteration? if so it didn't tick its primitive
            if blocked_this_tick.get(i, False):
                continue
            prim = a.built
            a.ti += 1
            if a.ti >= prim.n_ticks:
                # primitive finished -> evaluate its success_check (if any)
                if prim.success_check is not None:
                    ok = bool(prim.success_check(env, i, prim))
                else:
                    ok = None
                a.success[a.qi] = ok
                if verbose:
                    print(f"[step {step}] arm{i} finished {prim.name} success={ok}")
                a.qi += 1
                a.ti = 0
                a.built = None  # next primitive is re-built JIT at its entry

        if terminated or truncated:
            break

    # --- SETTLE: after the program completes, hold ALL arms still for settle_steps
    # RECORDED steps so the tower settles before the final success check. This (a)
    # lets a genuinely-good stack pass an is_static success gate that the last MOVING
    # frame would blip, and (b) correctly REJECTS a tower that topples once the arms
    # stop holding it. Skipped if the env already terminated (e.g. auto-terminate on
    # success) or truncated. Labelled WAIT (benign hold-still frames). ---
    if settle_steps > 0 and not (terminated or truncated):
        for _ in range(settle_steps):
            if step >= max_steps:
                break
            action_dict = {
                f"{action_prefix}-{i}": _action_for(
                    control_mode, arms[i].frozen_qpos, arms[i].frozen_grip)
                for i in range(num_arms)
            }
            obs, reward, terminated, truncated, info = env.step(action_dict)
            recorder.append({
                i: (vocab.WAIT, 0, vocab.render(vocab.WAIT, 0, _task_name()))
                for i in range(num_arms)
            })
            step += 1
            if terminated or truncated:
                break

    # assemble success report
    success_report: Dict[int, Dict[int, Optional[bool]]] = {
        i: dict(arms[i].success) for i in range(num_arms)
    }
    checked = [v for arm in success_report.values() for v in arm.values() if v is not None]
    all_success = (len(checked) > 0) and all(checked)

    return {
        "steps": step,
        "success": success_report,
        "all_success": all_success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
        "completed": _all_done(),
        "n_jitter_steps": int(n_jitter_steps),
    }


# --------------------------------------------------------------------------- helpers


def _lint_success_coverage(programs: Dict[int, List[QueuedRecipe]], planner, env,
                           mode: str = "warn"):
    """Flag primitives that CLAIM a spatial/grasp outcome but carry no success_check.

    Closes the liar-label loophole: a primitive with verb in
    VERBS_REQUIRING_SUCCESS_CHECK but ``success_check is None`` is silently treated
    as passing in ``all_success`` (it contributes nothing to the ``checked`` list),
    so a failed grasp/approach/lift/place could be written as if it happened.

    Because recipes are built lazily, this builds a throwaway PROBE primitive per
    recipe (against the current pre-run state) to inspect its verb/success_check.
    ``dry_run`` planning inside the probe only plans (does not advance physics). If a
    probe build raises, we skip linting that recipe (it'll surface at run time).

    ``mode``: "warn" -> UserWarning listing offenders; "error" -> ValueError.
    """
    offenders: List[str] = []
    for arm, queue in programs.items():
        for qi, qr in enumerate(queue):
            try:
                prim = qr.recipe(planner, env, arm)
            except Exception:
                # cannot probe this recipe pre-run (e.g. needs prior arm state);
                # the run itself will surface any real failure. Skip lint for it.
                continue
            if (prim.verb_id in VERBS_REQUIRING_SUCCESS_CHECK
                    and prim.success_check is None):
                offenders.append(f"arm{arm}[{qi}] {prim.name} (verb={prim.verb_id})")
    if not offenders:
        return
    msg = (
        "run_program: these primitives claim a spatial/grasp outcome but have NO "
        "success_check, so a failed primitive would be mislabelled (liar-label "
        f"loophole): {offenders}. Attach a success_check (approach->check_tcp_near, "
        "lift->check_*_lifted, place->placed) or pass check_success_coverage='off' "
        "if this is intentional. (close_gripper is exempt: is_grasping is premature "
        "right after the close ramp; a non-grasp is caught downstream by the lift.)"
    )
    if mode == "error":
        raise ValueError(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)


def _resolve_task_name(programs: Dict[int, List[QueuedRecipe]], planner, env) -> str:
    """Resolve the program's task name once (for rendering idle "wait" labels).

    Probe-builds the first recipe of the first non-empty arm to read its ``.task``;
    if that fails (or there are no recipes), default to "LiftBarrier". This is a
    throwaway probe (dry_run only plans, doesn't advance physics) and is called once
    per run, not per tick. ``render(WAIT, 0, task)`` is literally "wait" for every
    known task, so the only thing this protects against is an unknown-task KeyError.
    """
    for arm, prog in sorted(programs.items()):
        if prog:
            try:
                return prog[0].recipe(planner, env, arm).task
            except Exception:
                break
    return "LiftBarrier"


def _live_qpos(planner, arm: int) -> np.ndarray:
    q = planner.robot[arm].get_qpos()
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    q = np.asarray(q)
    if q.ndim == 2:
        q = q[0]
    return q[:7].astype(np.float32)


def _eval_gate(predicate: Callable, env, state) -> bool:
    """Evaluate a wait_for predicate. Accepts predicate(env, state)."""
    return bool(predicate(env, state))
