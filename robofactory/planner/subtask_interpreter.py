"""Subtask interpreter: owns the env step loop, runs per-arm primitive queues,
labels every action tick, and records the per-arm subtask stream live.

Design (see plan ``1-this-is-not-witty-quokka.md`` "Architecture: invert control"):

* Each arm has a QUEUE of :class:`~.subtask_primitives.Primitive`. One outer loop
  advances every arm by one tick, assembles a single ``action_dict`` keyed
  ``f"{action_prefix}-{i}"`` (format per follow_path:118-138), and does ONE
  ``env.step(action_dict)`` per tick (wrapped env -> RecordEpisodeMA records it).
* **Idle / blocked arm HOLDS the FROZEN last commanded qpos** (NOT live qpos) and
  is labelled ``wait``. An arm is blocked when its next primitive has an unmet
  ``wait_for=(other_arm, predicate)`` gate.
* **Labels align to the ACTION index** (length T; obs is T+1). The label for
  action[t] is the subtask that DROVE action[t] (matches the eval's "subtask
  active for THIS step" convention). Guaranteed by construction: we append the
  label at the same loop iteration we build and step the action.
* Between primitives the interpreter calls an optional
  ``boundary_hook(env, state)`` where the user's DART arm-noise / object-pose
  perturbation runs on the UNWRAPPED env (unrecorded). The NEXT primitive re-plans
  from the current state, so labels remain correct.
* Per-primitive **success checks** (run after the primitive's last tick) drop
  trajectories whose labelled subtask did not actually happen (the liar-label
  guard / the crux).

This module imports numpy at top; it does NOT import sapien. It only *uses* the
passed-in wrapped ``env`` / ``planner`` at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import subtask_vocab as vocab
from .subtask_primitives import Primitive, OPEN

# A per-primitive gate: advance this primitive only once predicate(env, state)
# is True for the named other arm's progress. predicate signature:
#   predicate(env, state) -> bool
WaitFor = Tuple[int, Callable]


@dataclass
class QueuedPrimitive:
    """A primitive plus its optional cross-arm gate.

    ``wait_for`` is (other_arm, predicate); the primitive does not start ticking
    until predicate(env, state) is True. ``predicate`` is typically a success
    check on the other arm (e.g. "arm A is grasping"), enabling sequential or
    hold-while-other-works choreography.
    """

    primitive: Primitive
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


def check_tcp_near(target_pose_fn: Callable, tol: float = 0.05) -> Callable:
    """approach success: arm tcp within ``tol`` (m) of target_pose_fn(env, arm).

    target_pose_fn returns a length-3 xyz (or 7-vec; first 3 used).
    """
    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        tcp_p = u.agent.agents[arm].tcp.pose.p
        tcp_p = tcp_p.detach().cpu().numpy() if hasattr(tcp_p, "detach") else np.asarray(tcp_p)
        tcp_p = np.asarray(tcp_p).reshape(-1)[:3]
        tgt = np.asarray(target_pose_fn(env, arm)).reshape(-1)[:3]
        return float(np.linalg.norm(tcp_p - tgt)) <= tol
    return _check


def check_barrier_lifted(dz: float = 0.15) -> Callable:
    """lift success (LiftBarrier): barrier z > arm0 base z + dz (matches env.evaluate)."""
    def _check(env, arm: int, primitive: Primitive) -> bool:
        u = _unwrap(env)
        bz = _scalar(u.barrier.pose.p[..., 2])
        base_z = _scalar(u.agent.agents[0].robot.pose.p[0, 2])
        return bz > base_z + dz
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

    queue: List[QueuedPrimitive]
    qi: int = 0                       # index of the current primitive in queue
    ti: int = 0                       # tick index within the current primitive
    frozen_qpos: Optional[np.ndarray] = None  # last commanded qpos (for holds)
    frozen_grip: float = OPEN         # last commanded grip
    done: bool = False
    # success flags per primitive index (None=not yet checked)
    success: Dict[int, Optional[bool]] = field(default_factory=dict)

    @property
    def current(self) -> Optional[QueuedPrimitive]:
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
    programs: Dict[int, List[QueuedPrimitive]],
    recorder: SubtaskRecorder,
    max_steps: int,
    boundary_hook: Optional[Callable] = None,
    action_prefix: str = "panda",
    control_mode: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """Run per-arm primitive programs through the wrapped env, labelling live.

    Parameters
    ----------
    env : the WRAPPED env (RecordEpisodeMA). env.step(action_dict) is recorded.
    planner : PandaArmMotionPlanningSolver (for current qpos of idle arms).
    programs : {arm: [QueuedPrimitive, ...]}. Each arm runs its queue in order.
    recorder : SubtaskRecorder (already reset). One append per env.step.
    max_steps : hard cap on env.steps (safety).
    boundary_hook : optional callable(env, state) invoked BETWEEN primitives (when
        any arm advances to a new primitive). Runs on the UNWRAPPED env, unrecorded.
        The next primitive re-plans from the current state.
    action_prefix : agent key prefix; action_dict keys are f"{action_prefix}-{i}".
    control_mode : env control mode; defaults to planner.control_mode.

    Returns
    -------
    dict with keys:
        steps : number of env.steps taken (== recorder.length)
        success : {arm: {primitive_index: bool}} per-primitive success results
        all_success : bool, True iff every primitive with a success_check passed
        terminated, truncated, info : last env.step outputs
        completed : bool, True iff all arms emptied their queues before max_steps
    """
    if control_mode is None:
        control_mode = getattr(planner, "control_mode", "pd_joint_pos")

    num_arms = recorder.num_arms
    arms: Dict[int, _ArmState] = {}
    for i in range(num_arms):
        arms[i] = _ArmState(queue=programs.get(i, []))
        # seed frozen qpos with current live qpos so a leading wait holds the start
        arms[i].frozen_qpos = _live_qpos(planner, i)
        arms[i].frozen_grip = OPEN

    # state passed to boundary_hook / predicates (extensible scratchpad)
    state: Dict = {"step": 0, "arms": arms, "planner": planner}

    info = {}
    terminated = truncated = False
    step = 0

    def _all_done() -> bool:
        return all(a.current is None for a in arms.values())

    # If any arm starts a brand-new primitive this iteration, we first call the
    # boundary hook (between-primitives perturbation), THEN plan/tick. We track the
    # last primitive index each arm has *entered* to detect transitions.
    entered: Dict[int, int] = {i: -1 for i in range(num_arms)}

    while step < max_steps and not _all_done():
        state["step"] = step

        # --- detect primitive transitions; call boundary_hook BETWEEN primitives ---
        transition = False
        for i in range(num_arms):
            a = arms[i]
            cur = a.current
            if cur is not None and a.ti == 0 and entered[i] != a.qi:
                transition = True
        if transition and boundary_hook is not None:
            boundary_hook(_unwrap(env), state)

        # --- gate check + per-arm tick assembly ---
        action_dict: Dict[str, np.ndarray] = {}
        labels: Dict[int, Tuple[int, int, str]] = {}
        # cache the per-arm blocked decision so the advance phase reuses it
        # (don't re-eval the gate predicate twice per tick — matters for any
        # predicate with side effects, and avoids redundant sim reads).
        blocked_this_tick: Dict[int, bool] = {}

        for i in range(num_arms):
            a = arms[i]
            cur = a.current

            blocked = False
            if cur is not None and cur.wait_for is not None and a.ti == 0:
                _other_arm, predicate = cur.wait_for  # other_arm is documentation only
                if not _eval_gate(predicate, env, state):
                    blocked = True
            blocked_this_tick[i] = blocked

            if cur is None or blocked:
                # idle/blocked: HOLD frozen last-commanded qpos, labelled wait
                qpos = a.frozen_qpos
                grip = a.frozen_grip
                action_dict[f"{action_prefix}-{i}"] = _action_for(control_mode, qpos, grip)
                labels[i] = (vocab.WAIT, 0, vocab.render(vocab.WAIT, 0, _task_of(cur, programs, i)))
                continue

            # entering this primitive for the first time
            if a.ti == 0:
                entered[i] = a.qi

            prim = cur.primitive
            qpos7, grip = prim.ticks[a.ti]
            qpos7 = np.asarray(qpos7, dtype=np.float32)
            action_dict[f"{action_prefix}-{i}"] = _action_for(control_mode, qpos7, grip)
            labels[i] = (prim.verb_id, prim.target_id, prim.text)
            # remember as the frozen command (for subsequent holds)
            a.frozen_qpos = qpos7.copy()
            a.frozen_grip = float(grip)

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
            a.ti += 1
            if a.ti >= cur.primitive.n_ticks:
                # primitive finished -> evaluate its success_check (if any)
                if cur.primitive.success_check is not None:
                    ok = bool(cur.primitive.success_check(env, i, cur.primitive))
                else:
                    ok = None
                a.success[a.qi] = ok
                if verbose:
                    print(f"[step {step}] arm{i} finished {cur.primitive.name} success={ok}")
                a.qi += 1
                a.ti = 0

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
    }


# --------------------------------------------------------------------------- helpers


def _task_of(cur, programs, arm) -> str:
    """Task name for label rendering of an idle arm (use any program's task)."""
    if cur is not None:
        return cur.primitive.task
    for q in programs.get(arm, []):
        return q.primitive.task
    # fall back to any arm's program
    for prog in programs.values():
        if prog:
            return prog[0].primitive.task
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
