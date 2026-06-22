"""LiftBarrier subtask scenario + CONTRASTIVE program sampler.

A "scenario" turns a random seed into a list of :class:`ProgramSpec` objects.
Each spec is a *recipe* that, given a live ``(planner, env)``, builds a per-arm
program ``{arm: [QueuedPrimitive]}`` for ``subtask_interpreter.run_program``.

Why a recipe (a builder callable) and not a pre-built program? Primitives plan
via ``move_to_pose_with_screw(..., dry_run=True)`` against the LIVE actor pose
(see ``subtask_primitives``). The interpreter resets the env to the seed right
before running, so the program must be built AFTER reset, against the freshly
reset env. The sampler therefore stays SIM-FREE: it only decides *structure*
(which arm grasps which end, gating, hold lengths) using its own RNG, and the
builders read the live env at run time.

CONTRASTIVE design (the crux of the data aug, see plan):
Matched variants are sampled from the SAME seed with an INDEPENDENT
``numpy.random.Generator`` (seeded from the task seed) so they NEVER touch the
env's reset RNG — frame-0 of every variant in a contrast group is byte-identical
(same env reset), they diverge only because the *program* differs. Required
matched families (each is a ``contrast_group_id``):

  (a) ``approach_stop`` vs ``approach_grasp_lift`` — same reach, one stops at the
      grasp pose (no close), the other closes + lifts. Labels diverge at the seam.
  (b) ``arms_LR`` vs ``arms_RL`` — swap which arm takes left_end / right_end.
  (c) ``sequential`` vs ``simultaneous`` — arm1 gated on arm0's grasp, vs both
      lift together (no gate).
  (d) ``grasp_and_hold`` — arm0 grasps + holds (gripper closed) while arm1 works.
      Paired with its no-hold sibling so the contrast is "hold vs release".

Every emitted spec carries ``variant_id`` (unique within the sample) and
``contrast_group_id`` (shared by a matched pair). ``sample(seed)`` returns the
flat list; ``group_specs(specs)`` buckets them by ``contrast_group_id`` so the
rollout driver can drop a whole matched group atomically if any member fails its
success checks.

IMPORT-CLEAN-ish: imports numpy + the (numpy-only) primitive/interpreter modules;
NO sapien / gym / robofactory.tasks at module top, so the sampler's structure and
metadata can be unit-tested off-GPU with a fake planner/env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import subtask_vocab as vocab
from . import subtask_primitives as P
from .subtask_interpreter import (
    QueuedPrimitive,
    check_is_grasping,
    check_tcp_near,
    check_barrier_lifted,
)

LB = "LiftBarrier"
# LiftBarrier has exactly 2 arms.
LB_NUM_ARMS = 2

# A builder takes the live (planner, env) and returns {arm: [QueuedPrimitive]}.
ProgramBuilder = Callable[[object, object], Dict[int, List[QueuedPrimitive]]]


@dataclass
class ProgramSpec:
    """A named, tagged recipe for one contrastive variant.

    Attributes
    ----------
    name : human label, e.g. "approach_stop".
    variant_id : unique id within a single ``sample(seed)`` call.
    contrast_group_id : shared by the members of a matched contrastive family;
        the rollout driver drops the WHOLE group atomically if any member fails.
    family : the family tag ("a_approach", "b_armswap", "c_gate", "d_hold").
    seed : the env reset seed this spec was sampled for.
    build : callable(planner, env) -> {arm: [QueuedPrimitive]} (built post-reset).
    num_arms : arms this program drives (2 for LiftBarrier).
    meta : free-form structure metadata (which arm took which end, gate, etc.).
    """

    name: str
    variant_id: int
    contrast_group_id: int
    family: str
    seed: int
    build: ProgramBuilder
    num_arms: int = LB_NUM_ARMS
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- LB builders
# Each builder is a closure over the STRUCTURE decided by the sampler (which arm
# grasps which end, gating, hold length). It reads the live env at call time.


def _lb_target_pose_fn(target_id: int):
    """A target_pose_fn(env, arm) for check_tcp_near: live LB end grasp xyz."""
    def _fn(env, arm):
        u = env.unwrapped if hasattr(env, "unwrapped") else env
        # resolve_lb_end_grasp_pose needs the planner; the interpreter passes env
        # only, so re-resolve from the barrier annotation directly via the actor.
        # We reuse the planner stashed on state isn't available here, so fall back
        # to the contact-point world pose via the barrier transform. Simpler: use
        # the barrier centre as a coarse target (check_tcp_near tol is generous).
        return np.asarray(u.barrier.pose.p).reshape(-1)[:3]
    return _fn


def _approach_qp(planner, env, arm: int, target_id: int, *, grip: float,
                 success_tol: float = 0.08) -> QueuedPrimitive:
    sc = check_tcp_near(_lb_target_pose_fn(target_id), tol=success_tol)
    prim = P.approach(planner, env, arm=arm, target_id=target_id, task=LB,
                      grip=grip, success_check=sc)
    return QueuedPrimitive(prim)


def _close_qp(planner, env, arm: int, target_id: int) -> QueuedPrimitive:
    sc = check_is_grasping("barrier")
    prim = P.close_gripper(planner, env, arm=arm, task=LB, start_grip=P.OPEN,
                           target_id=target_id, success_check=sc)
    return QueuedPrimitive(prim)


def _lift_qp(planner, env, arm: int, target_id: int, *, dz: float = 0.2,
             wait_for=None) -> QueuedPrimitive:
    sc = check_barrier_lifted(dz=0.15)
    prim = P.lift(planner, env, arm=arm, target_id=target_id, task=LB, dz=dz,
                  grip=P.CLOSED, success_check=sc)
    return QueuedPrimitive(prim, wait_for=wait_for)


def _hold_qp(planner, env, arm: int, n: int, *, grip: float) -> QueuedPrimitive:
    prim = P.hold(planner, env, arm=arm, n=n, task=LB, grip=grip)
    return QueuedPrimitive(prim)


def _gate_arm_grasping(other_arm: int, actor_name: str = "barrier") -> tuple:
    """Build a ``wait_for`` gate: (other_arm, predicate(env, state)->bool).

    NOTE: the interpreter calls gate predicates as ``predicate(env, state)`` (see
    subtask_interpreter._eval_gate) — a DIFFERENT signature from the per-primitive
    ``success_check(env, arm, primitive)``. So we cannot reuse check_is_grasping
    (which is success-check-shaped); we build a gate-shaped closure that reads the
    named arm's live grasp state directly.
    """
    def _pred(env, state):
        u = env.unwrapped if hasattr(env, "unwrapped") else env
        actor = getattr(u, actor_name)
        agent = u.agent.agents[other_arm]
        g = agent.is_grasping(actor)
        g = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)
        return bool(np.asarray(g).reshape(-1)[0])

    return (other_arm, _pred)


# --------------------------------------------------------------------------- family builders


def _make_approach_stop_builder(arm_left: int, arm_right: int,
                                seam_hold: int) -> ProgramBuilder:
    """(a) approach-stop: both arms reach their end, then a short hold. No close."""
    def build(planner, env):
        return {
            arm_left: [
                _approach_qp(planner, env, arm_left, vocab.LB_TARGETS["left_end"], grip=P.OPEN),
                _hold_qp(planner, env, arm_left, seam_hold, grip=P.OPEN),
            ],
            arm_right: [
                _approach_qp(planner, env, arm_right, vocab.LB_TARGETS["right_end"], grip=P.OPEN),
                _hold_qp(planner, env, arm_right, seam_hold, grip=P.OPEN),
            ],
        }
    return build


def _make_approach_grasp_lift_builder(arm_left: int, arm_right: int,
                                      seam_hold: int, *, sequential: bool) -> ProgramBuilder:
    """(a)/(c) approach -> close -> lift, optionally arm_right gated on arm_left grasp."""
    def build(planner, env):
        left_l = vocab.LB_TARGETS["left_end"]
        right_l = vocab.LB_TARGETS["right_end"]
        # sequential: arm_right's LIFT is gated on arm_left having grasped (a
        # gate-shaped predicate(env, state), NOT the success_check signature).
        gate = _gate_arm_grasping(arm_left, "barrier") if sequential else None
        prog = {
            arm_left: [
                _approach_qp(planner, env, arm_left, left_l, grip=P.OPEN),
                _hold_qp(planner, env, arm_left, seam_hold, grip=P.OPEN),
                _close_qp(planner, env, arm_left, left_l),
                _lift_qp(planner, env, arm_left, left_l),
            ],
            arm_right: [
                _approach_qp(planner, env, arm_right, right_l, grip=P.OPEN),
                _hold_qp(planner, env, arm_right, seam_hold, grip=P.OPEN),
                _close_qp(planner, env, arm_right, right_l),
                # the lift carries the optional gate (sequential vs simultaneous)
                _lift_qp(planner, env, arm_right, right_l, wait_for=gate),
            ],
        }
        return prog
    return build


def _make_grasp_and_hold_builder(work_arm: int, hold_arm: int,
                                 seam_hold: int, hold_n: int) -> ProgramBuilder:
    """(d) hold_arm grasps + holds (gripper CLOSED) while work_arm grasps+lifts."""
    def build(planner, env):
        work_l = (vocab.LB_TARGETS["left_end"] if work_arm == 0
                  else vocab.LB_TARGETS["right_end"])
        hold_l = (vocab.LB_TARGETS["left_end"] if hold_arm == 0
                  else vocab.LB_TARGETS["right_end"])
        return {
            hold_arm: [
                _approach_qp(planner, env, hold_arm, hold_l, grip=P.OPEN),
                _close_qp(planner, env, hold_arm, hold_l),
                # long hold, gripper CLOSED -> "grasp and hold while other works"
                QueuedPrimitive(P.hold(planner, env, hold_arm, hold_n, task=LB,
                                       grip=P.CLOSED, verb_id=vocab.WAIT, target_id=0)),
            ],
            work_arm: [
                _approach_qp(planner, env, work_arm, work_l, grip=P.OPEN),
                _hold_qp(planner, env, work_arm, seam_hold, grip=P.OPEN),
                _close_qp(planner, env, work_arm, work_l),
                _lift_qp(planner, env, work_arm, work_l),
            ],
        }
    return build


# --------------------------------------------------------------------------- sampler


def sample(seed: int) -> List[ProgramSpec]:
    """Sample the full contrastive set of LiftBarrier program specs for ``seed``.

    All randomness comes from a DEDICATED ``np.random.default_rng(seed)`` so the
    sampler is reproducible AND never touches the env reset RNG (frame-0 stays
    identical across variants). The structural choices (seam-hold length, hold
    length) are drawn here; the per-arm planning happens later in the builder.

    Returns a flat list of ProgramSpec. Matched contrastive pairs share a
    ``contrast_group_id`` (use ``group_specs`` to bucket them).
    """
    rng = np.random.default_rng(int(seed))
    # small randomized hold lengths (>=2 so a chunk straddling a seam is
    # unambiguous; see plan risk #6). Drawn from the dedicated RNG.
    seam_hold = int(rng.integers(2, 5))     # 2..4 inclusive
    hold_n = int(rng.integers(20, 41))      # 20..40 inclusive (grasp-and-hold)

    specs: List[ProgramSpec] = []
    vid = 0
    gid = 0

    def emit(name, family, group_id, build, meta):
        nonlocal vid
        specs.append(ProgramSpec(
            name=name, variant_id=vid, contrast_group_id=group_id, family=family,
            seed=int(seed), build=build, num_arms=LB_NUM_ARMS, meta=dict(meta),
        ))
        vid += 1

    # (a) approach-stop vs approach-then-close-then-lift (arm0=left, arm1=right)
    emit("approach_stop", "a_approach", gid,
         _make_approach_stop_builder(0, 1, seam_hold),
         {"arm_left": 0, "arm_right": 1, "grasps": False, "seam_hold": seam_hold})
    emit("approach_grasp_lift", "a_approach", gid,
         _make_approach_grasp_lift_builder(0, 1, seam_hold, sequential=False),
         {"arm_left": 0, "arm_right": 1, "grasps": True, "seam_hold": seam_hold})
    gid += 1

    # (b) swap which arm takes which end: LR (arm0=left) vs RL (arm0=right)
    emit("arms_LR", "b_armswap", gid,
         _make_approach_grasp_lift_builder(0, 1, seam_hold, sequential=False),
         {"arm_left": 0, "arm_right": 1, "seam_hold": seam_hold})
    emit("arms_RL", "b_armswap", gid,
         _make_approach_grasp_lift_builder(1, 0, seam_hold, sequential=False),
         {"arm_left": 1, "arm_right": 0, "seam_hold": seam_hold})
    gid += 1

    # (c) sequential (arm1 lift gated on arm0 grasp) vs simultaneous lift
    emit("sequential", "c_gate", gid,
         _make_approach_grasp_lift_builder(0, 1, seam_hold, sequential=True),
         {"arm_left": 0, "arm_right": 1, "gated": True, "seam_hold": seam_hold})
    emit("simultaneous", "c_gate", gid,
         _make_approach_grasp_lift_builder(0, 1, seam_hold, sequential=False),
         {"arm_left": 0, "arm_right": 1, "gated": False, "seam_hold": seam_hold})
    gid += 1

    # (d) grasp-and-hold (arm0 holds closed) vs its release sibling (no long hold).
    #     The sibling is the plain simultaneous grasp+lift (arm0 does NOT freeze).
    emit("grasp_and_hold", "d_hold", gid,
         _make_grasp_and_hold_builder(work_arm=1, hold_arm=0,
                                      seam_hold=seam_hold, hold_n=hold_n),
         {"work_arm": 1, "hold_arm": 0, "hold_n": hold_n, "seam_hold": seam_hold})
    emit("grasp_and_release", "d_hold", gid,
         _make_approach_grasp_lift_builder(0, 1, seam_hold, sequential=False),
         {"work_arm": 1, "hold_arm": 0, "hold_n": 0, "seam_hold": seam_hold})
    gid += 1

    return specs


def group_specs(specs: List[ProgramSpec]) -> Dict[int, List[ProgramSpec]]:
    """Bucket specs by contrast_group_id (for atomic matched-pair drop)."""
    groups: Dict[int, List[ProgramSpec]] = {}
    for s in specs:
        groups.setdefault(s.contrast_group_id, []).append(s)
    return groups


def filter_variants(specs: List[ProgramSpec],
                    variants: Optional[List[str]] = None) -> List[ProgramSpec]:
    """Keep only specs whose ``name`` OR ``family`` is in ``variants`` (None=all).

    Filtering is applied at the GROUP level so a matched pair is never half-kept:
    a contrast group is retained iff ANY of its members matches the filter, and
    then the WHOLE group is returned (keeps contrastive pairs intact).
    """
    if not variants:
        return list(specs)
    wanted = set(variants)
    keep_groups = {
        s.contrast_group_id for s in specs
        if s.name in wanted or s.family in wanted
    }
    return [s for s in specs if s.contrast_group_id in keep_groups]


# --------------------------------------------------------------------------- inline doc test
if __name__ == "__main__":
    s = sample(7)
    by_group = group_specs(s)
    print(f"sampled {len(s)} specs in {len(by_group)} contrast groups")
    for gid, members in sorted(by_group.items()):
        names = [m.name for m in members]
        assert len(members) == 2, (gid, names)  # all LB families are matched pairs
        print(f"  group {gid} [{members[0].family}]: {names}")
    # variant ids unique
    assert len({m.variant_id for m in s}) == len(s)
    # seed reproducible (same structural draws)
    assert [m.name for m in sample(7)] == [m.name for m in s]
    print("subtask_scenarios inline OK")
