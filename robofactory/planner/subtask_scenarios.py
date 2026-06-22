"""LiftBarrier subtask scenario + CONTRASTIVE program sampler.

A "scenario" turns a random seed into a list of :class:`ProgramSpec` objects.
Each spec is a *builder* that, given a live ``(planner, env)``, returns a per-arm
program ``{arm: [QueuedRecipe]}`` for ``subtask_interpreter.run_program``.

TWO levels of laziness:
  * ``spec.build(planner, env)`` returns the program STRUCTURE (per-arm lists of
    :class:`~.subtask_interpreter.QueuedRecipe`). It does NOT plan — each
    ``QueuedRecipe`` wraps a :data:`PrimitiveRecipe` callable.
  * The interpreter calls ``recipe(planner, env, arm)`` JUST-IN-TIME, when the arm
    ENTERS that primitive, so each primitive plans (per-arm ``dry_run`` screw plan)
    against the LIVE pose the arm is actually in — not the HOME pose at reset.

The sampler stays SIM-FREE: it only decides *structure* (gating, hold lengths)
using its own RNG; all planning happens later inside the recipes at run time.

EVERY VARIANT IS A COMPLETE TASK ROLLOUT (the barrier actually lifts)
====================================================================
LiftBarrier is ONE rigid bar; env success = barrier CENTRE z > 0.15m, which needs
BOTH ends to rise TOGETHER. A permanent one-arm hold (holding one end down while
the other lifts) tilts the bar and never succeeds — that was the old bug. So in
EVERY variant BOTH arms ULTIMATELY LIFT. Holds/waits are TRANSIENT (an arm waits
for its partner via a gate, then both lift). There is NO permanent-hold and NO
"approach-stop" (truncated) variant anymore: the contrast between matched variants
is COORDINATION / SEQUENCING (who waits, who leads), NOT truncation.

CONTRASTIVE design (the crux of the data aug):
Matched variants are sampled from the SAME seed with an INDEPENDENT
``numpy.random.Generator`` (seeded from the task seed) so they NEVER touch the
env's reset RNG — frame-0 of every variant in a contrast group is byte-identical
(same env reset); they diverge only because the *program* (its gating) differs.

NEW FAMILY SET (all complete rollouts; the baseline ``simultaneous`` is the shared
reference, re-emitted in each matched pair so each contrast is a clean A/B):

  * ``simultaneous`` (baseline): arm0 [approach(left)->close->lift(left)],
    arm1 [approach(right)->close->lift(right)], NO gates. Both lift together.
  * ``stagger_a_leads`` (vs simultaneous): arm0 LEADS ungated
    [approach(left)->close->lift(left, wait arm1 grasped)]; arm1 WAITS
    [approach(right, wait arm0 grasped)->close->lift(right, wait arm0 grasped)].
    arm1 is ``wait`` (gripper open) at frame 0 until arm0 grasps; arm0 holds its
    grasp (gripper closed, ``wait``) on its gated lift until arm1 grasps; then both
    lift together. COMPLETES. Counterfactual vs simultaneous: at frame 0 arm1 is
    ``wait`` here vs ``approach`` there.
  * ``stagger_b_leads`` (vs simultaneous): mirror — arm1 leads, arm0 waits.

DROPPED: ``sequential_lift``. For a rigid barrier BOTH ends must rise together or
the bar tilts and never reaches centre-z success, so a lift-timing-only contrast is
physically a NULL contrast: arm0 lifting alone immediately tilts the bar, so the
sampler can't even encode "arm0 lifts first". Frame-0 was identical to simultaneous
(both approaches ungated), so it added no usable coordination counterfactual.

GATE MECHANISM (deterministic program-progress, NOT physics):
The follower's wait gate is a DETERMINISTIC predicate on the LEAD arm's program
progress (its ``_ArmState.qi``), NOT a physics ``is_grasping`` probe. The lead
program is ``[approach(qi=0), close(qi=1), lift(qi=2)]`` and ``qi`` increments AFTER
a primitive completes, so "lead has finished grasping" == ``lead.qi >= 2`` (it has
finished approach+close and is now on its lift). ``GRASP_IDX = 2``. This makes the
follower ALWAYS ``wait`` at frame 0 (lead starts at qi=0 < 2), unlike the old
physics gate which opened early/inconsistently (the bar grasp seats late, so
``is_grasping`` flickered True before the close ramp finished and the follower
often did NOT actually wait).

DEADLOCK GUARD: exactly ONE arm has an UNGATED first primitive in every variant
(never gate arm0's first on arm1 AND arm1's first on arm0). ``gate_graph`` /
``assert_no_deadlock`` make this checkable in a pure unit test, and every variant
ends with BOTH arms holding a lift primitive (no permanent-hold variant).

Every emitted spec carries ``variant_id`` (unique within the sample) and
``contrast_group_id`` (shared by a matched pair). ``sample(seed)`` returns the flat
list; ``group_specs(specs)`` buckets them by ``contrast_group_id`` so the rollout
driver can drop a whole matched group atomically if any member fails its checks.

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
    QueuedRecipe,
    check_tcp_near,
    check_barrier_lifted,
)

LB = "LiftBarrier"
# LiftBarrier has exactly 2 arms.
LB_NUM_ARMS = 2

# A builder takes the live (planner, env) and returns {arm: [QueuedRecipe]}.
ProgramBuilder = Callable[[object, object], Dict[int, List[QueuedRecipe]]]


@dataclass
class ProgramSpec:
    """A named, tagged recipe for one contrastive variant.

    Attributes
    ----------
    name : human label, e.g. "stagger_a_leads".
    variant_id : unique id within a single ``sample(seed)`` call.
    contrast_group_id : shared by the members of a matched contrastive pair;
        the rollout driver drops the WHOLE pair atomically if any member fails.
    family : the family tag ("stagger_a", "stagger_b").
    seed : the env reset seed this spec was sampled for.
    build : callable(planner, env) -> {arm: [QueuedRecipe]} (the program STRUCTURE;
        the per-primitive planning is deferred to each recipe, called JIT by the
        interpreter at primitive entry).
    num_arms : arms this program drives (2 for LiftBarrier).
    meta : free-form structure metadata (which arm leads, gating, etc.).
    """

    name: str
    variant_id: int
    contrast_group_id: int
    family: str
    seed: int
    build: ProgramBuilder
    num_arms: int = LB_NUM_ARMS
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- LB recipes
# Each ``_*_qp`` returns a QueuedRecipe wrapping a PrimitiveRecipe closure
# ``recipe(planner, env, arm) -> Primitive``. The closure captures the STRUCTURE
# decided by the sampler (target end, grip, gate); the interpreter calls it JIT at
# primitive entry so it plans (per-arm dry_run screw plan) from the LIVE pose.


def _approach_qp(target_id: int, *, grip: float = P.OPEN,
                 success_tol: float = 0.08, wait_for=None) -> QueuedRecipe:
    """approach recipe: plan a reach to the target end's grasp pose (no close).

    success_check = check_tcp_near() with NO fn -> reads the primitive's stored
    ``target_pose`` (the resolved grasp xyz). TCP converges to that planned pose, so
    "TCP within tol of target_pose" is the correct check.
    """
    def recipe(planner, env, arm: int) -> P.Primitive:
        sc = check_tcp_near(tol=success_tol)
        return P.approach(planner, env, arm=arm, target_id=target_id, task=LB,
                          grip=grip, success_check=sc)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _close_qp(target_id: int, *, wait_for=None) -> QueuedRecipe:
    # NO success_check on close: is_grasping does NOT register immediately after the
    # close ramp (the grasp seats during the subsequent lift). A close that did not
    # actually grasp is caught DOWNSTREAM: no grasp -> the lift never raises the
    # barrier -> the env never reaches success -> the group is dropped.
    def recipe(planner, env, arm: int) -> P.Primitive:
        return P.close_gripper(planner, env, arm=arm, task=LB, start_grip=P.OPEN,
                               target_id=target_id, success_check=None)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _lift_qp(target_id: int, *, dz: float = 0.2, wait_for=None) -> QueuedRecipe:
    def recipe(planner, env, arm: int) -> P.Primitive:
        sc = check_barrier_lifted(dz=0.15)
        return P.lift(planner, env, arm=arm, target_id=target_id, task=LB, dz=dz,
                      grip=P.CLOSED, success_check=sc)
    return QueuedRecipe(recipe, wait_for=wait_for)


# Index into the lead program at which the lead arm has FINISHED grasping.
# Lead program is [approach (qi=0), close (qi=1), lift (qi=2)]; ``_ArmState.qi``
# increments AFTER a primitive completes (see subtask_interpreter.run_program: on
# primitive completion ``a.qi += 1``). So the arm reaches ``qi >= 2`` exactly once
# it has finished BOTH approach and close — i.e. it has grasped and is now on its
# lift. There is NO seam-hold primitive between approach and close (the builders
# emit approach->close->lift directly), so the index is the raw program position 2.
GRASP_IDX = 2


def _gate_other_reached(other_arm: int, min_qi: int) -> tuple:
    """Build a DETERMINISTIC ``wait_for`` gate: (other_arm, predicate(env, state)).

    The gate opens once the OTHER arm's PROGRAM PROGRESS has reached ``min_qi`` —
    i.e. its ``_ArmState.qi`` (index of its CURRENT primitive) is >= ``min_qi``.
    This is read from ``state["arms"][other_arm].qi`` (the interpreter passes its
    live per-arm states in ``state``; see run_program's ``state = {..., "arms":
    arms}``), NOT from any physics probe. ``qi`` increments AFTER a primitive
    completes, so ``qi >= GRASP_IDX (=2)`` means the other arm has FINISHED its
    approach+close and is now on its lift — it has grasped.

    DETERMINISTIC vs the old physics gate: the lead starts at ``qi=0`` so the
    follower ALWAYS waits at frame 0 (gate closed), and the gate opens at the exact
    program step the lead's close completes — never early/flaky like
    ``is_grasping`` (which seats late and flickered, so the follower often did not
    actually wait).

    The interpreter calls gate predicates as ``predicate(env, state)`` (see
    subtask_interpreter._eval_gate) — a DIFFERENT signature from the per-primitive
    ``success_check(env, arm, primitive)``.
    """
    def _pred(env, state):
        st = state.get("arms", {}) if hasattr(state, "get") else {}
        a = st.get(other_arm)
        return bool(a is not None and a.qi >= min_qi)

    return (other_arm, _pred)


# --------------------------------------------------------------------------- family builders
# Arm0 always takes left_end, arm1 always takes right_end (no arm-swap: cross-reach
# is infeasible on LiftBarrier). Variants differ ONLY in which lifts/approaches are
# gated (the coordination contrast). EVERY variant ends with both arms lifting.


def _left() -> int:
    return vocab.LB_TARGETS["left_end"]


def _right() -> int:
    return vocab.LB_TARGETS["right_end"]


def _make_simultaneous_builder() -> ProgramBuilder:
    """Baseline: both arms approach->close->lift, NO gates. Both lift together."""
    def build(planner, env):
        return {
            0: [_approach_qp(_left()), _close_qp(_left()), _lift_qp(_left())],
            1: [_approach_qp(_right()), _close_qp(_right()), _lift_qp(_right())],
        }
    return build


def _make_stagger_builder(lead_arm: int) -> ProgramBuilder:
    """One arm LEADS ungated; the partner WAITS to approach until the leader grasps.

    Both lifts are gated on the LEADER having grasped, so once the partner has also
    grasped both ends rise TOGETHER (no tilt). The partner's approach is the only
    place the wait shows up at frame 0 (it sits ``wait``, gripper open). The leader,
    after closing, sits on its gated lift = holds its grasp (gripper closed,
    labelled ``wait``) until the partner has grasped too. COMPLETES (both lift).

    DEADLOCK-SAFE: the LEADER's first primitive (approach) is UNGATED, so the
    program can always start; the follower's approach gate opens as soon as the
    leader grasps.
    """
    follow_arm = 1 - lead_arm
    lead_end = _left() if lead_arm == 0 else _right()
    follow_end = _left() if follow_arm == 0 else _right()

    def build(planner, env):
        # gate on the LEADER finishing its grasp (qi >= GRASP_IDX). DETERMINISTIC:
        # the leader starts at qi=0, so the follower ALWAYS waits at frame 0.
        gate_on_lead = _gate_other_reached(lead_arm, GRASP_IDX)
        return {
            # leader: ungated approach + close, then a lift gated on the FOLLOWER
            # having grasped (so both ends rise together).
            lead_arm: [
                _approach_qp(lead_end),
                _close_qp(lead_end),
                _lift_qp(lead_end, wait_for=_gate_other_reached(follow_arm, GRASP_IDX)),
            ],
            # follower: approach gated on the LEADER grasped, then close, then a lift
            # also gated on the leader grasped (already true by then -> rises with).
            follow_arm: [
                _approach_qp(follow_end, wait_for=gate_on_lead),
                _close_qp(follow_end),
                _lift_qp(follow_end, wait_for=gate_on_lead),
            ],
        }
    return build


# --------------------------------------------------------------------------- deadlock / lift guards
# Pure, sim-free structural checks on a built program (the {arm:[QueuedRecipe]}).


def gate_graph(program: Dict[int, List[QueuedRecipe]]) -> Dict[int, Optional[int]]:
    """Return {arm: other_arm} the FIRST primitive of each arm waits on (None=ungated).

    Only the FIRST primitive matters for startup: if every arm's first primitive is
    gated on another arm, nothing can begin (deadlock). Later gates can always open
    because some arm makes progress first.
    """
    out: Dict[int, Optional[int]] = {}
    for arm, queue in program.items():
        if not queue:
            out[arm] = None
            continue
        wf = queue[0].wait_for
        out[arm] = None if wf is None else int(wf[0])
    return out


def assert_no_deadlock(program: Dict[int, List[QueuedRecipe]]) -> None:
    """Raise if NO arm has an ungated first primitive (the program can't start).

    The deadlock guard: at least one arm's first QueuedRecipe must have
    wait_for=None, otherwise every arm blocks on another arm forever.
    """
    first_gates = gate_graph(program)
    if not first_gates:
        return
    if all(g is not None for g in first_gates.values()):
        raise AssertionError(
            f"DEADLOCK: every arm's first primitive is gated on another arm "
            f"(first-gate graph {first_gates}); no arm can start. Exactly one arm "
            f"must have an UNGATED first primitive."
        )


def assert_both_arms_lift(program: Dict[int, List[QueuedRecipe]], planner, env) -> None:
    """Raise unless EVERY arm's last primitive is a LIFT (no permanent-hold variant).

    Builds each arm's LAST recipe against the given (planner, env) — pass the FAKES
    off-GPU — and asserts its verb is LIFT. Guarantees every variant is a complete
    both-ends-rise rollout.
    """
    for arm, queue in program.items():
        assert queue, f"arm {arm} has an EMPTY queue (must end with a lift)"
        last = queue[-1].recipe(planner, env, arm)
        if last.verb_id != vocab.LIFT:
            raise AssertionError(
                f"arm {arm}'s last primitive is {last.name!r} (verb={last.verb_id}), "
                f"not a LIFT. Every variant must end with BOTH arms lifting "
                f"(no permanent-hold / truncated variant)."
            )


# --------------------------------------------------------------------------- sampler


def sample(seed: int) -> List[ProgramSpec]:
    """Sample the full contrastive set of LiftBarrier program specs for ``seed``.

    All randomness comes from a DEDICATED ``np.random.default_rng(seed)`` so the
    sampler is reproducible AND never touches the env reset RNG (frame-0 stays
    identical across variants). NOTE: today the structure is fixed (gating only),
    so the RNG draw is reserved for forward-compat (e.g. randomized hold lengths)
    and to keep reproducibility tests meaningful; it does NOT change which variants
    are emitted.

    Returns a flat list of ProgramSpec. Each matched contrastive pair
    ``{simultaneous, <variant>}`` shares a ``contrast_group_id`` (use
    ``group_specs`` to bucket them). The ``simultaneous`` baseline is RE-EMITTED in
    each pair so every contrast is a clean A/B from the same reset.
    """
    rng = np.random.default_rng(int(seed))
    # reserved structural draw (kept for reproducibility tests / forward-compat).
    _reserved = int(rng.integers(2, 5))

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

    # Each contrast group is a matched PAIR: the simultaneous baseline + one
    # coordination variant, from the SAME seed (frame-0 identical). The contrast is
    # who waits / who leads, NOT truncation — every member completes the lift.

    # (0) simultaneous  vs  stagger_a_leads (arm0 leads)
    emit("simultaneous", "stagger_a", gid, _make_simultaneous_builder(),
         {"lead_arm": None, "gated": False, "complete": True})
    emit("stagger_a_leads", "stagger_a", gid, _make_stagger_builder(lead_arm=0),
         {"lead_arm": 0, "follow_arm": 1, "complete": True})
    gid += 1

    # (1) simultaneous  vs  stagger_b_leads (arm1 leads, the mirror)
    emit("simultaneous", "stagger_b", gid, _make_simultaneous_builder(),
         {"lead_arm": None, "gated": False, "complete": True})
    emit("stagger_b_leads", "stagger_b", gid, _make_stagger_builder(lead_arm=1),
         {"lead_arm": 1, "follow_arm": 0, "complete": True})
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
        assert len(members) == 2, (gid, names)  # every group is a matched pair
        assert "simultaneous" in names, (gid, names)  # baseline in every pair
        print(f"  group {gid} [{members[0].family}]: {names}")
    assert len(s) == 4 and len(by_group) == 2, (len(s), len(by_group))
    assert {m.family for m in s} == {"stagger_a", "stagger_b"}
    # variant ids unique
    assert len({m.variant_id for m in s}) == len(s)
    # seed reproducible (same structural draws)
    assert [m.name for m in sample(7)] == [m.name for m in s]
    print("subtask_scenarios inline OK")
