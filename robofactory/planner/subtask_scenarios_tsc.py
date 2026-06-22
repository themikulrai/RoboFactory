"""ThreeRobotsStackCube (TSC) subtask scenario + COLLISION-FREE contrastive sampler.

Mirror of ``subtask_scenarios`` (the LiftBarrier sampler) for the 3-arm
ThreeRobotsStackCube task, REBUILT to reproduce the proven collision-free
choreography of ``solutions/three_robots_stack_cube.py`` (the original solver).
A "scenario" turns a random seed into a list of
:class:`~.subtask_scenarios.ProgramSpec` builders; each builder, given a live
``(planner, env)``, returns a per-arm program ``{arm: [QueuedRecipe]}`` for
``subtask_interpreter.run_program``. Same two levels of laziness as the LB sampler
(structure now, JIT per-primitive planning later) and the same SIM-FREE contract.

WHY THIS REWRITE: the previous TSC sampler dropped the original solver's
collision-free structure — it lifted only +0.2 and let all three arms descend into
the SAME stacking region, so episodes COLLIDED. We now mirror the original exactly.

THE ORIGINAL SOLVER's COLLISION-FREE STRUCTURE (solutions/three_robots_stack_cube.py):
  (1) all 3 grasp SIMULTANEOUSLY at their SEPARATE cubes (arm0->cubeA, arm1->cubeB,
      arm2->cubeC);
  (2) all 3 lift +0.5 SIMULTANEOUSLY (HIGH, well out of the stacking region);
  (3) PLACE STRICTLY SERIALLY: arm0 -> goal_region, then arm0 RETREATS back UP to
      grasp_poseA+0.5; THEN arm1 -> onto cubeA, retreats UP; THEN arm2 -> onto
      cubeB. Only ONE arm is ever in the stacking region at a time; the others
      stay parked HIGH (+0.5) and the placer RETREATS UP before the next descends.

COLLISION-FREE INVARIANTS (enforced in EVERY variant):
  A. Pick (approach+close) and the +0.5 lift MAY be simultaneous (separate cubes)
     OR staggered. TSC lift dz = 0.5 (NOT 0.2) so an idle/parked arm is HIGH.
  B. A grasped arm that is not the one currently placing is PARKED HIGH at +0.5 and
     WAITs there. This happens AUTOMATICALLY: after lift(0.5), the arm's PLACE is
     gated; while blocked the interpreter holds the frozen +0.5 qpos (gripper
     closed) labelled ``wait`` = "wait at top".
  C. PLACE IS STRICTLY SERIAL: the arm at stack-rank k may begin its place ONLY
     after the arm at rank k-1 has PLACED **AND RETREATED** (region clear). The
     gate is a deterministic program-progress predicate on the previous placer's
     qi reaching its RETREAT-done index, NOT just its place-done index. Stack
     order: cubeA(goal, rank0) -> cubeB(on A, rank1) -> cubeC(on B, rank2).
  D. Every place is followed by a RETREAT-UP out of the region (straight up ~+0.5,
     gripper open), mirroring the original's move-back-to-grasp_pose+0.5.

PER-ARM PROGRAM (base, stack-rank order arm0=A rank0, arm1=B rank1, arm2=C rank2):
  rank0 (arm0): approach(blue)  -> close -> lift(0.5) -> place(goal_region) -> open -> retreat_up
  rank1 (arm1): approach(green) -> close -> lift(0.5) -> place(on cubeA, gated) -> open -> retreat_up
  rank2 (arm2): approach(red)   -> close -> lift(0.5) -> place(on cubeB, gated) -> open -> retreat_up
The full program is [approach(0), close(1), lift(2), place(3), open(4), retreat_up(5)].
``_ArmState.qi`` increments AFTER a primitive completes, so:
  * GRASP_IDX       = 2 : finished approach+close (grasped), now on the lift.
  * an arm has FINISHED its retreat (region clear) once qi >= 6 (the program has 6
    primitives, retreat_up at index 5, qi==6 means it completed). We COMPUTE this
    per-arm via ``_retreat_done_idx`` (== len(program)) because the
    ``direct_place`` variant DROPS arm0's lift, shortening its program to 5 (retreat
    at index 4, retreat-done at qi >= 5) — the gate index must track the ACTUAL
    previous-arm program length, never a hardcoded 6.

CONTRAST FAMILIES (matched pairs, SAME seed; placing serial in BOTH members):
  1. ``pickcoord``: simultaneous_pick (all 3 approach+close+lift UNGATED together)
     VS staggered_pick (arm0 leads ungated; arm1 & arm2's approach gated on arm0
     GRASPED (qi >= GRASP_IDX) -> they WAIT at home at frame 0). Both then place
     serially A->B->C with retreats. Frame-0 counterfactual: arm1/arm2 WAIT vs
     APPROACH.
  2. ``raisedirect``: for the FIRST placer (arm0, rank0): raise_and_wait (arm0 does
     lift(0.5) then place — it raises HIGH first) VS direct_place (arm0 SKIPS the
     lift(0.5) and places directly from the grasp, since the region is clear and it
     is FIRST). The others ALWAYS raise. Counterfactual: arm0 after grasp -> LIFT
     (raise/wait-at-top) vs PLACE (descend directly). Both stay collision-free (arm0
     places first into the empty region; the others wait HIGH).

GATE MECHANISM / GUARDS (inherited from the LB module):
gates are DETERMINISTIC predicates on another arm's ``_ArmState.qi``, never a
physics probe. DEADLOCK GUARD: >= 1 arm has an UNGATED first primitive (the bottom
placer always grasps ungated) and the gate graph is acyclic (B waits A retreated, C
waits B retreated — a chain). The serial-place guard ``assert_serial_place_gated``
checks every place above rank 0 is gated on the PREVIOUS rank's RETREAT-done.

IMPORT-CLEAN-ish: imports numpy + the (numpy-only) primitive/interpreter modules
and the LB scenario module; NO sapien / gym / robofactory.tasks at module top, so
the sampler's structure + metadata are unit-testable off-GPU with a fake planner/env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from . import subtask_vocab as vocab
from . import subtask_primitives as P
from .subtask_interpreter import QueuedRecipe, check_tcp_near
# REUSE the LB module's shared machinery (one source of truth).
from .subtask_scenarios import (
    ProgramSpec,
    GRASP_IDX,
    _gate_other_reached,
    gate_graph,
    assert_no_deadlock,
    group_specs,
    filter_variants,
)

TSC = "ThreeRobotsStackCube"
# ThreeRobotsStackCube has exactly 3 arms.
TSC_NUM_ARMS = 3

# TSC lift / retreat raise height. 0.5 (NOT 0.2) so a parked/idle arm clears the
# stacking region entirely — matches the original solver's grasp_pose[2] += 0.5.
TSC_LIFT_DZ = 0.5

# A builder takes the live (planner, env) and returns {arm: [QueuedRecipe]}.
ProgramBuilder = Callable[[object, object], Dict[int, List[QueuedRecipe]]]

# Program-position indices for the FULL 6-primitive per-arm program
# [approach(0), close(1), lift(2), place(3), open(4), retreat_up(5)].
# qi increments AFTER a primitive completes, so:
PLACE_IDX = 3        # the place primitive sits at index 3 (full program)
PLACE_DONE_IDX = 4   # qi >= 4 means the place is DONE (now on its open)
# RETREAT-done index is COMPUTED per-arm (== len(program)) because the
# ``direct_place`` variant shortens arm0's program by dropping its lift. See
# ``_retreat_done_idx``.

# --------------------------------------------------------------------------- target/dest tables
# color -> TSC target id for the cube (the approach/lift target).
_COLOR_TID = {
    "blue": vocab.TSC_TARGETS["blue"],
    "green": vocab.TSC_TARGETS["green"],
    "red": vocab.TSC_TARGETS["red"],
}
# dest actor name -> the PLACE target id (the prompt destination phrase).
# goal_region -> "on the goal"; a cube dest -> "on the <color> cube".
_DEST_PLACE_TID = {
    "goal_region": vocab.TSC_TARGETS["goal_region"],
    "cubeA": vocab.TSC_TARGETS["on_blue"],   # cubeA is blue
    "cubeB": vocab.TSC_TARGETS["on_green"],  # cubeB is green
    "cubeC": vocab.TSC_TARGETS["on_red"],    # cubeC is red
}


def _color_of_cube(cube: str) -> str:
    return vocab.TSC_CUBE_COLOR[cube]


# --------------------------------------------------------------------------- TSC recipes
# Each ``_*_qp`` returns a QueuedRecipe wrapping a PrimitiveRecipe closure
# ``recipe(planner, env, arm) -> Primitive``. The closure captures the STRUCTURE
# decided by the sampler (which cube/color, which dest, gate); the interpreter
# calls it JIT at primitive entry so it plans from the LIVE pose.


def _approach_qp(color: str, *, grip: float = P.OPEN,
                 success_tol: float = 0.08, wait_for=None) -> QueuedRecipe:
    """approach the cube of ``color`` (OBB grasp pose). success_check = TCP near."""
    tid = _COLOR_TID[color]

    def recipe(planner, env, arm: int) -> P.Primitive:
        sc = check_tcp_near(tol=success_tol)
        return P.approach(planner, env, arm=arm, target_id=tid, task=TSC,
                          grip=grip, success_check=sc)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _close_qp(*, wait_for=None) -> QueuedRecipe:
    # NO success_check on close (is_grasping seats late; a non-grasp is caught
    # downstream: no grasp -> no stack -> env never succeeds -> group dropped).
    def recipe(planner, env, arm: int) -> P.Primitive:
        return P.close_gripper(planner, env, arm=arm, task=TSC, start_grip=P.OPEN,
                               success_check=None)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _lift_qp(color: str, *, dz: float = TSC_LIFT_DZ, wait_for=None) -> QueuedRecipe:
    # NO success_check: env stack-success is the real gate, and a per-cube height
    # check would need the cube's base z which the sampler does not know off-GPU.
    tid = _COLOR_TID[color]

    def recipe(planner, env, arm: int) -> P.Primitive:
        return P.lift(planner, env, arm=arm, target_id=tid, task=TSC, dz=dz,
                      grip=P.CLOSED, success_check=None)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _place_qp(color: str, dest_actor_name: str, *, wait_for=None) -> QueuedRecipe:
    """place the carried cube of ``color`` onto ``dest_actor_name``.

    ``now_pose`` (the carried cube's grasp pose) is re-resolved JIT from the LIVE
    cube pose via resolve_cube_grasp_pose, so object perturbation is absorbed. NO
    success_check (rely on env stack-success)."""
    tid = _DEST_PLACE_TID[dest_actor_name]

    def recipe(planner, env, arm: int) -> P.Primitive:
        u = env.unwrapped if hasattr(env, "unwrapped") else env
        now_pose = P.resolve_cube_grasp_pose(planner, u, arm, color)
        return P.place(planner, u, arm=arm, target_id=tid, task=TSC,
                       now_pose=now_pose, dest_actor_name=dest_actor_name,
                       grip=P.CLOSED, success_check=None)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _open_qp(*, wait_for=None) -> QueuedRecipe:
    def recipe(planner, env, arm: int) -> P.Primitive:
        return P.open_gripper(planner, env, arm=arm, task=TSC, start_grip=P.CLOSED,
                              success_check=None)
    return QueuedRecipe(recipe, wait_for=wait_for)


def _retreat_qp(*, dz: float = TSC_LIFT_DZ, wait_for=None) -> QueuedRecipe:
    """retreat the arm STRAIGHT UP out of the stacking region after a place+open.

    Mirrors the original solver's move-back-to-grasp_pose+0.5. Gripper OPEN (the
    cube was just released). Plans from the LIVE tcp pose (no grasp re-resolve)."""
    def recipe(planner, env, arm: int) -> P.Primitive:
        return P.retreat_up(planner, env, arm=arm, task=TSC, dz=dz,
                            grip=P.OPEN, success_check=None)
    return QueuedRecipe(recipe, wait_for=wait_for)


# --------------------------------------------------------------------------- assignment helper
# An "assignment" maps each arm -> (cube, dest_actor_name) and records the stack
# ORDER (which arm places first/second/third).

@dataclass(frozen=True)
class _Assign:
    """One arm's role: the cube it handles + where it places it + its stack rank.

    stack_rank 0 = bottom (placed into goal first); 1 = middle (onto bottom cube);
    2 = top (onto middle cube). The PLACE gate chains on the arm one rank below
    having RETREATED (region clear), not merely placed.
    """
    cube: str
    dest_actor_name: str
    stack_rank: int


# Base assignment (matches the canonical solver):
#   arm0 -> cubeA(blue) onto goal_region  (bottom, rank 0)
#   arm1 -> cubeB(green) onto cubeA        (middle, rank 1)
#   arm2 -> cubeC(red)  onto cubeB         (top, rank 2)
_BASE_ASSIGN: Dict[int, _Assign] = {
    0: _Assign(cube="cubeA", dest_actor_name="goal_region", stack_rank=0),
    1: _Assign(cube="cubeB", dest_actor_name="cubeA", stack_rank=1),
    2: _Assign(cube="cubeC", dest_actor_name="cubeB", stack_rank=2),
}


def _arm_at_rank(assign: Dict[int, _Assign], rank: int) -> int:
    """Return the arm whose stack_rank == ``rank`` (unique per assignment)."""
    for arm, a in assign.items():
        if a.stack_rank == rank:
            return arm
    raise KeyError(f"no arm at stack_rank {rank} in assignment {assign}")


def _retreat_done_idx(prog_len: int) -> int:
    """The qi at which an arm whose program has ``prog_len`` primitives has FINISHED
    its retreat. qi increments AFTER each primitive completes and the retreat is the
    LAST primitive, so the arm reaches qi == prog_len exactly when it finishes its
    retreat (region clear). This is COMPUTED (not hardcoded 6) so the
    ``direct_place`` variant's shorter (5-primitive) arm0 gates correctly."""
    return int(prog_len)


# --------------------------------------------------------------------------- per-arm program builder


def _arm_program(color: str, dest: str, *, grasp_gate, place_gate,
                 raise_first: bool) -> List[QueuedRecipe]:
    """Build ONE arm's full program.

    approach -> close -> [lift(0.5) if raise_first] -> place -> open -> retreat_up.

    ``raise_first`` False DROPS the lift (the ``direct_place`` arm0): it places
    directly from the grasp. ``grasp_gate`` gates the FIRST primitive (approach);
    ``place_gate`` gates the PLACE (serial-place stack-order gate)."""
    queue: List[QueuedRecipe] = [
        _approach_qp(color, wait_for=grasp_gate),
        _close_qp(),
    ]
    if raise_first:
        queue.append(_lift_qp(color))
    queue.append(_place_qp(color, dest, wait_for=place_gate))
    queue.append(_open_qp())
    queue.append(_retreat_qp())
    return queue


def _build_program(assign: Dict[int, _Assign], *,
                   grasp_lead_arm: Optional[int] = None,
                   direct_place_arm: Optional[int] = None) -> ProgramBuilder:
    """Build a TSC program (the collision-free serial-place choreography).

    ``grasp_lead_arm``:
      * None -> SIMULTANEOUS pick: every arm approaches+closes+lifts UNGATED.
      * an arm id -> STAGGERED pick: that arm's grasp chain is UNGATED (it leads);
        EVERY OTHER arm's approach waits until the lead arm has GRASPED
        (qi >= GRASP_IDX). So at frame 0 the followers ``wait`` at home.

    ``direct_place_arm``: if set, that arm SKIPS its lift (places directly from the
    grasp). Only valid for the rank-0 (first) placer — the region is clear and it
    places first, so the descend is safe. The others always raise.

    PLACES are ALWAYS serial: the arm at rank k waits until the arm at rank k-1 has
    PLACED AND RETREATED (qi >= that arm's retreat-done index). The retreat-done
    index is computed from the PREVIOUS arm's actual program length (so a shortened
    direct-place arm gates at the right index).
    """

    def build(planner, env):
        # First pass: determine each arm's program length (depends on raise_first)
        # so place gates can target the previous arm's RETREAT-done index.
        prog_len: Dict[int, int] = {}
        for arm in assign:
            raise_first = (direct_place_arm is None) or (arm != direct_place_arm)
            # approach, close, [lift], place, open, retreat_up
            prog_len[arm] = 6 if raise_first else 5

        prog: Dict[int, List[QueuedRecipe]] = {}
        for arm, a in assign.items():
            color = _color_of_cube(a.cube)

            # grasp-phase gate: None for simultaneous OR the lead arm; followers
            # wait on the lead arm's grasp completion.
            if grasp_lead_arm is None or arm == grasp_lead_arm:
                grasp_gate = None
            else:
                grasp_gate = _gate_other_reached(grasp_lead_arm, GRASP_IDX)

            # serial-place gate: wait until the rank-below arm has RETREATED.
            rank = a.stack_rank
            if rank == 0:
                place_gate = None  # bottom cube places first, ungated
            else:
                below_arm = _arm_at_rank(assign, rank - 1)
                place_gate = _gate_other_reached(
                    below_arm, _retreat_done_idx(prog_len[below_arm]))

            raise_first = (direct_place_arm is None) or (arm != direct_place_arm)
            prog[arm] = _arm_program(
                color, a.dest_actor_name,
                grasp_gate=grasp_gate, place_gate=place_gate,
                raise_first=raise_first,
            )
        return prog

    return build


# --------------------------------------------------------------------------- structural guards
# Pure, sim-free checks on a built program (the {arm:[QueuedRecipe]}).


def assert_all_arms_retreat(program: Dict[int, List[QueuedRecipe]], planner, env) -> None:
    """Raise unless EVERY arm: (a) PLACEs its cube, then (b) OPENs, then (c) ends
    with a RETREAT-UP. The retreat-up is the collision-clearing motion the original
    solver performs after each serial place; without it the next arm would descend
    into an occupied region.

    Layout: the LAST primitive is the retreat (verb LIFT, name starts 'retreat_up'),
    the second-to-last is the open, and the third-to-last is the place. (The
    direct_place arm0 has no lift, so we identify by walking from the END, which is
    layout-independent.)"""
    for arm, queue in program.items():
        assert len(queue) >= 4, (
            f"arm {arm} program too short ({len(queue)}); need at least "
            f"approach,close,place,open,retreat_up")
        retreat = queue[-1].recipe(planner, env, arm)
        if not (retreat.verb_id == vocab.LIFT and retreat.name.startswith("retreat_up")):
            raise AssertionError(
                f"arm {arm}'s LAST primitive is {retreat.name!r} (verb="
                f"{retreat.verb_id}), not a retreat_up. Every TSC variant must end "
                f"each arm with a RETREAT-UP out of the stacking region.")
        opn = queue[-2].recipe(planner, env, arm)
        if opn.verb_id != vocab.OPEN_GRIPPER:
            raise AssertionError(
                f"arm {arm}'s second-to-last primitive is {opn.name!r} (verb="
                f"{opn.verb_id}), not an open_gripper. Each arm must release before "
                f"retreating (env success requires all grippers released).")
        place = queue[-3].recipe(planner, env, arm)
        if place.verb_id != vocab.PLACE:
            raise AssertionError(
                f"arm {arm}'s third-to-last primitive is {place.name!r} (verb="
                f"{place.verb_id}), not a PLACE. Each arm must PLACE its cube before "
                f"the open+retreat.")


def assert_serial_place_gated(program: Dict[int, List[QueuedRecipe]],
                              assign: Dict[int, _Assign]) -> None:
    """Raise unless every PLACE above stack-rank 0 is gated on the PREVIOUS rank's
    arm having RETREATED (region clear). rank-0's place must be UNGATED (it is
    first). This is the collision-free serial-place guard.

    The PLACE QueuedRecipe is located by walking from the END (it is the
    third-to-last primitive: place, open, retreat_up), which is robust to the
    direct_place arm dropping its lift."""
    # map each arm -> its program length so we can check the gate target index.
    plen = {arm: len(q) for arm, q in program.items()}
    for arm, a in assign.items():
        place_qr = program[arm][-3]  # place, open, retreat_up
        rank = a.stack_rank
        if rank == 0:
            if place_qr.wait_for is not None:
                raise AssertionError(
                    f"rank-0 arm {arm}'s place is GATED ({place_qr.wait_for}); the "
                    f"first placer must be UNGATED.")
            continue
        below_arm = _arm_at_rank(assign, rank - 1)
        wf = place_qr.wait_for
        if wf is None or int(wf[0]) != below_arm:
            raise AssertionError(
                f"arm {arm} (rank {rank}) place must be gated on the rank-{rank-1} "
                f"arm {below_arm} having retreated; got gate {wf}.")
        # the gate must open at the below-arm's RETREAT-done index (== its prog len),
        # i.e. NOT merely at its place-done. Probe the predicate at boundary qis.
        _, predicate = wf
        below_retreat_done = plen[below_arm]

        class _S:
            def __init__(self, qi):
                self.qi = qi

        def _state(qi):
            return {"arms": {below_arm: _S(qi)}}

        # below arm only PLACED (not retreated) -> gate still CLOSED.
        if predicate(None, _state(PLACE_DONE_IDX)) is not False:
            raise AssertionError(
                f"arm {arm}'s place gate OPENED on the below-arm merely PLACING "
                f"(qi={PLACE_DONE_IDX}); it must wait for the below-arm to RETREAT "
                f"(qi>={below_retreat_done}).")
        # below arm RETREATED -> gate OPENS.
        if predicate(None, _state(below_retreat_done)) is not True:
            raise AssertionError(
                f"arm {arm}'s place gate did NOT open after the below-arm retreated "
                f"(qi={below_retreat_done}).")


# --------------------------------------------------------------------------- sampler


def sample(seed: int) -> List[ProgramSpec]:
    """Sample the collision-free contrastive set of TSC program specs for ``seed``.

    All randomness comes from a DEDICATED ``np.random.default_rng(seed)`` so the
    sampler is reproducible AND never touches the env reset RNG (frame-0 stays
    identical across variants). Structure is fixed (gating only), so the RNG draw is
    reserved for forward-compat / reproducibility.

    Returns a flat list of ProgramSpec, two matched contrastive PAIRS:
      * ``pickcoord``  : simultaneous_pick  vs  staggered_pick
      * ``raisedirect``: raise_and_wait     vs  direct_place
    Both members of each pair place STRICTLY SERIALLY (A->B->C) with retreats and so
    are collision-free; they differ only at the contrast axis (frame-0 pick timing
    for pickcoord; arm0's raise-vs-descend for raisedirect).
    """
    rng = np.random.default_rng(int(seed))
    _reserved = int(rng.integers(2, 5))  # reserved structural draw (forward-compat)

    specs: List[ProgramSpec] = []
    vid = 0
    gid = 0

    def emit(name, family, group_id, build, meta):
        nonlocal vid
        specs.append(ProgramSpec(
            name=name, variant_id=vid, contrast_group_id=group_id, family=family,
            seed=int(seed), build=build, num_arms=TSC_NUM_ARMS, meta=dict(meta),
        ))
        vid += 1

    # (0) pickcoord: simultaneous_pick  vs  staggered_pick
    #   simultaneous_pick: all 3 grasp+lift ungated together; serial place A->B->C.
    #   staggered_pick:    arm0 leads grasp ungated; arm1,arm2 wait for arm0 grasped.
    emit("simultaneous_pick", "pickcoord", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=None),
         {"pick": "simultaneous", "lead_arm": None, "complete": True,
          "assignment": "base"})
    emit("staggered_pick", "pickcoord", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=0),
         {"pick": "staggered", "lead_arm": 0, "follow_arms": [1, 2],
          "complete": True, "assignment": "base"})
    gid += 1

    # (1) raisedirect: raise_and_wait  vs  direct_place (FIRST placer arm0 only)
    #   raise_and_wait: arm0 lifts(0.5) then places (raises high first).
    #   direct_place:   arm0 SKIPS the lift, places directly from the grasp (region
    #                   clear, it is first). The others ALWAYS raise.
    #   Both use a SIMULTANEOUS pick so the ONLY contrast is arm0's raise-vs-descend.
    emit("raise_and_wait", "raisedirect", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=None),
         {"first_placer": 0, "arm0": "raise_then_place", "complete": True,
          "assignment": "base"})
    emit("direct_place", "raisedirect", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=None, direct_place_arm=0),
         {"first_placer": 0, "arm0": "direct_place_no_lift", "complete": True,
          "assignment": "base"})
    gid += 1

    return specs


# group_specs / filter_variants are re-exported from the LB module (imported above)
# so the runner can call scenarios_tsc.group_specs / filter_variants uniformly.


# --------------------------------------------------------------------------- inline doc test
if __name__ == "__main__":
    s = sample(7)
    by_group = group_specs(s)
    print(f"sampled {len(s)} specs in {len(by_group)} contrast groups")
    for gid_, members in sorted(by_group.items()):
        names = [m.name for m in members]
        assert len(members) == 2, (gid_, names)
        print(f"  group {gid_} [{members[0].family}]: {names}")
    assert len(s) == 4 and len(by_group) == 2, (len(s), len(by_group))
    assert {m.family for m in s} == {"pickcoord", "raisedirect"}
    assert {m.name for m in s} == {
        "simultaneous_pick", "staggered_pick", "raise_and_wait", "direct_place"}
    assert len({m.variant_id for m in s}) == len(s)
    assert [m.name for m in sample(7)] == [m.name for m in s]
    print("subtask_scenarios_tsc inline OK")
