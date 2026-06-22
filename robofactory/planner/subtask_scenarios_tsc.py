"""ThreeRobotsStackCube (TSC) subtask scenario + CONTRASTIVE program sampler.

Mirror of ``subtask_scenarios`` (the LiftBarrier sampler) for the 3-arm
ThreeRobotsStackCube task. A "scenario" turns a random seed into a list of
:class:`~.subtask_scenarios.ProgramSpec` objects; each spec is a *builder* that,
given a live ``(planner, env)``, returns a per-arm program ``{arm: [QueuedRecipe]}``
for ``subtask_interpreter.run_program``. Same two levels of laziness as the LB
sampler (structure now, JIT per-primitive planning later) and the same SIM-FREE
contract (the sampler only decides STRUCTURE — gating — with its own RNG; all
planning happens inside the recipes at run time).

We REUSE the LB module's shared machinery directly (``ProgramSpec``, ``QueuedRecipe``,
the deterministic program-progress gate ``_gate_other_reached``, and the structural
guards ``gate_graph`` / ``assert_no_deadlock`` / ``group_specs`` / ``filter_variants``)
so there is one source of truth for those.

EVERY VARIANT IS A COMPLETE TASK ROLLOUT (the cubes actually stack)
==================================================================
TSC env success = cubeB on cubeA, cubeC on cubeB, cubeB over goal_region, and all
three grippers released (see ThreeRobotsStackCubeEnv.evaluate). The canonical
solver stacks bottom-up: goal_region <- cubeA <- cubeB <- cubeC (A into the goal,
B on A, C on B). So in EVERY variant all three arms ULTIMATELY PLACE their cube
(end with a PLACE) and the env reaches its stack success. Holds/waits are
TRANSIENT (coordination), never permanent.

THE TASK TRUTH (read from tasks/three_robots_stack_cube.py + the solver):
  * 3 arms: panda-0 (left, y=-0.8), panda-1 (right, y=+0.8), panda-2 (middle, x=+0.8).
  * 3 cubes: cubeA=blue, cubeB=green, cubeC=red (see TSC_CUBE_COLOR).
  * Solver assignment + place destinations (solutions/three_robots_stack_cube.py):
      - arm0 -> cubeA (blue), placed onto ``goal_region``  (BOTTOM of the stack)
      - arm1 -> cubeB (green), placed onto ``cubeA``       (middle)
      - arm2 -> cubeC (red),  placed onto ``cubeB``        (top)
    i.e. dest actor names goal_region / cubeA / cubeB respectively.

STACK DEPENDENCY (the new gate axis vs LB's coordination gate):
Placement MUST be bottom-up: cubeA placed into the goal FIRST, then cubeB onto A,
then cubeC onto B. We enforce this with a DETERMINISTIC program-progress gate on
the PLACE primitive of each non-bottom cube: cubeB's place waits for cubeA's
placer to have FINISHED its place; cubeC's place waits for cubeB's placer. This is
the same ``_gate_other_reached`` mechanism the LB stagger gate uses, just pointed
at the PLACE-done program index instead of the grasp-done index.

PER-ARM PROGRAM (the cube X handler):
    [approach(color_of_X), close, lift, place(onto dest_of_X), open]
Program indices: approach=0, close=1, lift=2, place=3, open=4. ``_ArmState.qi``
increments AFTER a primitive completes, so:
  * GRASP_IDX = 2  : the arm has FINISHED approach+close and is on its lift (grasped).
  * PLACE_DONE_IDX = 4 : the arm has FINISHED its place and is on its open (placed).

VARIANTS (matched pairs; each vs the simultaneous baseline, SAME seed):
  1. ``simultaneous`` (baseline): all 3 arms approach+close+lift UNGATED together;
     PLACES gated ONLY by stack order (B-place waits A placed; C-place waits B
     placed). The bottom placer (cubeA->goal) is fully ungated. Frame-0: all 3
     arms ``approach``.
  2. ``stagger_grasp`` (coordination contrast, parallels LB's stagger): arm0 (the
     cubeA/bottom handler) LEADS grasp UNGATED; the other two arms WAIT (labelled
     ``wait``, gripper open) until arm0 has GRASPED (qi >= GRASP_IDX), then they
     grasp; PLACES still gated by stack order. Counterfactual vs simultaneous: at
     frame 0 arm1/arm2 are ``wait`` here vs ``approach`` there.
  3. ``target_swap`` (color counterfactual — ATTEMPT, may be infeasible, kept
     FILTERABLE): a DIFFERENT arm<->cube assignment from the SAME start —
     arm0 -> cubeC (red), arm1 -> cubeB (green), arm2 -> cubeA (blue) — but the
     STACK ORDER stays valid: whoever holds cubeA places into the goal FIRST (here
     arm2), then the cubeB holder onto cubeA (arm1), then the cubeC holder onto
     cubeB (arm0). The PROMPT reflects the cube COLOR, so arm0 reads "approach the
     red cube" here vs "approach the blue cube" in simultaneous -> a color-grounding
     counterfactual. CONCERN: cross-assignment can be infeasible on a 3-arm table
     (reachability: arm0 is left, cubeC may sit far right; and the stack point is
     fixed at the goal so the placer reach is unchanged but the GRASP reach changes).
     Implemented anyway and tagged so it can be dropped after sim validation.

GATE MECHANISM / DEADLOCK GUARD (inherited from the LB module, unchanged):
The gates are DETERMINISTIC predicates on another arm's ``_ArmState.qi`` (program
progress), never a physics probe. DEADLOCK GUARD: every variant has >= 1 arm with
an UNGATED first primitive (the bottom-cube handler always starts ungated) and the
gate graph is acyclic (B waits A, C waits B — a chain, never a cycle). EVERY arm
ends with a PLACE (the stack-complete guard ``assert_all_arms_place``).

PER-PRIMITIVE CHECKS:
  * approach -> check_tcp_near (reads the primitive's stored target_pose).
  * close    -> none (a non-grasp is caught downstream: no grasp -> no stack ->
    env never reaches success -> group dropped).
  * lift     -> none (env stack-success is the real gate; a height check would need
    the per-cube base z which the sampler does not know off-GPU).
  * place    -> none (rely on env stack-success).
Episode kept via env_success (proper stack) OR completed (see the runner's
``_member_passes``).

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

# A builder takes the live (planner, env) and returns {arm: [QueuedRecipe]}.
ProgramBuilder = Callable[[object, object], Dict[int, List[QueuedRecipe]]]

# Program-position index at which an arm has FINISHED its PLACE and is on its open.
# Per-arm program is [approach(0), close(1), lift(2), place(3), open(4)]; qi
# increments AFTER a primitive completes, so qi >= 4 means the place is DONE.
PLACE_DONE_IDX = 4

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


def _lift_qp(color: str, *, dz: float = 0.2, wait_for=None) -> QueuedRecipe:
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


# --------------------------------------------------------------------------- assignment helper
# An "assignment" maps each arm -> (cube, dest_actor_name) and records the stack
# ORDER (which arm places first/second/third). The builders below turn an
# assignment + a grasp-gating policy into a {arm:[QueuedRecipe]} program.

@dataclass(frozen=True)
class _Assign:
    """One arm's role: the cube it handles + where it places it + its stack rank.

    stack_rank 0 = bottom (placed into goal first); 1 = middle (onto bottom cube);
    2 = top (onto middle cube). The PLACE gate chains on the arm one rank below.
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

# target_swap assignment: SAME cubes/dests but a DIFFERENT arm holds each cube.
#   arm0 -> cubeC(red)  onto cubeB   (top, rank 2)
#   arm1 -> cubeB(green) onto cubeA   (middle, rank 1)   [unchanged arm for cubeB]
#   arm2 -> cubeA(blue) onto goal_region (bottom, rank 0)
# Stack order stays valid: the cubeA holder (arm2) places into the goal FIRST,
# then the cubeB holder (arm1) onto cubeA, then the cubeC holder (arm0) onto cubeB.
_SWAP_ASSIGN: Dict[int, _Assign] = {
    0: _Assign(cube="cubeC", dest_actor_name="cubeB", stack_rank=2),
    1: _Assign(cube="cubeB", dest_actor_name="cubeA", stack_rank=1),
    2: _Assign(cube="cubeA", dest_actor_name="goal_region", stack_rank=0),
}


def _arm_at_rank(assign: Dict[int, _Assign], rank: int) -> int:
    """Return the arm whose stack_rank == ``rank`` (unique per assignment)."""
    for arm, a in assign.items():
        if a.stack_rank == rank:
            return arm
    raise KeyError(f"no arm at stack_rank {rank} in assignment {assign}")


def _place_gate(assign: Dict[int, _Assign], arm: int):
    """Stack-order gate for ``arm``'s PLACE: wait until the arm one rank BELOW has
    FINISHED its place (qi >= PLACE_DONE_IDX). The bottom arm (rank 0) is UNGATED."""
    rank = assign[arm].stack_rank
    if rank == 0:
        return None  # bottom cube places first, ungated
    below_arm = _arm_at_rank(assign, rank - 1)
    return _gate_other_reached(below_arm, PLACE_DONE_IDX)


# --------------------------------------------------------------------------- family builders


def _build_program(assign: Dict[int, _Assign],
                   grasp_lead_arm: Optional[int]) -> ProgramBuilder:
    """Build a TSC program from an assignment + a grasp-gating policy.

    ``grasp_lead_arm``:
      * None -> SIMULTANEOUS grasp: every arm approaches+closes+lifts UNGATED.
      * an arm id -> STAGGER grasp: that arm's grasp chain is UNGATED (it leads);
        EVERY OTHER arm's approach (and close+lift, redundantly) waits until the
        lead arm has GRASPED (qi >= GRASP_IDX). So at frame 0 the followers ``wait``.

    PLACES are ALWAYS gated by stack order (``_place_gate``) in both policies, so
    the cubes stack bottom-up regardless of the grasp timing.
    """

    def build(planner, env):
        prog: Dict[int, List[QueuedRecipe]] = {}
        for arm, a in assign.items():
            color = _color_of_cube(a.cube)
            # grasp-phase gate: None for simultaneous OR the lead arm; followers
            # wait on the lead arm's grasp completion.
            if grasp_lead_arm is None or arm == grasp_lead_arm:
                grasp_gate = None
            else:
                grasp_gate = _gate_other_reached(grasp_lead_arm, GRASP_IDX)
            prog[arm] = [
                _approach_qp(color, wait_for=grasp_gate),
                _close_qp(),
                _lift_qp(color),
                _place_qp(color, a.dest_actor_name, wait_for=_place_gate(assign, arm)),
                _open_qp(),
            ]
        return prog

    return build


# --------------------------------------------------------------------------- stack-complete guard
# Pure, sim-free structural check: EVERY arm must end with a PLACE (no truncated /
# permanent-hold variant). The TSC analogue of LB's assert_both_arms_lift.


def assert_all_arms_place(program: Dict[int, List[QueuedRecipe]], planner, env) -> None:
    """Raise unless EVERY arm's program CONTAINS a PLACE whose next primitive is the
    final open (i.e. the arm places then releases). We assert the PLACE sits at
    index PLACE_DONE_IDX-1 (== 3) and the last primitive is an open_gripper, so the
    stack actually completes and the gripper releases (env success needs release)."""
    for arm, queue in program.items():
        assert queue, f"arm {arm} has an EMPTY queue (must place then open)"
        # the PLACE must be the second-to-last primitive (place -> open).
        place_qr = queue[PLACE_DONE_IDX - 1]
        place_prim = place_qr.recipe(planner, env, arm)
        if place_prim.verb_id != vocab.PLACE:
            raise AssertionError(
                f"arm {arm}'s primitive at index {PLACE_DONE_IDX - 1} is "
                f"{place_prim.name!r} (verb={place_prim.verb_id}), not a PLACE. "
                f"Every TSC variant must have each arm PLACE its cube."
            )
        last = queue[-1].recipe(planner, env, arm)
        if last.verb_id != vocab.OPEN_GRIPPER:
            raise AssertionError(
                f"arm {arm}'s last primitive is {last.name!r} (verb={last.verb_id}), "
                f"not an open_gripper. Each arm must release after placing (env "
                f"success requires all grippers released)."
            )


# --------------------------------------------------------------------------- sampler


def sample(seed: int) -> List[ProgramSpec]:
    """Sample the contrastive set of ThreeRobotsStackCube program specs for ``seed``.

    All randomness comes from a DEDICATED ``np.random.default_rng(seed)`` so the
    sampler is reproducible AND never touches the env reset RNG (frame-0 stays
    identical across variants). As in the LB sampler, the structure is fixed
    (gating only), so the RNG draw is reserved for forward-compat / reproducibility.

    Returns a flat list of ProgramSpec. Each matched contrastive pair
    ``{simultaneous, <variant>}`` shares a ``contrast_group_id`` (bucket with
    ``group_specs``). The ``simultaneous`` baseline is RE-EMITTED in each pair so
    every contrast is a clean A/B from the same reset.
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

    # The simultaneous baseline (re-built per pair so each contrast is a clean A/B
    # from the same seed). Builder is deterministic, so re-building is free.
    def _sim_builder():
        return _build_program(_BASE_ASSIGN, grasp_lead_arm=None)

    # (0) simultaneous  vs  stagger_grasp (arm0/bottom leads the grasp)
    emit("simultaneous", "stagger", gid, _sim_builder(),
         {"lead_arm": None, "grasp_gated": False, "complete": True,
          "assignment": "base"})
    emit("stagger_grasp", "stagger", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=0),
         {"lead_arm": 0, "follow_arms": [1, 2], "complete": True,
          "assignment": "base"})
    gid += 1

    # (1) simultaneous  vs  target_swap (color counterfactual; ATTEMPT, filterable).
    # Same simultaneous baseline (base assignment); target_swap uses the swapped
    # arm<->cube assignment with a SIMULTANEOUS grasp (so the ONLY contrast vs the
    # baseline is WHICH cube/color each arm grasps -> a color-grounding counterfactual).
    emit("simultaneous", "target_swap", gid, _sim_builder(),
         {"lead_arm": None, "grasp_gated": False, "complete": True,
          "assignment": "base"})
    emit("target_swap", "target_swap", gid,
         _build_program(_SWAP_ASSIGN, grasp_lead_arm=None),
         {"lead_arm": None, "grasp_gated": False, "complete": True,
          "assignment": "swap",
          # CONCERN: cross-assignment may be infeasible (reachability/stack). Kept
          # filterable so it can be dropped after sim validation (filter by name or
          # family 'target_swap').
          "feasibility": "ATTEMPT_may_be_infeasible"})
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
        assert "simultaneous" in names, (gid_, names)
        print(f"  group {gid_} [{members[0].family}]: {names}")
    assert len(s) == 4 and len(by_group) == 2, (len(s), len(by_group))
    assert {m.family for m in s} == {"stagger", "target_swap"}
    assert len({m.variant_id for m in s}) == len(s)
    assert [m.name for m in sample(7)] == [m.name for m in s]
    print("subtask_scenarios_tsc inline OK")
