"""TwoRobotsStackCube (2SC) subtask scenario + COLLISION-FREE contrastive sampler.

The 2-arm sibling of ``subtask_scenarios_tsc`` (the 3-arm ThreeRobotsStackCube
sampler). 2SC stacks TWO cubes: cubeA(blue) -> goal_region (bottom, rank 0), then
cubeB(green) -> onto cubeA (top, rank 1). Same SIM-FREE, two-levels-of-laziness
contract; same collision-free serial-place choreography; same causal-data goal: make
behavior decodable ONLY from the per-arm subtask command, never the sim/image alone,
so the low-level pi0.5 policy is FORCED to read the language channel.

WHY A THIN MODULE: every piece of the choreography machinery (recipes, the program
builder ``_build_program``, the serial-place gates, the structural guards) is
ARM-COUNT-AGNOSTIC — it iterates over an ``assign`` dict keyed by arm id. So this
module REUSES it wholesale from ``subtask_scenarios_tsc`` and only supplies the
2SC-specific pieces: a 2-arm base assignment, a 2-cube reorder, num_arms=2, and the
per-seed contrast group. (CLAUDE.md: reuse, do not rebuild.)

PRIMITIVE TASK STRING: the primitives resolve cube geometry + render prompts by a
``task`` string and only implement ``"LiftBarrier"`` / ``"ThreeRobotsStackCube"``.
2SC's cubes (blue, green) and place destinations (goal_region, on cubeA, on cubeB)
are a STRICT SUBSET of TSC's vocab, and ``resolve_cube_grasp_pose`` already maps
blue->cubeA / green->cubeB via ``TSC_CUBE_COLOR``. So the 2SC programs reuse
``task="ThreeRobotsStackCube"`` for the primitive calls (via the imported recipes):
the rendered prompts ("approach the blue cube", "place on the goal", "place on the
blue cube") are correct English for 2SC, with ZERO vocab edits. The EPISODE-level
task identity comes from the env id ("TwoRobotsStackCube-rf") via the runner's
TASK_MAP, NOT this string — the string only drives geometry + label rendering.

CONTRAST GROUP (5 members, ONE group per seed; single shared simultaneous baseline):
  baseline ``simultaneous_pick``: both arms approach+close+lift UNGATED together, then
     place serially A->B with retreats; the idle grasped arm parks HIGH (+0.5).
  1. ``pickcoord`` (staggered_pick / staggered_lead1): one arm leads the grasp
     ungated, the OTHER waits on the leader GRASPED (qi >= GRASP_IDX) -> the follower
     WAITs at home at frame 0. Two lead permutations (arm0 leads, arm1 leads). Which
     arm leads is decodable only from the frame-0 command (approach vs wait).
  2. ``waitcoord`` (wait_hold): identical choreography to the baseline, but at a
     random mid-PLACE point one arm FREEZES (labelled `wait`) for K ticks, then
     resumes and completes the stack. Whether/when an arm waits is decodable ONLY
     from the `wait` command. Config rides in meta["wait_inject"].
  3. ``stackorder`` (reorder_stack): the commanded bottom->top order is INVERTED
     (cubeB on the bottom, cubeA on top = order (1,0)). With only 2 cubes there is
     exactly ONE non-canonical order, so reorder_stack is ALWAYS (1,0) (the canonical
     (0,1) is already the baseline -> no duplicate). Which cube goes where is
     decodable ONLY from the place commands. The env's evaluate() is parameterized by
     meta["intended_order"] (set on the env by the runner after reset).

COLLISION-FREE / DEADLOCK invariants are identical to TSC and enforced by the SAME
imported guards (``assert_no_deadlock``, ``assert_serial_place_gated``): >=1 arm has
an ungated first primitive (the bottom placer always grasps ungated under either lead
choice), the gate graph is acyclic, and every place above rank 0 waits for the
rank-below arm to have RETREATED (region clear), not merely placed.
"""

from __future__ import annotations

from dataclasses import dataclass  # noqa: F401  (kept for parity / future local use)
from typing import Dict, List

import numpy as np

from . import subtask_vocab as vocab
# REUSE the arm-count-agnostic TSC machinery (one source of truth). All of these
# iterate over an ``assign`` dict keyed by arm id, so they work unchanged for 2 arms.
from .subtask_scenarios_tsc import (
    _Assign,
    _build_program,
    _arm_at_rank,
    assert_serial_place_gated,
    assert_all_arms_retreat,
)
# Shared LB machinery (re-exported so the runner can call uniformly).
from .subtask_scenarios import (
    ProgramSpec,
    assert_no_deadlock,
    group_specs,
    filter_variants,
)

TWOSC = "TwoRobotsStackCube"
# TwoRobotsStackCube has exactly 2 arms / 2 cubes.
TWOSC_NUM_ARMS = 2

# cube-index (0=A, 1=B) -> actor name. Arm i grasps cube i.
_CUBE_NAME = {0: "cubeA", 1: "cubeB"}
# The TWO bottom->top orders. Index 0 == (0,1) == canonical A-B (== baseline). Index
# 1 == (1,0) == B-A (the only non-canonical order). reorder_stack always uses (1,0).
_PERMS = ((0, 1), (1, 0))

# Base assignment (matches the canonical solver):
#   arm0 -> cubeA(blue) onto goal_region  (bottom, rank 0)
#   arm1 -> cubeB(green) onto cubeA        (top, rank 1)
_BASE_ASSIGN: Dict[int, _Assign] = {
    0: _Assign(cube="cubeA", dest_actor_name="goal_region", stack_rank=0),
    1: _Assign(cube="cubeB", dest_actor_name="cubeA", stack_rank=1),
}


def _reorder_assign(order) -> Dict[int, _Assign]:
    """Permuted _Assign for a bottom->top cube ``order`` (e.g. (1,0) == B-A).

    Each arm still grasps its OWN cube (arm i <-> cube i; parking stays arm-indexed so
    the serial-place choreography is collision-free for either order). Only the
    destination chain + stack_rank change: bottom -> goal_region (rank 0), top -> onto
    the bottom cube (rank 1)."""
    b, t = int(order[0]), int(order[1])
    return {
        b: _Assign(cube=_CUBE_NAME[b], dest_actor_name="goal_region", stack_rank=0),
        t: _Assign(cube=_CUBE_NAME[t], dest_actor_name=_CUBE_NAME[b], stack_rank=1),
    }


# --------------------------------------------------------------------------- sampler


def sample(seed: int) -> List[ProgramSpec]:
    """Sample the collision-free contrastive set of 2SC program specs for ``seed``.

    Returns a flat list of 5 ProgramSpec in ONE contrast group sharing a SINGLE
    ``simultaneous_pick`` baseline (frame-0 identical across all members; emitted
    once). Mirrors ``subtask_scenarios_tsc.sample`` adapted to 2 arms / 2 cubes:
    the 3 staggered lead perms collapse to 2 (arm0 / arm1 leads), and reorder has a
    single non-canonical order (B-A). All members place STRICTLY SERIALLY with
    retreats and so are collision-free; ``family`` tags each variant's contrast axis.
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
            seed=int(seed), build=build, num_arms=TWOSC_NUM_ARMS, meta=dict(meta),
        ))
        vid += 1

    # Shared simultaneous baseline (emitted once; frame-0 identical across members).
    emit("simultaneous_pick", "baseline", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=None),
         {"pick": "simultaneous", "lead_arm": None, "first_placer": 0,
          "complete": True, "assignment": "base"})

    # STAGGERED-PICK PERMUTATIONS: each arm leads the grasp in turn; the OTHER waits
    # on the leader's grasp (qi >= GRASP_IDX). Which arm leads is decodable only from
    # the frame-0 command (approach vs wait). Serial place order is unchanged, so both
    # permutations stay collision-free (assert_no_deadlock).
    for _lead in (0, 1):
        _name = "staggered_pick" if _lead == 0 else f"staggered_lead{_lead}"
        emit(_name, "pickcoord", gid,
             _build_program(_BASE_ASSIGN, grasp_lead_arm=_lead),
             {"pick": "staggered", "lead_arm": _lead,
              "follow_arms": [a for a in (0, 1) if a != _lead],
              "complete": True, "assignment": "base"})

    # WAIT_HOLD: same simultaneous choreography as the baseline, but at a random
    # mid-PLACE point one arm FREEZES (labelled `wait`) for K ticks, then resumes and
    # completes the stack. Whether/when an arm waits is decodable ONLY from the `wait`
    # command. Config rides in meta["wait_inject"]; the runner applies it via
    # run_program ONLY for this variant (clean contrast vs the no-wait baseline). The
    # 200-tick single event matches the TSC recipe; 2SC episodes are shorter, so the
    # injected-WAIT FRACTION is larger than TSC's ~31% -> verify in the pilot and dial
    # dur / the env max_episode_steps if it risks truncation.
    emit("wait_hold", "waitcoord", gid,
         _build_program(_BASE_ASSIGN, grasp_lead_arm=None),
         {"pick": "simultaneous_wait", "complete": True, "assignment": "base",
          "wait_inject": {"frac": 0.05, "dur_min": 200, "dur_max": 200,
                          "max_events": 1, "verb_id": vocab.PLACE, "seed": 0}})

    # REORDER_STACK: the commanded bottom->top order is INVERTED to (1,0) == B-A (the
    # only non-canonical 2-cube order; canonical A-B is the baseline). Which cube goes
    # where is decodable ONLY from the place commands. evaluate() is parameterized by
    # meta["intended_order"] (set on the env by the runner after reset). Collision-free
    # (parking is arm-indexed; places serial).
    _order = _PERMS[1]  # (1, 0) == B-A, always non-canonical
    emit("reorder_stack", "stackorder", gid,
         _build_program(_reorder_assign(_order), grasp_lead_arm=None),
         {"pick": "simultaneous", "complete": True, "assignment": "reorder",
          "intended_order": list(_order)})
    gid += 1

    return specs


# group_specs / filter_variants are re-exported from the LB module (imported above)
# so the runner can call scenarios_2sc.group_specs / filter_variants uniformly.


# --------------------------------------------------------------------------- inline doc test
if __name__ == "__main__":
    s = sample(7)
    by_group = group_specs(s)
    print(f"sampled {len(s)} specs in {len(by_group)} contrast groups")
    for gid_, members in sorted(by_group.items()):
        names = [m.name for m in members]
        # baseline + 2 staggered + wait_hold + reorder_stack
        assert len(members) == 5, (gid_, names)
        assert "simultaneous_pick" in names, (gid_, names)  # single shared baseline
        print(f"  group {gid_}: {names}")
    assert len(s) == 5 and len(by_group) == 1, (len(s), len(by_group))
    assert {m.family for m in s} == {
        "baseline", "pickcoord", "waitcoord", "stackorder"}
    assert {m.name for m in s} == {
        "simultaneous_pick", "staggered_pick", "staggered_lead1",
        "wait_hold", "reorder_stack"}
    assert len({m.variant_id for m in s}) == len(s)
    assert all(m.num_arms == 2 for m in s)
    assert [m.name for m in sample(7)] == [m.name for m in s]
    # every order builds a valid keyed _Assign (bottom/top distinct arms, ranks 0,1)
    for _o in _PERMS:
        _a = _reorder_assign(_o)
        assert set(_a.keys()) == {0, 1}, (_o, _a)
        assert [_a[_arm_at_rank(_a, r)].stack_rank for r in (0, 1)] == [0, 1]
    # structural guards on the built programs (sim-free): build each member with a
    # trivial fake planner/env is NOT possible here (recipes are lazy), but the gate
    # structure is checkable via assert_no_deadlock on the spec builders' gate graph.
    # Deadlock/serial-place guards run in the pilot with a live (planner, env).
    print("subtask_scenarios_2sc inline OK")
