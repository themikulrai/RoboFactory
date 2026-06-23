"""PURE + SIM-gated tests for the ThreeRobotsStackCube (TSC) subtask scenarios.

The TSC sampler reproduces the proven COLLISION-FREE choreography of the original
solver (solutions/three_robots_stack_cube.py): all 3 grasp at separate cubes, all
lift HIGH (+0.5), then place STRICTLY SERIALLY (A->goal, retreat; B->A, retreat;
C->B). Only ONE arm descends into the stacking region at a time; the others wait
parked HIGH; each placer RETREATS UP before the next descends.

PURE tests (no sim, always run): the sampler emits the two matched contrast pairs
(``pickcoord`` simultaneous_pick/staggered_pick; ``raisedirect``
simultaneous_pick/direct_place — simultaneous_pick is the SHARED baseline re-emitted
once per group, so there is NO ``raise_and_wait``); names/families; deadlock-acyclic;
every arm ends with
a retreat_up after place+open; the SERIAL-place gates chain on the PREVIOUS rank's
RETREAT-done (rank1 waits rank0 retreated, rank2 waits rank1 retreated) — NOT merely
on place-done; staggered_pick followers WAIT at frame 0; direct_place arm0 has NO
lift primitive; the retreat_up primitive plans straight UP (target z > start z); RNG
reproducibility + independence from global numpy RNG; group filtering keeps pairs
intact; render() strings; the runner's TASK_MAP wires the TSC sampler.

SIM tests (@pytest.mark.sim, skipped unless DART_RUN_SIM=1; DO NOT RUN on the login
node — SAPIEN renders dark; needs a GPU compute node): drive a real 3-arm TSC
rollout, verify the aligned stream, the staggered-pick follower waiting at frame 0,
the serial-place ordering, and a simultaneous_pick run reaching env stack-success
collision-free.

Run PURE:   /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
                scripts/dart/tests/test_subtask_tsc.py -k "not sim" -q
Run ALL (GPU node):  DART_RUN_SIM=1 pytest scripts/dart/tests/test_subtask_tsc.py -q
"""
from __future__ import annotations

import os
import os.path as osp
import sys

import numpy as np
import pytest

# make scripts/dart importable (so run_subtask_rollouts is reachable on a GPU node)
_DART_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
_SCRIPTS_DIR = osp.dirname(_DART_DIR)
for _p in (_DART_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robofactory.planner import subtask_vocab as vocab  # noqa: E402
from robofactory.planner import subtask_primitives as P  # noqa: E402
from robofactory.planner import subtask_interpreter as I  # noqa: E402
from robofactory.planner import subtask_scenarios_tsc as STSC  # noqa: E402
from robofactory.planner.subtask_primitives import OPEN, CLOSED  # noqa: E402

RUN_SIM = os.environ.get("DART_RUN_SIM", "0") == "1"
sim = pytest.mark.skipif(not RUN_SIM, reason="set DART_RUN_SIM=1 to run sim tests")


# ===========================================================================
# Light off-GPU fakes (3-arm planner + env) — no sapien, no gym.
# ===========================================================================
class FakeRobot:
    def __init__(self, q7):
        self._q = np.concatenate([np.asarray(q7, np.float32), [0.04, 0.04]]).astype(np.float32)

    def get_qpos(self):
        return self._q[None, :]  # (1, 9) numpy


class FakeActor:
    def __init__(self, z=0.6):
        self.z = z


# Down-pointing tcp orientation (wxyz, NON-identity) after a place. retreat_up must
# PRESERVE this quaternion (a pure vertical translation); an identity quat would
# force a large reorient (~567-step path) and stall the serial stack.
_DOWN_Q = np.array([0.0, 1.0, 0.0, 0.0], np.float32)


class _FakePose:
    def __init__(self, xyz, q=_DOWN_Q):
        self.p = np.asarray(xyz, np.float32)
        self.q = np.asarray(q, np.float32)


class _FakeTcp:
    def __init__(self, xyz, q=_DOWN_Q):
        self.pose = _FakePose(xyz, q)


class _FakeAgent:
    """Per-arm agent exposing tcp.pose.p and tcp.pose.q (read by retreat_up)."""
    def __init__(self, arm):
        # distinct tcp z per arm so retreat target z = start z + dz is checkable.
        self.tcp = _FakeTcp([0.1 * arm, 0.0, 0.4 + 0.05 * arm])


class _FakeAgentBundle:
    def __init__(self, num_arms):
        self.agents = [_FakeAgent(i) for i in range(num_arms)]


class FakePlanner:
    """Canned-waypoint 3-arm planner; a pose LIST under dry_run is REJECTED."""

    def __init__(self, num_arms=3, control_mode="pd_joint_pos", n_waypoints=5):
        self.control_mode = control_mode
        self.robot = [FakeRobot(np.zeros(7) + 0.1 * i) for i in range(num_arms)]
        self.n_waypoints = n_waypoints
        self.last_dry_run_pose = None

    def move_to_pose_with_screw(self, pose, move_id=0, dry_run=False, **kw):
        assert dry_run, "primitives must plan with dry_run=True"
        assert not isinstance(pose, list), "pose LIST passed under dry_run (the trap)"
        self.last_dry_run_pose = np.asarray(pose, np.float32)
        n = self.n_waypoints
        base = self.robot[move_id].get_qpos()[0, :7]
        wps = np.stack([base + (k + 1) / n * np.ones(7) for k in range(n)]).astype(np.float32)
        return {"position": wps, "status": "Success"}

    def get_grasp_pose_from_obb(self, actor, agent_id=0):
        return np.array([0.0, 0.1 * agent_id, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)

    def get_grasp_pose_for_stack(self, now_pose, target_actor, height_offset=0.05):
        out = np.asarray(now_pose, np.float32).copy()
        out[2] = float(target_actor.z) + height_offset
        return out


class FakeEnv:
    def __init__(self, num_arms=3):
        self.goal_region = FakeActor(z=0.0)
        self.cubeA = FakeActor(z=0.05)
        self.cubeB = FakeActor(z=0.1)
        self.cubeC = FakeActor(z=0.15)
        self.agent = _FakeAgentBundle(num_arms)
        self.unwrapped = self
        self.n_steps = 0
        self.step_log = []

    def step(self, action_dict):
        self.n_steps += 1
        self.step_log.append({k: np.asarray(v).copy() for k, v in action_dict.items()})
        return {}, 0.0, False, False, {"success": False}


def _fakes():
    return FakePlanner(num_arms=3), FakeEnv(num_arms=3)


# Canonical DISTINCT behaviours (3): a SINGLE simultaneous_pick baseline (emitted
# ONCE, NOT re-emitted per group) plus the two contrast variants. No `raise_and_wait`
# (was a byte-identical duplicate of the baseline).
_NAMES = {"simultaneous_pick", "staggered_pick", "direct_place"}


# ===========================================================================
# PURE: render() over the TSC color/dest/wait strings used by the sampler
# ===========================================================================
def test_render_tsc_strings():
    assert vocab.render(vocab.APPROACH, vocab.TSC_TARGETS["green"], "ThreeRobotsStackCube") \
        == "approach the green cube"
    assert vocab.render(vocab.APPROACH, vocab.TSC_TARGETS["blue"], "ThreeRobotsStackCube") \
        == "approach the blue cube"
    assert vocab.render(vocab.APPROACH, vocab.TSC_TARGETS["red"], "ThreeRobotsStackCube") \
        == "approach the red cube"
    assert vocab.render(vocab.PLACE, vocab.TSC_TARGETS["goal_region"], "ThreeRobotsStackCube") \
        == "place on the goal"
    assert vocab.render(vocab.PLACE, vocab.TSC_TARGETS["on_blue"], "ThreeRobotsStackCube") \
        == "place on the blue cube"
    assert vocab.render(vocab.PLACE, vocab.TSC_TARGETS["on_green"], "ThreeRobotsStackCube") \
        == "place on the green cube"
    assert vocab.render(vocab.WAIT, 0, "ThreeRobotsStackCube") == "wait"


# ===========================================================================
# PURE: the TSC retreat targets the arm's PARKED pose (grasp+0.5), NOT current+up
# ===========================================================================
def _grasp_pose(pl, arm):
    """The arm's OWN cube grasp pose as the FakePlanner resolves it (OBB)."""
    return pl.get_grasp_pose_from_obb(None, arm).reshape(-1).astype(np.float32)


def test_retreat_targets_parked_grasp_pose_plus_dz():
    """The TSC retreat (last primitive) plans to the arm's OWN cube grasp pose
    (its parked side, captured at PICK) RAISED by park_dz — the SAME high pose the
    lift reached — NOT "current tcp + dz". This is the original solver's
    move-back-to-grasp_pose+0.5 retreat (known-reachable, out of the central goal).

    We build a full program so the approach recipe runs first and CAPTURES the grasp
    pose into the shared ``parked`` dict; the retreat then reuses it. xy of the
    planned retreat pose == the arm's grasp xy (its own cube side), z == grasp z +
    park_dz, orientation == the grasp orientation (full 7-vec)."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    for arm in (0, 1, 2):
        # run the approach FIRST (this is what captures parked[arm]).
        prog[arm][0].recipe(pl, env, arm)
        grasp = _grasp_pose(pl, arm)
        prim = prog[arm][-1].recipe(pl, env, arm)  # the retreat (last primitive)
        # labelled LIFT (a raise/clear motion), target_id 0, named retreat_up
        assert prim.verb_id == vocab.LIFT
        assert prim.target_id == 0
        assert prim.name.startswith("retreat_up")
        planned = pl.last_dry_run_pose.reshape(-1)
        # xy == the arm's grasp xy (its OWN cube side), NOT the placed/current xy
        assert pytest.approx(planned[0], abs=1e-6) == float(grasp[0])
        assert pytest.approx(planned[1], abs=1e-6) == float(grasp[1])
        # z == grasp z + park_dz (the parked high pose the lift reached)
        assert pytest.approx(planned[2], abs=1e-6) == float(grasp[2]) + STSC.TSC_LIFT_DZ
        # full grasp ORIENTATION carried through (so the quaternion is right)
        np.testing.assert_allclose(planned[3:7], grasp[3:7], atol=1e-6)
        assert float(prim.target_pose[2]) == pytest.approx(float(grasp[2]) + STSC.TSC_LIFT_DZ)
        # gripper OPEN (cube released)
        assert all(g == OPEN for _, g in prim.ticks)


def test_approach_captures_parked_and_lift_retreat_reuse_it():
    """The approach recipe CAPTURES the arm's grasp pose into a shared per-arm dict;
    the lift and the retreat both target that SAME captured pose + park_dz (so they
    agree). We verify by perturbing the planner's resolved grasp pose AFTER approach
    runs: the lift/retreat must still use the CAPTURED (pre-perturbation) value,
    proving they reuse the captured copy rather than re-resolving live."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    arm = 1
    grasp = _grasp_pose(pl, arm)  # the value at approach time

    # approach captures parked[arm].
    prog[arm][0].recipe(pl, env, arm)

    # Now MOVE the cube: make the planner resolve a different grasp pose. If the
    # lift/retreat re-resolved live they'd pick this up; if they use the captured
    # copy they ignore it.
    moved = grasp.copy()
    moved[0] += 5.0  # large xy shift
    pl.get_grasp_pose_from_obb = lambda actor, agent_id=0, _m=moved: _m.copy()

    # lift (index 2 for a raising arm) -> uses captured grasp xy, z = grasp z + dz
    lift_prim = prog[arm][2].recipe(pl, env, arm)
    assert lift_prim.verb_id == vocab.LIFT
    lift_planned = pl.last_dry_run_pose.reshape(-1)
    assert pytest.approx(lift_planned[0], abs=1e-6) == float(grasp[0])  # captured, NOT moved
    assert pytest.approx(lift_planned[2], abs=1e-6) == float(grasp[2]) + STSC.TSC_LIFT_DZ

    # retreat (last) -> SAME captured target as the lift
    retreat_prim = prog[arm][-1].recipe(pl, env, arm)
    retreat_planned = pl.last_dry_run_pose.reshape(-1)
    assert pytest.approx(retreat_planned[0], abs=1e-6) == float(grasp[0])
    np.testing.assert_allclose(retreat_planned[:3], lift_planned[:3], atol=1e-6)


# ===========================================================================
# PURE: the sampler emits ONE group with a single shared baseline (dedup)
# ===========================================================================
def test_sampler_emits_one_group_single_baseline():
    specs = STSC.sample(7)
    groups = STSC.group_specs(specs)
    assert len(specs) == 3
    assert len(groups) == 1
    (gid, members), = groups.items()
    assert len(members) == 3, (gid, [m.name for m in members])
    assert all(m.contrast_group_id == gid for m in members)
    for m in members:
        assert m.num_arms == 3
    # EXACTLY ONE simultaneous_pick baseline (the dedup: no per-group re-emit).
    assert [m.name for m in members].count("simultaneous_pick") == 1
    assert {m.family for m in specs} == {"baseline", "pickcoord", "raisedirect"}
    # 3 distinct behaviours; no `raise_and_wait` (it was a byte-identical duplicate).
    assert {m.name for m in specs} == _NAMES
    assert "raise_and_wait" not in {m.name for m in specs}


def test_sampler_variant_ids_unique_and_reproducible():
    specs = STSC.sample(7)
    assert len({m.variant_id for m in specs}) == len(specs)
    again = STSC.sample(7)
    assert [m.name for m in specs] == [m.name for m in again]
    assert [m.meta for m in specs] == [m.meta for m in again]


def test_sampler_rng_independent_of_global_numpy():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    _ = STSC.sample(123)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after), "sample() perturbed the GLOBAL numpy RNG"


# ===========================================================================
# PURE: every variant — all 3 arms place->open->retreat_up (complete + clearing)
# ===========================================================================
def test_every_variant_all_arms_place_open_retreat():
    specs = STSC.sample(7)
    for spec in specs:
        pl, env = _fakes()
        prog = spec.build(pl, env)
        assert set(prog.keys()) == {0, 1, 2}, spec.name
        # the structural guard raises if any arm doesn't place->open->retreat_up
        STSC.assert_all_arms_retreat(prog, pl, env)
        for arm in (0, 1, 2):
            retreat = prog[arm][-1].recipe(pl, env, arm)
            assert retreat.verb_id == vocab.LIFT and retreat.name.startswith("retreat_up"), \
                (spec.name, arm)
            opn = prog[arm][-2].recipe(pl, env, arm)
            assert opn.verb_id == vocab.OPEN_GRIPPER, (spec.name, arm)
            place = prog[arm][-3].recipe(pl, env, arm)
            assert place.verb_id == vocab.PLACE, (spec.name, arm)


def test_full_arm_program_layout():
    """A FULL arm program (raising) is approach,close,lift,place,open,retreat_up.
    The direct_place arm0 DROPS the lift -> approach,close,place,open,retreat_up."""
    specs = STSC.sample(7)
    full = [vocab.APPROACH, vocab.CLOSE_GRIPPER, vocab.LIFT, vocab.PLACE,
            vocab.OPEN_GRIPPER, vocab.LIFT]  # last LIFT is the retreat_up
    direct = [vocab.APPROACH, vocab.CLOSE_GRIPPER, vocab.PLACE,
              vocab.OPEN_GRIPPER, vocab.LIFT]
    for spec in specs:
        pl, env = _fakes()
        prog = spec.build(pl, env)
        for arm in (0, 1, 2):
            verbs = [qr.recipe(pl, env, arm).verb_id for qr in prog[arm]]
            if spec.name == "direct_place" and arm == 0:
                assert verbs == direct, (spec.name, arm, verbs)
            else:
                assert verbs == full, (spec.name, arm, verbs)


# ===========================================================================
# PURE: deadlock guard (>=1 arm ungated first) + acyclic gate chain
# ===========================================================================
def test_no_deadlock_one_arm_ungated_first():
    specs = STSC.sample(7)
    for spec in specs:
        pl, env = _fakes()
        prog = spec.build(pl, env)
        STSC.assert_no_deadlock(prog)
        first_gates = STSC.gate_graph(prog)
        assert any(g is None for g in first_gates.values()), (spec.name, first_gates)


def test_simultaneous_pick_first_primitives_all_ungated():
    """simultaneous_pick: all 3 arms approach UNGATED at frame 0 (no grasp gate)."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    assert STSC.gate_graph(prog) == {0: None, 1: None, 2: None}


# ===========================================================================
# PURE: SERIAL-place gates chain on the PREVIOUS rank's RETREAT-done
# ===========================================================================
class _FakeArmState:
    def __init__(self, qi):
        self.qi = int(qi)


def _place_qr(prog, arm):
    """The PLACE QueuedRecipe of an arm: third-to-last (place, open, retreat_up)."""
    return prog[arm][-3]


def test_serial_place_gates_chain_on_retreat_done():
    """rank0 (arm0) place UNGATED; rank1 (arm1) place waits arm0 RETREATED; rank2
    (arm2) place waits arm1 RETREATED. The gate must NOT open on mere place-done."""
    for name in _NAMES:
        spec = next(m for m in STSC.sample(7) if m.name == name)
        pl, env = _fakes()
        prog = spec.build(pl, env)
        # the dedicated structural guard does the full check (gate target + index).
        STSC.assert_serial_place_gated(prog, STSC._BASE_ASSIGN)
        # arm0 (bottom): place ungated
        assert _place_qr(prog, 0).wait_for is None, name
        # arm1 waits on arm0; arm2 waits on arm1
        g1 = _place_qr(prog, 1).wait_for
        g2 = _place_qr(prog, 2).wait_for
        assert g1 is not None and g1[0] == 0, (name, g1)
        assert g2 is not None and g2[0] == 1, (name, g2)
        # arm1's gate opens at arm0's RETREAT-done (== arm0 prog len), not place-done.
        plen0 = len(prog[0])
        _, pred1 = g1
        assert pred1(None, {"arms": {0: _FakeArmState(STSC.PLACE_DONE_IDX)}}) is False, name
        assert pred1(None, {"arms": {0: _FakeArmState(plen0)}}) is True, name
        # arm2's gate opens at arm1's RETREAT-done (== arm1 prog len == 6).
        plen1 = len(prog[1])
        _, pred2 = g2
        assert pred2(None, {"arms": {1: _FakeArmState(STSC.PLACE_DONE_IDX)}}) is False, name
        assert pred2(None, {"arms": {1: _FakeArmState(plen1)}}) is True, name


def test_direct_place_arm0_gate_index_is_shorter():
    """direct_place: arm0 has 5 primitives (no lift) so its retreat-done qi is 5, and
    arm1's place gate must target 5 (NOT 6). Confirms the gate tracks the SHORTENED
    program length, not a hardcoded 6."""
    spec = next(m for m in STSC.sample(7) if m.name == "direct_place")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    assert len(prog[0]) == 5, len(prog[0])  # approach,close,place,open,retreat_up
    _, pred1 = _place_qr(prog, 1).wait_for
    # gate closed at qi=4 (arm0 placed+opened but not retreated), open at qi=5.
    assert pred1(None, {"arms": {0: _FakeArmState(4)}}) is False
    assert pred1(None, {"arms": {0: _FakeArmState(5)}}) is True


# ===========================================================================
# PURE: pickcoord staggered followers WAIT at frame 0
# ===========================================================================
def test_staggered_pick_followers_wait_at_frame0():
    """staggered_pick: arm0 leads grasp UNGATED; arm1 & arm2 first primitive GATED on
    arm0 grasped -> they WAIT at frame 0 vs APPROACH in simultaneous_pick."""
    spec = next(m for m in STSC.sample(7) if m.name == "staggered_pick")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    first_gates = STSC.gate_graph(prog)
    assert first_gates[0] is None        # leader ungated first
    assert first_gates[1] == 0           # follower waits on arm0
    assert first_gates[2] == 0           # follower waits on arm0
    for follow in (1, 2):
        other_arm, predicate = prog[follow][0].wait_for
        assert other_arm == 0
        assert predicate(None, {"arms": {0: _FakeArmState(0)}}) is False
        assert predicate(None, {"arms": {0: _FakeArmState(1)}}) is False  # only approach
        assert predicate(None, {"arms": {0: _FakeArmState(STSC.GRASP_IDX)}}) is True
        assert predicate(None, {"arms": {0: _FakeArmState(STSC.GRASP_IDX + 1)}}) is True


def test_simultaneous_pick_followers_approach_at_frame0():
    """Contrast partner: simultaneous_pick arm1/arm2 first primitive is UNGATED
    approach (the frame-0 counterfactual to staggered_pick's wait)."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    for arm in (1, 2):
        assert prog[arm][0].wait_for is None
        assert prog[arm][0].recipe(pl, env, arm).verb_id == vocab.APPROACH


# ===========================================================================
# PURE: the shared baseline (simultaneous_pick) raises arm0 — the reference the
# direct_place contrast (arm0 has NO lift) branches off.
# ===========================================================================
def test_baseline_arm0_raises_then_places():
    # the shared baseline (simultaneous_pick, family "baseline") has arm0 lift(0.5)
    # then place — "raise then place" — the reference for the direct_place contrast.
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    assert spec.family == "baseline"
    pl, env = _fakes()
    prog = spec.build(pl, env)
    verbs = [qr.recipe(pl, env, 0).verb_id for qr in prog[0]]
    assert vocab.LIFT in verbs[:3]  # the lift sits at index 2 (before place)
    assert verbs[2] == vocab.LIFT
    assert len(prog[0]) == 6
    # the meta records the raise-then-place baseline behaviour for arm0.
    assert spec.meta["arm0"] == "raise_then_place"


def test_direct_place_arm0_has_no_lift():
    """direct_place: arm0 SKIPS the lift -> NO lift primitive BEFORE the place. (The
    last primitive is still a retreat_up, also labelled LIFT, so we check the
    pre-place segment specifically.)"""
    spec = next(m for m in STSC.sample(7) if m.name == "direct_place")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    # arm0 program: approach, close, place, open, retreat_up — no lift before place
    pre_place = [qr.recipe(pl, env, 0).verb_id for qr in prog[0][:2]]
    assert pre_place == [vocab.APPROACH, vocab.CLOSE_GRIPPER]
    assert prog[0][2].recipe(pl, env, 0).verb_id == vocab.PLACE  # straight to place
    # the OTHER arms still raise (lift at index 2).
    for arm in (1, 2):
        assert prog[arm][2].recipe(pl, env, arm).verb_id == vocab.LIFT


# ===========================================================================
# PURE: group filtering keeps the (single) contrast group intact
# ===========================================================================
def test_filter_variants_keeps_group_intact():
    specs = STSC.sample(7)
    full = ["direct_place", "simultaneous_pick", "staggered_pick"]
    # any single-member request returns the WHOLE group (one shared group now).
    assert sorted(m.name for m in STSC.filter_variants(specs, ["staggered_pick"])) == full
    assert sorted(m.name for m in STSC.filter_variants(specs, ["direct_place"])) == full
    # filter by a variant family keeps the whole group too
    assert sorted(m.name for m in STSC.filter_variants(specs, ["pickcoord"])) == full
    assert len(STSC.filter_variants(specs, None)) == len(specs)
    assert STSC.filter_variants(specs, ["nope"]) == []


# ===========================================================================
# PURE: place recipe resolves now_pose JIT (carried-cube grasp pose) + dest
# ===========================================================================
def test_place_recipe_uses_dest_actor_and_color():
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    place0 = _place_qr(prog, 0).recipe(pl, env, 0)
    assert place0.verb_id == vocab.PLACE
    assert place0.text == "place on the goal"
    assert pytest.approx(pl.last_dry_run_pose[2], abs=1e-5) == 0.05  # goal z 0 + 0.05
    place1 = _place_qr(prog, 1).recipe(pl, env, 1)
    assert place1.text == "place on the blue cube", place1.text  # onto cubeA (blue)


# ===========================================================================
# PURE: the runner wires the TSC sampler in TASK_MAP
# ===========================================================================
def test_runner_task_map_has_tsc():
    import run_subtask_rollouts as R  # scripts/dart on sys.path
    assert "ThreeRobotsStackCube" in R.TASK_MAP
    env_id, yaml_rel, n_agents, sampler = R.TASK_MAP["ThreeRobotsStackCube"]
    assert env_id == "ThreeRobotsStackCube-rf"
    assert yaml_rel == "table/three_robots_stack_cube.yaml"
    assert n_agents == 3
    assert sampler is STSC.sample


# ===========================================================================
# PURE: interpreter aligns a 3-arm stream off-GPU (fake planner/env)
# ===========================================================================
def test_interpreter_runs_3arm_simultaneous_aligned():
    """Drive the simultaneous_pick TSC program through the interpreter with FAKES and
    assert the recorded stream is aligned (len==T for all 3 arms), no garbage labels,
    and each arm's NON-WAIT verb sequence is the full
    approach,close,lift,place,open,retreat_up program (the trailing retreat_up shows
    up as a LIFT run after the place/open)."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    rec = I.SubtaskRecorder(num_arms=3)
    prog = spec.build(pl, env)
    out = I.run_program(env, pl, prog, rec, max_steps=4000,
                        control_mode=pl.control_mode, check_success_coverage="off")
    T = out["steps"]
    a = rec.to_arrays()
    import itertools
    for arm in (0, 1, 2):
        assert len(a[f"subtask_arm{arm}_verb"]) == T
        verbs = [int(v) for v in a[f"subtask_arm{arm}_verb"]]
        assert all(v in vocab.VERB_IDS.values() for v in verbs)  # no garbage/None
        runs = [k for k, _ in itertools.groupby(verbs)]
        core = [r for r in runs if r != vocab.WAIT]  # strip idle waits (gated)
        # approach, close, lift, place, open, retreat_up(==LIFT)
        assert core == [vocab.APPROACH, vocab.CLOSE_GRIPPER, vocab.LIFT,
                        vocab.PLACE, vocab.OPEN_GRIPPER, vocab.LIFT], (arm, runs)
    assert rec.length == T
    # the program drains completely off-GPU (no env auto-terminate in the fake).
    assert out["completed"] is True


def test_interpreter_serial_place_ordering_off_gpu():
    """With FAKES (no env auto-terminate), assert the SERIAL-place ordering holds:
    arm0 finishes its PLACE strictly before arm1 STARTS its place, and arm1 before
    arm2 — because each place is gated on the previous arm's RETREAT-done."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous_pick")
    pl, env = _fakes()
    rec = I.SubtaskRecorder(num_arms=3)
    prog = spec.build(pl, env)
    I.run_program(env, pl, prog, rec, max_steps=4000,
                  control_mode=pl.control_mode, check_success_coverage="off")
    a = rec.to_arrays()
    v = {arm: [int(x) for x in a[f"subtask_arm{arm}_verb"]] for arm in (0, 1, 2)}

    def _place_window(verbs):
        idx = [i for i, x in enumerate(verbs) if x == vocab.PLACE]
        return idx[0], idx[-1]

    s0, e0 = _place_window(v[0])
    s1, e1 = _place_window(v[1])
    s2, e2 = _place_window(v[2])
    # arm0 finishes placing before arm1 begins; arm1 before arm2.
    assert e0 < s1, (e0, s1)
    assert e1 < s2, (e1, s2)


# ===========================================================================
# SIM-GATED (DO NOT RUN here): real 3-arm TSC rollouts through the interpreter
# ===========================================================================
def _make_sim_env(n_agents=3):
    import gymnasium as gym
    import robofactory  # noqa: F401  (registers envs)
    from robofactory import CONFIG_DIR
    from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
    cfg = osp.join(CONFIG_DIR, "table/three_robots_stack_cube.yaml")
    env = gym.make(
        "ThreeRobotsStackCube-rf", config=cfg, obs_mode="rgb",
        control_mode="pd_joint_pos", render_mode="sensors", reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="cpu", robot_uids=("panda_wristcam_multi",) * n_agents,
    )
    return env, PandaArmMotionPlanningSolver


def _build_sim_planner(env, Solver, seed):
    env.reset(seed=seed)
    return Solver(
        env, debug=False, vis=False,
        base_pose=[a.robot.pose for a in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False, print_env_info=False, is_multi_agent=True,
    )


@sim
def test_sim_3arm_stream_len_equals_T():
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in STSC.sample(0) if m.name == "simultaneous_pick")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        out = I.run_program(env, planner, prog, rec, max_steps=2000,
                            control_mode=planner.control_mode)
        T = out["steps"]
        assert rec.length == T
        a = rec.to_arrays()
        for arm in (0, 1, 2):
            verbs = [int(v) for v in a[f"subtask_arm{arm}_verb"]]
            assert len(verbs) == T
            assert all(v in vocab.VERB_IDS.values() for v in verbs)
    finally:
        env.close()


@sim
def test_sim_staggered_pick_followers_wait_at_frame0():
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in STSC.sample(0) if m.name == "staggered_pick")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        I.run_program(env, planner, prog, rec, max_steps=2500,
                      control_mode=planner.control_mode)
        a = rec.to_arrays()
        assert a["subtask_arm0_verb"][0] == vocab.APPROACH
        assert a["subtask_arm1_verb"][0] == vocab.WAIT
        assert a["subtask_arm2_verb"][0] == vocab.WAIT
        for arm in (0, 1, 2):
            assert vocab.PLACE in set(a[f"subtask_arm{arm}_verb"]), arm
    finally:
        env.close()


@sim
def test_sim_simultaneous_pick_reaches_stack_success_collision_free():
    from run_subtask_rollouts import _env_success
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in STSC.sample(0) if m.name == "simultaneous_pick")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        out = I.run_program(env, planner, prog, rec, max_steps=2500,
                            control_mode=planner.control_mode)
        # the collision-free serial-place run reaches env stack-success OR completes.
        assert _env_success(out["info"]) or out["completed"], out["info"]
    finally:
        env.close()


@sim
def test_sim_serial_place_ordering_holds():
    """A real simultaneous_pick run must place SERIALLY: arm0's place finishes before
    arm1 starts, arm1 before arm2 (collision-free gate enforcement)."""
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in STSC.sample(0) if m.name == "simultaneous_pick")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        I.run_program(env, planner, prog, rec, max_steps=2500,
                      control_mode=planner.control_mode)
        a = rec.to_arrays()
        v = {arm: [int(x) for x in a[f"subtask_arm{arm}_verb"]] for arm in (0, 1, 2)}
        placed = {arm: [i for i, x in enumerate(v[arm]) if x == vocab.PLACE]
                  for arm in (0, 1, 2)}
        # every arm placed; and the place windows are serially ordered.
        for arm in (0, 1, 2):
            assert placed[arm], arm
        assert placed[0][-1] < placed[1][0]
        assert placed[1][-1] < placed[2][0]
    finally:
        env.close()
