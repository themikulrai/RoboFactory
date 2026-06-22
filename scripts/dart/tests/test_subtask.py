"""Tests for the subtask-conditioned data-gen pipeline (Phase-1 harness).

PURE tests (no sim, always run): vocab render over every (verb x target) combo;
color-map-assert vs yaml + a negative drift case; primitive tick shape/length;
approach-stop has ZERO gripper-close ticks; gripper ramp monotonic+clamped; hold
repeats the frozen qpos byte-for-byte; the per-arm dry_run guard (a pose-LIST
under dry_run returns ONE arm's plan — regression-locked with a light stub); and
the contrastive scenario SAMPLER (matched pairs, seed-derived RNG independent of
env reset, unique variant ids, group-level filtering, atomic grouping).

SIM tests (@pytest.mark.sim, skipped unless DART_RUN_SIM=1; DO NOT RUN on the
login node — SAPIEN renders dark; needs a GPU compute node): drive a real short
LiftBarrier rollout through subtask_interpreter.run_program and verify the live
label stream (len==T, contiguous runs, no None, off-by-one), idle-arm wait+frozen
qpos, barrier gating, success checks firing, the contrastive frame-0-identical /
seam-divergence / stop-never-closes properties, atomic matched-pair drop, the
liar-label drop, and DART/object-perturb compatibility (jittered target still
reached with unchanged text/target_id; boundary_hook arm-noise NOT recorded).

Run PURE:   /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
                scripts/dart/tests/test_subtask.py -k "not sim" -q
Run ALL (GPU node):  DART_RUN_SIM=1 pytest scripts/dart/tests/test_subtask.py -q
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
from robofactory.planner import subtask_scenarios as S  # noqa: E402
from robofactory.planner.subtask_primitives import OPEN, CLOSED  # noqa: E402

RUN_SIM = os.environ.get("DART_RUN_SIM", "0") == "1"
sim = pytest.mark.skipif(not RUN_SIM, reason="set DART_RUN_SIM=1 to run sim tests")

# Panda 7-revolute joint limits (for the SIM action-bound sanity check).
PANDA_QLIMITS_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_QLIMITS_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


# ===========================================================================
# Light off-GPU fakes (planner + env) — no sapien, no gym.
# ===========================================================================
class FakeRobot:
    def __init__(self, q7):
        self._q = np.concatenate([np.asarray(q7, np.float32), [0.04, 0.04]]).astype(np.float32)

    def get_qpos(self):
        return self._q[None, :]  # (1, 9) numpy


class FakeActor:
    def __init__(self, z=0.6):
        self.z = z


class FakePlanner:
    """Canned-waypoint planner. A pose LIST under dry_run is REJECTED (the trap)."""

    def __init__(self, num_arms=2, control_mode="pd_joint_pos", n_waypoints=5):
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

    def get_grasp_pose_w_labeled_direction(self, actor, actor_data, pre_dis=0.0, id=0):
        return np.array([id * 0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)

    def get_grasp_pose_from_obb(self, actor, agent_id=0):
        return np.array([0.0, 0.1 * agent_id, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)

    def get_grasp_pose_for_stack(self, now_pose, target_actor, height_offset=0.05):
        out = np.asarray(now_pose, np.float32).copy()
        out[2] = float(target_actor.z) + height_offset
        return out


class FakeEnv:
    def __init__(self):
        self.barrier = FakeActor()
        self.goal_region = FakeActor(z=0.0)
        self.cubeA = FakeActor()
        self.cubeB = FakeActor()
        self.cubeC = FakeActor()
        self.annotation_data = {"barrier": {"scale": [0.6, 0.6, 0.2],
                                            "contact_points_pose": [np.eye(4)] * 4}}
        self.unwrapped = self
        self.n_steps = 0
        self.step_log = []

    def step(self, action_dict):
        self.n_steps += 1
        self.step_log.append({k: np.asarray(v).copy() for k, v in action_dict.items()})
        return {}, 0.0, False, False, {"success": False}


# ===========================================================================
# PURE: vocab render over EVERY (verb x target) combo
# ===========================================================================
def test_vocab_render_every_lb_combo():
    expect = {
        (vocab.WAIT, 0): "wait",
        (vocab.APPROACH, 1): "grasp the left end",
        (vocab.APPROACH, 2): "grasp the right end",
        (vocab.OPEN_GRIPPER, 0): "open the gripper",
        (vocab.CLOSE_GRIPPER, 0): "close the gripper",
        (vocab.LIFT, 1): "lift the left end",
        (vocab.LIFT, 2): "lift the right end",
        (vocab.PLACE, 1): "place the left end",
        (vocab.PLACE, 2): "place the right end",
    }
    for (v, t), s in expect.items():
        assert vocab.render(v, t, "LiftBarrier") == s, (v, t)


def test_vocab_render_every_tsc_combo():
    expect = {
        (vocab.WAIT, 0): "wait",
        (vocab.OPEN_GRIPPER, 0): "open the gripper",
        (vocab.CLOSE_GRIPPER, 0): "close the gripper",
        (vocab.APPROACH, vocab.TSC_TARGETS["blue"]): "approach the blue cube",
        (vocab.APPROACH, vocab.TSC_TARGETS["green"]): "approach the green cube",
        (vocab.APPROACH, vocab.TSC_TARGETS["red"]): "approach the red cube",
        (vocab.LIFT, vocab.TSC_TARGETS["blue"]): "lift the blue cube",
        (vocab.LIFT, vocab.TSC_TARGETS["green"]): "lift the green cube",
        (vocab.LIFT, vocab.TSC_TARGETS["red"]): "lift the red cube",
        (vocab.PLACE, vocab.TSC_TARGETS["goal_region"]): "place on the goal",
        (vocab.PLACE, vocab.TSC_TARGETS["on_blue"]): "place on the blue cube",
        (vocab.PLACE, vocab.TSC_TARGETS["on_green"]): "place on the green cube",
        (vocab.PLACE, vocab.TSC_TARGETS["on_red"]): "place on the red cube",
    }
    for (v, t), s in expect.items():
        assert vocab.render(v, t, "ThreeRobotsStackCube") == s, (v, t)


def test_vocab_render_rejects_unknown_task_and_verb():
    with pytest.raises(KeyError):
        vocab.render(vocab.APPROACH, 1, "NoSuchTask")
    with pytest.raises(KeyError):
        vocab.render(999, 1, "LiftBarrier")


def test_target_roundtrip():
    assert vocab.target_name("LiftBarrier", 2) == "right_end"
    assert vocab.target_id("ThreeRobotsStackCube", "green") == 2
    assert vocab.TSC_COLOR_CUBE["green"] == "cubeB"
    assert vocab.LB_TARGET_CONTACT_ID[vocab.LB_TARGETS["left_end"]] == 1
    assert vocab.LB_TARGET_CONTACT_ID[vocab.LB_TARGETS["right_end"]] == 2


# ===========================================================================
# PURE: color-map assert vs yaml (+ negative drift case)
# ===========================================================================
def test_color_map_matches_yaml():
    out = vocab.assert_cube_color_map()
    assert out == {"cubeA": "blue", "cubeB": "green", "cubeC": "red"}


def test_color_map_negative_drift(tmp_path):
    """A yaml whose cube colors disagree with TSC_CUBE_COLOR must raise loudly."""
    bad = tmp_path / "bad.yaml"
    # cubeA labelled blue in the map, but here painted RED -> dominant-channel drift
    bad.write_text(
        "scene:\n"
        "  primitives:\n"
        "    - name: cubeA\n"
        "      params: {color: [1.0, 0.0, 0.0, 1.0]}\n"
        "    - name: cubeB\n"
        "      params: {color: [0.0, 1.0, 0.0, 1.0]}\n"
        "    - name: cubeC\n"
        "      params: {color: [1.0, 0.0, 0.0, 1.0]}\n"
    )
    with pytest.raises(AssertionError):
        vocab.assert_cube_color_map(yaml_path=str(bad))


def test_color_map_negative_missing_cube(tmp_path):
    bad = tmp_path / "missing.yaml"
    bad.write_text(
        "scene:\n"
        "  primitives:\n"
        "    - name: cubeA\n"
        "      params: {color: [0.04705882, 0.16470588, 0.62745098, 1.0]}\n"
    )
    with pytest.raises(AssertionError):
        vocab.assert_cube_color_map(yaml_path=str(bad))


# ===========================================================================
# PURE: primitive tick shape / length / content
# ===========================================================================
def test_approach_stop_has_zero_close_ticks():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.approach(pl, env, arm=0, target_id=vocab.LB_TARGETS["left_end"],
                      task="LiftBarrier")
    assert prim.verb_id == vocab.APPROACH
    assert prim.n_ticks == pl.n_waypoints > 0
    # every tick (qpos7, grip): qpos length 7, grip == OPEN (no close)
    for q, g in prim.ticks:
        assert np.asarray(q).shape == (7,)
        assert g == OPEN
    assert prim.text == "grasp the left end"


def test_gripper_close_ramp_monotonic_clamped_frozen():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.close_gripper(pl, env, arm=0, task="LiftBarrier", start_grip=OPEN)
    assert prim.n_ticks == P.GRIPPER_RAMP_TICKS
    grips = [g for _, g in prim.ticks]
    assert all(grips[k + 1] <= grips[k] + 1e-9 for k in range(len(grips) - 1)), grips
    assert min(grips) >= CLOSED - 1e-9
    assert grips[-1] == pytest.approx(CLOSED, abs=1e-6)
    q0 = prim.ticks[0][0]
    assert all(np.array_equal(q, q0) for q, _ in prim.ticks)


def test_gripper_open_ramp_monotonic_clamped():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.open_gripper(pl, env, arm=1, task="ThreeRobotsStackCube", start_grip=CLOSED)
    grips = [g for _, g in prim.ticks]
    assert all(grips[k + 1] >= grips[k] - 1e-9 for k in range(len(grips) - 1)), grips
    assert max(grips) <= OPEN + 1e-9
    assert grips[-1] == pytest.approx(OPEN, abs=1e-6)


def test_hold_repeats_frozen_qpos_byte_for_byte():
    pl, env = FakePlanner(), FakeEnv()
    qpos = np.array([0.5, -0.2, 0.1, -1.0, 0.0, 1.5, 0.7], np.float32)
    prim = P.hold(pl, env, arm=0, n=7, task="LiftBarrier", qpos=qpos, grip=CLOSED)
    assert prim.verb_id == vocab.WAIT
    assert prim.n_ticks == 7
    for q, g in prim.ticks:
        assert np.array_equal(q, qpos)  # byte-for-byte
        assert g == CLOSED
    assert prim.text == "wait"


def test_lift_replans_z_plus_dz():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.lift(pl, env, arm=0, target_id=vocab.LB_TARGETS["right_end"],
                  task="LiftBarrier", dz=0.2, grip=CLOSED)
    assert prim.verb_id == vocab.LIFT
    assert pytest.approx(pl.last_dry_run_pose[2], abs=1e-5) == 0.7  # 0.5 + 0.2
    assert all(g == CLOSED for _, g in prim.ticks)


def test_place_uses_dest_actor_pose():
    pl, env = FakePlanner(), FakeEnv()
    now = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)
    prim = P.place(pl, env, arm=0, target_id=vocab.TSC_TARGETS["goal_region"],
                   task="ThreeRobotsStackCube", now_pose=now,
                   dest_actor_name="goal_region", height_offset=0.05)
    assert prim.verb_id == vocab.PLACE
    assert pytest.approx(pl.last_dry_run_pose[2], abs=1e-5) == 0.05
    assert prim.text == "place on the goal"


# ===========================================================================
# PURE: per-arm dry_run guard (the silent single-arm-drop trap) — regression lock
# ===========================================================================
def test_dry_run_pose_list_rejected_by_helper():
    """subtask_primitives._plan_to_pose refuses a pose LIST (our hard guard)."""
    pl = FakePlanner()
    with pytest.raises(ValueError):
        P._plan_to_pose(pl, 0, [np.zeros(7), np.ones(7)])


def test_dry_run_pose_list_returns_single_plan_stub():
    """Regression-lock the UPSTREAM trap itself: a stub mirroring motionplanner's
    dry_run early-return shows a pose LIST yields only the FIRST arm's plan.

    This documents WHY subtask_primitives plans strictly per-arm. We mimic
    motionplanner.move_to_pose_with_screw lines 151/161/185-186: it normalizes a
    single pose to a 1-list, loops over poses, and `return result` INSIDE the loop
    on the first iteration when dry_run -> only pose[0]'s plan ever comes back.
    """
    class _StubPlanner:
        def __init__(self):
            self.planned = []

        def move_to_pose_with_screw(self, pose, dry_run=False, move_id=0, **kw):
            pose = [pose] if not isinstance(pose, list) else pose
            move_id = [move_id] if not isinstance(move_id, list) else move_id
            for idx in range(len(pose)):
                # record what WOULD be planned, but dry_run returns on the FIRST
                result = {"position": np.full((3, 7), float(move_id[idx]), np.float32),
                          "for_move_id": move_id[idx]}
                self.planned.append(move_id[idx])
                if dry_run:
                    return result  # <-- the trap: returns inside the loop
            return result

    stub = _StubPlanner()
    res = stub.move_to_pose_with_screw([np.zeros(7), np.ones(7)],
                                       move_id=[0, 1], dry_run=True)
    # only arm-0's plan came back; arm-1 was silently dropped
    assert res["for_move_id"] == 0
    assert stub.planned == [0]


# ===========================================================================
# PURE: contrastive scenario sampler (all-complete-rollout coordination contrast)
#
# New family set (2 matched pairs, 4 variants). Every pair is {simultaneous, X}
# from ONE seed; the contrast is COORDINATION (who waits/leads), NOT truncation —
# every variant ends with BOTH arms lifting (a complete rollout). seq_lift was
# DROPPED: a rigid barrier needs both ends to rise together, so a lift-timing-only
# contrast is a physical null (frame-0 identical to simultaneous).
# ===========================================================================
def test_sampler_emits_two_matched_pairs():
    specs = S.sample(7)
    groups = S.group_specs(specs)
    assert len(specs) == 4
    assert len(groups) == 2
    for gid, members in groups.items():
        assert len(members) == 2, (gid, [m.name for m in members])
        # both members share family + contrast_group_id
        assert members[0].family == members[1].family
        assert members[0].contrast_group_id == members[1].contrast_group_id == gid
        # the simultaneous baseline is in EVERY pair (clean A/B)
        assert "simultaneous" in {m.name for m in members}, [m.name for m in members]
    families = {m.family for m in specs}
    assert families == {"stagger_a", "stagger_b"}
    names = {m.name for m in specs}
    assert names == {"simultaneous", "stagger_a_leads", "stagger_b_leads"}
    # the OLD families/variants are GONE (incl. the dropped seq_lift / sequential_lift)
    assert not (names & {"approach_stop", "approach_grasp_lift", "arms_LR",
                         "arms_RL", "grasp_and_hold", "grasp_and_release",
                         "sequential", "sequential_lift"})
    assert "seq_lift" not in families


def test_sampler_variant_ids_unique_and_reproducible():
    specs = S.sample(7)
    assert len({m.variant_id for m in specs}) == len(specs)
    # reproducible structural draws (names + meta) for the same seed
    again = S.sample(7)
    assert [m.name for m in specs] == [m.name for m in again]
    assert [m.meta for m in specs] == [m.meta for m in again]


def test_sampler_rng_independent_of_env_reset():
    """The sampler must use its OWN RNG (no global numpy state) so contrastive
    variants from a seed share the env reset (frame-0 identical). We assert
    sample() does not consume the GLOBAL numpy RNG."""
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    _ = S.sample(123)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after), "sample() perturbed the GLOBAL numpy RNG"


def test_sampler_arm_assignment_fixed_no_swap():
    """No arm-swap family: arm0 always takes the LEFT end, arm1 the RIGHT end.

    We inspect every variant's approach target_ids per arm (built off-GPU)."""
    specs = S.sample(7)
    for spec in specs:
        pl, env = FakePlanner(num_arms=2), FakeEnv()
        prog = spec.build(pl, env)
        # arm0's first approach targets left_end; arm1's first approach right_end
        a0 = next(qr.recipe(pl, env, 0) for qr in prog[0]
                  if qr.recipe(pl, env, 0).verb_id == vocab.APPROACH)
        a1 = next(qr.recipe(pl, env, 1) for qr in prog[1]
                  if qr.recipe(pl, env, 1).verb_id == vocab.APPROACH)
        assert a0.target_id == vocab.LB_TARGETS["left_end"], spec.name
        assert a1.target_id == vocab.LB_TARGETS["right_end"], spec.name


def test_filter_variants_keeps_groups_intact():
    specs = S.sample(7)
    # ask for ONE member by name; the WHOLE matched group must come back
    out = S.filter_variants(specs, ["stagger_a_leads"])
    names = sorted(m.name for m in out)
    assert names == ["simultaneous", "stagger_a_leads"]
    # ask by family
    out2 = S.filter_variants(specs, ["stagger_b"])
    assert sorted(m.name for m in out2) == ["simultaneous", "stagger_b_leads"]
    # None -> everything
    assert len(S.filter_variants(specs, None)) == len(specs)
    # unknown -> nothing
    assert S.filter_variants(specs, ["nope"]) == []


def _prog_verbs(prog, planner, env):
    """Build every recipe in a program (against the fakes) and collect verb ids.

    The builders return {arm:[QueuedRecipe]} (structure only); the per-primitive
    planning is deferred to each recipe. We invoke recipe(planner, env, arm) here
    to inspect the verbs the program will emit. Safe off-GPU (fake planner)."""
    verbs = set()
    for arm, queue in prog.items():
        for qr in queue:
            verbs.add(qr.recipe(planner, env, arm).verb_id)
    return verbs


def test_sampler_builders_callable_against_fakes():
    """The builders must produce a {arm:[QueuedRecipe]} program when given a
    live (planner, env). We drive them (and build their recipes) with the FAKE
    planner/env (no sim) so the structure (gating) is validated off-GPU."""
    specs = S.sample(7)
    for spec in specs:
        pl, env = FakePlanner(num_arms=2), FakeEnv()
        prog = spec.build(pl, env)
        assert set(prog.keys()) == {0, 1}, spec.name
        for arm, queue in prog.items():
            assert len(queue) >= 1
            for qr in queue:
                assert isinstance(qr, I.QueuedRecipe)
                assert callable(qr.recipe)
        verbs = _prog_verbs(prog, pl, env)
        # EVERY variant grasps (approach -> close -> lift on both arms)
        assert vocab.CLOSE_GRIPPER in verbs, spec.name
        assert vocab.LIFT in verbs, spec.name


def test_sampler_every_variant_both_arms_lift():
    """EVERY variant must end with BOTH arms on a LIFT (complete rollout; no
    permanent-hold / truncated variant). Uses the pure structural guard."""
    specs = S.sample(7)
    for spec in specs:
        pl, env = FakePlanner(num_arms=2), FakeEnv()
        prog = spec.build(pl, env)
        # the guard raises if any arm's last primitive is not a LIFT
        S.assert_both_arms_lift(prog, pl, env)
        # and explicitly: both arms' last primitive is a lift
        for arm in (0, 1):
            last = prog[arm][-1].recipe(pl, env, arm)
            assert last.verb_id == vocab.LIFT, (spec.name, arm, last.name)


def test_sampler_no_deadlock_one_arm_ungated_first():
    """DEADLOCK GUARD: in EVERY variant at least one arm's FIRST primitive is
    ungated (wait_for=None), so the program can always start; never gate both
    arms' first action on each other."""
    specs = S.sample(7)
    for spec in specs:
        pl, env = FakePlanner(num_arms=2), FakeEnv()
        prog = spec.build(pl, env)
        # the guard raises on a deadlock; also assert the acyclicity directly
        S.assert_no_deadlock(prog)
        first_gates = S.gate_graph(prog)
        assert any(g is None for g in first_gates.values()), (spec.name, first_gates)


def test_sampler_simultaneous_has_no_gates():
    """The baseline simultaneous variant carries NO wait_for gates anywhere."""
    specs = S.sample(7)
    sim_ = next(m for m in specs if m.name == "simultaneous")
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    prog = sim_.build(pl, env)
    gates = [qr.wait_for for q in prog.values() for qr in q if qr.wait_for]
    assert gates == []


def test_sampler_stagger_a_follower_waits_at_frame0():
    """stagger_a_leads: arm0 (lead) starts UNGATED; arm1 (follow) first primitive
    is GATED on arm0 -> arm1 is wait at frame 0 vs approach in simultaneous."""
    specs = S.sample(7)
    spec = next(m for m in specs if m.name == "stagger_a_leads")
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    prog = spec.build(pl, env)
    first_gates = S.gate_graph(prog)
    assert first_gates[0] is None       # leader (arm0) ungated first
    assert first_gates[1] == 0          # follower (arm1) waits on arm0
    # both lifts in the stagger are gated (so ends rise together)
    lift_gates = [qr.wait_for for arm, q in prog.items() for qr in q
                  if qr.recipe(pl, env, arm).verb_id == vocab.LIFT and qr.wait_for]
    assert len(lift_gates) == 2


def test_sampler_stagger_b_is_mirror():
    """stagger_b_leads: arm1 leads ungated, arm0 follows (the mirror of a)."""
    specs = S.sample(7)
    spec = next(m for m in specs if m.name == "stagger_b_leads")
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    prog = spec.build(pl, env)
    first_gates = S.gate_graph(prog)
    assert first_gates[1] is None       # leader (arm1) ungated first
    assert first_gates[0] == 1          # follower (arm0) waits on arm1


class _FakeArmState:
    """Minimal stand-in for subtask_interpreter._ArmState exposing only ``.qi``
    (the index of the arm's CURRENT primitive). Lets us probe the follower's
    wait gate deterministically OFF-GPU, with no sim and no _ArmState import."""

    def __init__(self, qi):
        self.qi = int(qi)


def test_stagger_follower_wait_gate_is_deterministic_program_progress():
    """LOCK the deterministic-wait off-GPU: the follower's FIRST QueuedRecipe has a
    wait_for whose predicate is FALSE while the lead arm sits at qi=0 (not grasped)
    and TRUE once lead.qi >= GRASP_IDX (finished approach+close = grasped). This is
    the fix for the flaky physics gate — it depends ONLY on program progress, never
    on a physics is_grasping probe.

    We check BOTH stagger variants (a leads -> arm0 leads, follower arm1; b leads ->
    arm1 leads, follower arm0)."""
    specs = S.sample(7)
    assert S.GRASP_IDX == 2  # the program-position index of the lift = grasp done
    for name, lead, follow in (("stagger_a_leads", 0, 1),
                               ("stagger_b_leads", 1, 0)):
        spec = next(m for m in specs if m.name == name)
        pl, env = FakePlanner(num_arms=2), FakeEnv()
        prog = spec.build(pl, env)
        # the follower's FIRST primitive (approach) must carry a wait_for on the lead
        gate = prog[follow][0].wait_for
        assert gate is not None, name
        other_arm, predicate = gate
        assert other_arm == lead, (name, other_arm)
        # construct fake interpreter state dicts exposing the lead arm's .qi
        def _state(lead_qi):
            return {"arms": {lead: _FakeArmState(lead_qi),
                             follow: _FakeArmState(0)}}
        # lead not started / not grasped -> follower WAITS (gate closed)
        assert predicate(env, _state(0)) is False, name
        assert predicate(env, _state(1)) is False, name  # only finished approach
        # lead finished approach+close (qi >= GRASP_IDX) -> gate OPENS
        assert predicate(env, _state(S.GRASP_IDX)) is True, name
        assert predicate(env, _state(S.GRASP_IDX + 1)) is True, name
        # robust to a missing/empty arms map -> gate stays closed (no crash)
        assert predicate(env, {}) is False, name
        assert predicate(env, {"arms": {}}) is False, name


# ===========================================================================
# PURE: keep-criterion helpers (_env_success + _member_passes) in the driver.
#
# The LiftBarrier env AUTO-TERMINATES on the barrier lift (~step 69) BEFORE the
# per-arm primitive queues empty, so a SUCCESSFUL lift returns completed=False.
# The keep predicate must therefore accept env-success OR completed (not completed
# alone, which dropped every real success). These tests are pure (fake out dicts).
# ===========================================================================
def _import_driver():
    """Import the rollout driver (gym + robofactory at module top; import-only,
    no gym.make -> no GPU). Imported lazily so collection of the pure suite doesn't
    pay for it unless these tests run."""
    import run_subtask_rollouts as R  # noqa: F401  (scripts/dart on sys.path)
    return R


def test_env_success_coerces_various_info_shapes():
    R = _import_driver()
    # None / missing / empty info -> False
    assert R._env_success(None) is False
    assert R._env_success({}) is False
    assert R._env_success({"other": 1}) is False
    assert R._env_success({"success": None}) is False
    # python bool / int
    assert R._env_success({"success": True}) is True
    assert R._env_success({"success": False}) is False
    assert R._env_success({"success": 1}) is True
    assert R._env_success({"success": 0}) is False
    # numpy scalar / array (batched env returns shape (1,))
    assert R._env_success({"success": np.array([True])}) is True
    assert R._env_success({"success": np.array([False])}) is False
    assert R._env_success({"success": np.asarray(True)}) is True

    # torch tensor (duck-typed via .detach) if torch is available
    try:
        import torch
        assert R._env_success({"success": torch.tensor([True])}) is True
        assert R._env_success({"success": torch.tensor([False])}) is False
    except ImportError:
        pass


def test_member_passes_keep_predicate_matrix():
    R = _import_driver()
    # 1) success-TERMINATION: env auto-terminated on the lift (completed False) but
    #    info.success True and approach checks passed -> KEEP.
    assert R._member_passes(
        {"all_success": True, "info": {"success": True}, "completed": False}) is True
    # 2) outright failure: a success_check returned False -> DROP regardless.
    assert R._member_passes(
        {"all_success": False, "info": {"success": True}, "completed": True}) is False
    assert R._member_passes(
        {"all_success": False, "info": {"success": False}, "completed": False}) is False
    # 3) gating delayed the lift past queue-drain without env terminating: queues
    #    completed and the approach checks passed -> KEEP (via completed).
    assert R._member_passes(
        {"all_success": True, "info": {}, "completed": True}) is True
    # 4) FAILED LIFT: program completed, lift's check_barrier_lifted RAN and was
    #    False -> all_success False, no env-success -> DROP.
    assert R._member_passes(
        {"all_success": False, "info": {"success": False}, "completed": True}) is False
    # 5) clean success with BOTH completed and env-success -> KEEP.
    assert R._member_passes(
        {"all_success": True, "info": {"success": True}, "completed": True}) is True
    # 6) all_success True but neither env-success nor completed (e.g. truncated mid
    #    flight before the lift confirmed) -> DROP (not a confirmed success).
    assert R._member_passes(
        {"all_success": True, "info": {"success": False}, "completed": False}) is False
    # 7) None out (plan/rollout raised) -> DROP.
    assert R._member_passes(None) is False


def test_group_all_passed_uses_member_predicate():
    R = _import_driver()
    rec = object()  # _group_all_passed only checks rec is not None
    succ_term = {"all_success": True, "info": {"success": True}, "completed": False}
    approach = {"all_success": True, "info": {}, "completed": True}
    failed = {"all_success": False, "info": {"success": False}, "completed": True}
    # a matched pair where BOTH pass (one via env-success, one via completed)
    assert R._group_all_passed([(rec, succ_term), (rec, approach)]) is True
    # one member fails its lift -> the WHOLE group drops (atomic)
    assert R._group_all_passed([(rec, succ_term), (rec, failed)]) is False
    # a None recorder (plan raised) -> drop
    assert R._group_all_passed([(None, None), (rec, approach)]) is False


# ===========================================================================
# SIM-GATED (DO NOT RUN here): real LiftBarrier rollouts through the interpreter
# ===========================================================================
def _make_sim_env(task="LiftBarrier", n_agents=2):
    import gymnasium as gym
    import robofactory  # noqa: F401  (registers envs)
    from robofactory import CONFIG_DIR
    from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
    cfg = osp.join(CONFIG_DIR, "table/lift_barrier.yaml")
    env = gym.make(
        "LiftBarrier-rf", config=cfg, obs_mode="rgb", control_mode="pd_joint_pos",
        render_mode="sensors", reward_mode="dense",
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
def test_sim_label_stream_len_equals_T_and_contiguous_no_none():
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in S.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=2)
        prog = spec.build(planner, env)
        out = I.run_program(env, planner, prog, rec, max_steps=600,
                            control_mode=planner.control_mode)
        T = out["steps"]
        assert rec.length == T
        a = rec.to_arrays()
        # the simultaneous program (both arms, ungated): approach -> close -> lift.
        # Collapsing consecutive duplicates yields that verb order, possibly with a
        # leading/trailing WAIT run (an arm idling while the other catches up). No
        # None / garbage verbs anywhere.
        import itertools
        for arm in (0, 1):
            verbs = [int(v) for v in a[f"subtask_arm{arm}_verb"]]
            assert len(verbs) == T
            assert all(v in vocab.VERB_IDS.values() for v in verbs)  # no None/garbage
            runs = [k for k, _ in itertools.groupby(verbs)]
            # strip leading/trailing idle-WAIT runs (legitimate cross-arm idling).
            if runs and runs[0] == vocab.WAIT:
                runs = runs[1:]
            if runs and runs[-1] == vocab.WAIT:
                runs = runs[:-1]
            assert runs == [vocab.APPROACH, vocab.CLOSE_GRIPPER, vocab.LIFT], runs
    finally:
        env.close()


@sim
def test_sim_offbyone_label_drives_action():
    """label[t] is the subtask that DROVE action[t]: the first non-wait label
    appears on the SAME step the first planned waypoint is sent."""
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in S.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=2)
        prog = spec.build(planner, env)
        # arm0 starts on approach immediately (no gate) -> label[0] == APPROACH
        out = I.run_program(env, planner, prog, rec, max_steps=600,
                            control_mode=planner.control_mode)
        a = rec.to_arrays()
        assert a["subtask_arm0_verb"][0] == vocab.APPROACH
        assert int(out["steps"]) == int(rec.length)
    finally:
        env.close()


@sim
def test_sim_idle_arm_wait_and_frozen():
    env, Solver = _make_sim_env()
    try:
        planner = _build_sim_planner(env, Solver, 0)
        rec = I.SubtaskRecorder(num_arms=2)
        prog = {
            0: [I.QueuedRecipe(lambda pl, e, arm: P.approach(
                pl, e, arm=arm, target_id=vocab.LB_TARGETS["left_end"],
                task="LiftBarrier"))],
            1: [],  # idle
        }
        I.run_program(env, planner, prog, rec, max_steps=200,
                      control_mode=planner.control_mode)
        a = rec.to_arrays()
        assert set(a["subtask_arm1_verb"]) == {vocab.WAIT}
    finally:
        env.close()


@sim
def test_sim_barrier_gating_blocks_until_grasp():
    """stagger_a_leads: the follower (arm1) is GATED on the leader (arm0) finishing
    its grasp (program progress qi >= GRASP_IDX). arm1 holds (wait) at frame 0 and
    only begins its approach after arm0 grasps; both arms ultimately lift."""
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in S.sample(0) if m.name == "stagger_a_leads")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=2)
        prog = spec.build(planner, env)
        I.run_program(env, planner, prog, rec, max_steps=800,
                      control_mode=planner.control_mode)
        a = rec.to_arrays()
        # arm1 (follower) WAITS at frame 0 (gate closed: arm0 not yet grasped).
        assert a["subtask_arm1_verb"][0] == vocab.WAIT
        # the leader (arm0) starts its approach immediately (ungated).
        assert a["subtask_arm0_verb"][0] == vocab.APPROACH
        # both arms ultimately lift (complete rollout, ends rise together).
        assert vocab.LIFT in set(a["subtask_arm1_verb"])
        assert vocab.LIFT in set(a["subtask_arm0_verb"])
    finally:
        env.close()


@sim
def test_sim_success_checks_fire():
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in S.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=2)
        prog = spec.build(planner, env)
        out = I.run_program(env, planner, prog, rec, max_steps=800,
                            control_mode=planner.control_mode)
        # at least the close_gripper / lift checks evaluated to a real bool
        checked = [v for arm in out["success"].values() for v in arm.values()
                   if v is not None]
        assert len(checked) > 0
    finally:
        env.close()


@sim
def test_sim_contrastive_frame0_identical_coordination_diverges():
    """simultaneous vs stagger_a_leads from the SAME seed: frame-0 obs identical,
    but the COORDINATION diverges — in stagger_a_leads the follower (arm1) is
    WAITING at frame 0 (gated on arm0's grasp, a DETERMINISTIC program-progress
    gate) whereas in simultaneous arm1 approaches immediately. BOTH variants reach
    env-success (a complete both-ends-lifted rollout). The deterministic gate makes
    the frame-0 WAIT reproducible (the old physics is_grasping gate opened early and
    arm1 often did NOT wait)."""
    from run_subtask_rollouts import _env_success
    env, Solver = _make_sim_env()
    try:
        specs = {m.name: m for m in S.sample(0)
                 if m.name in ("simultaneous", "stagger_a_leads")}
        recs = {}
        outs = {}
        first_qpos = {}
        for name, spec in specs.items():
            planner = _build_sim_planner(env, Solver, spec.seed)
            # frame-0 arm0 qpos right after reset (before any step)
            q0 = planner.robot[0].get_qpos()[0, :7]
            first_qpos[name] = (q0.cpu().numpy() if hasattr(q0, "cpu")
                                else np.asarray(q0)).copy()
            rec = I.SubtaskRecorder(num_arms=2)
            prog = spec.build(planner, env)
            outs[name] = I.run_program(env, planner, prog, rec, max_steps=800,
                                       control_mode=planner.control_mode)
            recs[name] = rec.to_arrays()
        # frame-0 identical (same reset seed, sampler RNG independent)
        assert np.allclose(first_qpos["simultaneous"],
                           first_qpos["stagger_a_leads"], atol=1e-5)
        # COORDINATION contrast at frame 0: simultaneous -> arm1 approaches; stagger
        # -> arm1 waits (DETERMINISTICALLY gated on arm0's program progress).
        assert recs["simultaneous"]["subtask_arm1_verb"][0] == vocab.APPROACH
        assert recs["stagger_a_leads"]["subtask_arm1_verb"][0] == vocab.WAIT
        # BOTH variants complete AND reach env-success (the barrier actually lifts).
        for name in ("simultaneous", "stagger_a_leads"):
            assert vocab.LIFT in set(recs[name]["subtask_arm0_verb"]), name
            assert vocab.LIFT in set(recs[name]["subtask_arm1_verb"]), name
            assert _env_success(outs[name]["info"]), name
    finally:
        env.close()


@sim
def test_sim_matched_pair_atomic_drop_and_liar_label():
    """A FAILED LIFT (the lift's check_barrier_lifted returns False) makes a variant
    fail; the rollout driver must drop the WHOLE contrast group, and the liar label
    is never written as a kept episode.

    NOTE: reframed from the old "forced-failed close grasp" case. The premature
    close is_grasping check was REMOVED (is_grasping does not register right after
    the close ramp; it false-failed every success). A close that did not grasp is
    now caught DOWNSTREAM: no grasp -> the lift does not raise the barrier -> the
    lift's check_barrier_lifted (which we force False here to simulate that) fails
    -> all_success False -> the group drops. We poison the LIFT, not the close.
    """
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in S.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=2)
        prog = spec.build(planner, env)
        # poison BOTH arms' LIFT success_check -> a failed lift. Recipes are built
        # JIT, so WRAP the recipe: build the real primitive, then force the lift's
        # success_check to False (lift builds verb LIFT). Poisoning both arms makes
        # all_success False whenever the lift checks RUN, regardless of which arm
        # finished its lift first.
        def _poison(orig_recipe):
            def wrapped(pl, e, arm):
                prim = orig_recipe(pl, e, arm)
                if prim.verb_id == vocab.LIFT:
                    prim.success_check = lambda e, a, p: False
                return prim
            return wrapped
        for arm in (0, 1):
            prog[arm] = [I.QueuedRecipe(_poison(qr.recipe), wait_for=qr.wait_for)
                         for qr in prog[arm]]
        out = I.run_program(env, planner, prog, rec, max_steps=800,
                            control_mode=planner.control_mode)
        from run_subtask_rollouts import _env_success, _member_passes
        # CONTRACT: a failed lift drops the group. If the lift checks RAN (the env
        # did not auto-terminate on a real barrier lift first), all_success is False
        # and the member is dropped. If the env DID auto-terminate (a real lift),
        # env_success is True -> this would NOT be a failed-lift case, so we only
        # assert the drop when the env did not confirm success.
        if not _env_success(out["info"]):
            assert out["all_success"] is False
            assert _member_passes(out) is False
    finally:
        env.close()


@sim
def test_sim_dart_object_perturb_compat():
    """Jitter the barrier pose at reset (object perturbation): approach still
    reaches the moved actor and the recorded text/target_id is unchanged (label
    tracks the actor, not a fixed location). A boundary_hook arm-noise injection
    is NOT recorded as a label/action."""
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in S.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        u = env.unwrapped
        # jitter the barrier by a small xy offset. With JIT building the first
        # approach recipe is built at entry, so re-resolving the moved actor's grasp
        # pose works whether the jitter is at reset (here) OR inside a boundary_hook.
        import sapien
        p = u.barrier.pose
        new_p = np.asarray(p.p).reshape(-1)[:3] + np.array([0.02, 0.0, 0.0])
        u.barrier.set_pose(sapien.Pose(new_p, np.asarray(p.q).reshape(-1)[:4]))

        hook_calls = {"n": 0, "first_step": None}

        def hook(unwrapped_env, state):
            # a real DART hook would step unwrapped env here. Record the env-step
            # count at the first call to prove the hook fires BETWEEN primitives,
            # never before the first primitive runs.
            if hook_calls["first_step"] is None:
                hook_calls["first_step"] = int(state.get("step", -1))
            hook_calls["n"] += 1

        rec = I.SubtaskRecorder(num_arms=2)
        prog = spec.build(planner, env)  # plans against the MOVED barrier
        out = I.run_program(env, planner, prog, rec, max_steps=600,
                            boundary_hook=hook, control_mode=planner.control_mode)
        a = rec.to_arrays()
        # text/target_id for the approach unchanged by the jitter
        assert "grasp the left end" in set(a["subtask_arm0_text"])
        assert vocab.LB_TARGETS["left_end"] in set(a["subtask_arm0_target"])
        # boundary_hook fired (between primitives) but added NO recorded steps:
        # recorder length == env steps == sum of primitive ticks.
        assert rec.length == out["steps"]
        assert hook_calls["n"] >= 1
        # TIMING (regression-lock the blocker fix): the hook must NOT fire before
        # the first primitive executes. The first call must land on step > 0 (i.e.
        # after at least one env.step), never on step 0 / before the first reach.
        assert hook_calls["first_step"] is not None
        assert hook_calls["first_step"] > 0, (
            f"boundary_hook fired before the first primitive "
            f"(first_step={hook_calls['first_step']}); first reach plan would be stale")
    finally:
        env.close()
