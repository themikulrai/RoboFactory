"""PURE + SIM-gated tests for the ThreeRobotsStackCube (TSC) subtask scenarios.

PURE tests (no sim, always run): the TSC sampler emits the matched pairs; names /
families; deadlock-acyclic; all-arms-end-in-place; stack-order gates present
(B-place waits A placed, C-place waits B placed); grasp-stagger gate; RNG
reproducibility + independence from the global numpy RNG; group filtering keeps
pairs intact; render() produces the TSC color/dest/wait strings; the runner's
TASK_MAP wires the TSC sampler.

SIM tests (@pytest.mark.sim, skipped unless DART_RUN_SIM=1; DO NOT RUN on the login
node — SAPIEN renders dark; needs a GPU compute node): drive a real 3-arm TSC
rollout through subtask_interpreter.run_program and verify the aligned stream
(len==T per arm), the stagger follower waiting at frame 0, and a full simultaneous
run reaching env stack-success.

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
    def __init__(self):
        self.goal_region = FakeActor(z=0.0)
        self.cubeA = FakeActor(z=0.05)
        self.cubeB = FakeActor(z=0.1)
        self.cubeC = FakeActor(z=0.15)
        self.unwrapped = self
        self.n_steps = 0
        self.step_log = []

    def step(self, action_dict):
        self.n_steps += 1
        self.step_log.append({k: np.asarray(v).copy() for k, v in action_dict.items()})
        return {}, 0.0, False, False, {"success": False}


def _fakes():
    return FakePlanner(num_arms=3), FakeEnv()


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
# PURE: the sampler emits the matched pairs
# ===========================================================================
def test_sampler_emits_two_matched_pairs():
    specs = STSC.sample(7)
    groups = STSC.group_specs(specs)
    assert len(specs) == 4
    assert len(groups) == 2
    for gid, members in groups.items():
        assert len(members) == 2, (gid, [m.name for m in members])
        assert members[0].family == members[1].family
        assert members[0].contrast_group_id == members[1].contrast_group_id == gid
        assert "simultaneous" in {m.name for m in members}, [m.name for m in members]
        for m in members:
            assert m.num_arms == 3
    assert {m.family for m in specs} == {"stagger", "target_swap"}
    assert {m.name for m in specs} == {"simultaneous", "stagger_grasp", "target_swap"}


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
# PURE: every variant has all 3 arms end in PLACE (then open) — complete rollout
# ===========================================================================
def test_every_variant_all_arms_place():
    specs = STSC.sample(7)
    for spec in specs:
        pl, env = _fakes()
        prog = spec.build(pl, env)
        assert set(prog.keys()) == {0, 1, 2}, spec.name
        # the structural guard raises if any arm doesn't place->open
        STSC.assert_all_arms_place(prog, pl, env)
        for arm in (0, 1, 2):
            place_prim = prog[arm][STSC.PLACE_DONE_IDX - 1].recipe(pl, env, arm)
            assert place_prim.verb_id == vocab.PLACE, (spec.name, arm)
            last = prog[arm][-1].recipe(pl, env, arm)
            assert last.verb_id == vocab.OPEN_GRIPPER, (spec.name, arm)


def test_every_arm_program_is_approach_close_lift_place_open():
    """Each arm's 5-primitive program is exactly approach,close,lift,place,open."""
    specs = STSC.sample(7)
    expected = [vocab.APPROACH, vocab.CLOSE_GRIPPER, vocab.LIFT, vocab.PLACE,
                vocab.OPEN_GRIPPER]
    for spec in specs:
        pl, env = _fakes()
        prog = spec.build(pl, env)
        for arm in (0, 1, 2):
            verbs = [qr.recipe(pl, env, arm).verb_id for qr in prog[arm]]
            assert verbs == expected, (spec.name, arm, verbs)


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


def test_simultaneous_first_primitives_all_ungated():
    """simultaneous: all 3 arms approach UNGATED at frame 0 (no grasp gate)."""
    specs = STSC.sample(7)
    spec = next(m for m in specs if m.name == "simultaneous")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    first_gates = STSC.gate_graph(prog)
    assert first_gates == {0: None, 1: None, 2: None}, first_gates


# ===========================================================================
# PURE: stack-order PLACE gates present (B-place waits A placed, C-place waits B)
# ===========================================================================
def _place_qr(prog, arm):
    """The PLACE QueuedRecipe of an arm (index PLACE_DONE_IDX-1 == 3)."""
    return prog[arm][STSC.PLACE_DONE_IDX - 1]


def test_stack_order_place_gates_base_assignment():
    """In the BASE assignment (arm0=cubeA bottom, arm1=cubeB mid, arm2=cubeC top):
    arm0's place is UNGATED (bottom places first); arm1's place waits on arm0;
    arm2's place waits on arm1. The gate opens at PLACE_DONE_IDX (place finished)."""
    specs = STSC.sample(7)
    for name in ("simultaneous", "stagger_grasp"):
        spec = next(m for m in specs if m.name == name)
        pl, env = _fakes()
        prog = spec.build(pl, env)
        # arm0 (bottom): place ungated
        assert _place_qr(prog, 0).wait_for is None, name
        # arm1 (middle): place waits on arm0 reaching PLACE_DONE_IDX
        g1 = _place_qr(prog, 1).wait_for
        assert g1 is not None and g1[0] == 0, (name, g1)
        # arm2 (top): place waits on arm1
        g2 = _place_qr(prog, 2).wait_for
        assert g2 is not None and g2[0] == 1, (name, g2)
        # gate predicates: closed until the below-arm has FINISHED its place.
        _assert_place_gate_semantics(env, g1, below_arm=0)
        _assert_place_gate_semantics(env, g2, below_arm=1)


class _FakeArmState:
    def __init__(self, qi):
        self.qi = int(qi)


def _assert_place_gate_semantics(env, gate, below_arm):
    """The place gate opens iff the below-arm's qi >= PLACE_DONE_IDX (place done)."""
    other_arm, predicate = gate
    assert other_arm == below_arm
    def _state(below_qi):
        return {"arms": {below_arm: _FakeArmState(below_qi)}}
    # below arm still placing / not done -> gate CLOSED
    assert predicate(env, _state(0)) is False
    assert predicate(env, _state(STSC.PLACE_DONE_IDX - 1)) is False  # on its place
    # below arm FINISHED its place (qi >= PLACE_DONE_IDX) -> gate OPENS
    assert predicate(env, _state(STSC.PLACE_DONE_IDX)) is True
    assert predicate(env, _state(STSC.PLACE_DONE_IDX + 1)) is True
    # robust to missing arms map
    assert predicate(env, {}) is False
    assert predicate(env, {"arms": {}}) is False


def test_target_swap_keeps_stack_order_valid():
    """target_swap: arm2 holds cubeA (bottom, ungated place), arm1 holds cubeB
    (waits on arm2), arm0 holds cubeC (waits on arm1). Stack stays bottom-up."""
    specs = STSC.sample(7)
    spec = next(m for m in specs if m.name == "target_swap")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    # arm2 (cubeA, bottom): place ungated
    assert _place_qr(prog, 2).wait_for is None
    # arm1 (cubeB, middle): place waits on arm2 (the cubeA holder)
    g1 = _place_qr(prog, 1).wait_for
    assert g1 is not None and g1[0] == 2, g1
    # arm0 (cubeC, top): place waits on arm1 (the cubeB holder)
    g0 = _place_qr(prog, 0).wait_for
    assert g0 is not None and g0[0] == 1, g0


# ===========================================================================
# PURE: the COLOR / coordination counterfactuals
# ===========================================================================
def test_color_counterfactual_arm0_blue_vs_red():
    """simultaneous -> arm0 approaches the BLUE cube (cubeA); target_swap -> arm0
    approaches the RED cube (cubeC). The PROMPT color flips for the same arm."""
    specs = STSC.sample(7)
    sim_ = next(m for m in specs if m.name == "simultaneous")
    swap = next(m for m in specs if m.name == "target_swap")
    pl, env = _fakes()
    a0_sim = sim_.build(pl, env)[0][0].recipe(pl, env, 0)
    a0_swap = swap.build(pl, env)[0][0].recipe(pl, env, 0)
    assert a0_sim.text == "approach the blue cube", a0_sim.text
    assert a0_swap.text == "approach the red cube", a0_swap.text
    assert a0_sim.target_id == vocab.TSC_TARGETS["blue"]
    assert a0_swap.target_id == vocab.TSC_TARGETS["red"]


def test_stagger_grasp_followers_wait_at_frame0():
    """stagger_grasp: arm0 (bottom) leads grasp UNGATED; arm1 & arm2 first primitive
    is GATED on arm0 grasping -> they WAIT at frame 0 vs approach in simultaneous."""
    specs = STSC.sample(7)
    spec = next(m for m in specs if m.name == "stagger_grasp")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    first_gates = STSC.gate_graph(prog)
    assert first_gates[0] is None         # leader ungated first
    assert first_gates[1] == 0            # follower waits on arm0
    assert first_gates[2] == 0            # follower waits on arm0
    # the followers' approach gate opens at GRASP_IDX (lead finished approach+close)
    for follow in (1, 2):
        other_arm, predicate = prog[follow][0].wait_for
        assert other_arm == 0
        def _state(lead_qi):
            return {"arms": {0: _FakeArmState(lead_qi)}}
        assert predicate(env, _state(0)) is False
        assert predicate(env, _state(1)) is False               # only approach done
        assert predicate(env, _state(STSC.GRASP_IDX)) is True   # grasped
        assert predicate(env, _state(STSC.GRASP_IDX + 1)) is True


# ===========================================================================
# PURE: group filtering keeps matched pairs intact
# ===========================================================================
def test_filter_variants_keeps_groups_intact():
    specs = STSC.sample(7)
    out = STSC.filter_variants(specs, ["stagger_grasp"])
    assert sorted(m.name for m in out) == ["simultaneous", "stagger_grasp"]
    out2 = STSC.filter_variants(specs, ["target_swap"])
    assert sorted(m.name for m in out2) == ["simultaneous", "target_swap"]
    assert len(STSC.filter_variants(specs, None)) == len(specs)
    assert STSC.filter_variants(specs, ["nope"]) == []


# ===========================================================================
# PURE: place recipe resolves now_pose JIT (carried-cube grasp pose) + dest
# ===========================================================================
def test_place_recipe_uses_dest_actor_and_color():
    """arm0's place (base assignment) targets goal_region with the 'on the goal'
    prompt; the recipe builds a real PLACE primitive against the fakes."""
    specs = STSC.sample(7)
    spec = next(m for m in specs if m.name == "simultaneous")
    pl, env = _fakes()
    prog = spec.build(pl, env)
    place0 = _place_qr(prog, 0).recipe(pl, env, 0)
    assert place0.verb_id == vocab.PLACE
    assert place0.text == "place on the goal"
    # dest goal_region z=0.0 + offset 0.05 -> planned pose z ~0.05
    assert pytest.approx(pl.last_dry_run_pose[2], abs=1e-5) == 0.05
    # arm1 places onto cubeA -> "place on the blue cube"
    place1 = _place_qr(prog, 1).recipe(pl, env, 1)
    assert place1.text == "place on the blue cube", place1.text


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
    """Drive the simultaneous TSC program through the interpreter with FAKES and
    assert the recorded stream is aligned (len==T for all 3 arms) and contiguous."""
    spec = next(m for m in STSC.sample(7) if m.name == "simultaneous")
    pl, env = _fakes()
    rec = I.SubtaskRecorder(num_arms=3)
    prog = spec.build(pl, env)
    out = I.run_program(env, pl, prog, rec, max_steps=2000,
                        control_mode=pl.control_mode, check_success_coverage="off")
    T = out["steps"]
    a = rec.to_arrays()
    for arm in (0, 1, 2):
        assert len(a[f"subtask_arm{arm}_verb"]) == T
        verbs = [int(v) for v in a[f"subtask_arm{arm}_verb"]]
        assert all(v in vocab.VERB_IDS.values() for v in verbs)  # no garbage/None
        # the program executes; each arm hits all five verbs in order somewhere.
        import itertools
        runs = [k for k, _ in itertools.groupby(verbs)]
        # strip idle WAIT runs (arms idle while gated on stack order)
        core = [r for r in runs if r != vocab.WAIT]
        assert core == [vocab.APPROACH, vocab.CLOSE_GRIPPER, vocab.LIFT,
                        vocab.PLACE, vocab.OPEN_GRIPPER], (arm, runs)
    assert rec.length == T


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
        spec = next(m for m in STSC.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        out = I.run_program(env, planner, prog, rec, max_steps=1200,
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
def test_sim_stagger_followers_wait_at_frame0():
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in STSC.sample(0) if m.name == "stagger_grasp")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        I.run_program(env, planner, prog, rec, max_steps=1500,
                      control_mode=planner.control_mode)
        a = rec.to_arrays()
        # arm0 (lead) starts its approach immediately; arm1 & arm2 WAIT at frame 0.
        assert a["subtask_arm0_verb"][0] == vocab.APPROACH
        assert a["subtask_arm1_verb"][0] == vocab.WAIT
        assert a["subtask_arm2_verb"][0] == vocab.WAIT
        # all three arms ultimately place (complete rollout).
        for arm in (0, 1, 2):
            assert vocab.PLACE in set(a[f"subtask_arm{arm}_verb"]), arm
    finally:
        env.close()


@sim
def test_sim_simultaneous_reaches_stack_success():
    from run_subtask_rollouts import _env_success
    env, Solver = _make_sim_env()
    try:
        spec = next(m for m in STSC.sample(0) if m.name == "simultaneous")
        planner = _build_sim_planner(env, Solver, spec.seed)
        rec = I.SubtaskRecorder(num_arms=3)
        prog = spec.build(planner, env)
        out = I.run_program(env, planner, prog, rec, max_steps=1500,
                            control_mode=planner.control_mode)
        # a full simultaneous run reaches the env stack-success OR completes the queues.
        assert _env_success(out["info"]) or out["completed"], out["info"]
    finally:
        env.close()
