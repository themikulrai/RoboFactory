"""Off-GPU unit tests for the subtask engine (vocab + primitives + interpreter).

NO sapien, NO gym.make, NO real planner. We drive subtask_primitives and
subtask_interpreter with lightweight FAKES so the whole suite runs on the login
node. (Real-sim interpreter tests are authored separately and gated DART_RUN_SIM.)

Run:  /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
        robofactory/planner/test_subtask_engine.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from robofactory.planner import subtask_vocab as vocab
from robofactory.planner import subtask_primitives as P
from robofactory.planner import subtask_interpreter as I
from robofactory.planner.subtask_primitives import OPEN, CLOSED


# ---------------------------------------------------------------- fakes


class FakeRobot:
    def __init__(self, q7):
        self._q = np.concatenate([np.asarray(q7, np.float32), [0.04, 0.04]]).astype(np.float32)

    def get_qpos(self):
        return self._q[None, :]  # shape (1, 9) numpy (no .cpu -> handled)


class FakePlanner:
    """Minimal planner: returns canned waypoints for dry_run plans.

    move_to_pose_with_screw(pose, move_id=arm, dry_run=True) -> {"position": (n,7)}.
    A pose LIST is REJECTED (regression-locks the silent single-arm-drop trap by
    asserting our helper never passes a list).
    """

    def __init__(self, num_arms=2, control_mode="pd_joint_pos", n_waypoints=5):
        self.control_mode = control_mode
        self.robot = [FakeRobot(np.zeros(7) + 0.1 * i) for i in range(num_arms)]
        self.n_waypoints = n_waypoints
        self.last_dry_run_pose = None

    def move_to_pose_with_screw(self, pose, move_id=0, dry_run=False, **kw):
        assert dry_run, "primitives must plan with dry_run=True"
        # The trap: a LIST under dry_run silently returns one arm's plan. Our helper
        # must NEVER pass a list; assert that here.
        assert not isinstance(pose, list), "pose LIST passed under dry_run (the trap)"
        self.last_dry_run_pose = np.asarray(pose, np.float32)
        # canned monotonically-increasing waypoints toward the pose's xyz
        n = self.n_waypoints
        base = self.robot[move_id].get_qpos()[0, :7]
        wps = np.stack([base + (k + 1) / n * np.ones(7) for k in range(n)]).astype(np.float32)
        return {"position": wps, "status": "Success"}

    # grasp-pose resolvers
    def get_grasp_pose_w_labeled_direction(self, actor, actor_data, pre_dis=0.0, id=0):
        return np.array([id * 0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)

    def get_grasp_pose_from_obb(self, actor, agent_id=0):
        return np.array([0.0, 0.1 * agent_id, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)

    def get_grasp_pose_for_stack(self, now_pose, target_actor, height_offset=0.05):
        out = np.asarray(now_pose, np.float32).copy()
        out[2] = float(target_actor.z) + height_offset
        return out


class FakeActor:
    def __init__(self, z=0.6):
        self.z = z


class FakeEnv:
    """Stands in for env.unwrapped with barrier + annotation_data + step()."""

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
        self.step_log = []  # record action_dicts seen

    def step(self, action_dict):
        self.n_steps += 1
        self.step_log.append({k: np.asarray(v).copy() for k, v in action_dict.items()})
        return {}, 0.0, False, False, {"success": False}


# ---------------------------------------------------------------- primitives


def test_approach_stop_has_no_close_ticks():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.approach(pl, env, arm=0, target_id=vocab.LB_TARGETS["left_end"],
                      task="LiftBarrier")
    assert prim.verb_id == vocab.APPROACH
    assert prim.n_ticks == pl.n_waypoints > 0
    # approach-stop: every tick carries the SAME (open) grip -> zero close ticks
    grips = [g for _, g in prim.ticks]
    assert all(g == OPEN for g in grips), grips
    assert prim.text == "grasp the left end"


def test_close_gripper_ramp_monotonic_and_clamped():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.close_gripper(pl, env, arm=0, task="LiftBarrier", start_grip=OPEN)
    assert prim.n_ticks == P.GRIPPER_RAMP_TICKS
    grips = [g for _, g in prim.ticks]
    # monotonic non-increasing toward CLOSED, clamped at -1
    assert all(grips[k + 1] <= grips[k] + 1e-9 for k in range(len(grips) - 1)), grips
    assert min(grips) >= CLOSED - 1e-9
    # 1 -> -1 in 20 steps of 0.1 reaches the clamp (float accumulation: ~-1.0).
    # Matches upstream motionplanner.close_gripper exactly (same max(g-0.1, CLOSED)).
    assert grips[-1] == pytest.approx(CLOSED, abs=1e-6)
    # qpos frozen across the whole ramp (byte-for-byte)
    q0 = prim.ticks[0][0]
    assert all(np.array_equal(q, q0) for q, _ in prim.ticks)


def test_open_gripper_ramp_monotonic_and_clamped():
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
        assert np.array_equal(q, qpos)
        assert g == CLOSED
    assert prim.text == "wait"


def test_lift_raises_z_and_replans():
    pl, env = FakePlanner(), FakeEnv()
    prim = P.lift(pl, env, arm=0, target_id=vocab.LB_TARGETS["right_end"],
                  task="LiftBarrier", dz=0.2, grip=CLOSED)
    assert prim.verb_id == vocab.LIFT
    # the dry_run pose's z should be the grasp z (0.5) + dz (0.2)
    assert pytest.approx(pl.last_dry_run_pose[2], abs=1e-5) == 0.7
    assert all(g == CLOSED for _, g in prim.ticks)


def test_place_uses_dest_actor_pose():
    pl, env = FakePlanner(), FakeEnv()
    now = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32)
    prim = P.place(pl, env, arm=0, target_id=vocab.TSC_TARGETS["goal_region"],
                   task="ThreeRobotsStackCube", now_pose=now,
                   dest_actor_name="goal_region", height_offset=0.05)
    assert prim.verb_id == vocab.PLACE
    # dest goal_region z=0.0 + offset 0.05 -> planned pose z ~0.05
    assert pytest.approx(pl.last_dry_run_pose[2], abs=1e-5) == 0.05
    assert prim.text == "place on the goal"


def test_dry_run_pose_list_rejected():
    """Regression-lock: planning helper refuses a pose LIST under dry_run."""
    pl = FakePlanner()
    with pytest.raises(ValueError):
        P._plan_to_pose(pl, 0, [np.zeros(7), np.ones(7)])


def test_tsc_approach_color_targets():
    pl, env = FakePlanner(), FakeEnv()
    for color, tid in (("blue", 1), ("green", 2), ("red", 3)):
        prim = P.approach(pl, env, arm=0, target_id=tid, task="ThreeRobotsStackCube")
        assert prim.text == f"approach the {color} cube"


# ---------------------------------------------------------------- interpreter


def _qp(planner, arm):
    return planner.robot[arm].get_qpos()[0, :7].astype(np.float32)


def test_recorder_length_equals_steps_and_aligned():
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    rec = I.SubtaskRecorder(num_arms=2)
    progs = {
        0: [I.QueuedPrimitive(P.approach(pl, env, 0, vocab.LB_TARGETS["left_end"], "LiftBarrier")),
            I.QueuedPrimitive(P.close_gripper(pl, env, 0, "LiftBarrier"))],
        1: [I.QueuedPrimitive(P.approach(pl, env, 1, vocab.LB_TARGETS["right_end"], "LiftBarrier")),
            I.QueuedPrimitive(P.close_gripper(pl, env, 1, "LiftBarrier"))],
    }
    out = I.run_program(env, pl, progs, rec, max_steps=200)
    # both arms: 5 approach + 20 close = 25 ticks each, run simultaneously
    assert out["steps"] == 25
    assert rec.length == 25
    assert env.n_steps == 25
    # off-by-one: label[t] is the subtask driving action[t]. First 5 are approach
    # (verb 1), next 20 are close (verb 3), per arm.
    a0 = rec.to_arrays()
    assert list(a0["subtask_arm0_verb"][:5]) == [vocab.APPROACH] * 5
    assert list(a0["subtask_arm0_verb"][5:25]) == [vocab.CLOSE_GRIPPER] * 20
    # no None / contiguous runs
    assert set(a0["subtask_arm0_verb"]) <= {vocab.APPROACH, vocab.CLOSE_GRIPPER}


def test_idle_arm_labelled_wait_and_frozen():
    """Arm1 has no program -> labelled wait every tick AND holds frozen qpos."""
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    rec = I.SubtaskRecorder(num_arms=2)
    progs = {0: [I.QueuedPrimitive(P.approach(pl, env, 0, vocab.LB_TARGETS["left_end"], "LiftBarrier"))],
             1: []}
    out = I.run_program(env, pl, progs, rec, max_steps=50)
    a = rec.to_arrays()
    # arm1 always wait
    assert list(a["subtask_arm1_verb"]) == [vocab.WAIT] * out["steps"]
    # arm1 qpos byte-identical across all steps (frozen, not live-sag)
    arm1_q = [s["panda-1"][:7] for s in env.step_log]
    assert all(np.array_equal(q, arm1_q[0]) for q in arm1_q)


def test_barrier_gating_blocks_until_predicate():
    """Arm1's primitive is gated on arm0 grasping; arm1 stays wait until then."""
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    rec = I.SubtaskRecorder(num_arms=2)

    grasped = {"v": False}

    def arm0_grasped(env, state):
        return grasped["v"]

    # arm0: approach(5) then a close that flips the grasp predicate when it finishes
    close0 = P.close_gripper(pl, env, 0, "LiftBarrier")

    def flip_check(env, arm, prim):
        grasped["v"] = True
        return True
    close0.success_check = flip_check

    progs = {
        0: [I.QueuedPrimitive(P.approach(pl, env, 0, vocab.LB_TARGETS["left_end"], "LiftBarrier")),
            I.QueuedPrimitive(close0)],
        1: [I.QueuedPrimitive(
                P.approach(pl, env, 1, vocab.LB_TARGETS["right_end"], "LiftBarrier"),
                wait_for=(0, arm0_grasped))],
    }
    out = I.run_program(env, pl, progs, rec, max_steps=200)
    a = rec.to_arrays()
    # arm1 is wait for the first 25 ticks (approach 5 + close 20 of arm0), then approach
    first_approach1 = next(i for i, v in enumerate(a["subtask_arm1_verb"]) if v == vocab.APPROACH)
    assert first_approach1 == 25, (first_approach1, list(a["subtask_arm1_verb"]))
    assert list(a["subtask_arm1_verb"][:25]) == [vocab.WAIT] * 25


def test_success_check_records_per_primitive():
    pl, env = FakePlanner(num_arms=1), FakeEnv()
    rec = I.SubtaskRecorder(num_arms=1)
    # one approach with a success_check that returns False (liar-label guard)
    prim = P.approach(pl, env, 0, vocab.LB_TARGETS["left_end"], "LiftBarrier",
                      success_check=lambda env, arm, p: False)
    progs = {0: [I.QueuedPrimitive(prim)]}
    out = I.run_program(env, pl, progs, rec, max_steps=50)
    assert out["success"][0][0] is False
    assert out["all_success"] is False


def test_boundary_hook_called_between_primitives_unrecorded():
    """boundary_hook fires at primitive transitions and does NOT add recorded steps."""
    pl, env = FakePlanner(num_arms=1), FakeEnv()
    rec = I.SubtaskRecorder(num_arms=1)
    calls = {"n": 0}

    def hook(unwrapped_env, state):
        calls["n"] += 1

    progs = {0: [
        I.QueuedPrimitive(P.approach(pl, env, 0, vocab.LB_TARGETS["left_end"], "LiftBarrier")),
        I.QueuedPrimitive(P.close_gripper(pl, env, 0, "LiftBarrier")),
        I.QueuedPrimitive(P.hold(pl, env, 0, 3, "LiftBarrier")),
    ]}
    n_steps_before = env.n_steps
    out = I.run_program(env, pl, progs, rec, max_steps=200, boundary_hook=hook)
    # hook fires once per primitive entry (3 primitives) -> 3 calls
    assert calls["n"] == 3
    # recorded steps == sum of ticks (5 + 20 + 3); hook added none
    assert out["steps"] == 28
    assert rec.length == 28


def test_flush_length_guard_raises_on_mismatch():
    rec = I.SubtaskRecorder(num_arms=1)
    rec.append({0: (vocab.WAIT, 0, "wait")})
    with pytest.raises(AssertionError):
        rec.flush(episode_id=0, out_dir="/tmp/_subtask_test", expected_T=5)


def test_flush_npz_roundtrip(tmp_path):
    pl, env = FakePlanner(num_arms=2), FakeEnv()
    rec = I.SubtaskRecorder(num_arms=2)
    progs = {
        0: [I.QueuedPrimitive(P.approach(pl, env, 0, vocab.LB_TARGETS["left_end"], "LiftBarrier"))],
        1: [I.QueuedPrimitive(P.approach(pl, env, 1, vocab.LB_TARGETS["right_end"], "LiftBarrier"))],
    }
    out = I.run_program(env, pl, progs, rec, max_steps=50)
    path = rec.flush(episode_id=3, out_dir=str(tmp_path), expected_T=out["steps"])
    assert path.endswith("subtask_stream_traj_3.npz")
    z = np.load(path, allow_pickle=True)
    assert int(z["length"]) == out["steps"]
    assert len(z["subtask_arm0_text"]) == out["steps"]
    assert z["subtask_arm0_text"][0] == "grasp the left end"
    assert list(z["subtask_arm0_verb"]) == [vocab.APPROACH] * out["steps"]


# ---------------------------------------------------------------- vocab (kept here too)


def test_vocab_render_all_lb_combos():
    assert vocab.render(vocab.WAIT, 0, "LiftBarrier") == "wait"
    assert vocab.render(vocab.APPROACH, 1, "LiftBarrier") == "grasp the left end"
    assert vocab.render(vocab.APPROACH, 2, "LiftBarrier") == "grasp the right end"
    assert vocab.render(vocab.LIFT, 1, "LiftBarrier") == "lift the left end"
    assert vocab.render(vocab.CLOSE_GRIPPER, 0, "LiftBarrier") == "close the gripper"


def test_vocab_color_map_asserts_against_yaml():
    out = vocab.assert_cube_color_map()
    assert out == {"cubeA": "blue", "cubeB": "green", "cubeC": "red"}
