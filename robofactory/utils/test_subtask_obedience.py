"""Unit tests for the PURE subtask-obedience probe logic (no SAPIEN / GPU / server).

Run just this slice with:  ``pytest -k subtask_obedience``

Covers:
  * the exact trained prompt strings (and cross-check vs the real subtask_vocab.json if present);
  * the canonical barrier LEFT/RIGHT end world-position computation (identity / translate /
    rotate / scale / mutation-guard / shape-guards), matched to the data-gen planner;
  * TCP-trajectory -> end assignment (closest approach, ties, min-over-trajectory, empty guard);
  * the min-approach GATE (assign_end): near+moved -> assigned, frozen/far -> "none",
    thresholds=None recovers closest_end, plus net_displacement;
  * the in-distribution ENGAGEMENT signals (approach_progress, grasp_fraction) and the paired
    grasp-vs-wait engagement-contrast scoring (channel-alive vs channel-dead, pairing guards);
  * OOD command classification (arm0->left in-domain, opposite-end OOD);
  * prompt-vocab validation (recognized typo-guard + OOD annotation, no hard-fail on OOD);
  * A/B pairing (missing / duplicate / mislabeled rollouts) and the obedience + paired
    causal-shift scoring, including the prompt-ignoring null and a perfect-obeying arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robofactory.utils import subtask_obedience as so


# --------------------------------------------------------------------------- prompt strings


def test_prompt_constants_exact():
    assert so.GRASP_LEFT_END == "grasp the left end"
    assert so.GRASP_RIGHT_END == "grasp the right end"
    assert so.NEUTRAL_WAIT == "wait"
    assert so.ARM_INDOMAIN_GRASP == {0: "grasp the left end", 1: "grasp the right end"}
    assert so.COMMAND_PROMPT == {"left": "grasp the left end", "right": "grasp the right end"}
    assert so.ROLLOUT_COMMAND == {"A": "left", "B": "right"}


@pytest.mark.parametrize(
    "vocab_json",
    [
        "/iris/u/mikulrai/data/RoboFactory/lerobot/robofactory_lift_barrier_ws_subtaskdart_v1/subtask_vocab.json",
        "/iris/u/mikulrai/data/RoboFactory/lerobot/robofactory_lift_barrier_wc_subtaskdart_v1/subtask_vocab.json",
    ],
)
def test_probe_strings_in_real_vocab_if_present(vocab_json):
    """The probe strings must appear in the real training subtask vocab (ground truth)."""
    if not Path(vocab_json).exists():
        pytest.skip(f"dataset vocab not present: {vocab_json}")
    vocab = so.load_lb_subtask_vocab(vocab_json)
    assert so.GRASP_LEFT_END in vocab
    assert so.GRASP_RIGHT_END in vocab
    assert so.NEUTRAL_WAIT in vocab


# ----------------------------------------------------------------- barrier end geometry


def _contact_4x4(x, y=0.0, z=0.0):
    """A 4x4 contact-point matrix with translation (x, y, z) and identity rotation."""
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = [x, y, z]
    return m


def _actor_data(left_x=-0.5, right_x=0.5, scale=(1.0, 1.0, 1.0)):
    """Synthetic barrier annotation: id0/id3 dummy, id1=LEFT(-x), id2=RIGHT(+x)."""
    return {
        "scale": list(scale),
        "contact_points_pose": [
            _contact_4x4(0.0),       # id0 (unused by probe)
            _contact_4x4(left_x),    # id1 -> LEFT
            _contact_4x4(right_x),   # id2 -> RIGHT
            _contact_4x4(0.0),       # id3 (unused)
        ],
    }


def test_end_position_identity_pose():
    data = _actor_data(left_x=-0.5, right_x=0.5)
    left = so.barrier_end_world_position(np.eye(4), data, so.LEFT_CONTACT_ID)
    right = so.barrier_end_world_position(np.eye(4), data, so.RIGHT_CONTACT_ID)
    np.testing.assert_allclose(left, [-0.5, 0, 0], atol=1e-9)
    np.testing.assert_allclose(right, [0.5, 0, 0], atol=1e-9)


def test_end_position_scale_applied_to_translation():
    data = _actor_data(left_x=-0.5, right_x=0.5, scale=(2.0, 1.0, 1.0))
    left = so.barrier_end_world_position(np.eye(4), data, so.LEFT_CONTACT_ID)
    np.testing.assert_allclose(left, [-1.0, 0, 0], atol=1e-9)  # -0.5 * 2.0


def test_end_position_translation_pose():
    data = _actor_data(left_x=-0.5, right_x=0.5)
    actor = np.eye(4)
    actor[:3, 3] = [1.0, 2.0, 3.0]
    left = so.barrier_end_world_position(actor, data, so.LEFT_CONTACT_ID)
    np.testing.assert_allclose(left, [0.5, 2.0, 3.0], atol=1e-9)  # 1.0 + (-0.5)


def test_end_position_rotation_maps_x_to_y():
    # +90deg about z: model +x -> world +y, model -x -> world -y. Mirrors the real barrier
    # whose long (model-x) axis lies along world-y between the two arms.
    rot = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    data = _actor_data(left_x=-0.5, right_x=0.5)
    ends = so.barrier_end_positions(rot, data)
    np.testing.assert_allclose(ends["left"], [0, -0.5, 0], atol=1e-9)
    np.testing.assert_allclose(ends["right"], [0, 0.5, 0], atol=1e-9)
    assert ends["y_consistent"] is True  # left.y (-0.5) <= right.y (0.5)


def test_end_positions_y_consistency_flag_false_when_flipped():
    # Swap which contact id is "left": now id1 is at +x, id2 at -x, with the x->y rotation
    # the labeled-left end has the LARGER y -> y_consistent should be False.
    rot = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    data = _actor_data(left_x=0.5, right_x=-0.5)
    ends = so.barrier_end_positions(rot, data)
    assert ends["y_consistent"] is False


def test_end_position_does_not_mutate_annotation():
    data = _actor_data(left_x=-0.5, right_x=0.5, scale=(3.0, 1.0, 1.0))
    before = np.array(data["contact_points_pose"][so.LEFT_CONTACT_ID], dtype=np.float64).copy()
    _ = so.barrier_end_world_position(np.eye(4), data, so.LEFT_CONTACT_ID)
    after = np.asarray(data["contact_points_pose"][so.LEFT_CONTACT_ID], dtype=np.float64)
    np.testing.assert_allclose(after, before, atol=1e-12)  # scale must not leak into the dict


def test_end_position_bad_actor_shape_raises():
    data = _actor_data()
    with pytest.raises(ValueError, match="4x4"):
        so.barrier_end_world_position(np.eye(3), data, so.LEFT_CONTACT_ID)


def test_end_position_matches_real_planner_formula():
    """End-to-end check against the planner's exact matrix expression (lines 276-286)."""
    rng = np.random.default_rng(0)
    actor = np.eye(4)
    actor[:3, :3] = np.linalg.qr(rng.standard_normal((3, 3)))[0]  # random rotation
    actor[:3, 3] = rng.standard_normal(3)
    local = np.eye(4)
    local[:3, 3] = [0.37, 0.1, 0.37]
    data = {"scale": [0.6, 0.6, 0.2], "contact_points_pose": [local]}
    # Replicate the planner verbatim:
    expect_local = local.copy()
    expect_local[:3, 3] *= np.array([0.6, 0.6, 0.2])
    convert = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    expect = (actor @ expect_local @ convert)[:3, 3]
    got = so.barrier_end_world_position(actor, data, 0)
    np.testing.assert_allclose(got, expect, atol=1e-12)


# ------------------------------------------------------------------- TCP -> end assignment


def _ends():
    return {"left": np.array([0.0, -0.5, 0.2]), "right": np.array([0.0, 0.5, 0.2])}


def test_closest_end_left():
    traj = [[0.0, -0.45, 0.2], [0.0, -0.48, 0.2]]
    label, dl, dr = so.closest_end(traj, _ends())
    assert label == "left"
    assert dl < dr


def test_closest_end_right():
    traj = [[0.0, 0.40, 0.2], [0.0, 0.49, 0.2]]
    label, dl, dr = so.closest_end(traj, _ends())
    assert label == "right"
    assert dr < dl


def test_closest_end_uses_min_over_trajectory():
    # Passes very close to LEFT mid-trajectory, ends near center -> assigned LEFT (min wins).
    traj = [[0, -0.5, 0.2], [0, 0.0, 0.2], [0, 0.05, 0.2]]
    label, dl, dr = so.closest_end(traj, _ends())
    assert label == "left"
    assert dl == pytest.approx(0.0, abs=1e-9)


def test_closest_end_tie_prefers_left():
    traj = [[0.0, 0.0, 0.2]]  # equidistant
    label, dl, dr = so.closest_end(traj, _ends())
    assert label == "left"
    assert dl == pytest.approx(dr)


def test_closest_end_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        so.closest_end([], _ends())


# ----------------------------------------------------------------------- OOD classification


@pytest.mark.parametrize(
    "arm,end,expected",
    [(0, "left", False), (0, "right", True), (1, "left", True), (1, "right", False)],
)
def test_is_ood_command(arm, end, expected):
    assert so.is_ood_command(arm, end) is expected


def test_is_ood_command_bad_args():
    with pytest.raises(ValueError):
        so.is_ood_command(2, "left")
    with pytest.raises(ValueError):
        so.is_ood_command(0, "middle")


# ----------------------------------------------------------------------- vocab + validation


def test_load_lb_subtask_vocab(tmp_path):
    p = tmp_path / "subtask_vocab.json"
    p.write_text(json.dumps({
        "verb": {"0": "wait", "1": "grasp the left end", "4": "lift the left end"},
        "target": {"0": "wait", "1": "grasp the left end", "2": "grasp the right end"},
    }))
    vocab = so.load_lb_subtask_vocab(str(p))
    assert vocab == {"wait", "grasp the left end", "grasp the right end", "lift the left end"}


def test_validate_probe_prompts_ok_and_flags_ood():
    vocab = {so.GRASP_LEFT_END, so.GRASP_RIGHT_END, so.NEUTRAL_WAIT}
    out = so.validate_probe_prompts([0, 1], vocab)
    assert out["recognized_vocab_checked"] is True
    assert out["unrecognized_prompts"] == []
    flags = {(c["target_arm"], c["commanded_end"]): c["is_ood"] for c in out["combos"]}
    assert flags == {(0, "left"): False, (0, "right"): True, (1, "left"): True, (1, "right"): False}
    # exact prompt strings recorded per combo
    by = {(c["target_arm"], c["rollout"]): c["prompt"] for c in out["combos"]}
    assert by[(0, "A")] == so.GRASP_LEFT_END and by[(0, "B")] == so.GRASP_RIGHT_END


def test_validate_probe_prompts_unrecognized_raises():
    vocab = {"some other subtask"}  # probe strings absent
    with pytest.raises(ValueError, match="recognized subtask vocab"):
        so.validate_probe_prompts([0], vocab)


def test_validate_probe_prompts_skips_check_when_vocab_none():
    out = so.validate_probe_prompts([0, 1], None)
    assert out["recognized_vocab_checked"] is False
    assert out["unrecognized_prompts"] == []


def test_validate_probe_prompts_custom_nontarget_recognized():
    vocab = {so.GRASP_LEFT_END, so.GRASP_RIGHT_END, "lift the left end"}
    # nontarget "wait" NOT in vocab -> must raise (typo guard covers the neutral prompt too).
    with pytest.raises(ValueError):
        so.validate_probe_prompts([0], vocab, nontarget_prompt="wait")
    # a recognized nontarget passes.
    out = so.validate_probe_prompts([0], vocab, nontarget_prompt="lift the left end")
    assert out["nontarget_prompt"] == "lift the left end"


# ------------------------------------------------------------------------ A/B pairing


def _roll(arm, seed, rollout, assigned, dl, dr):
    return {
        "target_arm": arm,
        "seed": seed,
        "rollout": rollout,
        "commanded_end": so.ROLLOUT_COMMAND[rollout],
        "assigned_end": assigned,
        "min_dist_left": dl,
        "min_dist_right": dr,
    }


def test_pair_ab_rollouts_ok():
    rolls = [
        _roll(0, 100, "A", "left", 0.01, 0.5),
        _roll(0, 100, "B", "right", 0.5, 0.02),
    ]
    paired = so.pair_ab_rollouts(rolls)
    assert set(paired.keys()) == {(0, 100)}
    assert paired[(0, 100)]["A"].obeyed and paired[(0, 100)]["B"].obeyed


def test_pair_ab_rollouts_missing_b_raises():
    with pytest.raises(ValueError, match="missing rollout"):
        so.pair_ab_rollouts([_roll(0, 100, "A", "left", 0.01, 0.5)])


def test_pair_ab_rollouts_duplicate_raises():
    rolls = [_roll(0, 100, "A", "left", 0.01, 0.5), _roll(0, 100, "A", "left", 0.02, 0.4)]
    with pytest.raises(ValueError, match="duplicate"):
        so.pair_ab_rollouts(rolls)


def test_pair_ab_rollouts_mislabeled_raises():
    # An "A" rollout that recorded commanded_end="right" is inconsistent (A must command left).
    bad = _roll(0, 100, "A", "left", 0.01, 0.5)
    bad["commanded_end"] = "right"
    with pytest.raises(ValueError, match="must command"):
        so.pair_ab_rollouts([bad, _roll(0, 100, "B", "right", 0.5, 0.02)])


# --------------------------------------------------------------------------- scoring


def test_rollout_result_properties():
    r = so.RolloutResult(0, 100, "A", "left", "left", 0.01, 0.5)
    assert r.obeyed is True
    assert r.approach == pytest.approx(0.49)  # dist_right - dist_left
    assert r.is_ood is False
    r2 = so.RolloutResult(0, 100, "B", "right", "left", 0.01, 0.5)
    assert r2.obeyed is False  # commanded right, assigned left
    assert r2.is_ood is True   # arm0 + right end = OOD


def test_score_arm_perfect_obeying():
    # arm0 obeys: A->left (close to left), B->right (close to right) for every seed.
    rolls = []
    for s in (100, 200, 300):
        rolls.append(_roll(0, s, "A", "left", 0.01, 0.5))
        rolls.append(_roll(0, s, "B", "right", 0.5, 0.01))
    out = so.score_arm(rolls)
    assert out["target_arm"] == 0
    assert out["n_seeds"] == 3
    assert out["obedience"] == pytest.approx(1.0)
    assert out["left_command_obeyed"] == pytest.approx(1.0)
    assert out["right_command_obeyed"] == pytest.approx(1.0)
    assert out["causal_consistency"] == pytest.approx(1.0)  # A always leaner-left than B
    assert out["mean_approach_shift"] > 0
    # A leans left (approach +0.49), B leans right (approach -0.49) -> shift ~0.98
    assert out["mean_approach_shift"] == pytest.approx(0.98, abs=1e-6)


def test_score_arm_prompt_ignoring_null():
    # Prompt-ignoring arm0 always grasps its trained LEFT end regardless of command.
    # Identical A and B trajectories -> obedience 0.5, zero causal shift.
    rolls = []
    for s in (100, 200, 300, 400):
        rolls.append(_roll(0, s, "A", "left", 0.02, 0.6))   # commanded left, went left -> obey
        rolls.append(_roll(0, s, "B", "left", 0.02, 0.6))   # commanded right, went left -> disobey
    out = so.score_arm(rolls)
    assert out["obedience"] == pytest.approx(0.5)
    assert out["null_obedience"] == 0.5
    assert out["frac_assigned_left"] == pytest.approx(1.0)  # fixed bias to the trained end
    assert out["causal_consistency"] == pytest.approx(0.0)  # shift == 0, not > 0
    assert out["mean_approach_shift"] == pytest.approx(0.0)


def test_score_arm_single_arm_guard():
    rolls = [_roll(0, 1, "A", "left", 0.01, 0.5), _roll(1, 1, "B", "right", 0.5, 0.01)]
    with pytest.raises(ValueError, match="single target_arm"):
        so.score_arm(rolls)


def test_score_arm_empty_raises():
    with pytest.raises(ValueError, match="no rollouts"):
        so.score_arm([])


def test_score_obedience_both_arms():
    rolls = []
    for s in (100, 200):
        rolls.append(_roll(0, s, "A", "left", 0.01, 0.5))
        rolls.append(_roll(0, s, "B", "right", 0.5, 0.01))
        rolls.append(_roll(1, s, "A", "left", 0.01, 0.5))
        rolls.append(_roll(1, s, "B", "right", 0.5, 0.01))
    out = so.score_obedience(rolls)
    assert set(out.keys()) == {"arm0", "arm1"}
    assert out["arm0"]["obedience"] == pytest.approx(1.0)
    assert out["arm1"]["indomain_end"] == "right"
    assert out["arm1"]["n_ood_rollouts"] == 2  # arm1 + left (the A rollouts) are OOD


def test_score_arm_counts_no_approach():
    # A "none" assignment (gated-out passive arm) must NOT count as obeyed or assigned-left, and
    # must be tallied in n_no_approach.
    rolls = [
        _roll(0, 100, "A", "none", 0.4, 0.5),   # passive: never approached an end
        _roll(0, 100, "B", "right", 0.5, 0.01),
    ]
    out = so.score_arm(rolls)
    assert out["n_no_approach"] == 1
    assert out["frac_assigned_left"] == pytest.approx(0.0)  # "none" is not "left"
    # only B obeyed (A commanded left but assigned "none")
    assert out["obedience"] == pytest.approx(0.5)


# --------------------------------------------------- min-approach gate (assign_end) + net disp


def test_net_displacement_moving_and_frozen():
    assert so.net_displacement([[0, 0, 0], [0, 0, 0.3]]) == pytest.approx(0.3)
    assert so.net_displacement([[1, 1, 1], [1, 1, 1]]) == pytest.approx(0.0)


def test_net_displacement_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        so.net_displacement([])


def test_assign_end_moved_and_near_assigns():
    # starts 0.3 from LEFT, drives to within 0.02 -> moved (0.28) AND near (0.02) -> "left".
    traj = [[0.0, -0.2, 0.2], [0.0, -0.48, 0.2]]
    assigned, dl, dr = so.assign_end(traj, _ends())
    assert assigned == "left"
    assert dl < dr


def test_assign_end_frozen_far_is_none():
    # frozen arm parked far from both ends -> closest_end would mislabel by start; gate -> "none".
    traj = [[0.5, 0.1, 0.6], [0.5, 0.1, 0.6]]
    assigned, _, _ = so.assign_end(traj, _ends())
    assert assigned == "none"


def test_assign_end_near_but_not_moved_is_none():
    # TCP sits ON the left end the whole time (near) but never MOVES (disp 0) -> "none".
    traj = [[0.0, -0.5, 0.2], [0.0, -0.5, 0.2]]
    assigned, dl, _ = so.assign_end(traj, _ends())
    assert dl == pytest.approx(0.0)
    assert assigned == "none"  # failed the min-net-displacement check


def test_assign_end_moved_but_not_near_is_none():
    # moves a lot but stays far from both ends -> "none" (failed the min-approach-dist check).
    traj = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.5]]
    assigned, _, _ = so.assign_end(traj, _ends())
    assert assigned == "none"


def test_assign_end_thresholds_none_recovers_closest_end():
    # Disabling both checks must reproduce closest_end exactly (even for a frozen arm).
    traj = [[0.0, -0.45, 0.2], [0.0, -0.45, 0.2]]
    gated, gdl, gdr = so.assign_end(traj, _ends(), min_approach_dist=None, min_net_displacement=None)
    raw, rdl, rdr = so.closest_end(traj, _ends())
    assert gated == raw == "left"
    assert (gdl, gdr) == pytest.approx((rdl, rdr))


def test_assign_end_partial_threshold_disable():
    # near-check disabled, moved-check active: a frozen-on-end arm still fails (didn't move).
    traj = [[0.0, -0.5, 0.2], [0.0, -0.5, 0.2]]
    assigned, _, _ = so.assign_end(traj, _ends(), min_approach_dist=None)
    assert assigned == "none"
    # moved-check disabled, near-check active: same frozen-on-end arm now passes (it is near).
    assigned2, _, _ = so.assign_end(traj, _ends(), min_net_displacement=None)
    assert assigned2 == "left"


# ----------------------------------------------------------- engagement signals (PRIMARY)


def test_approach_progress_toward_end_and_passive():
    left = _ends()["left"]  # (0, -0.5, 0.2)
    # drives from 0.5 m away to ON the end -> progress 0.5
    assert so.approach_progress([[0, 0, 0.2], [0, -0.5, 0.2]], left) == pytest.approx(0.5)
    # never moves toward the end -> progress ~0
    assert so.approach_progress([[0, 0, 0.2], [0, 0, 0.2]], left) == pytest.approx(0.0)


def test_approach_progress_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        so.approach_progress([], _ends()["left"])


def test_grasp_fraction_basic_and_per_arm():
    trace = [[False, False], [True, False], [True, True]]
    assert so.grasp_fraction(trace, 0) == pytest.approx(2 / 3)
    assert so.grasp_fraction(trace, 1) == pytest.approx(1 / 3)


def test_grasp_fraction_empty_and_ragged():
    assert so.grasp_fraction([], 0) == 0.0
    assert so.grasp_fraction(None, 0) == 0.0
    # ragged rows missing the queried index are skipped, not crashed.
    assert so.grasp_fraction([[True], [True, False]], 1) == pytest.approx(0.0)  # only one valid row, False


# ---------------------------------------------------- engagement-contrast scoring (PRIMARY)


def _eng(arm, seed, condition, appr, gfrac):
    return {
        "target_arm": arm,
        "seed": seed,
        "condition": condition,
        "approach_progress": appr,
        "grasp_fraction": gfrac,
    }


def test_engagement_channel_alive():
    # grasp engages MORE than wait every seed -> positive shifts, consistency 1.0.
    rolls = []
    for s in (100, 200, 300):
        rolls.append(_eng(0, s, "grasp", appr=0.30, gfrac=0.80))
        rolls.append(_eng(0, s, "wait", appr=0.00, gfrac=0.00))
    out = so.score_engagement_arm(rolls)
    assert out["target_arm"] == 0
    assert out["n_seeds"] == 3
    assert out["mean_approach_engagement_shift"] == pytest.approx(0.30)
    assert out["approach_engagement_consistency"] == pytest.approx(1.0)
    assert out["mean_grasp_close_shift"] == pytest.approx(0.80)
    assert out["grasp_close_consistency"] == pytest.approx(1.0)
    assert out["mean_approach_progress_grasp"] == pytest.approx(0.30)
    assert out["mean_approach_progress_wait"] == pytest.approx(0.0)
    assert out["null_consistency"] == 0.5


def test_engagement_channel_dead():
    # grasp behaves identically to wait -> zero shift, consistency 0.0 (no seed engaged MORE).
    rolls = []
    for s in (100, 200, 300, 400):
        rolls.append(_eng(0, s, "grasp", appr=0.20, gfrac=0.50))
        rolls.append(_eng(0, s, "wait", appr=0.20, gfrac=0.50))
    out = so.score_engagement_arm(rolls)
    assert out["mean_approach_engagement_shift"] == pytest.approx(0.0)
    assert out["approach_engagement_consistency"] == pytest.approx(0.0)
    assert out["mean_grasp_close_shift"] == pytest.approx(0.0)
    assert out["grasp_close_consistency"] == pytest.approx(0.0)


def test_engagement_pair_missing_wait_raises():
    with pytest.raises(ValueError, match="missing engagement condition"):
        so.pair_engagement_rollouts([_eng(0, 100, "grasp", 0.3, 0.8)])


def test_engagement_pair_duplicate_raises():
    rolls = [_eng(0, 100, "grasp", 0.3, 0.8), _eng(0, 100, "grasp", 0.2, 0.7)]
    with pytest.raises(ValueError, match="duplicate engagement"):
        so.pair_engagement_rollouts(rolls)


def test_engagement_bad_condition_raises():
    with pytest.raises(ValueError, match="grasp'/'wait'"):
        so.pair_engagement_rollouts([_eng(0, 100, "lift", 0.3, 0.8)])


def test_engagement_single_arm_guard():
    rolls = [_eng(0, 1, "grasp", 0.3, 0.8), _eng(1, 1, "wait", 0.0, 0.0)]
    with pytest.raises(ValueError, match="single target_arm"):
        so.score_engagement_arm(rolls)


def test_engagement_empty_raises():
    with pytest.raises(ValueError, match="no engagement rollouts"):
        so.score_engagement_arm([])


def test_score_engagement_both_arms():
    rolls = []
    for s in (100, 200):
        rolls.append(_eng(0, s, "grasp", 0.25, 0.6))
        rolls.append(_eng(0, s, "wait", 0.02, 0.0))
        rolls.append(_eng(1, s, "grasp", 0.30, 0.7))
        rolls.append(_eng(1, s, "wait", 0.01, 0.0))
    out = so.score_engagement(rolls)
    assert set(out.keys()) == {"arm0", "arm1"}
    assert out["arm0"]["approach_engagement_consistency"] == pytest.approx(1.0)
    assert out["arm1"]["indomain_end"] == "right"
    assert out["arm1"]["mean_grasp_close_shift"] == pytest.approx(0.7)


def test_engagement_rollout_dataclass_coercion():
    r = so.EngagementRollout(0, 100, "grasp", 0.3, 0.8)
    # score should accept dataclass instances directly (mixed with its wait dict partner).
    out = so.score_engagement_arm([r, _eng(0, 100, "wait", 0.0, 0.0)])
    assert out["n_seeds"] == 1
    assert out["mean_approach_engagement_shift"] == pytest.approx(0.3)


# ============================================================= hold-after-grasp scenario


def test_hold_phase_prompt_schedule_exact():
    """Phase-1 grasp->close->lift->wait per arm uses the EXACT trained strings at the right steps."""
    g, c, l = 25, 20, 50
    # arm0 (LEFT)
    assert so.hold_phase_prompt(0, 0, g, c, l) == "grasp the left end"
    assert so.hold_phase_prompt(0, 24, g, c, l) == "grasp the left end"
    assert so.hold_phase_prompt(0, 25, g, c, l) == "close the gripper"
    assert so.hold_phase_prompt(0, 44, g, c, l) == "close the gripper"
    assert so.hold_phase_prompt(0, 45, g, c, l) == "lift the left end"
    assert so.hold_phase_prompt(0, 94, g, c, l) == "lift the left end"
    assert so.hold_phase_prompt(0, 95, g, c, l) == "wait"   # Phase 2 (the hold)
    assert so.hold_phase_prompt(0, 200, g, c, l) == "wait"
    # arm1 (RIGHT)
    assert so.hold_phase_prompt(1, 0, g, c, l) == "grasp the right end"
    assert so.hold_phase_prompt(1, 25, g, c, l) == "close the gripper"
    assert so.hold_phase_prompt(1, 45, g, c, l) == "lift the right end"
    assert so.hold_phase_prompt(1, 95, g, c, l) == "wait"


def test_hold_phase_prompt_bad_arm():
    with pytest.raises(ValueError):
        so.hold_phase_prompt(2, 0, 25, 20, 50)


def test_hold_phase_prompt_strings_in_real_data():
    """Every Phase-1 string this scenario injects must be a real trained subtask string."""
    # confirmed union of subtask_arm{i}_text across robofactory_lift_barrier_ws_subtaskdart_v1.
    arm0 = {"grasp the left end", "close the gripper", "lift the left end", "wait"}
    arm1 = {"grasp the right end", "close the gripper", "lift the right end", "wait"}
    g, c, l = 25, 20, 50
    used0 = {so.hold_phase_prompt(0, t, g, c, l) for t in range(0, 200, 3)}
    used1 = {so.hold_phase_prompt(1, t, g, c, l) for t in range(0, 200, 3)}
    assert used0 <= arm0
    assert used1 <= arm1


def _hold_traces(n, z_fn, grasp0_fn, grasp1_fn):
    bz = [z_fn(t) for t in range(n)]
    gt = [[grasp0_fn(t), grasp1_fn(t)] for t in range(n)]
    return bz, gt


def test_hold_metrics_holds():
    """Barrier lifted in Phase 1 and stays up + grippers stay closed in Phase 2 -> HOLDS."""
    K = 95
    # Phase 1: rises 0.0 -> 0.30; Phase 2: stays ~0.30; both arms grasping throughout the lift/hold.
    bz, gt = _hold_traces(
        145,
        z_fn=lambda t: 0.0 + min(t, K) / K * 0.30,
        grasp0_fn=lambda t: t >= 30,
        grasp1_fn=lambda t: t >= 30,
    )
    m = so.hold_after_grasp_metrics(bz, gt, K, base_z=0.0)
    assert m["phase1_lifted"] is True
    assert m["verdict"] == "HOLDS"
    assert m["phase2_held_up"] is True
    assert m["phase2_held_grasp"] is True
    assert m["phase2_grasp_frac"] == [1.0, 1.0]


def test_hold_metrics_lets_go_drops():
    """Lifted in Phase 1 but barrier falls back in Phase 2 -> LETS-GO (held_up False)."""
    K = 95
    bz, gt = _hold_traces(
        145,
        z_fn=lambda t: (min(t, K) / K * 0.30) if t < K else max(0.0, 0.30 - (t - K) * 0.02),
        grasp0_fn=lambda t: t >= 30 and t < K,   # releases at the wait switch
        grasp1_fn=lambda t: t >= 30 and t < K,
    )
    m = so.hold_after_grasp_metrics(bz, gt, K, base_z=0.0)
    assert m["phase1_lifted"] is True
    assert m["verdict"] == "LETS-GO"
    assert m["phase2_dz"] < 0
    assert m["phase2_held_up"] is False


def test_hold_metrics_lets_go_releases_but_no_drop():
    """Barrier stays up but grippers OPEN in Phase 2 -> LETS-GO via the grasp gate."""
    K = 95
    bz, gt = _hold_traces(
        145,
        z_fn=lambda t: min(t, K) / K * 0.30,     # stays up (resting on something)
        grasp0_fn=lambda t: t >= 30 and t < K,   # both open at the switch
        grasp1_fn=lambda t: t >= 30 and t < K,
    )
    m = so.hold_after_grasp_metrics(bz, gt, K, base_z=0.0)
    assert m["phase2_held_up"] is True
    assert m["phase2_held_grasp"] is False
    assert m["verdict"] == "LETS-GO"


def test_hold_metrics_phase1_failed():
    """Barrier never rises in Phase 1 -> PHASE1_FAILED (hold test moot)."""
    K = 95
    bz, gt = _hold_traces(145, z_fn=lambda t: 0.0, grasp0_fn=lambda t: False,
                          grasp1_fn=lambda t: False)
    m = so.hold_after_grasp_metrics(bz, gt, K, base_z=0.0)
    assert m["phase1_lifted"] is False
    assert m["verdict"] == "PHASE1_FAILED"


def test_hold_metrics_no_phase2_data():
    """A rollout that ended before Phase 2 reports NO_PHASE2_DATA, never a hold verdict."""
    K = 95
    bz, gt = _hold_traces(80, z_fn=lambda t: t / K * 0.3, grasp0_fn=lambda t: True,
                          grasp1_fn=lambda t: True)
    m = so.hold_after_grasp_metrics(bz, gt, K, base_z=0.0)
    assert m["verdict"] == "NO_PHASE2_DATA"


def test_hold_metrics_threshold_clearance():
    """phase1_cleared_threshold reflects base_z + 0.15 (LiftBarrierEnv success)."""
    K = 95
    bz, gt = _hold_traces(145, z_fn=lambda t: min(t, K) / K * 0.30,
                          grasp0_fn=lambda t: True, grasp1_fn=lambda t: True)
    # base_z=0.0 -> threshold 0.15, peak 0.30 clears it.
    assert so.hold_after_grasp_metrics(bz, gt, K, base_z=0.0)["phase1_cleared_threshold"] is True
    # base_z=0.5 -> threshold 0.65, peak 0.30 does NOT clear it.
    assert so.hold_after_grasp_metrics(bz, gt, K, base_z=0.5)["phase1_cleared_threshold"] is False
    # base_z=None -> field is None (skipped).
    assert so.hold_after_grasp_metrics(bz, gt, K, base_z=None)["phase1_cleared_threshold"] is None
