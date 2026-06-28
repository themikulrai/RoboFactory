"""PURE tests for eval_subtask_obedience (no sim / no gym.make / no GPU).

Covers the off-GPU logic of robofactory/policy/openpi_pi05/eval_subtask_obedience.py:

  SCHEDULE parsing / expansion:
    * default LB schedule validates and renders the right phases per arm
    * active_segment picks the last segment with start_step <= step
    * expand_schedule produces a per-step prompt list of the right length
    * validate_schedule rejects malformed schedules (not starting at 0,
      non-increasing starts, bad verb id)

  OBEDIENCE-decision logic (the headline metric) given FAKE L/R distances:
    * commanded-end-closer-in-BOTH-variants -> obedient
    * arm ignores prompt (always nearer one fixed end) -> NOT obedient
    * obedient in only one variant -> NOT obedient
    * the margin makes a borderline case fail
    * prompt_swap_delta sign/magnitude
    * aggregate_obedience rate + mean-delta math over several seeds

  VOCAB render of the LB left/right targets (the prompts the probe commands).

These import eval_subtask_obedience at MODULE TOP, exercising the requirement that
the module import does NOT call gym.make / build an env (it may load sapien like the
other pure tests in this repo, which is fine on the login node -- only RENDERING is
dark, not importing).

Run PURE:
    cd /iris/u/mikulrai/RoboFactory-subtask-wt && PYTHONPATH=. \\
      /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \\
      scripts/dart/tests/test_subtask_obedience.py -x -q
"""
from __future__ import annotations

import os.path as osp
import sys

import pytest

# make the repo root + the openpi_pi05 package dir importable
_THIS = osp.dirname(osp.abspath(__file__))
_ROOT = osp.dirname(osp.dirname(osp.dirname(_THIS)))  # .../RoboFactory-subtask-wt
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from robofactory.planner import subtask_vocab as vocab  # noqa: E402
import robofactory.policy.openpi_pi05.eval_subtask_obedience as m  # noqa: E402


# ===========================================================================
# Module-import sanity: importing must NOT have built an env / called gym.make.
# ===========================================================================
def test_module_import_is_sim_free():
    """Merely importing the eval must not build a sapien env (no gym.make).

    The hierarchical eval (which DOES import gym/sapien helpers eagerly) is
    lazy-imported only inside sim functions, so it must be absent right after
    importing the obedience module."""
    assert "eval_hierarchical_lift_barrier" not in sys.modules
    # the lazy accessor cache must still be empty
    assert m._HIER is None


# ===========================================================================
# VOCAB render of the LB targets the probe commands
# ===========================================================================
def test_lb_target_render():
    assert vocab.render(vocab.APPROACH, vocab.LB_TARGETS["left_end"], "LiftBarrier") == "grasp the left end"
    assert vocab.render(vocab.APPROACH, vocab.LB_TARGETS["right_end"], "LiftBarrier") == "grasp the right end"
    assert vocab.render(vocab.WAIT, 0, "LiftBarrier") == "wait"
    assert vocab.render(vocab.LIFT, vocab.LB_TARGETS["left_end"], "LiftBarrier") == "lift the left end"
    # contact-id mapping the probe relies on to resolve the 3D end points
    assert vocab.LB_TARGET_CONTACT_ID[vocab.LB_TARGETS["left_end"]] == 1
    assert vocab.LB_TARGET_CONTACT_ID[vocab.LB_TARGETS["right_end"]] == 2


# ===========================================================================
# SCHEDULE parsing / expansion
# ===========================================================================
def test_default_schedule_validates_and_phases():
    sched = m.default_lb_schedule(approach_steps=60, close_steps=30)
    m.validate_schedule(sched)  # must not raise
    # arm0 -> LEFT end, arm1 -> RIGHT end
    assert m.schedule_prompt(sched[0], 0) == "grasp the left end"
    assert m.schedule_prompt(sched[1], 0) == "grasp the right end"
    # phase boundaries: 0..59 approach, 60..89 close, 90+ lift
    assert m.schedule_prompt(sched[0], 59) == "grasp the left end"
    assert m.schedule_prompt(sched[0], 60) == "close the gripper"
    assert m.schedule_prompt(sched[0], 89) == "close the gripper"
    assert m.schedule_prompt(sched[0], 90) == "lift the left end"
    assert m.schedule_prompt(sched[0], 999) == "lift the left end"
    assert m.schedule_prompt(sched[1], 95) == "lift the right end"


def test_active_segment_picks_last_le():
    segs = [(0, vocab.APPROACH, 1), (10, vocab.CLOSE_GRIPPER, 0), (20, vocab.LIFT, 1)]
    assert m.active_segment(segs, 0) == segs[0]
    assert m.active_segment(segs, 9) == segs[0]
    assert m.active_segment(segs, 10) == segs[1]
    assert m.active_segment(segs, 19) == segs[1]
    assert m.active_segment(segs, 20) == segs[2]
    assert m.active_segment(segs, 1000) == segs[2]


def test_expand_schedule_length_and_content():
    sched = m.default_lb_schedule(approach_steps=5, close_steps=3)
    exp = m.expand_schedule(sched, n_steps=12)
    assert set(exp.keys()) == {0, 1}
    assert len(exp[0]) == 12 and len(exp[1]) == 12
    # arm0: 0..4 approach left, 5..7 close, 8..11 lift left
    assert exp[0][:5] == ["grasp the left end"] * 5
    assert exp[0][5:8] == ["close the gripper"] * 3
    assert exp[0][8:] == ["lift the left end"] * 4
    assert exp[1][0] == "grasp the right end"
    assert exp[1][8] == "lift the right end"


def test_validate_schedule_rejects_bad_start():
    with pytest.raises(ValueError):
        m.validate_schedule({0: [(3, vocab.APPROACH, 1)]})  # must start at 0


def test_validate_schedule_rejects_non_increasing():
    with pytest.raises(ValueError):
        m.validate_schedule({0: [(0, vocab.APPROACH, 1), (0, vocab.LIFT, 1)]})


def test_validate_schedule_rejects_bad_verb():
    with pytest.raises(ValueError):
        m.validate_schedule({0: [(0, 999, 1)]})


def test_validate_schedule_rejects_empty():
    with pytest.raises(ValueError):
        m.validate_schedule({})
    with pytest.raises(ValueError):
        m.validate_schedule({0: []})


# ===========================================================================
# OBEDIENCE-decision logic (headline metric) with fake L/R distances
# ===========================================================================
def test_obedient_when_commanded_closer_in_both():
    # variant L (commanded LEFT) ends near LEFT; variant R (commanded RIGHT) near RIGHT
    assert m.decide_obedience(dist_L_to_left=0.05, dist_L_to_right=0.40,
                              dist_R_to_left=0.42, dist_R_to_right=0.06) is True


def test_not_obedient_when_prompt_ignored_fixed_end():
    # An arm that ALWAYS goes to LEFT regardless of prompt: variant R ends nearer LEFT.
    assert m.decide_obedience(dist_L_to_left=0.05, dist_L_to_right=0.40,
                              dist_R_to_left=0.05, dist_R_to_right=0.40) is False
    # An arm that ALWAYS goes to RIGHT: variant L ends nearer RIGHT.
    assert m.decide_obedience(dist_L_to_left=0.40, dist_L_to_right=0.05,
                              dist_R_to_left=0.42, dist_R_to_right=0.06) is False


def test_not_obedient_when_only_one_variant_correct():
    # variant L correct (near left) but variant R ALSO ends near left -> not obedient
    assert m.decide_obedience(dist_L_to_left=0.05, dist_L_to_right=0.40,
                              dist_R_to_left=0.10, dist_R_to_right=0.20) is False


def test_margin_makes_borderline_fail():
    # gaps are 0.02 each; a 0.05 margin should make BOTH fail the strict test
    assert m.decide_obedience(0.18, 0.20, 0.20, 0.18, margin=0.0) is True
    assert m.decide_obedience(0.18, 0.20, 0.20, 0.18, margin=0.05) is False


def test_tie_is_not_obedient():
    # exactly equal distances -> strict < fails -> not obedient
    assert m.decide_obedience(0.20, 0.20, 0.20, 0.20) is False


def test_prompt_swap_delta_sign_and_value():
    # obedient case: avg of (0.40-0.05) and (0.42-0.06) = (0.35 + 0.36)/2 = 0.355
    assert m.prompt_swap_delta(0.05, 0.40, 0.42, 0.06) == pytest.approx(0.355)
    # ANTI-obedient (does the OPPOSITE in both variants): variant L ends near RIGHT,
    # variant R ends near LEFT -> both gaps negative -> overall negative.
    d = m.prompt_swap_delta(dist_L_to_left=0.40, dist_L_to_right=0.05,
                            dist_R_to_left=0.05, dist_R_to_right=0.40)
    assert d == pytest.approx(-0.35)
    # "always goes to the SAME end" cancels out to exactly zero (perfect on one
    # variant, equally wrong on the other) -- a useful sanity edge.
    assert m.prompt_swap_delta(0.05, 0.40, 0.05, 0.40) == pytest.approx(0.0)


def test_aggregate_rate_and_mean_delta():
    per_seed = [
        dict(seed=1, dL_left=0.05, dL_right=0.40, dR_left=0.42, dR_right=0.06),  # obedient
        dict(seed=2, dL_left=0.40, dL_right=0.05, dR_left=0.42, dR_right=0.06),  # not (L wrong)
        dict(seed=3, dL_left=0.10, dL_right=0.30, dR_left=0.30, dR_right=0.10),  # obedient
        dict(seed=4, dL_left=0.05, dL_right=0.40, dR_left=0.05, dR_right=0.40),  # not (always left)
    ]
    agg = m.aggregate_obedience(per_seed)
    assert agg["num_seeds"] == 4
    assert agg["num_obedient"] == 2
    assert agg["obedience_rate"] == pytest.approx(0.5)
    # every per-seed record gets an 'obedient' bool + 'prompt_swap_delta'
    assert [r["obedient"] for r in agg["per_seed"]] == [True, False, True, False]
    assert all("prompt_swap_delta" in r for r in agg["per_seed"])


def test_aggregate_empty_is_zero():
    agg = m.aggregate_obedience([])
    assert agg["num_seeds"] == 0
    assert agg["obedience_rate"] == 0.0
    assert agg["mean_prompt_swap_delta"] == 0.0


# ===========================================================================
# WAIT-vs-ACT probe logic (the SECOND gate probe) with fake motion magnitudes
# ===========================================================================
def test_waitact_ratio_floors_denominator():
    # perfectly still wait -> finite large ratio (not inf) via eps floor
    r = m.waitact_ratio(wait_motion=0.0, act_motion=0.2)
    assert r == pytest.approx(0.2 / 1e-4)
    # normal case
    assert m.waitact_ratio(0.01, 0.2) == pytest.approx(20.0)


def test_decide_waitact_obedient_when_act_moves_wait_still():
    # ideal: wait ~0, act clearly moves -> pass
    assert m.decide_waitact(wait_motion=0.005, act_motion=0.20) is True


def test_decide_waitact_fails_when_frozen_on_both():
    # arm frozen on BOTH prompts: ratio ~1 (< min_ratio) -> fail
    assert m.decide_waitact(wait_motion=0.001, act_motion=0.001) is False


def test_decide_waitact_fails_when_drifts_on_wait():
    # high ratio but the WAIT rollout itself moves a lot (> max_wait_motion) -> fail
    # (a policy that ignores 'wait' and drifts is NOT obedient even if act moves more)
    assert m.decide_waitact(wait_motion=0.10, act_motion=0.40, max_wait_motion=0.05) is False


def test_decide_waitact_threshold_boundaries():
    # exactly at min_ratio (3.0) with tiny wait -> pass (>=)
    assert m.decide_waitact(0.01, 0.03, min_ratio=3.0, max_wait_motion=0.05) is True
    # just under min_ratio -> fail
    assert m.decide_waitact(0.01, 0.029, min_ratio=3.0, max_wait_motion=0.05) is False
    # wait exactly at max_wait_motion -> still pass (<=); just over -> fail
    assert m.decide_waitact(0.05, 0.30, min_ratio=3.0, max_wait_motion=0.05) is True
    assert m.decide_waitact(0.051, 0.30, min_ratio=3.0, max_wait_motion=0.05) is False


def test_aggregate_waitact_rate_and_means():
    per_seed = [
        dict(seed=1, wait_motion=0.005, act_motion=0.20),  # pass
        dict(seed=2, wait_motion=0.001, act_motion=0.001),  # fail (frozen both)
        dict(seed=3, wait_motion=0.01, act_motion=0.30),   # pass
        dict(seed=4, wait_motion=0.10, act_motion=0.40),   # fail (drifts on wait)
    ]
    agg = m.aggregate_waitact(per_seed)
    assert agg["num_seeds"] == 4
    assert agg["num_pass"] == 2
    assert agg["pass_rate"] == pytest.approx(0.5)
    assert [r["waitact_pass"] for r in agg["per_seed"]] == [True, False, True, False]
    assert agg["mean_wait_motion"] == pytest.approx((0.005 + 0.001 + 0.01 + 0.10) / 4)
    assert agg["mean_act_motion"] == pytest.approx((0.20 + 0.001 + 0.30 + 0.40) / 4)


def test_aggregate_waitact_empty_is_zero():
    agg = m.aggregate_waitact([])
    assert agg["num_seeds"] == 0
    assert agg["pass_rate"] == 0.0
    assert agg["mean_ratio"] == 0.0


# ===========================================================================
# GATE VERDICT logic (combines left-vs-right obedience + wait-vs-act motion)
# ===========================================================================
def test_gate_verdict_pass_needs_both():
    # obedience high AND wait<<act -> PASS
    v = m.gate_verdict(obedience_rate=0.8, mean_wait_motion=0.01, mean_act_motion=0.20)
    assert v["verdict"] == "PASS"
    assert v["left_right_obedience_pass"] and v["wait_act_pass"]


def test_gate_verdict_fail_low_obedience():
    # obedience too low even though motion is fine -> FAIL
    v = m.gate_verdict(obedience_rate=0.5, mean_wait_motion=0.01, mean_act_motion=0.20)
    assert v["verdict"] == "FAIL"
    assert v["left_right_obedience_pass"] is False
    assert v["wait_act_pass"] is True


def test_gate_verdict_fail_no_motion_separation():
    # obedience fine but the arm moves on BOTH prompts (low ratio) -> FAIL
    v = m.gate_verdict(obedience_rate=0.9, mean_wait_motion=0.15, mean_act_motion=0.18)
    assert v["verdict"] == "FAIL"
    assert v["left_right_obedience_pass"] is True
    assert v["wait_act_pass"] is False


def test_gate_verdict_threshold_boundary():
    # exactly at obedience_threshold (0.7) and ratio threshold (3.0) with small wait -> PASS
    v = m.gate_verdict(obedience_rate=0.7, mean_wait_motion=0.01, mean_act_motion=0.03,
                       obedience_threshold=0.7, motion_ratio_threshold=3.0,
                       max_wait_motion=0.05)
    assert v["verdict"] == "PASS"


# ===========================================================================
# GRASP-vs-LIFT probe logic (VALID in-distribution verb test) with fake dz's
# ===========================================================================
def test_grasplift_gap_sign_and_value():
    # lift raises more than grasp -> positive gap
    assert m.grasplift_gap(dz_grasp=0.01, dz_lift=0.12) == pytest.approx(0.11)
    # lift LOWER than grasp (disobedient / swapped) -> negative gap
    assert m.grasplift_gap(dz_grasp=0.10, dz_lift=0.02) == pytest.approx(-0.08)
    # identical motion on both verbs (verb ignored) -> exactly zero gap
    assert m.grasplift_gap(0.05, 0.05) == pytest.approx(0.0)


def test_decide_grasplift_obedient_when_lift_rises_more():
    # lift clearly higher than grasp by >= min_gap -> obedient
    assert m.decide_grasplift(dz_grasp=0.01, dz_lift=0.12, min_gap=0.03) is True


def test_decide_grasplift_fails_when_verb_ignored():
    # arm does the SAME thing on both verbs (gap ~0 < min_gap) -> not obedient
    assert m.decide_grasplift(dz_grasp=0.05, dz_lift=0.05, min_gap=0.03) is False
    # lift rises LESS than grasp (anti-obedient) -> not obedient
    assert m.decide_grasplift(dz_grasp=0.10, dz_lift=0.02, min_gap=0.03) is False


def test_decide_grasplift_threshold_boundary():
    # exactly at min_gap -> pass (>=); just under -> fail
    assert m.decide_grasplift(0.00, 0.03, min_gap=0.03) is True
    assert m.decide_grasplift(0.00, 0.029, min_gap=0.03) is False


def test_aggregate_grasplift_rate_and_means():
    per_seed = [
        dict(seed=1, dz_grasp=0.01, dz_lift=0.12, tcp_sep=0.15),  # pass (gap 0.11)
        dict(seed=2, dz_grasp=0.05, dz_lift=0.05, tcp_sep=0.01),  # fail (gap 0.0)
        dict(seed=3, dz_grasp=0.02, dz_lift=0.20, tcp_sep=0.22),  # pass (gap 0.18)
        dict(seed=4, dz_grasp=0.10, dz_lift=0.02, tcp_sep=0.09),  # fail (gap -0.08)
    ]
    agg = m.aggregate_grasplift(per_seed, min_gap=0.03)
    assert agg["num_seeds"] == 4
    assert agg["num_pass"] == 2
    assert agg["pass_rate"] == pytest.approx(0.5)
    assert [r["grasplift_pass"] for r in agg["per_seed"]] == [True, False, True, False]
    assert agg["mean_dz_grasp"] == pytest.approx((0.01 + 0.05 + 0.02 + 0.10) / 4)
    assert agg["mean_dz_lift"] == pytest.approx((0.12 + 0.05 + 0.20 + 0.02) / 4)
    assert agg["mean_grasplift_gap"] == pytest.approx((0.11 + 0.0 + 0.18 - 0.08) / 4)
    # axis-agnostic insurance metric is aggregated when present
    assert agg["mean_tcp_sep"] == pytest.approx((0.15 + 0.01 + 0.22 + 0.09) / 4)


def test_aggregate_grasplift_empty_is_zero():
    agg = m.aggregate_grasplift([])
    assert agg["num_seeds"] == 0
    assert agg["pass_rate"] == 0.0
    assert agg["mean_grasplift_gap"] == 0.0


# ===========================================================================
# VALID verdict (decent ckpts): waitact + grasplift ONLY, no left-vs-right
# ===========================================================================
def test_valid_verdict_pass_needs_both():
    # waitact good (ratio 20, wait small, rate 1.0) AND grasplift good (gap 0.10, rate 1.0)
    v = m.valid_verdict(
        waitact_pass_rate=1.0, mean_wait_motion=0.01, mean_act_motion=0.20,
        grasplift_pass_rate=1.0, mean_grasplift_gap=0.10,
    )
    assert v["verdict"] == "PASS"
    assert v["wait_act_pass"] and v["grasp_lift_pass"]


def test_valid_verdict_fail_on_grasplift_even_if_waitact_ok():
    # waitact fine but the verb channel is dead (gap below threshold) -> FAIL
    v = m.valid_verdict(
        waitact_pass_rate=1.0, mean_wait_motion=0.01, mean_act_motion=0.20,
        grasplift_pass_rate=0.2, mean_grasplift_gap=0.005,
    )
    assert v["verdict"] == "FAIL"
    assert v["wait_act_pass"] is True
    assert v["grasp_lift_pass"] is False


def test_valid_verdict_fail_on_waitact_even_if_grasplift_ok():
    # grasplift fine but the arm drifts on 'wait' (low ratio / high wait_motion) -> FAIL
    v = m.valid_verdict(
        waitact_pass_rate=0.3, mean_wait_motion=0.15, mean_act_motion=0.18,
        grasplift_pass_rate=1.0, mean_grasplift_gap=0.10,
    )
    assert v["verdict"] == "FAIL"
    assert v["wait_act_pass"] is False
    assert v["grasp_lift_pass"] is True


def test_valid_verdict_uses_no_left_right():
    # The valid verdict dict must NOT carry any left-vs-right obedience field (it is OOD
    # for decent ckpts and must not influence the decision).
    v = m.valid_verdict(
        waitact_pass_rate=1.0, mean_wait_motion=0.01, mean_act_motion=0.20,
        grasplift_pass_rate=1.0, mean_grasplift_gap=0.10,
    )
    assert "obedience_rate" not in v
    assert "left_right_obedience_pass" not in v


def test_valid_verdict_threshold_boundary():
    # exactly at the rate / wait_motion / gap thresholds (ratio comfortably above 3.0 to
    # avoid the 0.15/0.05 == 2.9999.. float-underflow edge) -> PASS.
    v = m.valid_verdict(
        waitact_pass_rate=0.7, mean_wait_motion=0.05, mean_act_motion=0.20,  # ratio 4.0
        grasplift_pass_rate=0.7, mean_grasplift_gap=0.03,
        waitact_rate_threshold=0.7, motion_ratio_threshold=3.0, max_wait_motion=0.05,
        grasplift_rate_threshold=0.7, grasplift_gap_threshold=0.03,
    )
    assert v["verdict"] == "PASS"


# ===========================================================================
# WAIT prompt the waitact probe holds the OTHER arm on (vocab sanity)
# ===========================================================================
def test_waitact_default_act_prompt_is_vocab():
    # the default ACT prompt must be a real LB-vocab string (else the LL gets OOV)
    assert vocab.render(vocab.APPROACH, vocab.LB_TARGETS["left_end"], "LiftBarrier") == "grasp the left end"


def test_grasplift_prompts_are_vocab():
    # both grasp-vs-lift prompts must be real LB-vocab (same side, only the verb differs)
    assert vocab.render(vocab.APPROACH, vocab.LB_TARGETS["left_end"], "LiftBarrier") == "grasp the left end"
    assert vocab.render(vocab.LIFT, vocab.LB_TARGETS["left_end"], "LiftBarrier") == "lift the left end"
