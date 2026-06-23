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
