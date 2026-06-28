"""RUN ON A GPU COMPUTE NODE ONLY (shader_pack=default).

Subtask-OBEDIENCE eval for the subtask-conditioned low-level (LL) pi0.5 policy.

This is the proof that the trained LL FOLLOWS its per-arm subtask instruction --
the whole point of the subtask-conditioned dataset. It is a fork of the
hierarchical eval (eval_hierarchical_lift_barrier.py) with the high-level (HL)
Qwen orchestrator REPLACED by a SCRIPTED, known-ground-truth subtask schedule,
plus an A/B "obedience probe" that needs no task success to give signal.

It reuses the hierarchical eval's machinery wholesale (imported, not copied):
  - env build + shader_pack=default guard + login-node refusal  (_assert_*)
  - the per-arm LL websocket clients / --mock-ll stand-in       (_make_ll_policies)
  - the per-arm obs+prompt build that routes a subtask string into ONE arm's LL
    language input (_build_ll_obs ... obs_i["prompt"]=prompts[i])  <-- the channel
  - the delta-from-qpos action decode, the tiled multiview video + subtask overlay
  - the strict grasp-gated success (success_candidates.lift_barrier_success_strict)

Modes
-----
  schedule  Drive each arm with a fixed TIME-INDEXED subtask schedule (no HL).
            The default LB schedule should SOLVE the task (both arms: approach ->
            close -> lift). Reports STRICT success so it can be compared against a
            prompt-ignoring baseline. This shows the LL can be *driven* by prompts.

  probe     The HEADLINE metric (works even on an undertrained ckpt). For each of
            N seeds, reset to the SAME state and run two SHORT rollouts differing
            ONLY in arm0's commanded subtask:
              Variant L: arm0 = "grasp the left end",  arm1 = "wait"
              Variant R: arm0 = "grasp the right end", arm1 = "wait"
            Measure arm0's final TCP distance to the LEFT and RIGHT contact points
            (resolved once at the reset pose via the labeled-direction grasp pose).
            OBEDIENCE(seed) := variant L ends strictly nearer LEFT AND variant R
            ends strictly nearer RIGHT. Reports per-seed L/R distances, the
            obedience rate, and a prompt-swap delta. The probe does NOT require
            task success -- it only measures which end the arm moves toward.

  both      schedule then probe.

  VALIDITY NOTE (decentralized ckpts): the left-vs-right probe is INVALID for the
  per-arm DECENT ckpts -- arm0 ONLY ever trained on the LEFT end ("grasp/lift the
  left end" + "wait"/"close"), so commanding it "grasp the RIGHT end" is OUT-OF-
  DISTRIBUTION and a 0% there is an artifact, not disobedience. For decent ckpts use
  --mode valid, which combines two IN-DISTRIBUTION probes only:
    * wait-vs-act   ("wait" vs "grasp the left end")  -- does the arm hold vs move?
    * grasp-vs-lift ("grasp the left end" vs "lift the left end") -- same side, the
      VERB differs; obedience => "lift" raises the TCP clearly more than "grasp".
      Both variants are PRIMED with an identical approach->close grasp prefix first
      (grasplift_prime_steps), because the LL only ever saw "lift" AFTER a grasp -- a
      cold "lift" from the neutral reset pose would itself be OOD. dz is measured from
      the post-grasp pose. We also log tcp_sep = ||tcp_lift - tcp_grasp|| as an
      axis-agnostic insurance metric (distinct verbs should drive the TCP to distinct
      places even if both descend).
  --mode gate (left-vs-right + waitact) is kept for the LEGACY/centralized case only.

A prompt-IGNORING baseline checkpoint can be run through the SAME probe via
--baseline-checkpoint (served on --baseline-ports) for a side-by-side number.

Three-process design (same as the hierarchical eval, minus the HL server):
  - The LL pi0.5 server(s) run in the OPENPI env (numpy<2) behind openpi websockets
    (scripts/serve_policy.py policy:checkpoint --policy.config=... --policy.dir=...).
  - THIS eval runs in the RoboFactory env (numpy<2, SAPIEN) as a websocket client.

ALL gym.make / sapien / planner imports live INSIDE functions called from main, so
importing this module is sim-free (the pure obedience/schedule logic is unit-tested
off-GPU; see robofactory/policy/openpi_pi05/tests/test_subtask_obedience.py).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make the openpi_pi05 package dir + repo root importable. We do NOT import the
# hierarchical eval (or sapien/gym) at MODULE TOP: that module imports sapien at
# its top level (~10s + CUDA libs), which would make merely importing THIS module
# sim-bound and break the off-GPU pure tests. Instead we lazy-import it via _hier()
# inside the sim-touching functions only. subtask_vocab IS import-clean (pure
# python, no sapien) so it is safe at module top.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from robofactory.planner import subtask_vocab as vocab  # noqa: E402


TASK = "LiftBarrier"  # subtask_vocab task key (env id is "LiftBarrier-rf")

# Default LiftBarrier config (copied from eval_hierarchical_lift_barrier.Args.config
# so this module does NOT import the sapien-heavy hier module just to read a string).
_DEFAULT_CONFIG = (
    "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml"
)

_HIER = None


def _hier():
    """Lazy import of the hierarchical eval (pulls sapien). Sim-path only."""
    global _HIER
    if _HIER is None:
        import eval_hierarchical_lift_barrier as _h  # noqa: E402
        _HIER = _h
    return _HIER


# ===========================================================================
# PURE schedule logic (no sim) -- unit-tested off-GPU
# ===========================================================================

# A schedule is, per arm, an ordered list of (start_step, verb_id, target_id)
# segments. The active segment for a given step is the last one whose start_step
# <= step. We render (verb_id, target_id) -> NL prompt via vocab.render(task).

ScheduleSeg = Tuple[int, int, int]  # (start_step, verb_id, target_id)


def default_lb_schedule(
    approach_steps: int = 60, close_steps: int = 30
) -> Dict[int, List[ScheduleSeg]]:
    """Default LiftBarrier schedule that SHOULD solve the task.

    Both arms run approach -> close_gripper -> lift, arm0 on the LEFT end and arm1
    on the RIGHT end (contact ids 1 / 2). The phase boundaries are time-indexed
    (env steps), tuned to be generous; the LL chunks are short so it re-reads the
    prompt frequently. ``close_steps`` is the close-gripper window before lift.

    Returns {arm_index: [ (start_step, verb_id, target_id), ... ]}.
    """
    left = vocab.LB_TARGETS["left_end"]
    right = vocab.LB_TARGETS["right_end"]
    t0 = 0
    t1 = approach_steps
    t2 = approach_steps + close_steps
    return {
        0: [
            (t0, vocab.APPROACH, left),
            (t1, vocab.CLOSE_GRIPPER, 0),
            (t2, vocab.LIFT, left),
        ],
        1: [
            (t0, vocab.APPROACH, right),
            (t1, vocab.CLOSE_GRIPPER, 0),
            (t2, vocab.LIFT, right),
        ],
    }


def staggered_lb_schedule(
    lead_arm: int = 0, approach_steps: int = 60, close_steps: int = 30, lead_gap: int = 0
) -> Dict[int, List[ScheduleSeg]]:
    """STAGGERED LiftBarrier schedule: ``lead_arm`` runs approach->close->lift FIRST;
    the follower holds WAIT until the leader has grasped+lifted, then runs its own
    approach->close->lift. Tests whether the LL respects per-arm ORDERING (does the
    follower actually hold while the leader goes first?). NOT a left/right-end swap —
    arm0 stays on the left end, arm1 on the right; only the TIMING differs.
    lead_arm=0 -> "left arm goes first"; lead_arm=1 -> "right arm goes first".
    """
    left = vocab.LB_TARGETS["left_end"]
    right = vocab.LB_TARGETS["right_end"]
    tgt = {0: left, 1: right}
    if lead_gap <= 0:
        # follower starts approaching AFTER the leader has grasped (approach+close).
        lead_gap = approach_steps + close_steps + 10
    foll = 1 - lead_arm
    # CRITICAL (matches the dataset, verified): the bar can only be lifted once BOTH
    # arms have grasped — lifting one end before the other grasps moves the bar away
    # (no chance of success). So the leader grasps and HOLDS (close_gripper, no lift)
    # until the follower has also grasped, then BOTH lift TOGETHER at t_lift.
    t_lift = lead_gap + approach_steps + close_steps
    return {
        lead_arm: [
            (0, vocab.APPROACH, tgt[lead_arm]),
            (approach_steps, vocab.CLOSE_GRIPPER, 0),   # grasp, then hold (no lift yet)
            (t_lift, vocab.LIFT, tgt[lead_arm]),        # lift only after follower grasps
        ],
        foll: [
            (0, vocab.WAIT, 0),                          # wait while leader grasps
            (lead_gap, vocab.APPROACH, tgt[foll]),
            (lead_gap + approach_steps, vocab.CLOSE_GRIPPER, 0),
            (t_lift, vocab.LIFT, tgt[foll]),             # synchronized lift with leader
        ],
    }


def validate_schedule(schedule: Dict[int, List[ScheduleSeg]]) -> None:
    """Fail loudly on a malformed schedule (so a typo never silently no-ops).

    Each arm's segments must be non-empty, start at step 0, have strictly
    increasing start_steps, and use known verb/target ids.
    """
    if not schedule:
        raise ValueError("empty schedule")
    for arm, segs in schedule.items():
        if not segs:
            raise ValueError(f"arm {arm}: empty segment list")
        if segs[0][0] != 0:
            raise ValueError(f"arm {arm}: first segment must start at step 0, got {segs[0][0]}")
        prev = -1
        for (start, verb_id, target_id) in segs:
            if start <= prev:
                raise ValueError(
                    f"arm {arm}: start_steps must be strictly increasing, "
                    f"got {start} after {prev}"
                )
            prev = start
            if verb_id not in range(len(vocab.VERBS)):
                raise ValueError(f"arm {arm}: unknown verb_id {verb_id}")
            # render() validates (verb, target) jointly for the task.
            vocab.render(verb_id, target_id, TASK)


def active_segment(segs: List[ScheduleSeg], step: int) -> ScheduleSeg:
    """The active segment at ``step`` = the last segment with start_step <= step."""
    active = segs[0]
    for seg in segs:
        if seg[0] <= step:
            active = seg
        else:
            break
    return active


def schedule_prompt(segs: List[ScheduleSeg], step: int, task: str = TASK) -> str:
    """Rendered NL prompt for the active segment of ``segs`` at ``step``."""
    _, verb_id, target_id = active_segment(segs, step)
    return vocab.render(verb_id, target_id, task)


def expand_schedule(
    schedule: Dict[int, List[ScheduleSeg]], n_steps: int, task: str = TASK
) -> Dict[int, List[str]]:
    """Expand a per-arm segment schedule to a per-step list of NL prompts.

    Returns {arm: [prompt_step0, prompt_step1, ...]} of length ``n_steps`` each.
    Pure -- used both at rollout time and in unit tests.
    """
    out: Dict[int, List[str]] = {}
    for arm, segs in schedule.items():
        out[arm] = [schedule_prompt(segs, s, task) for s in range(n_steps)]
    return out


# ===========================================================================
# PURE obedience-decision logic (no sim) -- the headline metric
# ===========================================================================


def decide_obedience(
    dist_L_to_left: float,
    dist_L_to_right: float,
    dist_R_to_left: float,
    dist_R_to_right: float,
    margin: float = 0.0,
) -> bool:
    """OBEDIENCE for one seed's A/B pair.

    Obedient iff, in BOTH variants, the COMMANDED end is approached strictly
    closer than the alternative:
      * Variant L commanded LEFT  -> needs dist_L_to_left + margin < dist_L_to_right
      * Variant R commanded RIGHT -> needs dist_R_to_right + margin < dist_R_to_left

    ``margin`` (metres) is an optional strictness buffer (default 0 = strict <).
    """
    left_ok = (dist_L_to_left + margin) < dist_L_to_right
    right_ok = (dist_R_to_right + margin) < dist_R_to_left
    return bool(left_ok and right_ok)


def prompt_swap_delta(
    dist_L_to_left: float,
    dist_L_to_right: float,
    dist_R_to_left: float,
    dist_R_to_right: float,
) -> float:
    """How much CLOSER the commanded end is than the alternative, averaged over the
    two variants (positive => obeys on average; metres).

      variant L gap = dist_L_to_right - dist_L_to_left   (commanded LEFT)
      variant R gap = dist_R_to_left  - dist_R_to_right  (commanded RIGHT)
    """
    gap_L = dist_L_to_right - dist_L_to_left
    gap_R = dist_R_to_left - dist_R_to_right
    return float((gap_L + gap_R) / 2.0)


# ---------------------------------------------------------------------------
# WAIT-vs-ACT probe logic (no sim). The SECOND probe required by the gate:
# command the probed arm "wait" for K steps vs "grasp the left end" for K steps,
# holding the OTHER arm fixed, and compare the probed arm's TCP MOTION magnitude.
# Obedience => wait-motion ~ 0, act-motion clearly larger.
# ---------------------------------------------------------------------------


def waitact_ratio(wait_motion: float, act_motion: float, eps: float = 1e-4) -> float:
    """act / wait motion ratio (higher = more obedient). ``eps`` floors the
    denominator so a perfectly-still 'wait' (the ideal) gives a large finite
    ratio instead of inf."""
    return float(act_motion / max(wait_motion, eps))


def decide_waitact(wait_motion: float, act_motion: float, min_ratio: float = 3.0,
                   max_wait_motion: float = 0.05) -> bool:
    """Per-seed wait-vs-act PASS.

    Obedient iff the 'act' rollout moves the probed arm's TCP much more than the
    'wait' rollout (ratio >= ``min_ratio``) AND the 'wait' rollout barely moves
    (wait_motion <= ``max_wait_motion`` metres). Both gates matter: a policy that
    drifts on BOTH prompts can have a high ratio yet not be holding still on
    'wait', and a policy frozen on BOTH has a tiny ratio.
    """
    ratio_ok = waitact_ratio(wait_motion, act_motion) >= min_ratio
    wait_ok = wait_motion <= max_wait_motion
    return bool(ratio_ok and wait_ok)


def aggregate_waitact(per_seed: List[dict], min_ratio: float = 3.0,
                      max_wait_motion: float = 0.05) -> dict:
    """Aggregate per-seed wait-vs-act records.

    Each record needs ``wait_motion`` and ``act_motion`` (metres of TCP
    displacement). Returns rate + mean motions + mean ratio + per-seed pass.
    """
    n = len(per_seed)
    out_seeds = []
    n_pass = 0
    ratios, wmot, amot = [], [], []
    for rec in per_seed:
        wm, am = rec["wait_motion"], rec["act_motion"]
        r = waitact_ratio(wm, am)
        ok = decide_waitact(wm, am, min_ratio=min_ratio, max_wait_motion=max_wait_motion)
        n_pass += int(ok)
        ratios.append(r)
        wmot.append(wm)
        amot.append(am)
        out_seeds.append({**rec, "ratio": r, "waitact_pass": ok})
    return {
        "num_seeds": n,
        "num_pass": n_pass,
        "pass_rate": (n_pass / n) if n else 0.0,
        "mean_wait_motion": float(np.mean(wmot)) if wmot else 0.0,
        "mean_act_motion": float(np.mean(amot)) if amot else 0.0,
        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "min_ratio": min_ratio,
        "max_wait_motion": max_wait_motion,
        "per_seed": out_seeds,
    }


# ---------------------------------------------------------------------------
# GRASP-vs-LIFT probe logic (no sim). The VALID IN-DISTRIBUTION verb test for the
# DECENTRALIZED ckpts. arm0 ONLY ever trained on the LEFT end ("grasp the left
# end" / "lift the left end"), so left-vs-right is OOD and dropped from the decent
# verdict. grasp-vs-lift keeps the SAME side (left) and varies only the VERB:
# from a matched reset, command "grasp the left end" vs "lift the left end" and
# compare the probed arm's VERTICAL (z) TCP displacement. If the LL obeys the verb
# channel, "lift" should raise the TCP clearly MORE than "grasp" (which only
# approaches / stays low). Obedience => (dz_lift - dz_grasp) >= a clear margin.
# ---------------------------------------------------------------------------


def grasplift_gap(dz_grasp: float, dz_lift: float) -> float:
    """How much MORE the 'lift' rollout raised the probed arm's TCP than 'grasp'
    (metres of vertical displacement). Positive => verb obeyed (lift rises more)."""
    return float(dz_lift - dz_grasp)


def decide_grasplift(dz_grasp: float, dz_lift: float, min_gap: float = 0.03) -> bool:
    """Per-seed grasp-vs-lift PASS.

    Obedient iff the 'lift' rollout raised the probed arm's TCP at least ``min_gap``
    metres MORE than the 'grasp' rollout did (``dz_lift - dz_grasp >= min_gap``).
    Both verbs share the SAME side (left) so this is fully in-distribution for the
    decentralized arm0; only the VERB differs, so a difference is verb-obedience.
    """
    return bool(grasplift_gap(dz_grasp, dz_lift) >= min_gap)


def aggregate_grasplift(per_seed: List[dict], min_gap: float = 0.03) -> dict:
    """Aggregate per-seed grasp-vs-lift records.

    Each record needs ``dz_grasp`` and ``dz_lift`` (signed vertical TCP displacement,
    metres). Returns rate + mean gap + mean dz per verb + per-seed pass.
    """
    n = len(per_seed)
    out_seeds = []
    n_pass = 0
    gaps, gz, lz, seps = [], [], [], []
    for rec in per_seed:
        dg, dl = rec["dz_grasp"], rec["dz_lift"]
        gap = grasplift_gap(dg, dl)
        ok = decide_grasplift(dg, dl, min_gap=min_gap)
        n_pass += int(ok)
        gaps.append(gap)
        gz.append(dg)
        lz.append(dl)
        if "tcp_sep" in rec:
            seps.append(rec["tcp_sep"])
        out_seeds.append({**rec, "grasplift_gap": gap, "grasplift_pass": ok})
    return {
        "num_seeds": n,
        "num_pass": n_pass,
        "pass_rate": (n_pass / n) if n else 0.0,
        "mean_dz_grasp": float(np.mean(gz)) if gz else 0.0,
        "mean_dz_lift": float(np.mean(lz)) if lz else 0.0,
        "mean_grasplift_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_tcp_sep": float(np.mean(seps)) if seps else 0.0,
        "min_gap": min_gap,
        "per_seed": out_seeds,
    }


def valid_verdict(
    waitact_pass_rate: float,
    mean_wait_motion: float,
    mean_act_motion: float,
    grasplift_pass_rate: float,
    mean_grasplift_gap: float,
    *,
    waitact_rate_threshold: float = 0.7,
    motion_ratio_threshold: float = 3.0,
    max_wait_motion: float = 0.05,
    grasplift_rate_threshold: float = 0.7,
    grasplift_gap_threshold: float = 0.03,
) -> dict:
    """The VALID-ONLY obedience verdict for the DECENTRALIZED ckpts.

    Uses ONLY the two in-distribution probes (NO left-vs-right, which is OOD for a
    per-arm decent ckpt). PASS iff BOTH:
      * wait-vs-act: the arm holds still on 'wait' and moves on the verb --
        mean act/wait motion ratio >= ``motion_ratio_threshold`` AND
        mean wait_motion <= ``max_wait_motion``, AND per-seed pass rate >=
        ``waitact_rate_threshold``; AND
      * grasp-vs-lift: 'lift' raises the TCP clearly more than 'grasp' --
        mean gap >= ``grasplift_gap_threshold`` AND per-seed pass rate >=
        ``grasplift_rate_threshold``.

    Returns the verdict + sub-verdicts + the numbers that decided it.
    """
    motion_ratio = float(mean_act_motion / max(mean_wait_motion, 1e-4))
    waitact_pass = (
        (motion_ratio >= motion_ratio_threshold)
        and (mean_wait_motion <= max_wait_motion)
        and (waitact_pass_rate >= waitact_rate_threshold)
    )
    grasplift_pass = (
        (mean_grasplift_gap >= grasplift_gap_threshold)
        and (grasplift_pass_rate >= grasplift_rate_threshold)
    )
    return {
        "verdict": "PASS" if (waitact_pass and grasplift_pass) else "FAIL",
        "wait_act_pass": bool(waitact_pass),
        "grasp_lift_pass": bool(grasplift_pass),
        "waitact_pass_rate": float(waitact_pass_rate),
        "waitact_rate_threshold": float(waitact_rate_threshold),
        "motion_ratio": motion_ratio,
        "motion_ratio_threshold": float(motion_ratio_threshold),
        "mean_wait_motion": float(mean_wait_motion),
        "mean_act_motion": float(mean_act_motion),
        "max_wait_motion": float(max_wait_motion),
        "grasplift_pass_rate": float(grasplift_pass_rate),
        "grasplift_rate_threshold": float(grasplift_rate_threshold),
        "mean_grasplift_gap": float(mean_grasplift_gap),
        "grasplift_gap_threshold": float(grasplift_gap_threshold),
    }


def gate_verdict(obedience_rate: float, mean_wait_motion: float, mean_act_motion: float,
                 obedience_threshold: float = 0.7, motion_ratio_threshold: float = 3.0,
                 max_wait_motion: float = 0.05) -> dict:
    """The headline GATE: is the LL FOLLOWING the subtask language channel?

    PASS iff BOTH:
      * left-vs-right obedience_rate >= ``obedience_threshold`` (default 0.7), AND
      * wait-motion << act-motion: mean act/wait motion ratio >= ``motion_ratio_threshold``
        AND mean wait-motion <= ``max_wait_motion`` (the arm actually holds still on 'wait').

    Returns a dict with the verdict + the two sub-verdicts + the numbers that
    decided it, so the JSON reader needs no recompute.
    """
    motion_ratio = float(mean_act_motion / max(mean_wait_motion, 1e-4))
    obedience_pass = obedience_rate >= obedience_threshold
    waitact_pass = (motion_ratio >= motion_ratio_threshold) and (mean_wait_motion <= max_wait_motion)
    return {
        "verdict": "PASS" if (obedience_pass and waitact_pass) else "FAIL",
        "left_right_obedience_pass": bool(obedience_pass),
        "wait_act_pass": bool(waitact_pass),
        "obedience_rate": float(obedience_rate),
        "obedience_threshold": float(obedience_threshold),
        "motion_ratio": motion_ratio,
        "motion_ratio_threshold": float(motion_ratio_threshold),
        "mean_wait_motion": float(mean_wait_motion),
        "mean_act_motion": float(mean_act_motion),
        "max_wait_motion": float(max_wait_motion),
    }


def aggregate_obedience(per_seed: List[dict], margin: float = 0.0) -> dict:
    """Aggregate per-seed probe records into a rate + mean delta.

    Each record needs the four distances dL_left/dL_right/dR_left/dR_right.
    Returns {num_seeds, num_obedient, obedience_rate, mean_prompt_swap_delta,
    per_seed:[...]}.
    """
    n = len(per_seed)
    out_seeds = []
    n_ob = 0
    deltas = []
    for rec in per_seed:
        ob = decide_obedience(
            rec["dL_left"], rec["dL_right"], rec["dR_left"], rec["dR_right"], margin=margin
        )
        delta = prompt_swap_delta(
            rec["dL_left"], rec["dL_right"], rec["dR_left"], rec["dR_right"]
        )
        n_ob += int(ob)
        deltas.append(delta)
        out_seeds.append({**rec, "obedient": ob, "prompt_swap_delta": delta})
    return {
        "num_seeds": n,
        "num_obedient": n_ob,
        "obedience_rate": (n_ob / n) if n else 0.0,
        "mean_prompt_swap_delta": float(np.mean(deltas)) if deltas else 0.0,
        "margin": margin,
        "per_seed": out_seeds,
    }


# ===========================================================================
# Args
# ===========================================================================


@dataclasses.dataclass
class Args:
    task: str = "LiftBarrier-rf"
    config: str = _DEFAULT_CONFIG
    mode: str = "probe"  # {schedule, probe, waitact, grasplift, gate, valid, both}
    # "gate"  = the LEGACY headline GATE = left-vs-right probe + waitact + PASS/FAIL
    #           verdict. INVALID for DECENTRALIZED ckpts: arm0 only ever trained on the
    #           LEFT end, so "grasp the right end" is OOD and its 0% is an artifact, not
    #           disobedience. Kept behind this flag for the centralized/legacy case only.
    # "valid" = the VALID decent verdict = waitact + grasplift (NO left-vs-right). This is
    #           the downstream-VLM go/no-go for the per-arm decentralized ckpts.
    # "grasplift" = run only the in-distribution grasp-vs-lift verb probe.
    # "both"  = schedule + probe (legacy).
    # ---- LL pi0.5 server(s) (the TRAINED subtask-conditioned ckpt) ----
    checkpoint: str = ""  # informational; the served ckpt is selected at serve time
    config_name: str = ""  # informational; recorded into the result JSON
    host: str = "127.0.0.1"
    ports: str = "8000,8001"  # CSV: one port per arm
    expect_config: str = ""  # PR2 handshake: hard-fail if served config_name differs
    # ---- optional prompt-IGNORING baseline ckpt (served separately) ----
    baseline_checkpoint: str = ""  # informational
    baseline_ports: str = ""  # CSV ports for the baseline servers; "" => skip baseline
    baseline_expect_config: str = ""
    # ---- cameras ----
    camera_family: str = "workspace"  # {workspace, wristcam}; workspace = trained-on head cams
    # ---- rollout control ----
    num_arms: int = 2
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = "panda_wristcam_multi,panda_wristcam_multi"
    replan_after: int = 8
    sim_backend: str = "auto"
    # schedule mode
    schedule_seeds: str = "1000,1001,1002,1003,1004"  # CSV env seeds for schedule rollouts
    schedule_max_steps: int = 300
    schedule_variant: str = "simultaneous"  # simultaneous | arm0_leads (left first) | arm1_leads (right first)
    schedule_lead_gap: int = 0  # staggered: steps the follower WAITS before starting (0 => auto)
    schedule_log_contact: bool = False  # log LIVE per-arm contact + ends each frame -> contact_dir/seed{seed}.npz
    schedule_no_terminate: bool = False  # run schedule to schedule_max_steps (no early stop) for offline contact scoring
    contact_dir: str = ""
    approach_steps: int = 60  # phase-0 window before close_gripper
    close_steps: int = 30     # close-gripper window before lift
    # probe mode
    num_seeds: int = 10           # A/B pairs (alias: --n-seeds)
    probe_base_seed: int = 1000   # probe seeds = base, base+1, ...
    probe_max_steps: int = 80     # SHORT (approach phase only) -> cheap
    obedience_margin: float = 0.0  # metres; strictness buffer in decide_obedience
    # which arm is PROBED (0 or 1). The OTHER arm is held on "wait". The decent
    # ckpts are per-arm: arm0 sees "grasp the left/right end", arm1 too. The gate
    # primarily probes arm0; --probe-arm 1 documents/exercises arm1.
    probe_arm: int = 0
    # ---- wait-vs-act probe + GATE thresholds ----
    waitact_act_prompt: str = "grasp the left end"  # the ACT prompt (vs "wait")
    waitact_min_ratio: float = 3.0       # per-seed pass needs act/wait motion >= this
    waitact_max_wait_motion: float = 0.05  # metres; 'wait' must move LESS than this
    obedience_threshold: float = 0.7     # GATE: left-vs-right obedience_rate >= this
    motion_ratio_threshold: float = 3.0  # GATE: mean act/wait motion ratio >= this
    waitact_rate_threshold: float = 0.7  # VALID verdict: per-seed waitact pass rate >= this
    # ---- grasp-vs-lift probe (VALID in-distribution verb test for decent ckpts) ----
    grasplift_grasp_prompt: str = "grasp the left end"  # low/approach verb (same side)
    grasplift_lift_prompt: str = "lift the left end"    # raise verb (same side)
    grasplift_min_gap: float = 0.03      # metres; per-seed pass needs dz_lift-dz_grasp >= this
    grasplift_rate_threshold: float = 0.7  # VALID verdict: per-seed grasplift pass rate >= this
    # PRIME the approach->close grasp prefix before the measured verb so "lift" is in-dist
    # (the LL only saw "lift" post-grasp). Set grasplift_prime_steps<=0 for a cold-start ablation.
    grasplift_prime_steps: int = 40       # approach steps before the measured verb
    grasplift_prime_close_steps: int = 20  # close-gripper steps before the measured verb
    # ---- mock / plumbing ----
    mock_ll: bool = False  # zero-action LL stand-in (plumbing only; probe will be ~chance)
    # ---- output ----
    out: str = ""  # result JSON path; "" => <results_dir>/eval_subtask_obedience_<ts>.json
    results_dir: str = "/iris/u/mikulrai/projects/RoboFactory/eval_results"
    video_dir: str = ""  # defaults to <results_dir>/videos_obedience
    video_frame_stride: int = 2
    record_video: bool = True


# ===========================================================================
# Sim-touching helpers (imports stay inside; called only from main)
# ===========================================================================


def _build_env(args: Args):
    """Build the LiftBarrier env with shader_pack=default (login-node refused)."""
    import gymnasium as gym  # noqa: F401
    import sapien  # noqa: F401
    from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
    # Importing the hier module registers all robofactory tasks (it runs
    # `from robofactory.tasks import *` + `import robofactory.agents` at its top),
    # so the "LiftBarrier-rf" gym id is registered by the time we call gym.make.
    _hier()
    import robofactory.tasks  # noqa: F401  (registers tasks via package import)
    import robofactory.agents  # noqa: F401

    gym_make_kwargs = dict(
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids=tuple(args.robot_uids_csv.split(",")),
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    _hier()._assert_shader_pack_default(gym_make_kwargs)
    env = gym.make(args.task, **gym_make_kwargs)
    return env


def _tcp_xyz(env, arm: int) -> np.ndarray:
    """Current world TCP xyz of arm ``arm`` (robust torch/np conversion)."""
    p = env.unwrapped.agent.agents[arm].tcp.pose.p
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return np.asarray(p).reshape(-1)[:3].astype(np.float32)


def _resolve_lb_contact_points(env) -> Dict[str, np.ndarray]:
    """Resolve the LEFT and RIGHT end contact points (world xyz) at the CURRENT
    env pose, via the same labeled-direction grasp resolver the data-gen uses.

    Builds a per-arm planner on the already-reset env (matches
    run_subtask_rollouts._build_planner) and reads the [:3] of the resolved 7-vec.
    """
    from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
    from robofactory.planner.subtask_primitives import resolve_lb_end_grasp_pose

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True,
    )
    left = np.asarray(
        resolve_lb_end_grasp_pose(planner, env.unwrapped, vocab.LB_TARGETS["left_end"])
    ).reshape(-1)[:3].astype(np.float32)
    right = np.asarray(
        resolve_lb_end_grasp_pose(planner, env.unwrapped, vocab.LB_TARGETS["right_end"])
    ).reshape(-1)[:3].astype(np.float32)
    try:
        planner.close()
    except Exception:
        pass
    return {"left": left, "right": right}


def _ll_action(ll_policies, args, obs, cam_map, prompts, chunks, chunk_idxs):
    """One env-step worth of per-arm absolute joint targets, routing prompts[i] into
    arm i's LL. Mutates chunks/chunk_idxs in place (chunked replanning, like hier).

    Returns action_dict keyed by f"{action_prefix}-{i}". The delta->absolute decode
    is IDENTICAL to hier.run_episode (cur_q + delta).
    """
    h = _hier()
    for i in range(args.num_arms):
        if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
            obs_i = h._build_ll_obs(obs, args, cam_map, prompts[i])  # sets obs_i["prompt"]=prompts[i]
            chunks[i] = np.asarray(ll_policies[i].infer(obs_i)["actions"])
            chunk_idxs[i] = 0
    cur_q = h._cur_qpos(obs, args.num_arms, args.robot_uid)
    return cur_q


def _assemble_action(chunks, chunk_idxs, cur_q, action_prefix, num_arms):
    action_dict = {}
    for i in range(num_arms):
        step_i = chunks[i][chunk_idxs[i]]
        delta = step_i[:7]
        grip = step_i[7]
        target = np.concatenate([cur_q[i] + delta, np.array([grip], dtype=np.float32)])
        action_dict[f"{action_prefix}-{i}"] = target.astype(np.float32)
        chunk_idxs[i] += 1
    return action_dict


def _strict_success(env) -> bool:
    """Grasp-gated LiftBarrier success (falls back to env success if unavailable)."""
    try:
        from robofactory.tasks.success_candidates import lift_barrier_success_strict
        s = lift_barrier_success_strict(env)
        return bool(s.reshape(-1)[0]) if hasattr(s, "reshape") else bool(s)
    except Exception:
        info = env.unwrapped.evaluate()
        s = info.get("success", False)
        return bool(s.item() if hasattr(s, "item") else s)


# ===========================================================================
# SCHEDULE-mode rollout
# ===========================================================================


def _sched_contacts_and_ends(env, num_arms):
    """LIVE per-arm gripper<->barrier contact force + both-ends z + base z (read after
    an env.step). Mirrors the eval_decent_pi05 logger so the SAME offline median scorer
    applies to subtask-LL schedule rollouts."""
    import numpy as _np
    u = getattr(env, "unwrapped", env)
    barrier = u.barrier
    from robofactory.tasks.success_candidates import BARRIER_END_OFFSETS
    def t2n(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else _np.asarray(x)
    p = t2n(barrier.pose.p).reshape(-1)[:3]; q = t2n(barrier.pose.q).reshape(-1)[:4]
    w, x, y, z = q
    R = _np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    offs = t2n(BARRIER_END_OFFSETS)
    end_z = [float((p + R @ _np.asarray(o))[2]) for o in offs]
    base_z = float(t2n(u.agent.agents[0].robot.pose.p).reshape(-1)[2])
    forces = []
    for i in range(num_arms):
        ag = u.agent.agents[i]; tot = 0.0
        for l in ag.robot.get_links():
            if ("finger" in l.name.lower()) or ("hand" in l.name.lower()):
                try: tot += float(_np.linalg.norm(t2n(u.scene.get_pairwise_contact_forces(l, barrier))))
                except Exception: pass
        forces.append(tot)
    return forces, end_z, base_z


def run_schedule_episode(env, ll_policies, args, cam_map, action_prefix, seed,
                         schedule, view_sensors, video_path=None):
    """Drive each arm with the time-indexed ``schedule``; report strict success."""
    import numpy as _np
    from pathlib import Path as _P
    obs, _ = env.reset(seed=int(seed))
    n = args.schedule_max_steps
    per_step = expand_schedule(schedule, n)  # {arm: [prompt per step]}

    h = _hier()
    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    frames = []
    contact_trace = []
    success = False
    step = 0
    for step in range(n):
        prompts = [per_step[i][step] for i in range(args.num_arms)]
        if video_path is not None:
            from policy._shared.multiview_video import tile_views
            frame = tile_views([h._extract_image(obs, s) for s in view_sensors])
            frame = h._overlay_subtasks(
                frame, step, n, prompts[0], prompts[1] if args.num_arms > 1 else None
            )
            frames.append(frame)
        cur_q = _ll_action(ll_policies, args, obs, cam_map, prompts, chunks, chunk_idxs)
        action_dict = _assemble_action(chunks, chunk_idxs, cur_q, action_prefix, args.num_arms)
        obs, _, term, trunc, info = env.step(action_dict)
        success = _strict_success(env)
        if getattr(args, "schedule_log_contact", False):
            _f, _ez, _bz = _sched_contacts_and_ends(env, args.num_arms)
            contact_trace.append([float(step), float(_f[0]), float(_f[1]), float(_ez[0]), float(_ez[1]), float(_bz)])
        # schedule_no_terminate: run full horizon (no early stop) for offline contact scoring.
        if not getattr(args, "schedule_no_terminate", False):
            if success or bool(term) or bool(trunc):
                break
        elif bool(trunc):
            break

    if video_path is not None and frames:
        from policy._shared.multiview_video import subsample
        h._write_mp4(video_path, subsample(frames, args.video_frame_stride))

    if getattr(args, "schedule_log_contact", False) and contact_trace and getattr(args, "contact_dir", ""):
        _P(args.contact_dir).mkdir(parents=True, exist_ok=True)
        _np.savez(str(_P(args.contact_dir) / f"seed{int(seed)}.npz"),
                  trace=_np.asarray(contact_trace, dtype=_np.float32),
                  cols=_np.array(["step", "f_arm0", "f_arm1", "end0_z", "end1_z", "base_z"]))

    return {"seed": int(seed), "success": bool(success), "steps": int(step + 1)}


# ===========================================================================
# PROBE-mode: one A/B pair on one seed
# ===========================================================================


def _probe_one_variant(env, ll_policies, args, cam_map, action_prefix, seed,
                       probe_prompt, view_sensors, video_path=None, prime=None):
    """SHORT rollout: the PROBED arm (args.probe_arm) is driven by ``probe_prompt``,
    every OTHER arm is held on 'wait'. Returns (tcp_end, motion, tcp_measure_start).

    Resets to ``seed`` first (identical start for the A/B pair, ``env.reset(seed)`` is
    deterministic). If ``prime`` is given it is a list of ``(prompt, n_steps)`` phases run
    on the PROBED arm BEFORE the measured ``probe_prompt`` phase; ``tcp_measure_start`` is
    then captured AFTER the prime (so dz is measured from the primed pose, e.g. post-grasp).
    This is what makes the LIFT verb IN-DISTRIBUTION: the LL only ever saw "lift" after an
    APPROACH->CLOSE grasp, so we replay that grasp prefix identically for both variants and
    only then fork on grasp-vs-lift."""
    h = _hier()
    obs, _ = env.reset(seed=int(seed))
    wait_prompt = vocab.render(vocab.WAIT, 0, TASK)
    parm = int(args.probe_arm)
    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    frames = []

    def _run_phase(phase_prompt, n_steps, phase_label):
        nonlocal obs, chunks, chunk_idxs
        prompts = [wait_prompt] * args.num_arms
        prompts[parm] = phase_prompt
        for step in range(n_steps):
            if video_path is not None:
                from policy._shared.multiview_video import tile_views
                frame = tile_views([h._extract_image(obs, s) for s in view_sensors])
                frame = h._overlay_subtasks(
                    frame, step, n_steps, prompts[0],
                    prompts[1] if args.num_arms > 1 else None,
                )
                frames.append(frame)
            cur_q = _ll_action(ll_policies, args, obs, cam_map, prompts, chunks, chunk_idxs)
            action_dict = _assemble_action(chunks, chunk_idxs, cur_q, action_prefix, args.num_arms)
            obs, _, term, trunc, _ = env.step(action_dict)
            if bool(term) or bool(trunc):
                return False
        return True

    # optional PRIME phases (e.g. approach -> close) BEFORE the measured verb.
    if prime:
        for (pp, pn) in prime:
            if not _run_phase(pp, pn, "prime"):
                break
    tcp_start = _tcp_xyz(env, parm)  # measured AFTER the prime

    _run_phase(probe_prompt, args.probe_max_steps, "probe")

    tcp_end = _tcp_xyz(env, parm)
    motion = float(np.linalg.norm(tcp_end - tcp_start))
    if video_path is not None and frames:
        from policy._shared.multiview_video import subsample
        h._write_mp4(video_path, subsample(frames, args.video_frame_stride))
    return tcp_end, motion, tcp_start


def run_probe_seed(env, ll_policies, args, cam_map, action_prefix, seed,
                   view_sensors, video_dir=None, tag=""):
    """One A/B obedience pair on ``seed``. Returns the per-seed record with the four
    TCP-to-end distances. Contact points are resolved at the reset pose (identical
    for both variants -- env.reset(seed) is deterministic)."""
    # Resolve LEFT/RIGHT contact points at the canonical reset pose.
    env.reset(seed=int(seed))
    contacts = _resolve_lb_contact_points(env)
    left_pt, right_pt = contacts["left"], contacts["right"]

    vid_L = str(Path(video_dir) / f"{tag}probe_seed{seed}_L.mp4") if video_dir else None
    vid_R = str(Path(video_dir) / f"{tag}probe_seed{seed}_R.mp4") if video_dir else None

    prompt_left = vocab.render(vocab.APPROACH, vocab.LB_TARGETS["left_end"], TASK)
    prompt_right = vocab.render(vocab.APPROACH, vocab.LB_TARGETS["right_end"], TASK)

    tcp_L, _, _ = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix, seed,
                                     prompt_left, view_sensors, video_path=vid_L)
    tcp_R, _, _ = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix, seed,
                                     prompt_right, view_sensors, video_path=vid_R)

    return {
        "seed": int(seed),
        "probe_arm": int(args.probe_arm),
        "prompt_left": prompt_left,
        "prompt_right": prompt_right,
        "left_contact": left_pt.tolist(),
        "right_contact": right_pt.tolist(),
        "tcp_variant_L": tcp_L.tolist(),
        "tcp_variant_R": tcp_R.tolist(),
        "dL_left": float(np.linalg.norm(tcp_L - left_pt)),
        "dL_right": float(np.linalg.norm(tcp_L - right_pt)),
        "dR_left": float(np.linalg.norm(tcp_R - left_pt)),
        "dR_right": float(np.linalg.norm(tcp_R - right_pt)),
    }


# ===========================================================================
# WAIT-vs-ACT probe: one seed
# ===========================================================================


def run_waitact_seed(env, ll_policies, args, cam_map, action_prefix, seed,
                     view_sensors, video_dir=None, tag=""):
    """One wait-vs-act pair on ``seed``. Two SHORT rollouts that differ ONLY in the
    probed arm's prompt: 'wait' vs ``args.waitact_act_prompt`` (e.g. 'grasp the left
    end'). Every other arm is held on 'wait' in BOTH rollouts. Returns the probed
    arm's TCP displacement (metres) under each prompt.

    Obedience => wait_motion ~ 0 (arm holds still when told to wait) while
    act_motion is clearly larger (arm moves to execute the verb)."""
    wait_prompt = vocab.render(vocab.WAIT, 0, TASK)
    vid_w = str(Path(video_dir) / f"{tag}waitact_seed{seed}_wait.mp4") if video_dir else None
    vid_a = str(Path(video_dir) / f"{tag}waitact_seed{seed}_act.mp4") if video_dir else None

    # NOTE: _probe_one_variant resets to ``seed`` itself, so the two rollouts share
    # an identical start. The probed arm is args.probe_arm; the others are 'wait'.
    _, wait_motion, _ = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix,
                                           seed, wait_prompt, view_sensors, video_path=vid_w)
    _, act_motion, _ = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix,
                                          seed, args.waitact_act_prompt, view_sensors,
                                          video_path=vid_a)
    return {
        "seed": int(seed),
        "probe_arm": int(args.probe_arm),
        "wait_prompt": wait_prompt,
        "act_prompt": args.waitact_act_prompt,
        "wait_motion": float(wait_motion),
        "act_motion": float(act_motion),
    }


# ===========================================================================
# GRASP-vs-LIFT probe: one seed (VALID in-distribution verb test)
# ===========================================================================


def run_grasplift_seed(env, ll_policies, args, cam_map, action_prefix, seed,
                       view_sensors, video_dir=None, tag=""):
    """One grasp-vs-lift pair on ``seed``. Two SHORT rollouts that differ ONLY in the
    probed arm's VERB on the SAME side: ``args.grasplift_grasp_prompt`` ("grasp the
    left end") vs ``args.grasplift_lift_prompt`` ("lift the left end"). Every other
    arm is held on 'wait' in BOTH. Returns the probed arm's SIGNED VERTICAL (z) TCP
    displacement under each verb.

    Both prompts are in arm0's training distribution (left side only), so a difference
    is VERB-obedience, not an OOD artifact. Obedience => 'lift' raises the TCP clearly
    more than 'grasp' (dz_lift - dz_grasp >= min_gap)."""
    vid_g = str(Path(video_dir) / f"{tag}grasplift_seed{seed}_grasp.mp4") if video_dir else None
    vid_l = str(Path(video_dir) / f"{tag}grasplift_seed{seed}_lift.mp4") if video_dir else None

    # PRIME both variants IDENTICALLY with the approach->close grasp prefix BEFORE forking
    # on the measured verb. The LL only ever saw "lift the left end" AFTER an
    # approach->close grasp (training program approach->close_gripper->lift), so a cold
    # "lift" from the neutral reset pose would be OUT-OF-DISTRIBUTION (the same trap that
    # invalidated left-vs-right). Priming puts the arm in the post-grasp state the lift
    # verb was trained on; dz is then measured from that post-grasp pose. ``prime`` is
    # empty/disabled when grasplift_prime_steps <= 0 (cold-start, for ablation).
    prime = None
    if args.grasplift_prime_steps > 0:
        close_prompt = vocab.render(vocab.CLOSE_GRIPPER, 0, TASK)
        prime = [
            (args.grasplift_grasp_prompt, args.grasplift_prime_steps),  # approach
            (close_prompt, args.grasplift_prime_close_steps),           # close gripper
        ]

    # _probe_one_variant resets to ``seed`` itself, so both rollouts share an identical
    # start (and identical prime). The probed arm is args.probe_arm; others held on 'wait'.
    tcp_g, _, start_g = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix,
                                           seed, args.grasplift_grasp_prompt, view_sensors,
                                           video_path=vid_g, prime=prime)
    tcp_l, _, start_l = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix,
                                           seed, args.grasplift_lift_prompt, view_sensors,
                                           video_path=vid_l, prime=prime)
    dz_grasp = float(tcp_g[2] - start_g[2])
    dz_lift = float(tcp_l[2] - start_l[2])
    # axis-agnostic insurance metric: 3D separation of the two verbs' final TCP. Even if
    # both verbs descend (dz both negative -> small gap), distinct verbs should drive the
    # TCP to DIFFERENT places. A near-zero tcp_sep => the LL ignored the verb entirely.
    tcp_sep = float(np.linalg.norm(np.asarray(tcp_l) - np.asarray(tcp_g)))
    return {
        "seed": int(seed),
        "probe_arm": int(args.probe_arm),
        "grasp_prompt": args.grasplift_grasp_prompt,
        "lift_prompt": args.grasplift_lift_prompt,
        "primed": bool(prime is not None),
        "dz_grasp": dz_grasp,
        "dz_lift": dz_lift,
        "tcp_sep": tcp_sep,
    }


# ===========================================================================
# Orchestration
# ===========================================================================


def _make_policies_with_ports(args, ports_csv, expect_config):
    """Make LL clients on a given CSV of ports + run the PR2 identity handshake."""
    import copy
    a = copy.copy(args)
    a.ports = ports_csv
    a.expect_config = expect_config
    policies = _hier()._make_ll_policies(a)
    metadata = []
    for i, p in enumerate(policies):
        meta = dict(p.get_server_metadata() or {}) if hasattr(p, "get_server_metadata") else {}
        metadata.append(meta)
        print(f"[LL arm{i} ports={ports_csv}] server metadata: {meta}")
    if expect_config and not args.mock_ll:
        from robofactory.utils.server_identity import assert_server_identity_or_exit
        names = [c.strip() for c in expect_config.split(",")]
        if len(names) == 1:
            names = names * args.num_arms
        for i in range(args.num_arms):
            assert_server_identity_or_exit(metadata[i], names[i], expect_action_dim=None,
                                           label=f"arm{i}")
    return policies, metadata


def _run_probe_suite(env, args, cam_map, action_prefix, view_sensors, ll_policies,
                     video_dir, tag):
    seeds = [args.probe_base_seed + i for i in range(args.num_seeds)]
    per_seed = []
    for s in seeds:
        rec = run_probe_seed(env, ll_policies, args, cam_map, action_prefix, s,
                             view_sensors, video_dir=video_dir, tag=tag)
        per_seed.append(rec)
        print(f"[probe {tag}seed {s}] dL_left={rec['dL_left']:.3f} dL_right={rec['dL_right']:.3f} "
              f"dR_left={rec['dR_left']:.3f} dR_right={rec['dR_right']:.3f}")
    agg = aggregate_obedience(per_seed, margin=args.obedience_margin)
    print(f"[probe {tag}] obedience {agg['num_obedient']}/{agg['num_seeds']} "
          f"(rate={agg['obedience_rate']:.3f}) mean_swap_delta={agg['mean_prompt_swap_delta']:.4f}")
    return agg


def _run_waitact_suite(env, args, cam_map, action_prefix, view_sensors, ll_policies,
                       video_dir, tag):
    seeds = [args.probe_base_seed + i for i in range(args.num_seeds)]
    per_seed = []
    for s in seeds:
        rec = run_waitact_seed(env, ll_policies, args, cam_map, action_prefix, s,
                               view_sensors, video_dir=video_dir, tag=tag)
        per_seed.append(rec)
        print(f"[waitact {tag}seed {s}] wait_motion={rec['wait_motion']:.4f} "
              f"act_motion={rec['act_motion']:.4f}")
    agg = aggregate_waitact(per_seed, min_ratio=args.waitact_min_ratio,
                            max_wait_motion=args.waitact_max_wait_motion)
    print(f"[waitact {tag}] pass {agg['num_pass']}/{agg['num_seeds']} "
          f"(rate={agg['pass_rate']:.3f}) mean_wait={agg['mean_wait_motion']:.4f} "
          f"mean_act={agg['mean_act_motion']:.4f} mean_ratio={agg['mean_ratio']:.2f}")
    return agg


def _run_grasplift_suite(env, args, cam_map, action_prefix, view_sensors, ll_policies,
                         video_dir, tag):
    seeds = [args.probe_base_seed + i for i in range(args.num_seeds)]
    per_seed = []
    for s in seeds:
        rec = run_grasplift_seed(env, ll_policies, args, cam_map, action_prefix, s,
                                 view_sensors, video_dir=video_dir, tag=tag)
        per_seed.append(rec)
        print(f"[grasplift {tag}seed {s}] dz_grasp={rec['dz_grasp']:.4f} "
              f"dz_lift={rec['dz_lift']:.4f} tcp_sep={rec['tcp_sep']:.4f}")
    agg = aggregate_grasplift(per_seed, min_gap=args.grasplift_min_gap)
    print(f"[grasplift {tag}] pass {agg['num_pass']}/{agg['num_seeds']} "
          f"(rate={agg['pass_rate']:.3f}) mean_dz_grasp={agg['mean_dz_grasp']:.4f} "
          f"mean_dz_lift={agg['mean_dz_lift']:.4f} mean_gap={agg['mean_grasplift_gap']:.4f} "
          f"mean_tcp_sep={agg['mean_tcp_sep']:.4f}")
    return agg


def main(args: Args):
    h = _hier()
    h._assert_not_login_node()
    valid_modes = ("schedule", "probe", "waitact", "grasplift", "gate", "valid", "both")
    if args.mode not in valid_modes:
        raise ValueError(f"--mode must be one of {valid_modes}, got {args.mode!r}")
    if args.camera_family not in h.CAMERA_FAMILIES:
        raise ValueError(f"--camera-family must be one of {sorted(h.CAMERA_FAMILIES)}")
    if not (0 <= int(args.probe_arm) < args.num_arms):
        raise ValueError(f"--probe-arm must be in [0,{args.num_arms}), got {args.probe_arm}")

    from policy._shared.multiview_video import ordered_unique

    cam_map = dict(h.CAMERA_FAMILIES[args.camera_family])
    view_sensors = ordered_unique(list(cam_map.values()))
    print(f"[camera-family] {args.camera_family}: LL={cam_map} video_views={view_sensors}")

    if not args.video_dir:
        args.video_dir = str(Path(args.results_dir) / "videos_obedience")
    video_dir = args.video_dir if args.record_video else None
    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    env = _build_env(args)
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]

    # Primary (trained subtask-conditioned) LL policies + handshake.
    ll_policies, ll_metadata = _make_policies_with_ports(args, args.ports, args.expect_config)

    # Optional prompt-ignoring baseline LL policies.
    baseline_policies, baseline_metadata = None, None
    if args.baseline_ports:
        baseline_policies, baseline_metadata = _make_policies_with_ports(
            args, args.baseline_ports, args.baseline_expect_config
        )

    results: dict = {
        "task": args.task,
        "mode": args.mode,
        "camera_family": args.camera_family,
        "args": dataclasses.asdict(args),
        "checkpoint": args.checkpoint,
        "config_name": args.config_name,
        "server_metadata": {f"arm{i}": ll_metadata[i] for i in range(len(ll_metadata))},
        "baseline_checkpoint": args.baseline_checkpoint or None,
        "baseline_server_metadata": (
            {f"arm{i}": baseline_metadata[i] for i in range(len(baseline_metadata))}
            if baseline_metadata else None
        ),
        "mock_ll": args.mock_ll,
    }

    # ---- SCHEDULE mode ----
    if args.mode in ("schedule", "both"):
        if args.schedule_variant == "arm0_leads":
            schedule = staggered_lb_schedule(0, args.approach_steps, args.close_steps, args.schedule_lead_gap)
        elif args.schedule_variant == "arm1_leads":
            schedule = staggered_lb_schedule(1, args.approach_steps, args.close_steps, args.schedule_lead_gap)
        else:
            schedule = default_lb_schedule(args.approach_steps, args.close_steps)
        print(f"[schedule] variant={args.schedule_variant} schedule={schedule}")
        validate_schedule(schedule)
        sched_seeds = [int(s) for s in args.schedule_seeds.split(",") if s.strip()]
        sched_eps = []
        for s in sched_seeds:
            vp = str(Path(video_dir) / f"schedule_seed{s}.mp4") if video_dir else None
            rec = run_schedule_episode(env, ll_policies, args, cam_map, action_prefix, s,
                                       schedule, view_sensors, video_path=vp)
            sched_eps.append(rec)
            print(f"[schedule seed {s}] {rec}")
        n_ok = sum(1 for e in sched_eps if e["success"])
        results["schedule"] = {
            "schedule": {str(a): segs for a, segs in schedule.items()},
            "seeds": sched_seeds,
            "num_episodes": len(sched_eps),
            "num_success": n_ok,
            "success_rate": (n_ok / len(sched_eps)) if sched_eps else 0.0,
            "episodes": sched_eps,
        }
        print(f"[schedule] strict success {n_ok}/{len(sched_eps)}")

    # ---- PROBE mode (left-vs-right; headline obedience metric) ----
    if args.mode in ("probe", "both", "gate"):
        results["probe"] = _run_probe_suite(
            env, args, cam_map, action_prefix, view_sensors, ll_policies, video_dir, tag=""
        )
        if baseline_policies is not None:
            results["probe_baseline"] = _run_probe_suite(
                env, args, cam_map, action_prefix, view_sensors, baseline_policies,
                video_dir, tag="baseline_"
            )

    # ---- WAIT-vs-ACT mode (motion magnitude) ----
    if args.mode in ("waitact", "gate", "valid"):
        results["waitact"] = _run_waitact_suite(
            env, args, cam_map, action_prefix, view_sensors, ll_policies, video_dir, tag=""
        )
        if baseline_policies is not None:
            results["waitact_baseline"] = _run_waitact_suite(
                env, args, cam_map, action_prefix, view_sensors, baseline_policies,
                video_dir, tag="baseline_"
            )

    # ---- GRASP-vs-LIFT mode (VALID in-distribution verb test) ----
    if args.mode in ("grasplift", "valid"):
        results["grasplift"] = _run_grasplift_suite(
            env, args, cam_map, action_prefix, view_sensors, ll_policies, video_dir, tag=""
        )
        if baseline_policies is not None:
            results["grasplift_baseline"] = _run_grasplift_suite(
                env, args, cam_map, action_prefix, view_sensors, baseline_policies,
                video_dir, tag="baseline_"
            )

    # ---- GATE verdict (combines both probes) ----
    if args.mode == "gate":
        pr, wa = results["probe"], results["waitact"]
        verdict = gate_verdict(
            obedience_rate=pr["obedience_rate"],
            mean_wait_motion=wa["mean_wait_motion"],
            mean_act_motion=wa["mean_act_motion"],
            obedience_threshold=args.obedience_threshold,
            motion_ratio_threshold=args.motion_ratio_threshold,
            max_wait_motion=args.waitact_max_wait_motion,
        )
        results["verdict"] = verdict
        print(f"[GATE VERDICT] {verdict['verdict']}  "
              f"(left-right obedience={verdict['obedience_rate']:.3f}>={verdict['obedience_threshold']} "
              f"-> {verdict['left_right_obedience_pass']}; "
              f"motion_ratio={verdict['motion_ratio']:.2f}>={verdict['motion_ratio_threshold']} "
              f"& wait_motion={verdict['mean_wait_motion']:.4f}<={verdict['max_wait_motion']} "
              f"-> {verdict['wait_act_pass']})")
        if baseline_policies is not None and "waitact_baseline" in results:
            results["verdict_baseline"] = gate_verdict(
                obedience_rate=results["probe_baseline"]["obedience_rate"],
                mean_wait_motion=results["waitact_baseline"]["mean_wait_motion"],
                mean_act_motion=results["waitact_baseline"]["mean_act_motion"],
                obedience_threshold=args.obedience_threshold,
                motion_ratio_threshold=args.motion_ratio_threshold,
                max_wait_motion=args.waitact_max_wait_motion,
            )

    # ---- VALID verdict (decent ckpts): waitact + grasplift ONLY (NO left-vs-right) ----
    if args.mode == "valid":
        wa, gl = results["waitact"], results["grasplift"]
        verdict = valid_verdict(
            waitact_pass_rate=wa["pass_rate"],
            mean_wait_motion=wa["mean_wait_motion"],
            mean_act_motion=wa["mean_act_motion"],
            grasplift_pass_rate=gl["pass_rate"],
            mean_grasplift_gap=gl["mean_grasplift_gap"],
            waitact_rate_threshold=args.waitact_rate_threshold,
            motion_ratio_threshold=args.motion_ratio_threshold,
            max_wait_motion=args.waitact_max_wait_motion,
            grasplift_rate_threshold=args.grasplift_rate_threshold,
            grasplift_gap_threshold=args.grasplift_min_gap,
        )
        results["verdict"] = verdict
        print(f"[VALID VERDICT] {verdict['verdict']}  "
              f"(waitact: ratio={verdict['motion_ratio']:.2f}>={verdict['motion_ratio_threshold']} "
              f"& wait_motion={verdict['mean_wait_motion']:.4f}<={verdict['max_wait_motion']} "
              f"& rate={verdict['waitact_pass_rate']:.2f}>={verdict['waitact_rate_threshold']} "
              f"-> {verdict['wait_act_pass']}; "
              f"grasplift: gap={verdict['mean_grasplift_gap']:.4f}>={verdict['grasplift_gap_threshold']} "
              f"& rate={verdict['grasplift_pass_rate']:.2f}>={verdict['grasplift_rate_threshold']} "
              f"-> {verdict['grasp_lift_pass']})")
        if baseline_policies is not None and "grasplift_baseline" in results:
            results["verdict_baseline"] = valid_verdict(
                waitact_pass_rate=results["waitact_baseline"]["pass_rate"],
                mean_wait_motion=results["waitact_baseline"]["mean_wait_motion"],
                mean_act_motion=results["waitact_baseline"]["mean_act_motion"],
                grasplift_pass_rate=results["grasplift_baseline"]["pass_rate"],
                mean_grasplift_gap=results["grasplift_baseline"]["mean_grasplift_gap"],
                waitact_rate_threshold=args.waitact_rate_threshold,
                motion_ratio_threshold=args.motion_ratio_threshold,
                max_wait_motion=args.waitact_max_wait_motion,
                grasplift_rate_threshold=args.grasplift_rate_threshold,
                grasplift_gap_threshold=args.grasplift_min_gap,
            )

    env.close()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(args.results_dir) / f"eval_subtask_obedience_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[result] -> {out_path}")
    return results


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
