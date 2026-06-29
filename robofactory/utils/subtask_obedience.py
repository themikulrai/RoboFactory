"""Pure (GPU/sim-free) logic for the per-arm subtask-obedience probe.

This module holds every piece of the Lift-Barrier subtask-obedience probe that does
NOT need SAPIEN, a GPU, or a policy server, so it can be unit-tested in isolation:

  * the EXACT subtask prompt strings the decentralized pi0.5 LoRA policies were trained
    on (verified against the LeRobot dataset `subtask_arm{i}_text` columns);
  * the canonical barrier LEFT/RIGHT end world-position computation, replicated bit for
    bit from the data-gen planner (PandaArmMotionPlanningSolver.get_grasp_pose_w_labeled_direction);
  * TCP-trajectory -> end assignment WITH a min-approach gate ("did the arm REALLY approach an
    end, or is it a frozen arm being mislabeled by its start position?");
  * the in-distribution ENGAGEMENT contrast (PRIMARY metric, see below) + paired scoring;
  * the OOD direction A/B obedience + paired causal-shift scoring (SECONDARY metric).

The driver `robofactory/policy/openpi_pi05/eval_subtask_obedience.py` imports these helpers.

WHAT THE PROBE TESTS -- TWO METRICS, PRIMARY FIRST (READ THIS)
--------------------------------------------------------------
The open question is whether the per-arm SUBTASK PROMPT actually steers an arm, or whether the
policy ignores the text and shortcuts off proprioception/vision. We answer it with TWO contrasts
run from the SAME env reset:

PRIMARY -- IN-DISTRIBUTION ENGAGEMENT CONTRAST (grasp-own-end vs wait).
  For the target arm, compare its OWN-end grasp prompt (in-distribution: arm0 -> "grasp the left
  end", arm1 -> "grasp the right end") against the neutral "wait" prompt. Both the prompt and the
  commanded motion are IN-DISTRIBUTION, so the result is interpretable in BOTH directions:
    * engagement(grasp) ~ engagement(wait)  -> the prompt buys NO extra engagement: channel DEAD.
    * engagement(grasp)  >  engagement(wait) -> the grasp prompt makes the arm actively engage
      (drive its TCP toward its own end and/or close the gripper) where "wait" leaves it passive:
      channel ALIVE.
  ENGAGEMENT is measured from the rollout traces as (a) `approach_progress` -- metres the TCP
  moved toward the arm's own end, and (b) `grasp_fraction` -- fraction of steps the arm was
  grasping. We report the PAIRED (per-seed) grasp-minus-wait shift on each, with null = 0 (shift)
  / 0.5 (consistency). This is the review's #1 fix: unlike the OOD direction probe below, a NULL
  here is a clean "channel dead", not "channel alive but cannot execute an unseen motion".

SECONDARY -- OOD DIRECTION A/B (kept for completeness; do NOT read it as the headline).
  From the same reset the target arm is commanded "grasp the left end" (A) vs "grasp the right
  end" (B); a prompt-IGNORING policy produces identical A/B (obedience == 0.5 null, zero shift),
  a prompt-OBEYING policy shifts toward the commanded end.
  OOD CAVEAT: each arm trained ONLY on its OWN end (arm0/LEFT only saw "grasp the left end",
  arm1/RIGHT only "grasp the right end"), so commanding the opposite end is OUT OF DISTRIBUTION.
  Half of every A/B pair is therefore OOD, biasing direction obedience toward the null: a low
  value does NOT prove the channel is dead (the policy may understand the text yet be unable to
  execute the unseen motion). Read the paired A-vs-B causal shift (not raw obedience) and treat
  every OOD number as an upper bound on incapacity, not a lower bound on disobedience. The
  PRIMARY engagement contrast above is the metric to trust for "is the channel alive?".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------- prompt strings

# EXACT per-frame `subtask_arm{i}_text` strings the SubtaskColumnInjector fed the policy as the
# pi0.5 `prompt` during training. Verified by reading the literal text columns of the LeRobot
# datasets robofactory_lift_barrier_{ws,wc}_subtaskdart_v1 (data/chunk-*/episode_*.parquet):
#   arm0 (LEFT)  text values: {"grasp the left end", "close the gripper", "lift the left end",  "wait"}
#   arm1 (RIGHT) text values: {"grasp the right end","close the gripper", "lift the right end", "wait"}
# and against subtask_vocab.json (verb/target maps). They are NOT free text -- do not paraphrase.
GRASP_LEFT_END = "grasp the left end"
GRASP_RIGHT_END = "grasp the right end"
NEUTRAL_WAIT = "wait"  # the canonical no-op subtask; in-vocab for BOTH arms.

# The grasp subtask each arm actually saw in training (its in-distribution command).
ARM_INDOMAIN_GRASP = {0: GRASP_LEFT_END, 1: GRASP_RIGHT_END}

# Map a commanded END side -> the exact prompt string to inject into the TARGET arm.
COMMAND_PROMPT = {"left": GRASP_LEFT_END, "right": GRASP_RIGHT_END}

# Rollout label -> commanded end. A commands LEFT, B commands RIGHT (the probe's A/B contract).
ROLLOUT_COMMAND = {"A": "left", "B": "right"}

# --------------------------------------------------------- canonical barrier-end geometry

# Contact-point ids in the steel-barrier annotation (model_data.json):
#   id=1 -> the LEFT end  (data-gen planner assigns it to move_id 0 == arm0, the LEFT arm)
#   id=2 -> the RIGHT end (assigned to move_id 1 == arm1, the RIGHT arm)
# See robofactory/planner/solutions/lift_barrier.py (pose1=id1->arm0, pose2=id2->arm1).
LEFT_CONTACT_ID = 1
RIGHT_CONTACT_ID = 2

# The fixed model->grasp convert matrix from the planner (rotation only; never touches the
# translation, but replicated verbatim for fidelity with get_grasp_pose_w_labeled_direction).
_CONVERT_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def barrier_end_world_position(
    actor_matrix: np.ndarray, actor_data: Mapping, contact_id: int
) -> np.ndarray:
    """World-frame XYZ of a labeled barrier contact point (an "end"), pre_dis=0.

    Faithful re-implementation of the position branch of the data-gen planner's
    ``get_grasp_pose_w_labeled_direction`` (motionplanner.py:275-286):

        actor_matrix (4x4 world transform of the barrier actor)
        @ local_contact_matrix (with translation scaled by actor_data['scale'])
        @ convert_matrix
      -> take [:3, 3]

    Parameters
    ----------
    actor_matrix : (4, 4) world transform of the barrier actor (``actor.pose.to_transformation_matrix()[0]``).
    actor_data   : the loaded model_data.json for the barrier (``env.annotation_data['barrier']``);
                   must contain 'contact_points_pose' (list of 4x4) and 'scale' (len-3).
    contact_id   : index into 'contact_points_pose' (use LEFT_CONTACT_ID / RIGHT_CONTACT_ID).
    """
    m = np.asarray(actor_matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"actor_matrix must be 4x4, got {m.shape}")
    # .copy() so we never mutate the caller's annotation dict (the planner mutates a fresh
    # np.asarray; if the cell is already an ndarray, asarray aliases it -- copy defends us).
    local = np.asarray(actor_data["contact_points_pose"][contact_id], dtype=np.float64).copy()
    if local.shape != (4, 4):
        raise ValueError(
            f"contact_points_pose[{contact_id}] must be 4x4, got {local.shape}"
        )
    local[:3, 3] *= np.asarray(actor_data["scale"], dtype=np.float64)
    global_mat = m @ local @ _CONVERT_MATRIX
    return global_mat[:3, 3].astype(np.float64)


def barrier_end_positions(
    actor_matrix: np.ndarray,
    actor_data: Mapping,
    left_id: int = LEFT_CONTACT_ID,
    right_id: int = RIGHT_CONTACT_ID,
) -> dict:
    """Return {'left': xyz, 'right': xyz} for the barrier's two ends in world frame.

    Also records a y-ordering cross-check: the LEFT arm (arm0) sits at world y<0 and the
    RIGHT arm (arm1) at world y>0, so the canonical LEFT end should have the smaller world y.
    `y_consistent` is False if the id-labeling disagrees with that geometric expectation
    (a loud signal that the labeling/annotation drifted)."""
    left = barrier_end_world_position(actor_matrix, actor_data, left_id)
    right = barrier_end_world_position(actor_matrix, actor_data, right_id)
    return {
        "left": left,
        "right": right,
        "y_consistent": bool(left[1] <= right[1]),
    }


def closest_end(tcp_traj: Sequence, ends: Mapping) -> tuple:
    """Assign a TCP trajectory to the LEFT or RIGHT end by closest approach.

    Returns ``(assigned_end, min_dist_left, min_dist_right)`` where ``assigned_end`` is the
    end with the smaller min-over-trajectory Euclidean distance (ties -> "left"). Using the
    *minimum* distance over the whole trajectory (not just the final frame) captures the
    approach even if the arm later slips or the barrier is knocked away.
    """
    tcp = np.asarray(tcp_traj, dtype=np.float64).reshape(-1, 3)
    if tcp.shape[0] == 0:
        raise ValueError("empty TCP trajectory")
    left = np.asarray(ends["left"], dtype=np.float64).reshape(3)
    right = np.asarray(ends["right"], dtype=np.float64).reshape(3)
    dl = float(np.linalg.norm(tcp - left, axis=1).min())
    dr = float(np.linalg.norm(tcp - right, axis=1).min())
    return ("left" if dl <= dr else "right", dl, dr)


# ----------------------------------------------------- min-approach gate on end assignment

# Heuristic gate thresholds (metres). DOCUMENTED DEFAULTS -- override via the driver CLI
# (--min-approach-dist / --min-net-displacement). They separate a GENUINE approach from a
# frozen arm that merely STARTED nearer one end. These are conservative first guesses, NOT
# calibrated against real grasp traces: if you change the probe horizon, re-check them against
# in-distribution rollouts (the in-distribution grasp should clear the gate, "wait" should not).
DEFAULT_MIN_APPROACH_DIST = 0.15      # TCP must come within 15 cm of an end to "approach" it
DEFAULT_MIN_NET_DISPLACEMENT = 0.05   # ...and the TCP must travel >= 5 cm from its start at all


def net_displacement(tcp_traj: Sequence) -> float:
    """Max Euclidean distance the TCP travelled from its START position over the trajectory.

    ~0 for a frozen/passive arm; positive once the arm moves. Used by the min-approach gate to
    reject a stationary arm (which the ungated assignment would mislabel by whichever end it
    happened to start nearest)."""
    tcp = np.asarray(tcp_traj, dtype=np.float64).reshape(-1, 3)
    if tcp.shape[0] == 0:
        raise ValueError("empty TCP trajectory")
    return float(np.linalg.norm(tcp - tcp[0], axis=1).max())


def assign_end(
    tcp_traj: Sequence,
    ends: Mapping,
    min_approach_dist: float | None = DEFAULT_MIN_APPROACH_DIST,
    min_net_displacement: float | None = DEFAULT_MIN_NET_DISPLACEMENT,
) -> tuple:
    """Closest-approach end assignment WITH a min-approach gate (validity fix #2).

    Returns ``(assigned_end, min_dist_left, min_dist_right)``. ``assigned_end`` is the closest
    end ("left"/"right", ties -> "left") ONLY when the TCP genuinely approached it; otherwise it
    is ``"none"`` (passive / no-approach). The gate requires BOTH:

      (a) the TCP came within ``min_approach_dist`` of that end  (it got NEAR), AND
      (b) the TCP net displacement from start >= ``min_net_displacement``  (it actually MOVED).

    Pass ``None`` for either threshold to disable that check; ``assign_end(.., None, None)``
    recovers the ungated ``closest_end`` label exactly.

    WHY: the ungated ``closest_end`` labels a FROZEN arm by its start position, manufacturing a
    fake-clean null (the arm is credited with "approaching" whichever end it started nearer, even
    though it never moved). The gate labels such an arm "none" instead, so passivity reads as
    passivity rather than as (dis)obedience."""
    label, dl, dr = closest_end(tcp_traj, ends)
    dmin = dl if label == "left" else dr
    disp = net_displacement(tcp_traj)
    near = (min_approach_dist is None) or (dmin <= float(min_approach_dist))
    moved = (min_net_displacement is None) or (disp >= float(min_net_displacement))
    return (label if (near and moved) else "none"), dl, dr


# ------------------------------------------ in-distribution ENGAGEMENT signals (PRIMARY metric)


def approach_progress(tcp_traj: Sequence, end: Sequence) -> float:
    """Metres of progress the TCP made TOWARD ``end``: dist(tcp[0], end) - min_t dist(tcp[t], end).

    Always >= 0 (the start frame is in the min). ~0 for an arm that never moves toward the end;
    larger the closer the arm drives its TCP to the end. This is the continuous per-rollout
    engagement signal for the in-distribution grasp-vs-wait contrast."""
    tcp = np.asarray(tcp_traj, dtype=np.float64).reshape(-1, 3)
    if tcp.shape[0] == 0:
        raise ValueError("empty TCP trajectory")
    e = np.asarray(end, dtype=np.float64).reshape(3)
    d = np.linalg.norm(tcp - e, axis=1)
    return float(d[0] - d.min())


def grasp_fraction(grasp_trace: Sequence, arm: int) -> float:
    """Fraction of recorded steps where ``arm`` was grasping the barrier (is_grasping == True).

    ``grasp_trace`` is the driver's per-step ``[is_grasping(arm0), is_grasping(arm1), ...]`` list;
    it may be shorter than the rollout, or empty, if contact recording failed (-> returns 0.0).
    The gripper-close half of the engagement signal: under "wait" the arm should not grasp; under
    its in-distribution grasp prompt it should."""
    if not grasp_trace:
        return 0.0
    vals = []
    for row in grasp_trace:
        try:
            vals.append(bool(row[arm]))
        except (IndexError, TypeError, KeyError):
            continue
    if not vals:
        return 0.0
    return float(np.mean(vals))


# ------------------------------------------------------------------------- prompt vocab guard


def load_lb_subtask_vocab(vocab_json_path: str) -> set:
    """Load the union of recognized subtask strings from a subtask_vocab.json.

    The file maps {"verb": {id: str}, "target": {id: str}}. The probe uses the UNION of all
    verb + target strings as the set of *recognized* subtask strings (a typo guard): a probe
    prompt must be one of these, otherwise it is a paraphrase the policy never saw and the
    probe would silently test the wrong thing.
    """
    data = json.loads(Path(vocab_json_path).read_text())
    vocab: set = set()
    for group in ("verb", "target"):
        vocab.update(str(v) for v in data.get(group, {}).values())
    return vocab


def is_ood_command(target_arm: int, commanded_end: str) -> bool:
    """True iff commanding ``target_arm`` to grasp ``commanded_end`` is out of distribution.

    arm0 only ever trained on the LEFT end, arm1 only on the RIGHT end, so the opposite-end
    command is OOD (see module docstring)."""
    if target_arm not in (0, 1):
        raise ValueError(f"target_arm must be 0 or 1, got {target_arm}")
    if commanded_end not in ("left", "right"):
        raise ValueError(f"commanded_end must be 'left'/'right', got {commanded_end!r}")
    indomain = "left" if target_arm == 0 else "right"
    return commanded_end != indomain


def validate_probe_prompts(
    target_arms: Iterable[int],
    recognized_vocab: set | None,
    nontarget_prompt: str = NEUTRAL_WAIT,
) -> dict:
    """Validate the probe prompts and annotate OOD combos.

    Hard-fails (ValueError) only if a probe prompt is NOT a recognized subtask string (typo
    guard). It deliberately does NOT fail on OOD combos -- the probe REQUIRES commanding the
    opposite (unseen) end. Returns a JSON-safe dict recording, per (arm, command), whether the
    command is OOD, plus the exact strings used and the recognized-vocab check.
    """
    probe_strings = {GRASP_LEFT_END, GRASP_RIGHT_END, nontarget_prompt}
    unrecognized = []
    if recognized_vocab is not None:
        unrecognized = sorted(s for s in probe_strings if s not in recognized_vocab)
        if unrecognized:
            raise ValueError(
                "subtask-obedience probe prompt(s) not in the recognized subtask vocab "
                f"{sorted(recognized_vocab)}: {unrecognized}. These are NOT the exact strings "
                "the policy was trained on -- fix the prompt or point --vocab-json at the right "
                "subtask_vocab.json."
            )
    combos = []
    for arm in target_arms:
        for label, end in ROLLOUT_COMMAND.items():
            combos.append(
                {
                    "target_arm": int(arm),
                    "rollout": label,
                    "commanded_end": end,
                    "prompt": COMMAND_PROMPT[end],
                    "is_ood": is_ood_command(int(arm), end),
                }
            )
    return {
        "grasp_left_prompt": GRASP_LEFT_END,
        "grasp_right_prompt": GRASP_RIGHT_END,
        "nontarget_prompt": nontarget_prompt,
        "recognized_vocab_checked": recognized_vocab is not None,
        "unrecognized_prompts": unrecognized,
        "combos": combos,
    }


# ----------------------------------------------------------------------------- A/B scoring


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    """One A or B rollout of the target arm (the minimal fields the scorer needs).

    The driver produces richer dicts (with raw traces); the scorer only consumes these keys
    and ignores extras. `approach` = min_dist_right - min_dist_left (larger == closer to LEFT)."""

    target_arm: int
    seed: int
    rollout: str  # "A" or "B"
    commanded_end: str  # "left" or "right"
    assigned_end: str  # "left" or "right"
    min_dist_left: float
    min_dist_right: float

    @property
    def obeyed(self) -> bool:
        return self.assigned_end == self.commanded_end

    @property
    def approach(self) -> float:
        return self.min_dist_right - self.min_dist_left

    @property
    def is_ood(self) -> bool:
        return is_ood_command(self.target_arm, self.commanded_end)


def _coerce(r) -> RolloutResult:
    if isinstance(r, RolloutResult):
        return r
    return RolloutResult(
        target_arm=int(r["target_arm"]),
        seed=int(r["seed"]),
        rollout=str(r["rollout"]),
        commanded_end=str(r["commanded_end"]),
        assigned_end=str(r["assigned_end"]),
        min_dist_left=float(r["min_dist_left"]),
        min_dist_right=float(r["min_dist_right"]),
    )


def pair_ab_rollouts(rollouts: Sequence) -> dict:
    """Group rollouts by (target_arm, seed); require EXACTLY one A and one B per pair.

    Raises ValueError on a missing/duplicate/mislabeled rollout (e.g. an A that did not command
    LEFT). Returns {(arm, seed): {"A": RolloutResult, "B": RolloutResult}}."""
    grouped: dict = {}
    for raw in rollouts:
        r = _coerce(raw)
        if r.rollout not in ("A", "B"):
            raise ValueError(f"rollout label must be 'A'/'B', got {r.rollout!r}")
        want = ROLLOUT_COMMAND[r.rollout]
        if r.commanded_end != want:
            raise ValueError(
                f"rollout {r.rollout} for arm {r.target_arm} seed {r.seed} commanded "
                f"{r.commanded_end!r}, but {r.rollout} must command {want!r} (A=left, B=right)"
            )
        key = (r.target_arm, r.seed)
        slot = grouped.setdefault(key, {})
        if r.rollout in slot:
            raise ValueError(f"duplicate rollout {r.rollout} for arm {r.target_arm} seed {r.seed}")
        slot[r.rollout] = r
    for key, slot in grouped.items():
        missing = {"A", "B"} - set(slot)
        if missing:
            raise ValueError(
                f"arm {key[0]} seed {key[1]} is missing rollout(s) {sorted(missing)}; "
                "every seed needs a paired A (left) and B (right)"
            )
    return grouped


def score_arm(rollouts: Sequence) -> dict:
    """Aggregate obedience + paired causal-shift metrics for ONE target arm.

    All input rollouts must share the same target_arm. Metrics:
      * ``obedience``            -- fraction of rollouts whose assigned end == commanded end.
                                    Prompt-IGNORING null == 0.5 (an arm that always goes to one
                                    fixed end obeys exactly one of {A,B}).
      * ``null_obedience``       -- 0.5 (the no-causal-effect reference).
      * ``left_command_obeyed``  -- fraction of A (left-commanded) rollouts assigned LEFT.
      * ``right_command_obeyed`` -- fraction of B (right-commanded) rollouts assigned RIGHT.
      * ``frac_assigned_left``   -- fraction of ALL rollouts assigned LEFT regardless of command
                                    (the prompt-ignoring "fixed bias": ~0.5 split independent of
                                    command, or pinned to the arm's trained end == disobedience).
      * ``causal_consistency``   -- PAIRED, the primary "is the channel alive?" signal: fraction
                                    of seeds where the LEFT-commanded rollout (A) leaned MORE left
                                    than the RIGHT-commanded one (B), i.e. approach_A > approach_B.
                                    No causal effect -> ~0.5.
      * ``mean_approach_shift``  -- mean over seeds of (approach_A - approach_B) in metres; >0 means
                                    the prompt pushes the TCP toward the commanded end.
    """
    rs = [_coerce(r) for r in rollouts]
    if not rs:
        raise ValueError("no rollouts to score")
    arms = {r.target_arm for r in rs}
    if len(arms) != 1:
        raise ValueError(f"score_arm expects a single target_arm, got {sorted(arms)}")
    arm = arms.pop()
    paired = pair_ab_rollouts(rs)

    n_roll = len(rs)
    n_obey = sum(1 for r in rs if r.obeyed)
    n_a = sum(1 for r in rs if r.rollout == "A")
    n_b = sum(1 for r in rs if r.rollout == "B")
    a_obey = sum(1 for r in rs if r.rollout == "A" and r.obeyed)
    b_obey = sum(1 for r in rs if r.rollout == "B" and r.obeyed)
    n_left_assigned = sum(1 for r in rs if r.assigned_end == "left")
    # With the min-approach gate (assign_end), a passive arm is "none" rather than mislabeled by
    # start position; track how many rollouts never genuinely approached an end.
    n_no_approach = sum(1 for r in rs if r.assigned_end not in ("left", "right"))

    per_seed = []
    shifts = []
    n_consistent = 0
    for (a_arm, seed), slot in sorted(paired.items(), key=lambda kv: kv[0][1]):
        a, b = slot["A"], slot["B"]
        shift = a.approach - b.approach  # >0 == A leaned more left than B == prompt-obeying
        consistent = shift > 0
        n_consistent += int(consistent)
        shifts.append(shift)
        per_seed.append(
            {
                "seed": seed,
                "A_assigned": a.assigned_end,
                "B_assigned": b.assigned_end,
                "A_obeyed": a.obeyed,
                "B_obeyed": b.obeyed,
                "A_min_dist_left": a.min_dist_left,
                "A_min_dist_right": a.min_dist_right,
                "B_min_dist_left": b.min_dist_left,
                "B_min_dist_right": b.min_dist_right,
                "approach_shift": shift,
                "directionally_consistent": consistent,
                "A_is_ood": a.is_ood,
                "B_is_ood": b.is_ood,
            }
        )

    n_seeds = len(paired)
    return {
        "target_arm": arm,
        "indomain_end": "left" if arm == 0 else "right",
        "n_seeds": n_seeds,
        "n_rollouts": n_roll,
        "obedience": n_obey / n_roll,
        "null_obedience": 0.5,
        "left_command_obeyed": (a_obey / n_a) if n_a else None,
        "right_command_obeyed": (b_obey / n_b) if n_b else None,
        "frac_assigned_left": n_left_assigned / n_roll,
        "causal_consistency": (n_consistent / n_seeds) if n_seeds else None,
        "mean_approach_shift": float(np.mean(shifts)) if shifts else None,
        "std_approach_shift": float(np.std(shifts)) if shifts else None,
        "n_ood_rollouts": sum(1 for r in rs if r.is_ood),
        "n_no_approach": n_no_approach,
        "per_seed": per_seed,
    }


def score_obedience(rollouts: Sequence) -> dict:
    """Score every target arm present in ``rollouts``; returns {"arm0": {...}, "arm1": {...}}."""
    by_arm: dict = {}
    for raw in rollouts:
        r = _coerce(raw)
        by_arm.setdefault(r.target_arm, []).append(r)
    return {f"arm{arm}": score_arm(rs) for arm, rs in sorted(by_arm.items())}


# --------------------------------------------------- ENGAGEMENT contrast scoring (PRIMARY)


@dataclasses.dataclass(frozen=True)
class EngagementRollout:
    """One rollout for the in-distribution engagement contrast (the PRIMARY metric).

    ``condition`` is "grasp" (target arm given its OWN-end, in-distribution grasp prompt) or
    "wait" (target arm given the neutral "wait" prompt) -- both from the SAME env reset, with the
    non-target arm holding "wait" in both. The two engagement components are computed by the
    driver from the rollout traces:
      * ``approach_progress`` -- metres the TCP moved toward the arm's OWN end (so.approach_progress)
      * ``grasp_fraction``    -- fraction of steps the arm was grasping (so.grasp_fraction)
    The scorer only consumes these fields and ignores any extras the driver attaches."""

    target_arm: int
    seed: int
    condition: str  # "grasp" or "wait"
    approach_progress: float
    grasp_fraction: float


def _coerce_engagement(r) -> EngagementRollout:
    if isinstance(r, EngagementRollout):
        return r
    return EngagementRollout(
        target_arm=int(r["target_arm"]),
        seed=int(r["seed"]),
        condition=str(r["condition"]),
        approach_progress=float(r["approach_progress"]),
        grasp_fraction=float(r["grasp_fraction"]),
    )


def pair_engagement_rollouts(rollouts: Sequence) -> dict:
    """Group engagement rollouts by (target_arm, seed); require EXACTLY one grasp + one wait.

    Raises ValueError on a missing/duplicate/mislabeled condition. Returns
    {(arm, seed): {"grasp": EngagementRollout, "wait": EngagementRollout}}."""
    grouped: dict = {}
    for raw in rollouts:
        r = _coerce_engagement(raw)
        if r.condition not in ("grasp", "wait"):
            raise ValueError(f"engagement condition must be 'grasp'/'wait', got {r.condition!r}")
        key = (r.target_arm, r.seed)
        slot = grouped.setdefault(key, {})
        if r.condition in slot:
            raise ValueError(
                f"duplicate engagement condition {r.condition!r} for arm {r.target_arm} seed {r.seed}"
            )
        slot[r.condition] = r
    for key, slot in grouped.items():
        missing = {"grasp", "wait"} - set(slot)
        if missing:
            raise ValueError(
                f"arm {key[0]} seed {key[1]} is missing engagement condition(s) {sorted(missing)}; "
                "every seed needs a paired grasp (own-end) and wait rollout"
            )
    return grouped


def score_engagement_arm(rollouts: Sequence) -> dict:
    """Paired in-distribution ENGAGEMENT contrast for ONE target arm (the PRIMARY metric).

    For each seed it contrasts the arm's OWN-end grasp prompt (in-distribution) against the
    neutral "wait" prompt from the same reset. Because both the prompt AND the motion are
    in-distribution, the result is interpretable in BOTH directions (unlike the OOD direction A/B):

      * shift ~ 0  AND  consistency ~ 0.5  -> the grasp prompt buys NO extra engagement over
        "wait": the prompt channel looks DEAD (the policy acts the same either way).
      * shift  > 0  AND  consistency -> 1.0 -> the grasp prompt makes the arm ACTIVELY engage
        (drive its TCP toward its own end and/or close the gripper) where "wait" leaves it
        passive: the prompt channel is ALIVE.

    Metrics (paired by seed; "consistency" null = 0.5, "shift" null = 0.0):
      * mean_approach_engagement_shift / approach_engagement_consistency
            -- on approach_progress (metres toward the arm's own end)
      * mean_grasp_close_shift / grasp_close_consistency
            -- on grasp_fraction (fraction of steps grasping)
      * mean_*_grasp / mean_*_wait -- per-condition means for context."""
    rs = [_coerce_engagement(r) for r in rollouts]
    if not rs:
        raise ValueError("no engagement rollouts to score")
    arms = {r.target_arm for r in rs}
    if len(arms) != 1:
        raise ValueError(f"score_engagement_arm expects a single target_arm, got {sorted(arms)}")
    arm = arms.pop()
    paired = pair_engagement_rollouts(rs)

    per_seed = []
    appr_shifts, grip_shifts = [], []
    appr_g, appr_w, grip_g, grip_w = [], [], [], []
    n_appr_consistent = n_grip_consistent = 0
    for (_a, seed), slot in sorted(paired.items(), key=lambda kv: kv[0][1]):
        g, w = slot["grasp"], slot["wait"]
        a_shift = g.approach_progress - w.approach_progress
        gp_shift = g.grasp_fraction - w.grasp_fraction
        appr_shifts.append(a_shift)
        grip_shifts.append(gp_shift)
        appr_g.append(g.approach_progress)
        appr_w.append(w.approach_progress)
        grip_g.append(g.grasp_fraction)
        grip_w.append(w.grasp_fraction)
        n_appr_consistent += int(a_shift > 0)
        n_grip_consistent += int(gp_shift > 0)
        per_seed.append({
            "seed": seed,
            "grasp_approach_progress": g.approach_progress,
            "wait_approach_progress": w.approach_progress,
            "approach_engagement_shift": a_shift,
            "grasp_grasp_fraction": g.grasp_fraction,
            "wait_grasp_fraction": w.grasp_fraction,
            "grasp_close_shift": gp_shift,
            "approach_engaged_more": bool(a_shift > 0),
            "grasp_closed_more": bool(gp_shift > 0),
        })

    n_seeds = len(paired)
    return {
        "target_arm": arm,
        "indomain_end": "left" if arm == 0 else "right",
        "metric": "engagement_contrast (in-distribution grasp-own-end vs wait) -- PRIMARY",
        "n_seeds": n_seeds,
        "null_shift": 0.0,
        "null_consistency": 0.5,
        "mean_approach_progress_grasp": float(np.mean(appr_g)) if appr_g else None,
        "mean_approach_progress_wait": float(np.mean(appr_w)) if appr_w else None,
        "mean_approach_engagement_shift": float(np.mean(appr_shifts)) if appr_shifts else None,
        "std_approach_engagement_shift": float(np.std(appr_shifts)) if appr_shifts else None,
        "approach_engagement_consistency": (n_appr_consistent / n_seeds) if n_seeds else None,
        "mean_grasp_fraction_grasp": float(np.mean(grip_g)) if grip_g else None,
        "mean_grasp_fraction_wait": float(np.mean(grip_w)) if grip_w else None,
        "mean_grasp_close_shift": float(np.mean(grip_shifts)) if grip_shifts else None,
        "grasp_close_consistency": (n_grip_consistent / n_seeds) if n_seeds else None,
        "per_seed": per_seed,
    }


def score_engagement(rollouts: Sequence) -> dict:
    """Score the engagement contrast for every target arm present; {"arm0":{...},"arm1":{...}}."""
    by_arm: dict = {}
    for raw in rollouts:
        r = _coerce_engagement(raw)
        by_arm.setdefault(r.target_arm, []).append(r)
    return {f"arm{arm}": score_engagement_arm(rs) for arm, rs in sorted(by_arm.items())}


# =========================================================================================
# HOLD-AFTER-GRASP scenario (pure logic) -- the "grasp-then-wait" hold probe.
# -----------------------------------------------------------------------------------------
# The old decentralized policies, once both arms had grasped+lifted the barrier, DRIFTED when
# commanded the neutral "wait" subtask (let go / dropped the barrier). This scenario runs the
# EXACT trained per-arm Phase-1 schedule (grasp own end -> close the gripper -> lift own end) on
# BOTH arms simultaneously until the barrier is up, then switches BOTH arms to "wait" (Phase 2)
# and asks: does the policy HOLD (barrier stays up + grippers stay closed) or LET GO?
#
# The per-arm Phase-1 verb strings are the EXACT trained subtask strings, verified against the
# LeRobot subtask_arm{i}_text columns of robofactory_lift_barrier_ws_subtaskdart_v1 (every
# episode's natural schedule is grasp own end -> close the gripper -> lift own end -> wait).
# =========================================================================================

HOLD_CLOSE_GRIPPER = "close the gripper"          # shared by both arms (Phase-1 close step)
HOLD_LIFT_END = {0: "lift the left end", 1: "lift the right end"}   # per-arm lift (Phase-1)
HOLD_GRASP_END = {0: GRASP_LEFT_END, 1: GRASP_RIGHT_END}           # per-arm grasp (Phase-1)


def hold_phase_prompt(arm: int, t: int, grasp_steps: int, close_steps: int, lift_steps: int) -> str:
    """The trained subtask string for ``arm`` at env step ``t`` in the hold-after-grasp scenario.

    Phase-1 schedule (own-end, exact trained strings):
        t in [0, grasp_steps)                                   -> grasp own end
        t in [grasp_steps, grasp_steps+close_steps)             -> "close the gripper"
        t in [+close_steps, grasp_steps+close_steps+lift_steps) -> lift own end
    Phase-2 (the hold test): t >= grasp_steps+close_steps+lift_steps -> "wait" (NEUTRAL_WAIT).
    """
    if arm not in (0, 1):
        raise ValueError(f"arm must be 0 or 1, got {arm}")
    if t < grasp_steps:
        return HOLD_GRASP_END[arm]
    if t < grasp_steps + close_steps:
        return HOLD_CLOSE_GRIPPER
    if t < grasp_steps + close_steps + lift_steps:
        return HOLD_LIFT_END[arm]
    return NEUTRAL_WAIT


# Hold-verdict default thresholds (metres / fraction). Conservative first guesses -- override via
# the driver CLI. lift_dz: Phase-1 must raise the barrier centre at least this far above its start
# for the hold test to be meaningful. hold_drop_tol: in Phase-2 the barrier may sag at most this
# far below its Phase-2-start height and still count as "held". grasp_hold_frac: BOTH grippers
# must stay grasping at least this fraction of Phase-2 steps to count as "held".
DEFAULT_HOLD_LIFT_DZ = 0.08
DEFAULT_HOLD_DROP_TOL = 0.05
DEFAULT_HOLD_GRASP_FRAC = 0.5


def hold_after_grasp_metrics(
    barz_trace: Sequence[float],
    grasp_trace: Sequence[Sequence[bool]],
    K: int,
    num_arms: int = 2,
    base_z: float | None = None,
    lift_dz: float = DEFAULT_HOLD_LIFT_DZ,
    hold_drop_tol: float = DEFAULT_HOLD_DROP_TOL,
    grasp_hold_frac: float = DEFAULT_HOLD_GRASP_FRAC,
    late_window: int = 10,
) -> dict:
    """Score one hold-after-grasp rollout into Phase-1 (lift) + Phase-2 (hold) numbers + verdict.

    ``barz_trace[t]`` / ``grasp_trace[t]`` are recorded AFTER executing env step ``t`` (so index
    ``K-1`` is the last Phase-1 step and index ``K`` the first Phase-2 step). ``base_z`` (robot
    base z) sets the env success threshold ``base_z + 0.15`` (LiftBarrierEnv.evaluate); pass None
    to skip the threshold-clearance fields.

    Verdict:
      * "NO_PHASE2_DATA" -- the rollout ended before Phase 2 (no hold window recorded).
      * "PHASE1_FAILED"  -- the barrier never rose >= ``lift_dz`` in Phase 1 (hold test moot).
      * "HOLDS"          -- Phase 1 lifted AND in Phase 2 the barrier sagged <= ``hold_drop_tol``
                            AND BOTH grippers stayed grasping >= ``grasp_hold_frac`` of the time.
      * "LETS-GO"        -- Phase 1 lifted but Phase 2 failed the hold (dropped and/or released).
    """
    bz = [float(z) for z in barz_trace]
    n = len(bz)

    def _gfrac(lo: int, hi: int, arm: int):
        rows = [r for r in list(grasp_trace)[lo:hi] if len(r) > arm]
        if not rows:
            return None
        return float(np.mean([bool(r[arm]) for r in rows]))

    success_threshold = (float(base_z) + 0.15) if base_z is not None else None

    if n == 0:
        return {"verdict": "NO_PHASE2_DATA", "n_steps": 0, "K": int(K),
                "success_threshold": success_threshold}

    k1_hi = min(K, n)                       # exclusive end of Phase-1 frames present
    barz_t0 = bz[0]
    barz_peak_phase1 = max(bz[:k1_hi]) if k1_hi > 0 else bz[0]
    phase1_lift_dz = barz_peak_phase1 - barz_t0
    phase1_lifted = phase1_lift_dz >= lift_dz
    phase1_cleared_threshold = (
        None if success_threshold is None else bool(barz_peak_phase1 >= success_threshold)
    )
    phase1_late_grasp = [_gfrac(max(0, k1_hi - late_window), k1_hi, i) for i in range(num_arms)]

    has_phase2 = n > K
    if not has_phase2:
        return {
            "verdict": "NO_PHASE2_DATA",
            "n_steps": n, "K": int(K),
            "barz_t0": barz_t0, "barz_peak_phase1": barz_peak_phase1,
            "phase1_lift_dz": phase1_lift_dz, "phase1_lifted": bool(phase1_lifted),
            "phase1_cleared_threshold": phase1_cleared_threshold,
            "phase1_late_grasp_frac": phase1_late_grasp,
            "success_threshold": success_threshold,
        }

    barz_phase2_start = bz[K - 1]           # state entering Phase 2 (after last lift step)
    barz_phase2_end = bz[-1]
    phase2_min_barz = min(bz[K:])
    phase2_dz = barz_phase2_end - barz_phase2_start
    phase2_grasp_frac = [_gfrac(K, n, i) for i in range(num_arms)]
    grasp_vals = [g for g in phase2_grasp_frac if g is not None]
    min_grasp_frac = min(grasp_vals) if grasp_vals else None

    held_up = phase2_dz >= -abs(hold_drop_tol)
    held_grasp = (min_grasp_frac is not None) and (min_grasp_frac >= grasp_hold_frac)
    phase2_held = bool(held_up and held_grasp)

    if not phase1_lifted:
        verdict = "PHASE1_FAILED"
    elif phase2_held:
        verdict = "HOLDS"
    else:
        verdict = "LETS-GO"

    return {
        "verdict": verdict,
        "n_steps": n, "K": int(K),
        # ---- Phase 1 (grasp + lift) ----
        "barz_t0": barz_t0,
        "barz_peak_phase1": barz_peak_phase1,
        "phase1_lift_dz": phase1_lift_dz,
        "phase1_lifted": bool(phase1_lifted),
        "phase1_cleared_threshold": phase1_cleared_threshold,
        "phase1_late_grasp_frac": phase1_late_grasp,
        # ---- Phase 2 (hold = both arms "wait") ----
        "barz_phase2_start": barz_phase2_start,
        "barz_phase2_end": barz_phase2_end,
        "phase2_min_barz": phase2_min_barz,
        "phase2_dz": phase2_dz,
        "phase2_grasp_frac": phase2_grasp_frac,
        "phase2_min_grasp_frac": min_grasp_frac,
        "phase2_held_up": bool(held_up),
        "phase2_held_grasp": bool(held_grasp),
        "success_threshold": success_threshold,
        # ---- thresholds used (for provenance) ----
        "thresholds": {
            "lift_dz": lift_dz, "hold_drop_tol": hold_drop_tol,
            "grasp_hold_frac": grasp_hold_frac, "late_window": late_window,
        },
    }
