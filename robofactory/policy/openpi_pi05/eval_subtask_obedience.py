"""Per-arm SUBTASK-OBEDIENCE probe for the decentralized pi0.5 LoRA Lift-Barrier policies.

Forked from eval_hierarchical_lift_barrier.py (env setup, the SAPIEN shader_pack="default"
gate, the login-node guard, _build_ll_obs, the LL websocket serve handshake, grasp/barz/TCP
recording, and the always-on tiled-multiview video). The high-level Qwen orchestrator is
REMOVED: this driver injects fixed per-arm subtask prompts itself.

WHAT IT TESTS -- TWO METRICS, PRIMARY FIRST
-------------------------------------------
Whether the per-arm subtask PROMPT actually steers an arm, vs the policy ignoring it and
shortcutting off proprioception/vision. For a target arm, from the SAME env reset/seed it runs
THREE short rollouts:

    Rollout A: target arm prompt = "grasp the left end"   (commands the LEFT end)
    Rollout B: target arm prompt = "grasp the right end"  (commands the RIGHT end)
    Rollout W: target arm prompt = "wait"                 (neutral / passive baseline)

The non-target arm holds the neutral "wait" prompt in ALL rollouts (DESIGN CHOICE -- see below).

PRIMARY -- IN-DISTRIBUTION ENGAGEMENT CONTRAST (grasp-own-end vs wait).
  Compares the target arm's OWN-end grasp rollout (in-distribution: A for arm0/LEFT, B for
  arm1/RIGHT) against its wait rollout W. ENGAGEMENT = how much the arm (a) drives its TCP toward
  its own end (approach_progress, metres) and (b) closes its gripper (grasp_fraction). We report
  the PAIRED per-seed grasp-minus-wait shift on each (null shift 0, null consistency 0.5):
    shift ~ 0  -> grasp prompt buys no engagement over "wait": channel DEAD.
    shift  > 0 -> grasp prompt engages the arm where "wait" leaves it passive: channel ALIVE.
  Both prompt and motion are IN-DISTRIBUTION, so a NULL here is a clean "channel dead", unlike
  the OOD direction probe below. This is the review's #1 fix -- read THIS as the headline.

SECONDARY -- OOD DIRECTION A/B (kept for completeness, NOT the headline).
  Obedience = fraction of A/B rollouts whose TCP approaches the COMMANDED end (A->left, B->right);
  a prompt-IGNORING policy yields identical A/B (obedience 0.5 null, zero paired shift).
  OOD CAVEAT (read robofactory/utils/subtask_obedience.py too): each arm trained ONLY on its OWN
  end, so one of every A/B pair commands an OOD opposite end. A low OOD obedience does NOT prove
  the channel is dead -- the policy may understand the text yet be unable to execute the unseen
  motion. Every OOD number is an upper bound on incapacity, not disobedience.

DESIGN CHOICES (documented; override via CLI where noted)
  * Non-target arm prompt = "wait" (--nontarget-prompt). A passive non-target arm minimizes
    barrier disturbance so the target arm's TCP-vs-end signal is clean. The alternative
    (matched-correct subtask) would actively move the barrier and confound the measurement.
  * The prompt is held CONSTANT for the whole short rollout (--probe-steps, default 24). The
    in-distribution grasp phase is short (~7 steps); 24 steps (~3x) gives the arm room to travel
    toward its end and begin closing the gripper without the long stale-end drift of the old
    100-step horizon (which was ~14x the grasp phase and reused t=0 ends for the whole window).
    Over 24 steps the barrier is ~static, so the t=0 end positions stay a valid target.
  * Each rollout resets with the SAME env seed (byte-identical start observation), so the only
    *initial-condition* difference is the prompt string. Each rollout passes a distinct,
    deterministic per-rollout policy RNG seed (seed*3 + {A:0,B:1,W:2}): the LL server re-keys
    jax.random only when the seed CHANGES (policy.py:79-85), so distinct seeds force a fresh
    re-key per rollout (reproducible, no RNG carry-over). pi0.5 flow-matching draws a noise prior,
    so the rollouts carry independent (reproducible) sampling noise; the PAIRED metric averaged
    over seeds turns that into variance, not bias.
  * End assignment uses the GATED closest approach (so.assign_end, --min-approach-dist /
    --min-net-displacement) to the t=0 end positions: a frozen arm that never genuinely
    approaches an end is labelled "none" rather than mislabeled by its start position. Each
    record also stores ends_final + end_drift as a cheap t=0-validity check.

This driver CONNECTS to already-running LL websocket servers (like the base eval; pass --ports);
run_subtask_obedience_probe.sh serves the two arms on free ports and invokes it.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import sys as _sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import sapien  # noqa: F401
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from robofactory.tasks import *  # noqa: F401, F403
import robofactory.agents  # noqa: F401

_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy._shared.multiview_video import ordered_unique, tile_views, subsample  # noqa: E402

from robofactory.utils import subtask_obedience as so  # noqa: E402

IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")

# Camera families (identical to the base eval): slot name -> SAPIEN sensor name.
CAMERA_FAMILIES = {
    "wristcam": {
        "base_0_rgb_raw": "head_camera_global",
        "left_wrist_0_rgb_raw": "hand_camera_0",
        "right_wrist_0_rgb_raw": "hand_camera_1",
        "extra_0_rgb_raw": None,
    },
    "workspace": {
        "base_0_rgb_raw": "head_camera_global",
        "left_wrist_0_rgb_raw": "head_camera_agent0",
        "right_wrist_0_rgb_raw": "head_camera_agent1",
        "extra_0_rgb_raw": None,
    },
}
# Checkpoint-name shorthands: ws == workspace head cams, wc == wristcam egocentric.
CAMERA_FAMILY_ALIASES = {"ws": "workspace", "wc": "wristcam",
                         "workspace": "workspace", "wristcam": "wristcam"}

DEFAULT_VOCAB_JSON = (
    "/iris/u/mikulrai/data/RoboFactory/lerobot/"
    "robofactory_lift_barrier_ws_subtaskdart_v1/subtask_vocab.json"
)


@dataclasses.dataclass
class Args:
    task: str = "LiftBarrier-rf"
    config: str = "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml"
    # ---- LL pi0.5 servers (connect to already-running servers; the launcher serves them) ----
    host: str = "127.0.0.1"
    # REQUIRED (tyro.MISSING, no default): a hardcoded default silently routes a
    # driver to colliding ports when co-scheduled. run_subtask_obedience_probe.sh
    # allocates + passes job-unique free ports explicitly (PR1).
    ports: Annotated[str, tyro.conf.arg(help="comma-separated ports, one per arm (REQUIRED)")] = tyro.MISSING  # CSV: one websocket port per arm
    config_arm0: str = "pi05_robofactory_lb_ws_subtaskdart_decent_arm0"
    config_arm1: str = "pi05_robofactory_lb_ws_subtaskdart_decent_arm1"
    ckpt_arm0: str = ""  # informational (recorded in JSON); the launcher serves these
    ckpt_arm1: str = ""
    replan_after: int = 8
    num_arms: int = 2
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = "panda_wristcam_multi,panda_wristcam_multi"
    # ---- scenario selector (additive; "obedience" keeps the original A/B/W probe untouched) ----
    scenario: str = "obedience"   # {"obedience", "hold-after-grasp"}
    # ---- hold-after-grasp Phase-1 schedule (env steps; trained schedule ~25/20/47) + Phase-2 hold
    grasp_steps: int = 25      # Phase-1: own-end grasp ("grasp the {left,right} end")
    close_steps: int = 20      # Phase-1: "close the gripper"
    lift_steps: int = 50       # Phase-1: own-end lift ("lift the {left,right} end")
    hold_steps: int = 50       # Phase-2 (M): BOTH arms switched to "wait" -- THE HOLD TEST
    # hold-verdict thresholds (metres / fraction); see subtask_obedience.hold_after_grasp_metrics
    hold_lift_dz: float = so.DEFAULT_HOLD_LIFT_DZ
    hold_drop_tol: float = so.DEFAULT_HOLD_DROP_TOL
    hold_grasp_frac: float = so.DEFAULT_HOLD_GRASP_FRAC
    # ---- probe control ----
    target_arm: str = "both"   # {"0", "1", "both"}
    probe_steps: int = 24      # rollout length (~3x the ~7-step in-distribution grasp phase; see docstring)
    nontarget_prompt: str = "wait"   # neutral prompt for the non-target arm in ALL rollouts
    vocab_json: str = DEFAULT_VOCAB_JSON  # subtask_vocab.json for the prompt typo guard ("" skips)
    # ---- end-assignment min-approach gate (so.assign_end; metres). <0 disables that check. ----
    min_approach_dist: float = so.DEFAULT_MIN_APPROACH_DIST      # TCP must come within this of an end
    min_net_displacement: float = so.DEFAULT_MIN_NET_DISPLACEMENT  # ...and travel >= this from start
    # ---- seeds (PR4 single source of truth) ----
    seed_pool: str = ""        # named frozen pool (e.g. fresh_ood_60)
    env_seeds: str = ""        # ad-hoc final-env-seed list (comma/space)
    n_seeds: int = 24          # cap on resolved seeds run (raised 10->24 for statistical power)
    allow_train_seeds: bool = False
    # ---- cameras / sim ----
    camera_family: str = "ws"  # {ws|wc} (or workspace|wristcam)
    max_env_steps: int = 500
    sim_backend: str = "auto"
    # ---- outputs ----
    out_json: str = ""         # explicit result-JSON path ("" => results_dir/eval_subtask_obedience_<ts>.json)
    results_dir: str = "/iris/u/mikulrai/projects/RoboFactory/eval_results"
    video_dir: str = ""        # tiled-multiview videos (always recorded); "" => <results_dir>/videos
    video_frame_stride: int = 2
    # ---- modes / guards ----
    mock_ll: bool = False      # zero-action chunks instead of contacting LL servers (plumbing smoke)
    expect_config: str = ""    # PR2 server-identity handshake (single name or CSV per arm)


# ----------------------------------------------------------------------------- obs helpers
# (identical to eval_hierarchical_lift_barrier.py -- the LL obs contract must match training)


def _gripper(q):
    return float((q[7] + q[8]) / 2.0)


def _build_state(obs, num_arms, robot_uid):
    parts = []
    for i in range(num_arms):
        q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
        parts.append(q[:7].astype(np.float32))
        parts.append(np.array([_gripper(q)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _extract_image(obs, cam):
    img = obs["sensor_data"][cam]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _cur_qpos(obs, num_arms, robot_uid):
    return [
        np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()[:7].astype(np.float32)
        for i in range(num_arms)
    ]


def _build_ll_obs(obs, args, cam_map, prompt):
    """Build one LL observation dict (4 image slots + state + prompt)."""
    out = {"state": _build_state(obs, args.num_arms, args.robot_uid), "prompt": prompt}
    ref = None
    for slot, cam in cam_map.items():
        if cam is None:
            continue
        img = _extract_image(obs, cam)
        out[slot] = img
        if ref is None:
            ref = img.shape
    if ref is None:
        ref = (224, 224, 3)
    for slot in IMAGE_SLOTS:
        if slot not in out:
            out[slot] = np.zeros(ref, dtype=np.uint8)
    return out


def _scal(x):
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return a.reshape(-1)


def _tcp_pos(env, arm):
    """World XYZ of arm `arm`'s tool-center-point (matches motionplanner's env_agent[i].tcp)."""
    agent = env.unwrapped.agent.agents[arm]
    p = agent.tcp.pose.p
    return _scal(p)[:3].astype(np.float64)


def _barrier_world_matrix(env):
    barrier = env.unwrapped.barrier
    m = barrier.pose.to_transformation_matrix()
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    if m.ndim == 3:
        m = m[0]
    return m.astype(np.float64)


def _barrier_ends(env):
    """Canonical {'left','right',...} barrier-end world positions (reused planner formula)."""
    adata = env.unwrapped.annotation_data["barrier"]
    return so.barrier_end_positions(_barrier_world_matrix(env), adata)


# ---------------------------------------------------------------------------------- video


def _write_mp4(path, frames):
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _overlay(img, step, total, header, left, right):
    """Burn the probe header (target arm + commanded end) + per-arm prompt onto a frame."""
    import cv2

    img = np.ascontiguousarray(img)
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(img, f"{header}  step {step + 1}/{total}", (5, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"L: {left or '-'}", (5, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"R: {right or '-'}", (5, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------------------- LL clients
# (identical to the base eval -- same websocket contract / mock stand-in)


class _MockLLPolicy:
    """Zero-action LL stand-in for --mock-ll plumbing tests (no openpi server needed)."""

    def __init__(self, horizon: int = 16, dim: int = 8):
        self.horizon = horizon
        self.dim = dim

    def infer(self, obs):
        return {"actions": np.zeros((self.horizon, self.dim), dtype=np.float32)}


def _make_ll_policies(args):
    if args.mock_ll:
        return [_MockLLPolicy() for _ in range(args.num_arms)]
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    ports = [int(p) for p in args.ports.split(",")]
    if len(ports) < args.num_arms:
        raise ValueError(f"need {args.num_arms} LL ports, got {ports}")
    return [WebsocketClientPolicy(host=args.host, port=ports[i]) for i in range(args.num_arms)]


# ----------------------------------------------------------------------------------- guards
# (verbatim from eval_hierarchical_lift_barrier.py -- the SAPIEN fidelity gates are mandatory)

REQUIRED_SHADER_PACK = "default"
_SHADER_CONFIG_KEYS = ("sensor_configs", "human_render_camera_configs", "viewer_camera_configs")


def _assert_shader_pack_default(gym_make_kwargs: dict) -> None:
    bad = []
    for key in _SHADER_CONFIG_KEYS:
        cfg = gym_make_kwargs.get(key)
        sp = cfg.get("shader_pack") if isinstance(cfg, dict) else None
        if sp != REQUIRED_SHADER_PACK:
            bad.append(f"{key}.shader_pack={sp!r}")
    if bad:
        raise RuntimeError(
            "SAPIEN shader_pack guard FAILED (Gate G2): expected shader_pack="
            f"{REQUIRED_SHADER_PACK!r} on {list(_SHADER_CONFIG_KEYS)}, but got: "
            f"{', '.join(bad)}. ManiSkill's 'minimal' fallback blacks out the upper region of "
            "head_camera_global vs the gray clear used at data-gen, silently corrupting eval. "
            "See memory/feedback_sapien_shader_pack_eval_mismatch.md."
        )


def _is_compute_node() -> bool:
    import os

    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"):
        return True
    if os.environ.get("ROBOFACTORY_ALLOW_NON_SLURM_NODE") == "1":
        return True
    host = socket.gethostname().split(".")[0]
    if host.startswith("iris-ws"):
        return True
    return False


def _assert_not_login_node():
    if _is_compute_node():
        return
    host = socket.gethostname()
    raise RuntimeError(
        f"REFUSING to run SAPIEN eval on host {host!r}: no SLURM_JOB_ID and not a workstation "
        "node. The iris login node renders ~32% darker / blacks out head_camera_global, which "
        "silently corrupts eval. Run inside a SLURM allocation on a COMPUTE node (sbatch / "
        "srun / interactive). If you are knowingly on a correctly-rendering interactive node, "
        "set ROBOFACTORY_ALLOW_NON_SLURM_NODE=1. See "
        "memory/feedback_sapien_shader_pack_eval_mismatch.md."
    )


# ------------------------------------------------------------------------------ rollout


# Per-rollout policy-RNG seed offsets. Distinct values force the LL server to re-key jax.random
# per rollout (policy.py:79-85); A/B/W must differ so none inherits another's advanced RNG.
_ROLLOUT_SEED_OFFSET = {"A": 0, "B": 1, "W": 2}


def run_rollout(env, ll_policies, args, cam_map, action_prefix, view_sensors,
                target_arm, seed, rollout_label, target_prompt, commanded_end, video_path):
    """One rollout: drive `target_arm` with `target_prompt` from reset `seed`.

    `rollout_label` in {"A","B","W"}; `commanded_end` in {"left","right"} for A/B or None for the
    "wait" baseline (W). Returns a record dict consumable by both subtask_obedience.score_obedience
    (the A/B direction metric) and score_engagement (the in-distribution grasp-vs-wait metric),
    plus raw traces. End assignment is GATED (so.assign_end): a frozen arm reads "none"."""
    target_arm = int(target_arm)
    obs, _ = env.reset(seed=int(seed))

    ends0 = _barrier_ends(env)  # t=0 end positions = the target the arm should approach

    # Fixed per-arm prompts for the whole short rollout (non-target arm always neutral).
    prompts = [args.nontarget_prompt] * args.num_arms
    prompts[target_arm] = target_prompt

    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    episode_seed = int(seed) * 3 + _ROLLOUT_SEED_OFFSET[rollout_label]

    try:
        _barrier = env.unwrapped.barrier
        _subagents = env.unwrapped.agent.agents
    except Exception:
        _barrier, _subagents = None, None

    frames = []
    tcp_traj = []
    grasp_trace, barz_trace = [], []
    success = False
    step = 0
    header = f"arm{target_arm} cmd={commanded_end or 'wait'} ({rollout_label})"
    n_steps = min(args.probe_steps, args.max_env_steps)

    for step in range(n_steps):
        frame = tile_views([_extract_image(obs, s) for s in view_sensors])
        frame = _overlay(frame, step, n_steps, header,
                         prompts[0], prompts[1] if args.num_arms > 1 else None)
        frames.append(frame)

        # record target-arm TCP BEFORE stepping (closest-approach over the trajectory)
        tcp_traj.append(_tcp_pos(env, target_arm).tolist())

        for i in range(args.num_arms):
            if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
                obs_i = _build_ll_obs(obs, args, cam_map, prompts[i])
                obs_i["_episode_seed"] = episode_seed  # distinct per-rollout -> forces a fresh re-key
                chunks[i] = np.asarray(ll_policies[i].infer(obs_i)["actions"])
                chunk_idxs[i] = 0

        cur_q = _cur_qpos(obs, args.num_arms, args.robot_uid)
        action_dict = {}
        for i in range(args.num_arms):
            step_i = chunks[i][chunk_idxs[i]]
            delta = step_i[:7]
            grip = step_i[7]
            target = np.concatenate([cur_q[i] + delta, np.array([grip], dtype=np.float32)])
            action_dict[f"{action_prefix}-{i}"] = target.astype(np.float32)
            chunk_idxs[i] += 1

        obs, _, term, trunc, info = env.step(action_dict)
        s = info.get("success", False)
        success = bool(s.item() if hasattr(s, "item") else s)
        if _barrier is not None:
            try:
                barz_trace.append(float(_scal(_barrier.pose.p)[2]))
                grasp_trace.append([bool(_scal(_subagents[i].is_grasping(_barrier))[0])
                                    for i in range(args.num_arms)])
            except Exception:
                pass
        if success or term or trunc:
            break

    # final TCP after the last step
    tcp_traj.append(_tcp_pos(env, target_arm).tolist())
    ends_final = _barrier_ends(env)

    if frames:
        _write_mp4(video_path, subsample(frames, args.video_frame_stride))

    # GATED end assignment: a frozen arm that never genuinely approached an end -> "none".
    gate_kw = dict(
        min_approach_dist=(None if args.min_approach_dist < 0 else args.min_approach_dist),
        min_net_displacement=(None if args.min_net_displacement < 0 else args.min_net_displacement),
    )
    assigned_end, dl, dr = so.assign_end(tcp_traj, ends0, **gate_kw)

    # ENGAGEMENT signals (PRIMARY metric): progress toward the arm's OWN end + gripper-close.
    own_end_side = "left" if target_arm == 0 else "right"
    appr_prog = so.approach_progress(tcp_traj, ends0[own_end_side])
    gfrac = so.grasp_fraction(grasp_trace, target_arm)
    disp = so.net_displacement(tcp_traj)
    end_drift = {
        "left": float(np.linalg.norm(ends_final["left"] - ends0["left"])),
        "right": float(np.linalg.norm(ends_final["right"] - ends0["right"])),
    }

    return {
        "target_arm": target_arm,
        "seed": int(seed),
        "rollout": rollout_label,
        "commanded_end": commanded_end,
        "prompt": prompts[target_arm],
        "nontarget_prompt": args.nontarget_prompt,
        "assigned_end": assigned_end,
        "min_dist_left": dl,
        "min_dist_right": dr,
        "net_displacement": disp,
        "own_end": own_end_side,
        "approach_progress": appr_prog,   # toward own end (engagement; metres)
        "grasp_fraction": gfrac,          # fraction of steps target arm grasping (engagement)
        "is_ood": (so.is_ood_command(target_arm, commanded_end)
                   if commanded_end in ("left", "right") else None),
        "obeyed": (assigned_end == commanded_end
                   if commanded_end in ("left", "right") else None),
        "success": bool(success),
        "steps": int(step + 1),
        "ends_t0": {"left": ends0["left"].tolist(), "right": ends0["right"].tolist(),
                    "y_consistent": ends0["y_consistent"]},
        "ends_final": {"left": ends_final["left"].tolist(), "right": ends_final["right"].tolist()},
        "end_drift": end_drift,
        "tcp_traj": tcp_traj,
        "grasp_trace": grasp_trace,
        "barz_trace": barz_trace,
        "video": video_path,
    }


# ----------------------------------------------------------- hold-after-grasp scenario rollout


def _robot_base_z(env, arm: int = 0) -> float:
    """World z of arm `arm`'s robot base (LiftBarrierEnv.evaluate's success threshold = base_z+0.15)."""
    try:
        p = env.unwrapped.agent.agents[arm].robot.pose.p
        return float(_scal(p)[2])
    except Exception:
        return float("nan")


def run_hold_after_grasp_rollout(env, ll_policies, args, cam_map, action_prefix, view_sensors,
                                 seed, video_path):
    """Both arms grasp+lift (Phase 1), then BOTH commanded "wait" (Phase 2 = the HOLD test).

    Phase 1 (K = grasp_steps + close_steps + lift_steps): each arm runs its EXACT trained own-end
    schedule grasp -> "close the gripper" -> lift, SIMULTANEOUSLY, long enough to grasp + lift the
    barrier. Phase 2 (M = hold_steps): BOTH arms switched to "wait". A prompt change forces an
    immediate LL replan so the phase switch is crisp (no up-to-replan_after-step lag). Records the
    FULL tiled-multiview video plus per-step barrier-z + per-arm is_grasping over both phases, and
    scores them with subtask_obedience.hold_after_grasp_metrics (HOLDS vs LETS-GO)."""
    g, c, l, M = args.grasp_steps, args.close_steps, args.lift_steps, args.hold_steps
    K = g + c + l
    total = min(K + M, args.max_env_steps)

    obs, _ = env.reset(seed=int(seed))
    ends0 = _barrier_ends(env)
    base_z = _robot_base_z(env, 0)
    episode_seed = int(seed) * 7 + 5  # distinct RNG namespace from the A/B/W obedience rollouts

    try:
        _barrier = env.unwrapped.barrier
        _subagents = env.unwrapped.agent.agents
    except Exception:
        _barrier, _subagents = None, None

    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    last_prompt = [None] * args.num_arms

    frames = []
    barz_trace, grasp_trace = [], []
    tcp_traj = [[] for _ in range(args.num_arms)]
    success_first_step = None
    step = 0
    for step in range(total):
        phase = 1 if step < K else 2
        prompts = [so.hold_phase_prompt(i, step, g, c, l) for i in range(args.num_arms)]
        phase_label = "P1 grasp+lift" if phase == 1 else "P2 HOLD(wait)"
        header = f"hold-grasp s{seed} [{phase_label}] K{K} M{M}"

        frame = tile_views([_extract_image(obs, s) for s in view_sensors])
        frame = _overlay(frame, step, total, header,
                         prompts[0], prompts[1] if args.num_arms > 1 else None)
        frames.append(frame)
        for i in range(args.num_arms):
            tcp_traj[i].append(_tcp_pos(env, i).tolist())

        for i in range(args.num_arms):
            # Force a fresh chunk when the prompt CHANGES (crisp phase switch), else every replan_after.
            need = (chunks[i] is None or chunk_idxs[i] >= args.replan_after
                    or prompts[i] != last_prompt[i])
            if need:
                obs_i = _build_ll_obs(obs, args, cam_map, prompts[i])
                obs_i["_episode_seed"] = episode_seed  # forces a fresh re-key per seed
                chunks[i] = np.asarray(ll_policies[i].infer(obs_i)["actions"])
                chunk_idxs[i] = 0
                last_prompt[i] = prompts[i]

        cur_q = _cur_qpos(obs, args.num_arms, args.robot_uid)
        action_dict = {}
        for i in range(args.num_arms):
            step_i = chunks[i][chunk_idxs[i]]
            target = np.concatenate([cur_q[i] + step_i[:7], np.array([step_i[7]], dtype=np.float32)])
            action_dict[f"{action_prefix}-{i}"] = target.astype(np.float32)
            chunk_idxs[i] += 1

        obs, _, term, trunc, info = env.step(action_dict)
        s = info.get("success", False)
        s = bool(s.item() if hasattr(s, "item") else s)
        if s and success_first_step is None:
            success_first_step = step
        if _barrier is not None:
            try:
                barz_trace.append(float(_scal(_barrier.pose.p)[2]))
                grasp_trace.append([bool(_scal(_subagents[i].is_grasping(_barrier))[0])
                                    for i in range(args.num_arms)])
            except Exception:
                pass
        # NOTE: do NOT break on env-success or `term`. LiftBarrier success is an instantaneous
        # z>base+0.15 check that fires DURING the lift; we must keep stepping into the wait phase
        # to test the hold. gym.make adds only MSTimeLimit (no auto-reset), so stepping past `term`
        # is safe. Only `trunc` (env max_episode_steps=500) ends us early -- total (~145) << 500.
        if trunc:
            break

    if frames:
        _write_mp4(video_path, subsample(frames, args.video_frame_stride))

    hold = so.hold_after_grasp_metrics(
        barz_trace, grasp_trace, K, num_arms=args.num_arms, base_z=base_z,
        lift_dz=args.hold_lift_dz, hold_drop_tol=args.hold_drop_tol,
        grasp_hold_frac=args.hold_grasp_frac,
    )

    return {
        "scenario": "hold-after-grasp",
        "seed": int(seed),
        "base_z": base_z,
        "phase_schedule": {"grasp_steps": g, "close_steps": c, "lift_steps": l,
                           "K": K, "hold_steps": M, "total": total},
        "success_first_step": success_first_step,
        "hold": hold,
        "barz_trace": barz_trace,
        "grasp_trace": grasp_trace,
        "ends_t0": {"left": ends0["left"].tolist(), "right": ends0["right"].tolist(),
                    "y_consistent": ends0["y_consistent"]},
        "video": video_path,
    }


def _run_hold_after_grasp(env, ll_policies, args, cam_map, view_sensors, action_prefix,
                          seeds, fam, seed_provenance, ll_server_metadata):
    """Driver for the hold-after-grasp scenario: one rollout + video per seed, then a JSON summary."""
    g, c, l, M = args.grasp_steps, args.close_steps, args.lift_steps, args.hold_steps
    K = g + c + l
    print(f"[hold-after-grasp] PHASE1 grasp={g}+close={c}+lift={l} -> K={K} steps; "
          f"PHASE2 hold(wait) M={M}; total={K + M} "
          f"(prompts arm0: {so.HOLD_GRASP_END[0]!r}->{so.HOLD_CLOSE_GRIPPER!r}->{so.HOLD_LIFT_END[0]!r}->'wait'; "
          f"arm1: {so.HOLD_GRASP_END[1]!r}->...->{so.HOLD_LIFT_END[1]!r}->'wait')")

    records = []
    for seed in seeds:
        vp = str(Path(args.video_dir) / f"hold_after_grasp_seed{seed}.mp4")
        rec = run_hold_after_grasp_rollout(env, ll_policies, args, cam_map, action_prefix,
                                           view_sensors, seed, vp)
        records.append(rec)
        h = rec["hold"]
        print(f"[seed{seed}] verdict={h['verdict']} | P1: z0={h.get('barz_t0'):.3f} "
              f"peak={h.get('barz_peak_phase1'):.3f} dz={h.get('phase1_lift_dz'):.3f} "
              f"lifted={h.get('phase1_lifted')} cleared_thr={h.get('phase1_cleared_threshold')} "
              f"late_grasp={h.get('phase1_late_grasp_frac')} | "
              f"P2: z_start={h.get('barz_phase2_start')} z_end={h.get('barz_phase2_end')} "
              f"dz={h.get('phase2_dz')} grasp_frac={h.get('phase2_grasp_frac')} "
              f"first_success_step={rec['success_first_step']} video={vp}")
    env.close()

    # Pick the best seed for the headline clip: prefer a clean Phase-1 lift, then the most
    # informative Phase-2 (largest grasp retention if HOLDS, else clearest drop if LETS-GO).
    def _rank(rec):
        h = rec["hold"]
        lifted = 1 if h.get("phase1_lifted") else 0
        grasp = h.get("phase2_min_grasp_frac") or 0.0
        return (lifted, h.get("phase1_lift_dz") or 0.0, grasp)
    best = max(records, key=_rank) if records else None

    summary = {
        "task": args.task,
        "probe": "hold_after_grasp",
        "scenario": "hold-after-grasp",
        "args": dataclasses.asdict(args),
        "camera_family_resolved": fam,
        "seed_provenance": seed_provenance,
        "phase_schedule": {"grasp_steps": g, "close_steps": c, "lift_steps": l,
                           "K": K, "hold_steps": M},
        "hold_thresholds": {"lift_dz": args.hold_lift_dz, "drop_tol": args.hold_drop_tol,
                            "grasp_frac": args.hold_grasp_frac},
        "server_metadata": {f"arm{i}": ll_server_metadata[i] for i in range(len(ll_server_metadata))},
        "best_seed": (best["seed"] if best else None),
        "best_video": (best["video"] if best else None),
        "per_seed_verdict": {str(r["seed"]): r["hold"]["verdict"] for r in records},
        "records": records,
    }

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_json) if args.out_json else (
        Path(args.results_dir) / f"eval_hold_after_grasp_{args.task}_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[result] wrote {out_path}")
    if best:
        print(f"[BEST] seed={best['seed']} verdict={best['hold']['verdict']} video={best['video']}")
    return summary


def _resolve_target_arms(target_arm: str):
    if target_arm == "both":
        return [0, 1]
    if target_arm in ("0", "1"):
        return [int(target_arm)]
    raise ValueError(f"--target-arm must be one of {{0,1,both}}, got {target_arm!r}")


# --------------------------------------------------------------------------------- main


def main(args: Args):
    _assert_not_login_node()

    fam = CAMERA_FAMILY_ALIASES.get(args.camera_family)
    if fam is None:
        raise ValueError(
            f"--camera-family must be one of {sorted(CAMERA_FAMILY_ALIASES)}, got {args.camera_family!r}"
        )
    cam_map = dict(CAMERA_FAMILIES[fam])
    view_sensors = ordered_unique(list(cam_map.values()))
    target_arms = _resolve_target_arms(args.target_arm)
    print(f"[camera-family] {args.camera_family} -> {fam}: LL={cam_map} video_views={view_sensors}")

    # Prompt guard: probe strings must be the EXACT trained subtask strings (typo guard); OOD
    # combos are flagged but NOT rejected (the probe must command the opposite/unseen end).
    recognized_vocab = None
    if args.vocab_json:
        recognized_vocab = so.load_lb_subtask_vocab(args.vocab_json)
    prompt_validation = so.validate_probe_prompts(
        target_arms, recognized_vocab, nontarget_prompt=args.nontarget_prompt
    )
    n_ood = sum(1 for c in prompt_validation["combos"] if c["is_ood"])
    print(f"[prompts] grasp_left={so.GRASP_LEFT_END!r} grasp_right={so.GRASP_RIGHT_END!r} "
          f"nontarget={args.nontarget_prompt!r} | {n_ood}/{len(prompt_validation['combos'])} "
          "(arm,command) combos are OOD (opposite-end). OOD obedience is an upper bound on "
          "incapacity, NOT disobedience -- read the paired causal shift.")

    # Seeds (PR4 single source of truth). --seed-pool / --env-seeds required (no silent default).
    if not (args.seed_pool or args.env_seeds):
        raise ValueError("provide --seed-pool NAME or --env-seeds CSV (no implicit default seeds)")
    from robofactory.utils.eval_seeds import resolve_seeds
    resolved, seed_provenance = resolve_seeds(
        pool=args.seed_pool or None, env_seeds=args.env_seeds or None,
        allow_train=args.allow_train_seeds,
    )
    seeds = list(resolved)[: args.n_seeds] if args.n_seeds > 0 else list(resolved)
    print(f"[seeds] running {len(seeds)} seed(s): {seeds}")

    if not args.video_dir:
        args.video_dir = str(Path(args.results_dir) / "videos")

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
    _assert_shader_pack_default(gym_make_kwargs)
    env = gym.make(args.task, **gym_make_kwargs)
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]

    ll_policies = _make_ll_policies(args)

    # PR2 server-identity handshake (skipped under --mock-ll). Per-arm decent dim is 8; the
    # config_name is the real guard (a wrong/port-collided server hard-fails before rollout 1).
    ll_server_metadata = []
    for i, p in enumerate(ll_policies):
        meta = dict(p.get_server_metadata() or {}) if hasattr(p, "get_server_metadata") else {}
        ll_server_metadata.append(meta)
        print(f"[arm{i}] server metadata: {meta}")
    expect = args.expect_config
    if not expect and not args.mock_ll:
        expect = f"{args.config_arm0},{args.config_arm1}"  # default: the two configs we expect
    if expect and not args.mock_ll:
        expect_configs = [c.strip() for c in expect.split(",")]
        if len(expect_configs) == 1:
            expect_configs = expect_configs * args.num_arms
        if len(expect_configs) != args.num_arms:
            print(f"SERVER IDENTITY MISMATCH: --expect-config has {len(expect_configs)} entries "
                  f"but --num-arms={args.num_arms}; refusing to run", file=_sys.stderr, flush=True)
            _sys.exit(3)
        from robofactory.utils.server_identity import assert_server_identity_or_exit
        for i in range(args.num_arms):
            assert_server_identity_or_exit(
                ll_server_metadata[i], expect_configs[i], expect_action_dim=None, label=f"arm{i}"
            )

    # ---- hold-after-grasp scenario: both arms grasp+lift, then both "wait" (the HOLD test) ----
    if args.scenario == "hold-after-grasp":
        return _run_hold_after_grasp(
            env, ll_policies, args, cam_map, view_sensors, action_prefix,
            seeds, fam, seed_provenance, ll_server_metadata,
        )
    if args.scenario != "obedience":
        raise ValueError(f"--scenario must be 'obedience' or 'hold-after-grasp', got {args.scenario!r}")

    # Per (arm, seed): A (grasp left), B (grasp right), W (wait). A/B feed the SECONDARY OOD
    # direction metric; the arm's OWN-end grasp (A for arm0, B for arm1) + W feed the PRIMARY
    # in-distribution engagement contrast.
    rollout_specs = [
        ("A", so.GRASP_LEFT_END, "left"),
        ("B", so.GRASP_RIGHT_END, "right"),
        ("W", args.nontarget_prompt, None),
    ]
    rollouts = []
    for target_arm in target_arms:
        for seed in seeds:
            for rollout_label, target_prompt, commanded_end in rollout_specs:
                vp = str(Path(args.video_dir) / f"arm{target_arm}_seed{seed}_{rollout_label}.mp4")
                rec = run_rollout(env, ll_policies, args, cam_map, action_prefix, view_sensors,
                                  target_arm, seed, rollout_label, target_prompt, commanded_end, vp)
                rollouts.append(rec)
                print(f"[arm{target_arm} seed{seed} {rollout_label}] cmd={commanded_end or 'wait'} "
                      f"assigned={rec['assigned_end']} obeyed={rec['obeyed']} ood={rec['is_ood']} "
                      f"dL={rec['min_dist_left']:.3f} dR={rec['min_dist_right']:.3f} "
                      f"appr={rec['approach_progress']:.3f} grip={rec['grasp_fraction']:.2f} "
                      f"disp={rec['net_displacement']:.3f} success={rec['success']} steps={rec['steps']}")

    env.close()

    # PRIMARY: in-distribution engagement contrast (own-end grasp vs wait, paired per seed).
    engagement_rollouts = []
    for r in rollouts:
        own_label = "A" if r["target_arm"] == 0 else "B"  # the arm's in-distribution grasp
        if r["rollout"] == own_label:
            engagement_rollouts.append({**r, "condition": "grasp"})
        elif r["rollout"] == "W":
            engagement_rollouts.append({**r, "condition": "wait"})
    engagement_scores = so.score_engagement(engagement_rollouts)
    for arm_key, sc in engagement_scores.items():
        print(f"[ENGAGEMENT {arm_key}] (PRIMARY) approach_shift={sc['mean_approach_engagement_shift']} "
              f"(grasp {sc['mean_approach_progress_grasp']} vs wait {sc['mean_approach_progress_wait']}) "
              f"approach_consistency={sc['approach_engagement_consistency']} (null 0.5) | "
              f"grip_shift={sc['mean_grasp_close_shift']} grip_consistency={sc['grasp_close_consistency']} "
              f"n_seeds={sc['n_seeds']}")

    # SECONDARY: OOD direction A/B obedience + paired causal shift.
    direction_rollouts = [r for r in rollouts if r["rollout"] in ("A", "B")]
    scores = so.score_obedience(direction_rollouts)
    for arm_key, sc in scores.items():
        print(f"[direction {arm_key}] (SECONDARY/OOD) obedience={sc['obedience']:.3f} "
              f"(null {sc['null_obedience']}) causal_consistency={sc['causal_consistency']} "
              f"mean_shift={sc['mean_approach_shift']} n_no_approach={sc['n_no_approach']} "
              f"n_seeds={sc['n_seeds']} frac_left={sc['frac_assigned_left']:.3f}")

    results = {
        "task": args.task,
        "probe": "subtask_obedience",
        "args": dataclasses.asdict(args),
        "camera_family_resolved": fam,
        "seed_provenance": seed_provenance,
        "prompt_validation": prompt_validation,
        "server_metadata": {f"arm{i}": ll_server_metadata[i] for i in range(len(ll_server_metadata))},
        "mock_ll": args.mock_ll,
        "n_seeds": len(seeds),
        "primary_metric": "engagement_scores (in-distribution grasp-own-end vs wait)",
        "engagement_scores": engagement_scores,
        "scores": scores,
        "rollouts": rollouts,
    }

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_json) if args.out_json else (
        Path(args.results_dir) / f"eval_subtask_obedience_{args.task}_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[result] wrote {out_path}")
    return results


if __name__ == "__main__":
    main(tyro.cli(Args))
