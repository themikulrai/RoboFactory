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
    mode: str = "both"  # {schedule, probe, both}
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
    approach_steps: int = 60  # phase-0 window before close_gripper
    close_steps: int = 30     # close-gripper window before lift
    # probe mode
    num_seeds: int = 10           # A/B pairs
    probe_base_seed: int = 1000   # probe seeds = base, base+1, ...
    probe_max_steps: int = 80     # SHORT (approach phase only) -> cheap
    obedience_margin: float = 0.0  # metres; strictness buffer in decide_obedience
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


def run_schedule_episode(env, ll_policies, args, cam_map, action_prefix, seed,
                         schedule, view_sensors, video_path=None):
    """Drive each arm with the time-indexed ``schedule``; report strict success."""
    obs, _ = env.reset(seed=int(seed))
    n = args.schedule_max_steps
    per_step = expand_schedule(schedule, n)  # {arm: [prompt per step]}

    h = _hier()
    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    frames = []
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
        if success or bool(term) or bool(trunc):
            break

    if video_path is not None and frames:
        from policy._shared.multiview_video import subsample
        h._write_mp4(video_path, subsample(frames, args.video_frame_stride))

    return {"seed": int(seed), "success": bool(success), "steps": int(step + 1)}


# ===========================================================================
# PROBE-mode: one A/B pair on one seed
# ===========================================================================


def _probe_one_variant(env, ll_policies, args, cam_map, action_prefix, seed,
                       arm0_prompt, view_sensors, video_path=None):
    """SHORT rollout: arm0 driven by ``arm0_prompt``, arm1 = 'wait'. Returns arm0's
    FINAL TCP xyz. Resets to ``seed`` first (identical start for the A/B pair)."""
    h = _hier()
    obs, _ = env.reset(seed=int(seed))
    wait_prompt = vocab.render(vocab.WAIT, 0, TASK)
    prompts = [arm0_prompt] + [wait_prompt] * (args.num_arms - 1)

    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    frames = []
    for step in range(args.probe_max_steps):
        if video_path is not None:
            from policy._shared.multiview_video import tile_views
            frame = tile_views([h._extract_image(obs, s) for s in view_sensors])
            frame = h._overlay_subtasks(
                frame, step, args.probe_max_steps, prompts[0],
                prompts[1] if args.num_arms > 1 else None,
            )
            frames.append(frame)
        cur_q = _ll_action(ll_policies, args, obs, cam_map, prompts, chunks, chunk_idxs)
        action_dict = _assemble_action(chunks, chunk_idxs, cur_q, action_prefix, args.num_arms)
        obs, _, term, trunc, _ = env.step(action_dict)
        if bool(term) or bool(trunc):
            break

    tcp0 = _tcp_xyz(env, 0)
    if video_path is not None and frames:
        from policy._shared.multiview_video import subsample
        h._write_mp4(video_path, subsample(frames, args.video_frame_stride))
    return tcp0


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

    tcp_L = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix, seed,
                               prompt_left, view_sensors, video_path=vid_L)
    tcp_R = _probe_one_variant(env, ll_policies, args, cam_map, action_prefix, seed,
                               prompt_right, view_sensors, video_path=vid_R)

    return {
        "seed": int(seed),
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


def main(args: Args):
    h = _hier()
    h._assert_not_login_node()
    if args.mode not in ("schedule", "probe", "both"):
        raise ValueError(f"--mode must be schedule|probe|both, got {args.mode!r}")
    if args.camera_family not in h.CAMERA_FAMILIES:
        raise ValueError(f"--camera-family must be one of {sorted(h.CAMERA_FAMILIES)}")

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
        schedule = default_lb_schedule(args.approach_steps, args.close_steps)
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

    # ---- PROBE mode (headline) ----
    if args.mode in ("probe", "both"):
        results["probe"] = _run_probe_suite(
            env, args, cam_map, action_prefix, view_sensors, ll_policies, video_dir, tag=""
        )
        if baseline_policies is not None:
            results["probe_baseline"] = _run_probe_suite(
                env, args, cam_map, action_prefix, view_sensors, baseline_policies,
                video_dir, tag="baseline_"
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
