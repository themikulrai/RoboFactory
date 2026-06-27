"""Non-hierarchical (LL-only) pi0.5 eval for TwoRobotsStackCube (2SC): 2 per-arm
policy servers, one per port.

This is a faithful 2SC clone of eval_decent_pi05.py — only the Args defaults are
retargeted (task, config yaml, num_arms=2, max_env_steps=600, camera-mapping,
robot_uids_csv, wandb tags). All run/episode/validation logic is identical.

Each server runs a per-arm LoRA model (pi05_robofactory_2sc_wc_*_decent_arm{i}).
The server applies RoboFactoryDecentInputs internally, so this client sends the full
16-dim state (2 arms x 8) + all mapped images (same format as centralised eval). Each
server returns an 8-dim per-arm action chunk (delta joints + gripper).

Usage:
    # Start 2 servers first (e.g., in background via SLURM script), then:
    python eval_2sc_ll_only.py \\
        --ports 8000,8001 \\
        --prompt "stack the two cubes using two robot arms" \\
        --seeds 10010,10011,10012 \\
        --num-episodes 1
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import sapien  # noqa: F401
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robofactory.tasks import *  # noqa: F401, F403
import robofactory.agents  # noqa: F401  (registers panda_wristcam_multi via @register_agent)
from robofactory.utils.success_persistence import probe_sustained, SUSTAIN_K  # PR7

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from policy._shared.eval_context import WandbRun, VideoRecorder  # noqa: E402
from policy._shared.multiview_video import ordered_unique, tile_views, subsample  # noqa: E402


# PR6: the TSC DEFAULT_PROMPT that was silently used for EVERY task is DELETED.
# --prompt is now REQUIRED and validated against the task / subtask vocab (see Args.prompt).
DEFAULT_CAMERA_MAPPING = {
    "base_0_rgb_raw": "head_camera_global",
    "left_wrist_0_rgb_raw": "hand_camera_0",
    "right_wrist_0_rgb_raw": "hand_camera_1",
    "extra_0_rgb_raw": "hand_camera_2",
}
# Default (wc/centralised) image slots. NOTE: these are NOT hardcoded into the obs
# construction anymore — the actual slots are driven by the --camera-mapping JSON keys
# (see _resolve_camera_mapping). This tuple is only the fallback used when no
# --camera-mapping is passed (i.e. DEFAULT_CAMERA_MAPPING above). The 5-camera "union"
# model uses different raw keys (global_rgb, wrist0_rgb, wrist1_rgb, head0_rgb, head1_rgb)
# that the server's RoboFactoryDecentUnionInputs reads verbatim, so its mapping JSON
# carries those 5 keys and they flow straight through to the obs dict.
DEFAULT_IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")

# SAPIEN shader_pack / login-node / black-sky fidelity guards now live in the shared
# robofactory.utils.eval_guards module (PR3): assert_shader_pack_default (hard),
# shader_bg_guard (PROMOTED warn -> hard fail; RF_ALLOW_SHADER_MISMATCH=1 downgrades),
# assert_not_login_node (hard). See memory/feedback_sapien_shader_pack_eval_mismatch.md.


@dataclasses.dataclass
class Args:
    task: str = "TwoRobotsStackCube-rf"
    config: str = (
        "/iris/u/mikulrai/RoboFactory-2sc-regen/robofactory/configs/table/two_robots_stack_cube_aug15.yaml"
    )
    host: str = "127.0.0.1"
    # REQUIRED (tyro.MISSING, no default): a hardcoded default silently routes a
    # driver to colliding ports when co-scheduled. Launchers/run_eval.py always
    # pass job-unique free ports explicitly (PR1).
    ports: Annotated[str, tyro.conf.arg(help="comma-separated ports, one per arm (REQUIRED)")] = tyro.MISSING
    num_episodes: int = 1
    # Seed resolution (PR4 — robofactory.utils.eval_seeds is the single source of truth).
    # Prefer --seed-pool; --env-seeds is an ad-hoc final-env-seed list. Resolved seeds are
    # FINAL env seeds handed straight to env.reset — NO x100_000 transform (that historical
    # transform is folded into the canonical_env_60 pool). --seeds = legacy alias.
    seed_pool: str = ""
    env_seeds: str = ""
    allow_train_seeds: bool = False
    seeds: Annotated[str, tyro.conf.arg(help="DEPRECATED alias for --env-seeds (final env seeds)")] = ""
    max_env_steps: int = 600
    """PR7: per-episode ENV-step budget. Default 600 = 2SC env max_episode_steps
    (TwoRobotsStackCube-rf), recorded in the manifest. The success-persistence probe
    shares this budget."""
    replan_after: int = 8
    # PR6: --prompt is REQUIRED (tyro.MISSING). The old TSC DEFAULT_PROMPT silently
    # tagged every task. Validated against the task prompts JSON (or the LB subtask
    # vocab via --subtask-vocab); OOV hard-fails unless --allow-oov-prompt (recorded in JSON).
    prompt: Annotated[str, tyro.conf.arg(help="task instruction (REQUIRED); validated against task/subtask vocab")] = tyro.MISSING
    prompts_json: str = ""  # explicit prompts-JSON path; "" => derive from --task/--camera-mapping
    subtask_vocab: bool = False  # validate against the LB sidecar subtask vocab (sidecar-trained ckpts)
    subtask_npz: str = ""  # override the subtask vocab npz path (defaults to lb_subtask_index.npz)
    allow_oov_prompt: bool = False  # permit an out-of-vocab prompt (recorded in JSON; for E1 prompt-swap probe)
    sim_backend: str = "auto"
    out_dir: str = "/iris/u/mikulrai/logs/eval_pi05_decent"
    video_dir: str = ""  # location override; defaults to <out_dir>/videos. Video is ALWAYS recorded.
    video_frame_stride: int = 2  # subsample recorded frames before writing mp4 (1 = every frame)
    video_max: int = 3   # DEPRECATED/ignored: recording is now always-on for every seed. Kept for launcher compat.
    video_all: bool = False  # DEPRECATED/ignored: every seed always records. Kept so --video-all still parses.
    run_id: str = ""  # disambiguates videos across runs; "" => $SLURM_JOB_ID or unix-ts
    num_arms: int = 2
    # If set, hard-fail (exit 3) before episode 1 unless each arm's server metadata
    # config_name matches (comma-separated, one per arm, aligned with --ports) AND its
    # action_dim == 8 (per-arm decent dim). Empty => no identity check (PR2).
    expect_config: Annotated[str, tyro.conf.arg(help="comma-separated config names, one per arm, aligned with --ports")] = ""
    camera_mapping: str = (
        "/iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/two_robots_stack_cube_wristcam.json"
    )
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = "panda_wristcam_multi,panda_wristcam_multi"
    trajectory_log_path: str = ""  # if set, write JSONL per-step trajectory data
    save_trajectory: bool = False
    """PR9: capture per-episode env_states + actions + proprio (qpos) to an h5 under
    --trajectory-root for future self-training. RGB is NOT recorded (re-renderable from
    env_states). The per-arm clients decode delta->absolute (cur_qpos + delta) BEFORE
    env.step, so the recorded actions are the ABSOLUTE joint targets — directly consumable
    by parse_h5_to_zarr_unified.py --state-source qpos. Default off."""
    trajectory_root: str = ""  # PR9: root for --save-trajectory h5s; "" => default eval_trajs root
    wandb: bool = False
    wandb_project: str = "openpi-robofactory"
    wandb_tags: str = "eval,pi05,decent,2sc,ll_only"
    wandb_name: str = ""  # override default name "eval_decent_{task}_run{run_id}" if set


def _gripper_from_qpos(qpos_step: np.ndarray) -> float:
    return float((qpos_step[7] + qpos_step[8]) / 2.0)


def _build_state(obs: dict, num_arms: int, robot_uid: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i in range(num_arms):
        q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
        parts.append(q[:7].astype(np.float32))
        parts.append(np.array([_gripper_from_qpos(q)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _extract_image(obs: dict, cam_name: str) -> np.ndarray:
    img = obs["sensor_data"][cam_name]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _build_obs_dict(obs: dict, prompt: str, num_arms: int, cam_map: dict[str, str], robot_uid: str) -> dict:
    out: dict = {
        "state": _build_state(obs, num_arms, robot_uid),
        "prompt": prompt,
    }
    # The image-slot keys are driven ENTIRELY by the camera-mapping JSON keys (cam_map),
    # NOT a hardcoded constant. This lets the same eval client drive any per-arm server
    # whose input transform reads its own raw image keys directly from the obs dict:
    #   - wc/centralised: base_0_rgb_raw, left_wrist_0_rgb_raw, right_wrist_0_rgb_raw, extra_0_rgb_raw
    #   - 5-cam union:    global_rgb, wrist0_rgb, wrist1_rgb, head0_rgb, head1_rgb
    # Slots mapped to null are zero-filled to match the masked input the model saw during
    # fit (e.g. extra_0_rgb_raw=null for 2-arm wc configs).
    ref_shape: tuple[int, int, int] | None = None
    for slot, cam_name in cam_map.items():
        if cam_name is not None:
            img = _extract_image(obs, cam_name)
            out[slot] = img
            if ref_shape is None:
                ref_shape = img.shape  # type: ignore[assignment]
    if ref_shape is None:
        ref_shape = (224, 224, 3)
    for slot in cam_map:
        if slot not in out:
            out[slot] = np.zeros(ref_shape, dtype=np.uint8)
    return out


def _current_qpos_per_arm(obs: dict, num_arms: int, robot_uid: str) -> list[np.ndarray]:
    return [
        np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()[:7].astype(np.float32)
        for i in range(num_arms)
    ]


def _hold_qpos_action(obs: dict, last_gripper_per_arm: list[float], num_arms: int, robot_uid: str, action_prefix: str) -> dict:
    """PR7: action commanding every arm to STAY at its current qpos (hold-qpos probe).

    See robofactory.utils.success_persistence. The gripper channel keeps the LAST
    commanded gripper per arm so an open/closed grasp is preserved during the hold."""
    cur = _current_qpos_per_arm(obs, num_arms, robot_uid)
    return {
        f"{action_prefix}-{i}": np.concatenate(
            [cur[i], np.array([last_gripper_per_arm[i]], dtype=np.float32)]
        ).astype(np.float32)
        for i in range(num_arms)
    }


def _write_mp4(path: str, frames: list[np.ndarray]) -> None:
    import cv2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _resolve_run_id(run_id: str) -> str:
    import os
    if run_id:
        return run_id
    return os.environ.get("SLURM_JOB_ID") or str(int(time.time()))


def _video_filename(task: str, run_id: str, seed_base: int, ep_i: int) -> str:
    return f"{task}_run{run_id}_seed{seed_base}_ep{ep_i:03d}.mp4"


def _resolve_camera_mapping(path: str) -> dict[str, str]:
    """Load the camera-mapping JSON. The JSON keys ARE the obs-dict image-slot keys
    (verbatim — no suffix appended) that the policy server's input transform reads.
    Values are the SAPIEN sensor names (or null to zero-fill that slot). Supports any
    number/names of cameras; the caller must ensure the keys match what the target
    server's input transform expects (wc *_raw keys vs union non-_raw keys)."""
    if not path:
        return dict(DEFAULT_CAMERA_MAPPING)
    mapping = json.loads(Path(path).read_text())
    if not mapping:
        raise ValueError(f"camera_mapping {path!r} is empty")
    return dict(mapping)


def _cube_xyz(env, name: str) -> list[float]:
    try:
        actor = getattr(env.unwrapped, name)
        p = actor.pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        p = np.asarray(p)
        if p.ndim == 2:
            p = p[0]
        return [float(p[0]), float(p[1]), float(p[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def _action_dict_to_per_arm_list(action_dict: dict, num_arms: int, action_prefix: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(num_arms):
        a = np.asarray(action_dict[f"{action_prefix}-{i}"]).reshape(-1).astype(float)
        out.append([float(x) for x in a.tolist()])
    return out


def run_episode(env, policies: list, args: Args, cam_map: dict[str, str], view_sensors: list[str], seed: int, action_prefix: str, video_path: str) -> dict:
    """Run one episode with 3 per-arm policy servers.

    `view_sensors` is the ordered, deduped union of every per-arm policy's
    non-masked image inputs (see main()); each recorded frame tiles all of them.
    """
    obs, _ = env.reset(seed=seed)
    # Hard-fail (once/process) if the first head_camera_global frame is black-skied (PR3).
    # NOTE: no try/except — a black sky must abort the run, not be silently swallowed.
    from robofactory.utils.eval_guards import shader_bg_guard
    _global_view = "head_camera_global" if "head_camera_global" in view_sensors else view_sensors[0]
    shader_bg_guard(_extract_image(obs, _global_view))
    success = False
    # PR7: persistence-probe state. last_gripper_per_arm holds the most-recently commanded
    # gripper so the hold action preserves the grasp; env_steps counts ENV steps (the unified
    # 400-step budget unit, shared with the probe).
    sustained_info: dict | None = None
    last_gripper_per_arm: list[float] = [0.0] * args.num_arms
    env_steps = 0
    t0 = time.time()

    chunks: list[np.ndarray | None] = [None] * args.num_arms
    chunk_idxs: list[int] = [args.replan_after] * args.num_arms
    video_frames: list[np.ndarray] = []

    traj_fp = None
    if args.trajectory_log_path:
        Path(args.trajectory_log_path).parent.mkdir(parents=True, exist_ok=True)
        traj_fp = open(args.trajectory_log_path, "a")

    try:
        for step in range(args.max_env_steps):
            video_frames.append(tile_views([_extract_image(obs, s) for s in view_sensors]))
            # Replan arms whose chunk is exhausted
            obs_dict = None
            replanned_per_arm = [False] * args.num_arms
            for i in range(args.num_arms):
                if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
                    if obs_dict is None:
                        obs_dict = _build_obs_dict(obs, args.prompt, args.num_arms, cam_map, args.robot_uid)
                    # PR5: re-key each per-arm server's rng per episode. Set fresh every call
                    # because Policy.infer pops the key in place (each arm is a separate server).
                    obs_dict["_episode_seed"] = int(seed)
                    result = policies[i].infer(obs_dict)
                    chunks[i] = np.asarray(result["actions"])  # (H, 8)
                    chunk_idxs[i] = 0
                    replanned_per_arm[i] = True

            cur_qpos = _current_qpos_per_arm(obs, args.num_arms, args.robot_uid)
            action_dict: dict[str, np.ndarray] = {}
            local_chunk_idxs = list(chunk_idxs)
            for i in range(args.num_arms):
                step_i = chunks[i][chunk_idxs[i]]  # (8,)
                delta = step_i[:7]
                gripper = step_i[7]
                target = np.concatenate([cur_qpos[i] + delta, np.array([gripper], dtype=np.float32)])
                action_dict[f"{action_prefix}-{i}"] = target.astype(np.float32)
                last_gripper_per_arm[i] = float(gripper)  # PR7: preserve grasp during hold probe
                chunk_idxs[i] += 1

            if traj_fp is not None:
                state_24 = _build_state(obs, args.num_arms, args.robot_uid)
                action_chunks_per_arm = [
                    ([[float(v) for v in r] for r in chunks[i].tolist()] if replanned_per_arm[i] else None)
                    for i in range(args.num_arms)
                ]
                row = {
                    "seed": int(seed),
                    "step": int(step),
                    "chunk_idx": int(min(local_chunk_idxs)),  # use min as a proxy; per-arm in chunk_idxs_per_arm
                    "chunk_idxs_per_arm": [int(x) for x in local_chunk_idxs],
                    "replanned": bool(any(replanned_per_arm)),
                    "replanned_per_arm": [bool(x) for x in replanned_per_arm],
                    "state_24": [float(x) for x in state_24.tolist()],
                    "action_chunks_per_arm": action_chunks_per_arm,
                    "applied_action_per_arm": _action_dict_to_per_arm_list(action_dict, args.num_arms, action_prefix),
                    "cube_xyz": {
                        "cubeA": _cube_xyz(env, "cubeA"),
                        "cubeB": _cube_xyz(env, "cubeB"),
                        "cubeC": _cube_xyz(env, "cubeC"),
                    },
                    "success_far": False,
                }

            obs, _, terminated, truncated, info = env.step(action_dict)
            env_steps += 1
            succ_field = info.get("success", False)
            if hasattr(succ_field, "item"):
                succ_field = succ_field.item()
            success = bool(succ_field)

            if traj_fp is not None:
                row["success_far"] = bool(success)
                traj_fp.write(json.dumps(row) + "\n")
                traj_fp.flush()

            if success or terminated or truncated:
                # PR7: on a first success, run K more hold-qpos env steps (within the shared
                # budget) and record success_sustained_10. Headline (success_first) unchanged.
                if success:
                    def _hold():
                        return _hold_qpos_action(obs, last_gripper_per_arm, args.num_arms, args.robot_uid, action_prefix)

                    def _step(act):
                        nonlocal obs, env_steps
                        obs, _r, _t, _tr, _info = env.step(act)
                        env_steps += 1
                        s = _info.get("success", False)
                        if hasattr(s, "item"):
                            s = s.item()
                        return bool(s), bool(_t), bool(_tr)

                    sustained_info = probe_sustained(_hold, _step, k=SUSTAIN_K, budget_left=args.max_env_steps - env_steps)
                break
    finally:
        if traj_fp is not None:
            traj_fp.close()

    if video_path and video_frames:
        _write_mp4(video_path, subsample(video_frames, args.video_frame_stride))

    out = {
        "seed": seed,
        "success": success,
        "success_first": success,  # PR7: explicit headline
        "steps": env_steps,        # PR7: ENV steps (unified budget unit)
        "wall_s": time.time() - t0,
    }
    if sustained_info is not None:
        out["success_sustained_10"] = bool(sustained_info["sustained"])
        out["sustained_info"] = sustained_info
    elif success:
        out["success_sustained_10"] = None  # success at last budget step, no room to probe
    return out


def main(args: Args) -> None:
    # Shared hard-fail eval fidelity guards (PR3): refuse the login node up front.
    from robofactory.utils.eval_guards import assert_not_login_node, assert_shader_pack_default
    assert_not_login_node()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # PR4: resolve FINAL env seeds via the single source of truth. No x100_000.
    from robofactory.utils.eval_seeds import resolve_seeds
    seeds, seed_provenance = resolve_seeds(
        pool=args.seed_pool or None,
        env_seeds=(args.env_seeds or args.seeds) or None,
        allow_train=args.allow_train_seeds,
    )

    # PR6: required, vocab-checked prompt. OOV hard-fails (prints the vocab) unless
    # --allow-oov-prompt is set. The validation dict is recorded in the result JSON.
    from robofactory.utils.eval_validity import load_prompt_vocab, validate_prompt
    _vocab, _vocab_src = load_prompt_vocab(
        prompts_json=args.prompts_json or None,
        task=args.camera_mapping or args.task,
        use_subtask_vocab=args.subtask_vocab,
        subtask_npz=args.subtask_npz or None,
    )
    prompt_validation = validate_prompt(
        args.prompt, _vocab, allow_oov=args.allow_oov_prompt, vocab_source=_vocab_src
    )

    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    assert len(ports) == args.num_arms, f"Need {args.num_arms} ports, got {ports}"

    cam_map = _resolve_camera_mapping(args.camera_mapping)
    # view_sensors = ordered, deduped union of every per-arm policy's non-masked
    # image inputs. The client sends ALL agents the same obs dict (the full union
    # of mapped sensors); each per-arm server masks to global+its-own-wrist
    # internally. So the union of distinct non-null cam_map values is exactly what
    # the robots collectively saw. ordered_unique drops null slots and dedups the
    # shared global; cam_map preserves JSON key order (global first).
    view_sensors = ordered_unique(list(cam_map.values()))
    if not view_sensors:
        raise ValueError(f"camera mapping has no non-null cameras: {cam_map!r}")
    print(f"view_sensors (tiled in video) = {view_sensors}", flush=True)

    # Video is ALWAYS recorded. Default the record dir to <out_dir>/videos so there
    # is no way to run eval without producing tiled multi-view videos.
    video_dir = args.video_dir or str(out_dir / "videos")

    env_kwargs = dict(
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        # CRITICAL: must match data-gen (robofactory.planner.run uses shader_pack="default").
        # ManiSkill default is shader_pack="minimal", which produces a pure-black framebuffer
        # clear -> the upper portion of head_camera_global (above the table geometry) renders
        # black, while training H5 frames have the "default" shader's gray clear. Without
        # this kwarg the eval base_0_rgb diverges from the training distribution by RMSE 0.53
        # in [-1,1] space. See memory/feedback_sapien_shader_pack_eval_mismatch.md.
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    if args.robot_uids_csv:
        env_kwargs["robot_uids"] = tuple(args.robot_uids_csv.split(","))
    assert_shader_pack_default(env_kwargs)
    env = gym.make(args.task, **env_kwargs)
    # ManiSkill uses URDF body name (e.g. "panda") for action_space keys but the
    # registered agent uid (e.g. "panda_wristcam_multi") for obs["agent"] keys.
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]
    print(f"obs_prefix='{args.robot_uid}' action_prefix='{action_prefix}'", flush=True)

    # PR9: optional self-training trajectory capture. The per-arm clients decode
    # delta->absolute BEFORE env.step, so RecordEpisodeMA buffers ABSOLUTE joint targets
    # (no converter-side fix). RGB dropped (re-renderable); qpos rides along via obs.
    traj_h5_path = None
    if args.save_trajectory:
        from robofactory.utils.eval_trajectory import trajectory_output_dir, wrap_record_trajectory
        _ts = int(time.time())
        _traj_label = f"eval_pi05_decent_{args.task}_run{_resolve_run_id(args.run_id)}_{_ts}"
        _traj_dir = trajectory_output_dir(args.trajectory_root or None, _traj_label)
        env, traj_h5_path = wrap_record_trajectory(env, _traj_dir, trajectory_name="trajectory")
        print(f"[eval_decent_pi05] PR9 --save-trajectory -> {traj_h5_path}", flush=True)

    policies = [WebsocketClientPolicy(host=args.host, port=p) for p in ports]
    server_metadata = [dict(p.get_server_metadata() or {}) for p in policies]
    for i, p in enumerate(policies):
        print(f"[arm{i}] server metadata: {server_metadata[i]}")
    # PR2 server-identity handshake (one expected config per arm, aligned with --ports).
    # Per-arm decent model action dim is 8. A mismatch (or arm/config count mismatch)
    # hard-fails before episode 1 instead of producing per-episode IndexErrors.
    expect_configs = [c.strip() for c in args.expect_config.split(",")] if args.expect_config else []
    if expect_configs and len(expect_configs) != args.num_arms:
        print(
            f"SERVER IDENTITY MISMATCH: --expect-config has {len(expect_configs)} entries "
            f"but --num-arms={args.num_arms} (one per arm, aligned with --ports); refusing to run",
            file=_sys.stderr, flush=True,
        )
        _sys.exit(3)
    from robofactory.utils.server_identity import assert_server_identity_or_exit
    for i in range(args.num_arms):
        assert_server_identity_or_exit(
            server_metadata[i],
            expect_configs[i] if expect_configs else None,
            # action_dim from serve_policy is the padded model dim (32), not task-specific;
            # config_name is the real guard (server reports a fixed real_action_dim too) —
            # PR2 dim-arm disabled, see E10 false-reject 2026-06-11.
            expect_action_dim=None,
            label=f"arm{i} (port {ports[i]})",
        )
    print(f"num_arms={args.num_arms} ports={ports} cam_map={cam_map}")

    run_id = _resolve_run_id(args.run_id)

    with WandbRun(
        enabled=args.wandb,
        project=args.wandb_project,
        job_type="eval",
        name=(args.wandb_name or f"eval_decent_{args.task}_run{run_id}"),
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
        config=dataclasses.asdict(args),
    ) as wandb_run, VideoRecorder(
        video_dir, all_seeds=True,
    ) as videos:
        # S1 collapse probe (metric-only; never blocks eval). Probe each per-arm
        # server under namespace collapse/arm{i}/*.
        try:
            import warnings as _warnings
            from robofactory.utils.preflight_collapse import probe_collapse_pi05_loaded_policy
            _calib = '/iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz'
            _ref_shape = (224, 224, 3)
            _state_dim = args.num_arms * 8
            _probe_slots = tuple(cam_map.keys())
            def _build_obs_for_probe(img_chw, qpos):
                state = np.resize(np.asarray(qpos, dtype=np.float32), _state_dim).astype(np.float32)
                hwc_u8 = (np.clip(np.moveaxis(img_chw, 0, -1), 0, 1) * 255.0).astype(np.uint8)
                out = {"state": state, "prompt": args.prompt}
                for slot in _probe_slots:
                    out[slot] = hwc_u8
                return out
            for _i, _p in enumerate(policies):
                _rep = probe_collapse_pi05_loaded_policy(
                    _p, _calib,
                    build_obs_dict=_build_obs_for_probe,
                    image_slots=_probe_slots, proprio_key="state", max_episodes=8,
                )
                wandb_run.log_raw(_rep.to_wandb_payload(prefix=f"collapse/arm{_i}"))
                _r = _rep.image_to_baseline_ratio
                if _r < 1.5:
                    _warnings.warn(f"[collapse arm{_i}] mse_zero_image/baseline = {_r:.2f} < 1.5 - image input may be ignored")
                print(f"[collapse-probe arm{_i}] {_rep.summary()}", flush=True)
        except Exception as _e:
            import traceback as _tb
            print(f"[collapse-probe] skipped: {type(_e).__name__}: {_e!r}", file=_sys.stderr)
            _tb.print_exc(file=_sys.stderr)
        results: list[dict] = []
        episode_global_idx = 0
        for seed_idx, seed in enumerate(seeds):
            for ep_i in range(args.num_episodes):
                # PR4: `seed` is the FINAL env seed; ep_i offsets additively only for
                # >1 episode/seed (identity for the canonical num_episodes==1). No x100_000.
                ep_seed = seed + ep_i
                # cap maps to (seed, ep_i): record first N (seed_idx, ep_i==0) pairs only
                # ALWAYS record every seed/episode (no per-seed cap). videos.record_dir
                # is video_dir, which always resolves to a real path (<out_dir>/videos).
                video_path = str(
                    Path(videos.record_dir)
                    / _video_filename(args.task, run_id, seed, ep_i)
                )
                try:
                    r = run_episode(env, policies, args, cam_map, view_sensors, ep_seed, action_prefix, video_path)
                except Exception as e:  # noqa: BLE001
                    r = {"seed": ep_seed, "success": False, "steps": -1, "error": repr(e)}
                if traj_h5_path is not None:
                    # PR9: episodes flush in order; episode_global_idx -> traj_{idx} in the h5.
                    r["trajectory_path"] = traj_h5_path
                    r["trajectory_group"] = f"traj_{episode_global_idx}"
                results.append(r)
                print(
                    f"[seed_base={seed} ep={ep_i:03d}] success={r['success']} "
                    f"steps={r['steps']} wall_s={r.get('wall_s', -1):.1f}"
                )
                wandb_run.log_episode(
                    episode_global_idx,
                    seed_base=seed,
                    seed_full=ep_seed,
                    success=int(bool(r["success"])),
                    steps=int(r["steps"]),
                    wall_s=float(r.get("wall_s", -1)),
                )
                episode_global_idx += 1

        n = len(results)
        n_succ = sum(1 for r in results if r["success"])
        # PR7: report BOTH success_first (headline) and success_sustained_10 (persistence).
        n_sustained = sum(1 for r in results if r.get("success_sustained_10") is True)
        print(
            f"\n=== summary ===\nepisodes: {n}\n"
            f"success_first: {n_succ}/{n} ({100.0 * n_succ / n:.1f}%)\n"
            f"success_sustained_10: {n_sustained}/{n} ({100.0 * n_sustained / n:.1f}%)"
        )

        from robofactory.utils.eval_guards import shader_mismatch_override_active
        # PR6: validity guard + full provenance (eval_protocol v2).
        from robofactory.utils.eval_validity import (
            build_provenance, classify_validity, finalize_validity,
        )
        validity = classify_validity(results)
        provenance = build_provenance(
            shader_pack="default",
            enable_shadow=False,
            sim_backend=args.sim_backend,
            seed_provenance=seed_provenance,
            prompts=[args.prompt],
            prompt_validation=prompt_validation,
            max_env_steps=args.max_env_steps,
            chunk_config={"replan_after": args.replan_after, "num_arms": args.num_arms},
            server_metadata={f"arm{i}": server_metadata[i] for i in range(len(server_metadata))},
        )
        out_file = out_dir / f"eval_decent_{args.task}_{int(time.time())}.json"
        out_file.write_text(json.dumps({
            "args": dataclasses.asdict(args),
            "seed_provenance": seed_provenance,  # PR4: pool name + sha + allow_train
            "shader_mismatch_override": shader_mismatch_override_active(),
            "server_metadata": {f"arm{i}": server_metadata[i] for i in range(len(server_metadata))},
            "provenance": provenance,            # PR6: eval_protocol v2
            "prompt_validation": prompt_validation,
            "validity": validity,                # PR6: valid / invalid_reason
            "save_trajectory": args.save_trajectory,  # PR9
            "trajectory_h5": traj_h5_path,            # PR9
            "results": results,
        }, indent=2))
        print(f"Saved {out_file}")

        wandb_run.log_summary(
            success_rate=(n_succ / n) if n else 0.0,          # == success_first rate
            success_first_rate=(n_succ / n) if n else 0.0,
            success_sustained_10_rate=(n_sustained / n) if n else 0.0,  # PR7
            n_episodes=n,
            n_success=n_succ,
            n_success_sustained_10=n_sustained,
            results_json_path=str(out_file),
            valid=validity["valid"],
        )

        env.close()
        # PR6: if >5% episodes invalid -> rename *_INVALID.json, tag wandb 'invalid',
        # exit 2. Done after env.close() so the env shuts down cleanly first.
        finalize_validity(validity, out_file, wandb_run=wandb_run)


if __name__ == "__main__":
    main(tyro.cli(Args))
