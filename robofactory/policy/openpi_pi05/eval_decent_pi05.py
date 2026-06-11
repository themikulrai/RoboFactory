"""Decentralised pi0.5 eval: 3 per-arm policy servers, one per port.

Each server runs a per-arm LoRA model (pi05_robofactory_decent_wristcam_lora_finetune_arm{i}).
The server applies RoboFactoryDecentInputs internally, so this client sends the full
24-dim state + all 4 images (same format as centralised eval).  Each server returns
an 8-dim per-arm action chunk (delta joints + gripper).

Usage:
    # Start 3 servers first (e.g., in background via SLURM script), then:
    python eval_decent_pi05.py \\
        --seeds 10010,10011,10012 \\
        --max-env-steps 200 \\
        --robot-uid panda_wristcam_multi \\
        --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi" \\
        --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/three_robots_stack_cube_wristcam.json \\
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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from policy._shared.eval_context import WandbRun, VideoRecorder  # noqa: E402
from policy._shared.multiview_video import ordered_unique, tile_views, subsample  # noqa: E402


DEFAULT_PROMPT = "stack the three cubes using three robot arms"
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

# One-shot guard for the SAPIEN shader_pack regression. Data-gen uses
# shader_pack="default" (gray clear). ManiSkill default is "minimal" (pure
# black clear), which leaves the upper rows of head_camera_global pitch black
# above the table geometry and silently desyncs eval from training. We check
# the upper sixth of the global cam on the very first frame; if it's almost
# entirely black warn loudly. See memory/feedback_sapien_shader_pack_eval_mismatch.md.
_SHADER_BG_CHECK_DONE = False

def _shader_bg_guard(global_img: np.ndarray) -> None:
    global _SHADER_BG_CHECK_DONE
    if _SHADER_BG_CHECK_DONE:
        return
    _SHADER_BG_CHECK_DONE = True
    if global_img is None or global_img.ndim < 3:
        return
    upper = global_img[: max(1, global_img.shape[0] // 6)]
    near_black_frac = float((upper.sum(axis=-1) < 6).mean())
    if near_black_frac > 0.9:
        import warnings as _w
        _w.warn(
            f"[shader_pack guard] head_camera_global upper region is {100*near_black_frac:.0f}% near-black. "
            "This is the SAPIEN shader_pack=minimal symptom. Pass "
            "sensor_configs=dict(shader_pack=\"default\") to gym.make to match data-gen. "
            "See ~/.claude/projects/-iris-u-mikulrai/memory/feedback_sapien_shader_pack_eval_mismatch.md",
            stacklevel=2,
        )


@dataclasses.dataclass
class Args:
    task: str = "ThreeRobotsStackCube-rf"
    config: str = (
        "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml"
    )
    host: str = "127.0.0.1"
    # REQUIRED (tyro.MISSING, no default): a hardcoded default silently routes a
    # driver to colliding ports when co-scheduled. Launchers/run_eval.py always
    # pass job-unique free ports explicitly (PR1).
    ports: Annotated[str, tyro.conf.arg(help="comma-separated ports, one per arm (REQUIRED)")] = tyro.MISSING
    num_episodes: int = 1
    seeds: Annotated[str, tyro.conf.arg(help="comma-separated seed list")] = "10010,10011,10012"
    max_env_steps: int = 200
    replan_after: int = 8
    prompt: str = DEFAULT_PROMPT
    sim_backend: str = "auto"
    out_dir: str = "/iris/u/mikulrai/logs/eval_pi05_decent"
    video_dir: str = ""  # location override; defaults to <out_dir>/videos. Video is ALWAYS recorded.
    video_frame_stride: int = 2  # subsample recorded frames before writing mp4 (1 = every frame)
    video_max: int = 3   # DEPRECATED/ignored: recording is now always-on for every seed. Kept for launcher compat.
    video_all: bool = False  # DEPRECATED/ignored: every seed always records. Kept so --video-all still parses.
    run_id: str = ""  # disambiguates videos across runs; "" => $SLURM_JOB_ID or unix-ts
    num_arms: int = 3
    # If set, hard-fail (exit 3) before episode 1 unless each arm's server metadata
    # config_name matches (comma-separated, one per arm, aligned with --ports) AND its
    # action_dim == 8 (per-arm decent dim). Empty => no identity check (PR2).
    expect_config: Annotated[str, tyro.conf.arg(help="comma-separated config names, one per arm, aligned with --ports")] = ""
    camera_mapping: str = ""
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = ""
    trajectory_log_path: str = ""  # if set, write JSONL per-step trajectory data
    wandb: bool = False
    wandb_project: str = "openpi-robofactory"
    wandb_tags: str = "eval,pi05,decent"
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
    try:
        _shader_bg_guard(_extract_image(obs, view_sensors[0]))
    except Exception:
        pass
    success = False
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
            succ_field = info.get("success", False)
            if hasattr(succ_field, "item"):
                succ_field = succ_field.item()
            success = bool(succ_field)

            if traj_fp is not None:
                row["success_far"] = bool(success)
                traj_fp.write(json.dumps(row) + "\n")
                traj_fp.flush()

            if success or terminated or truncated:
                break
    finally:
        if traj_fp is not None:
            traj_fp.close()

    if video_path and video_frames:
        _write_mp4(video_path, subsample(video_frames, args.video_frame_stride))

    return {"seed": seed, "success": success, "steps": step + 1, "wall_s": time.time() - t0}


def main(args: Args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
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
    env = gym.make(args.task, **env_kwargs)
    # ManiSkill uses URDF body name (e.g. "panda") for action_space keys but the
    # registered agent uid (e.g. "panda_wristcam_multi") for obs["agent"] keys.
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]
    print(f"obs_prefix='{args.robot_uid}' action_prefix='{action_prefix}'", flush=True)

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
            expect_action_dim=8,
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
                ep_seed = seed * 100_000 + ep_i
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
        print(f"\n=== summary ===\nepisodes: {n}\nsuccess: {n_succ}/{n} ({100.0 * n_succ / n:.1f}%)")

        out_file = out_dir / f"eval_decent_{args.task}_{int(time.time())}.json"
        out_file.write_text(json.dumps({
            "args": dataclasses.asdict(args),
            "server_metadata": {f"arm{i}": server_metadata[i] for i in range(len(server_metadata))},
            "results": results,
        }, indent=2))
        print(f"Saved {out_file}")

        wandb_run.log_summary(
            success_rate=(n_succ / n) if n else 0.0,
            n_episodes=n,
            n_success=n_succ,
            results_json_path=str(out_file),
        )

        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
