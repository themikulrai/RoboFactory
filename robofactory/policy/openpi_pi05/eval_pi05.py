"""Evaluate a fine-tuned pi0.5 RoboFactory policy in the ManiSkill sim.

Architecture:
- Connects to a `scripts/serve_policy.py` instance over WebSocket (openpi_client).
- For each episode, resets the env, then loops:
    1. Build a state of shape (num_arms * 8,) from current qpos of all arms (matches the converter).
    2. Build the 4-image observation dict from --camera_mapping (slot -> sim camera name).
    3. Call the policy server -> action chunk shape (action_horizon, active_dim).
    4. Receding-horizon: execute first `replan_after` steps; for each step convert delta -> absolute
       joint target (target_q = current_q + delta) and pass {panda-i: [...7 joints, gripper]} to env.step.
    5. Re-plan if the chunk runs out before episode end.

Usage (3-arm default):
    python eval_pi05.py --task ThreeRobotsStackCube-rf --num-episodes 50 --seeds 0,1,2

Usage (4-arm):
    python eval_pi05.py \\
        --task LongPipelineDelivery-rf \\
        --num-arms 4 \\
        --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/long_pipeline_delivery.json \\
        --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/long_pipeline_delivery.yaml \\
        --max-env-steps 1800 \\
        --num-episodes 50 --seeds 0,1,2
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import sapien  # noqa: F401  (sapien import side effects)
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robofactory.tasks import *  # noqa: F401, F403  (registers env IDs with gym)
import robofactory.agents  # noqa: F401  (registers panda_wristcam_multi via @register_agent)

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from policy._shared.eval_context import WandbRun, VideoRecorder  # noqa: E402
from policy._shared.multiview_video import tile_views, ordered_unique, subsample  # noqa: E402


DEFAULT_PROMPT = "stack the three cubes using three robot arms"

DEFAULT_CAMERA_MAPPING = {
    "base_0_rgb_raw": "head_camera_global",
    "left_wrist_0_rgb_raw": "head_camera_agent0",
    "right_wrist_0_rgb_raw": "head_camera_agent1",
    "extra_0_rgb_raw": "head_camera_agent2",
}
IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")


@dataclasses.dataclass
class Args:
    task: str = "ThreeRobotsStackCube-rf"
    config: str = (
        "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml"
    )
    host: str = "127.0.0.1"
    # REQUIRED (tyro.MISSING, no default): a hardcoded default silently routes a
    # driver to a colliding port when co-scheduled. Launchers/run_eval.py always
    # pass a job-unique free port explicitly (PR1). tyro.MISSING keeps dataclass
    # field ordering valid while marking --port required on the CLI.
    port: int = tyro.MISSING
    num_episodes: int = 50
    seeds: Annotated[str, tyro.conf.arg(help="comma-separated seed list")] = "0,1,2"
    max_env_steps: int = 1800
    replan_after: int = 8
    prompt: str = DEFAULT_PROMPT
    sim_backend: str = "auto"
    out_dir: str = "/iris/u/mikulrai/logs/eval_pi05"
    video_dir: str = ""  # output dir for per-episode mp4s (tiled multi-view); "" => <out_dir>/videos
    video_frame_stride: int = 2  # subsample recorded frames before writing mp4; 1 = every frame
    run_id: str = ""  # disambiguates videos across runs; "" => $SLURM_JOB_ID or unix-ts
    num_envs: int = 1  # >1 = vectorize; one batch of size num_envs runs in lockstep
    num_arms: int = 3
    active_dim: int = 0  # 0 => num_arms * 8
    expect_config: str = ""  # if set, hard-fail (exit 3) before episode 1 unless the
    # server's metadata config_name matches AND action_dim == num_arms*8 (PR2)
    state_pad_to: int = 0  # 0 => no padding; set to model's active_dim to pad env state w/ zeros (curriculum models)
    camera_mapping: str = ""  # path to JSON; empty => DEFAULT_CAMERA_MAPPING
    robot_uid: str = "panda"  # agent key prefix, e.g. "panda_wristcam_multi" for D2
    robot_uids_csv: str = ""  # comma-separated UIDs for gym.make; empty = use default
    trajectory_log_path: str = ""  # if set (and num_envs==1), write JSONL per-step trajectory data
    wandb: bool = False
    wandb_project: str = "openpi-robofactory"
    wandb_tags: str = "eval,pi05,cent"
    video_max: int = 3
    video_all: bool = False


def _gripper_from_qpos(qpos_step: np.ndarray) -> float:
    return float((qpos_step[7] + qpos_step[8]) / 2.0)


def _get_qpos(obs: dict, robot_uid: str, arm_i: int) -> np.ndarray:
    """Return qpos ndarray for arm_i. Supports three obs schemas:
    - Flat single-arm (PickMeat-rf):       obs['agent']['qpos']
    - Multi-arm indexed (TSC, LP, ...):    obs['agent'][f'{uid}-{i}']['qpos']
    - Single-arm named (alt. registration): obs['agent'][uid]['qpos']
    """
    agent = obs["agent"]
    if "qpos" in agent:
        return np.asarray(agent["qpos"])
    multi_key = f"{robot_uid}-{arm_i}"
    if multi_key in agent:
        return np.asarray(agent[multi_key]["qpos"])
    if robot_uid in agent:
        return np.asarray(agent[robot_uid]["qpos"])
    raise KeyError(f"qpos lookup failed: arm={arm_i} uid={robot_uid} agent_keys={list(agent.keys())}")


def _build_state(obs: dict, num_arms: int, robot_uid: str = "panda", pad_to: int = 0) -> np.ndarray:
    state_parts: list[np.ndarray] = []
    for i in range(num_arms):
        q = _get_qpos(obs, robot_uid, i).squeeze()
        state_parts.append(q[:7].astype(np.float32))
        state_parts.append(np.array([_gripper_from_qpos(q)], dtype=np.float32))
    out = np.concatenate(state_parts).astype(np.float32)
    if pad_to and out.shape[0] < pad_to:
        out = np.concatenate([out, np.zeros(pad_to - out.shape[0], dtype=np.float32)])
    return out


def _extract_image(obs: dict, cam_name: str) -> np.ndarray:
    img = obs["sensor_data"][cam_name]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _build_obs_dict(obs: dict, prompt: str, num_arms: int, cam_map: dict, robot_uid: str = "panda", state_pad_to: int = 0) -> dict:
    out: dict = {
        "state": _build_state(obs, num_arms, robot_uid, pad_to=state_pad_to),
        "prompt": prompt,
    }
    # Mirrors training-side RoboFactoryInputs: slots whose mapping is None are
    # zero-filled with the same shape as a real image. Required for sparse
    # configs like pick_meat.json (1-cam).
    ref_shape: tuple[int, int, int] | None = None
    for slot in IMAGE_SLOTS:
        cam_name = cam_map.get(slot)
        if cam_name is not None:
            img = _extract_image(obs, cam_name)
            out[slot] = img
            if ref_shape is None:
                ref_shape = img.shape  # type: ignore[assignment]
    if ref_shape is None:
        ref_shape = (224, 224, 3)
    for slot in IMAGE_SLOTS:
        if slot not in out:
            out[slot] = np.zeros(ref_shape, dtype=np.uint8)
    return out


def _delta_to_absolute_action(
    chunk_step: np.ndarray, current_qpos_per_arm: list[np.ndarray], num_arms: int, action_prefix: str = "panda"
) -> dict:
    """chunk_step: (num_arms*8,) = per-arm [delta_joints(7), gripper(1)].

    Always returns a dict keyed by f"{action_prefix}-{i}". Caller flattens to
    a single ndarray via _flatten_action_dict_for_box when env.action_space is Box.
    """
    out: dict[str, np.ndarray] = {}
    for i in range(num_arms):
        s = i * 8
        delta = chunk_step[s : s + 7]
        gripper = chunk_step[s + 7]
        target = np.concatenate([current_qpos_per_arm[i] + delta, np.array([gripper], dtype=np.float32)])
        out[f"{action_prefix}-{i}"] = target.astype(np.float32)
    return out


def _flatten_action_dict_for_box(action_dict: dict, num_arms: int, action_prefix: str) -> np.ndarray:
    """Box single-arm path: return the per-arm[0] entry as a flat (8,) array.

    PickMeat is the canonical single-arm task; env.action_space is Box(8,).
    """
    if num_arms != 1:
        raise RuntimeError(f"Box action space only valid for num_arms=1; got {num_arms}")
    return np.asarray(action_dict[f"{action_prefix}-0"], dtype=np.float32).reshape(-1)


def _current_qpos_per_arm(obs: dict, num_arms: int, robot_uid: str = "panda") -> list[np.ndarray]:
    return [_get_qpos(obs, robot_uid, i).squeeze()[:7].astype(np.float32) for i in range(num_arms)]


def _qpos_at(obs: dict, robot_uid: str, arm_i: int, env_idx: int) -> np.ndarray:
    q = _get_qpos(obs, robot_uid, arm_i)
    if q.ndim == 2:
        q = q[env_idx]
    elif q.ndim == 1:
        pass
    else:
        q = q.squeeze()
    return q


def _build_state_at(obs: dict, num_arms: int, robot_uid: str, env_idx: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i in range(num_arms):
        q = _qpos_at(obs, robot_uid, i, env_idx)
        parts.append(q[:7].astype(np.float32))
        parts.append(np.array([_gripper_from_qpos(q)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _extract_image_at(obs: dict, cam_name: str, env_idx: int) -> np.ndarray:
    img = obs["sensor_data"][cam_name]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[env_idx]
    return img.astype(np.uint8)


def _build_obs_dict_at(obs: dict, prompt: str, num_arms: int, cam_map: dict, robot_uid: str, env_idx: int) -> dict:
    out: dict = {"state": _build_state_at(obs, num_arms, robot_uid, env_idx), "prompt": prompt}
    ref_shape: tuple[int, int, int] | None = None
    for slot in IMAGE_SLOTS:
        cam_name = cam_map.get(slot)
        if cam_name is not None:
            img = _extract_image_at(obs, cam_name, env_idx)
            out[slot] = img
            if ref_shape is None:
                ref_shape = img.shape  # type: ignore[assignment]
    if ref_shape is None:
        ref_shape = (224, 224, 3)
    for slot in IMAGE_SLOTS:
        if slot not in out:
            out[slot] = np.zeros(ref_shape, dtype=np.uint8)
    return out


def _current_qpos_per_arm_at(obs: dict, num_arms: int, robot_uid: str, env_idx: int) -> list[np.ndarray]:
    return [_qpos_at(obs, robot_uid, i, env_idx)[:7].astype(np.float32) for i in range(num_arms)]


def _to_numpy_1d(x, n: int) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    arr = np.asarray(x).reshape(-1)
    if arr.size == 1 and n > 1:
        arr = np.full(n, bool(arr.item()))
    return arr.astype(bool)


def _resolve_camera_mapping(path: str) -> dict[str, str]:
    if not path:
        return dict(DEFAULT_CAMERA_MAPPING)
    mapping = json.loads(Path(path).read_text())
    missing = set(IMAGE_SLOTS) - set(mapping.keys())
    if missing:
        raise ValueError(f"camera_mapping missing slots: {missing}")
    return {slot: mapping[slot] for slot in IMAGE_SLOTS}


def _cube_xyz(env, name: str) -> list[float]:
    """Read cube xyz from env.unwrapped.<name> (e.g. 'cubeA'). Returns [x,y,z] floats."""
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


def run_episode(env, policy, args: Args, cam_map: dict[str, str], active_dim: int, seed: int, action_prefix: str, video_path: str = "", is_dict_action: bool = True) -> dict:
    obs, _ = env.reset(seed=seed)
    success = False
    t0 = time.time()
    chunk_idx: int = 10**9
    chunk: np.ndarray | None = None
    video_frames: list[np.ndarray] = []
    view_sensors = ordered_unique(list(cam_map.values()))

    # Hard-fail (once/process) if the first head_camera_global frame is black-skied (PR3).
    from robofactory.utils.eval_guards import shader_bg_guard
    _global_view = "head_camera_global" if "head_camera_global" in view_sensors else view_sensors[0]
    shader_bg_guard(_extract_image(obs, _global_view))

    # ---- trajectory logging (only when num_envs == 1) ----
    traj_fp = None
    if args.trajectory_log_path and args.num_envs == 1:
        Path(args.trajectory_log_path).parent.mkdir(parents=True, exist_ok=True)
        traj_fp = open(args.trajectory_log_path, "a")

    try:
        for step in range(args.max_env_steps):
            video_frames.append(tile_views([_extract_image(obs, s) for s in view_sensors]))
            replanned = False
            if chunk is None or chunk_idx >= args.replan_after:
                obs_dict = _build_obs_dict(obs, args.prompt, args.num_arms, cam_map, args.robot_uid, args.state_pad_to)
                result = policy.infer(obs_dict)
                chunk = np.asarray(result["actions"])[:, :active_dim]  # (H, active_dim)
                chunk_idx = 0
                replanned = True

            cur_qpos = _current_qpos_per_arm(obs, args.num_arms, args.robot_uid)
            action_dict = _delta_to_absolute_action(chunk[chunk_idx], cur_qpos, args.num_arms, action_prefix)
            local_chunk_idx = chunk_idx
            chunk_idx += 1

            if traj_fp is not None:
                state_24 = _build_state(obs, args.num_arms, args.robot_uid)
                row = {
                    "seed": int(seed),
                    "step": int(step),
                    "chunk_idx": int(local_chunk_idx),
                    "replanned": bool(replanned),
                    "state_24": [float(x) for x in state_24.tolist()],
                    "action_chunk": ([[float(v) for v in r] for r in chunk.tolist()] if replanned else None),
                    "applied_action_per_arm": _action_dict_to_per_arm_list(action_dict, args.num_arms, action_prefix),
                    "cube_xyz": {
                        "cubeA": _cube_xyz(env, "cubeA"),
                        "cubeB": _cube_xyz(env, "cubeB"),
                        "cubeC": _cube_xyz(env, "cubeC"),
                    },
                    "success_far": False,  # filled after env.step below
                }

            step_action = action_dict if is_dict_action else _flatten_action_dict_for_box(action_dict, args.num_arms, action_prefix)
            obs, reward, terminated, truncated, info = env.step(step_action)
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


def run_batch_episodes(env, policy, args: Args, cam_map: dict[str, str], active_dim: int, seeds: list[int], action_prefix: str, video_dir: str) -> list[dict]:
    """Run len(seeds) episodes in lockstep using a vectorised env (num_envs == len(seeds)).

    Server inference is per-env and sequential (no server-side batching). Sim is the
    parallel part. Once an env is done, its action stays the last command (env keeps
    stepping but we don't update success/steps).
    """
    n = len(seeds)
    obs, _ = env.reset(seed=seeds)
    t0 = time.time()
    chunks: list[np.ndarray | None] = [None] * n
    chunk_idxs: list[int] = [args.replan_after] * n
    done = [False] * n
    successes = [False] * n
    steps = [0] * n
    view_sensors = ordered_unique(list(cam_map.values()))

    # Hard-fail (once/process) if the first head_camera_global frame is black-skied (PR3).
    from robofactory.utils.eval_guards import shader_bg_guard
    _global_view = "head_camera_global" if "head_camera_global" in view_sensors else view_sensors[0]
    shader_bg_guard(_extract_image_at(obs, _global_view, 0))
    video_frames: list[list[np.ndarray]] = [[] for _ in range(n)]

    for step in range(args.max_env_steps):
        for i in range(n):
            if not done[i]:
                video_frames[i].append(tile_views([_extract_image_at(obs, s, i) for s in view_sensors]))

        for i in range(n):
            if done[i]:
                continue
            if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
                obs_dict = _build_obs_dict_at(obs, args.prompt, args.num_arms, cam_map, args.robot_uid, i)
                result = policy.infer(obs_dict)
                chunks[i] = np.asarray(result["actions"])[:, :active_dim]
                chunk_idxs[i] = 0

        # Build batched action dict (n, 8) per arm key
        batched: dict[str, np.ndarray] = {f"{action_prefix}-{a}": np.zeros((n, 8), dtype=np.float32) for a in range(args.num_arms)}
        for i in range(n):
            if done[i]:
                continue
            cur_qpos = _current_qpos_per_arm_at(obs, args.num_arms, args.robot_uid, i)
            step_act = chunks[i][chunk_idxs[i]]
            for a in range(args.num_arms):
                s = a * 8
                delta = step_act[s : s + 7]
                gripper = step_act[s + 7]
                batched[f"{action_prefix}-{a}"][i, :7] = cur_qpos[a] + delta
                batched[f"{action_prefix}-{a}"][i, 7] = gripper
            chunk_idxs[i] += 1

        obs, _, terminated, truncated, info = env.step(batched)
        succ = _to_numpy_1d(info.get("success", np.zeros(n, dtype=bool)), n)
        term = _to_numpy_1d(terminated, n)
        trunc = _to_numpy_1d(truncated, n)
        for i in range(n):
            if done[i]:
                continue
            steps[i] = step + 1
            if succ[i] or term[i] or trunc[i]:
                successes[i] = bool(succ[i])
                done[i] = True
        if all(done):
            break

    run_id = _resolve_run_id(args.run_id)
    results = []
    for i, sd in enumerate(seeds):
        if video_frames[i]:
            seed_base, ep_i = divmod(int(sd), 100_000)
            vp = str(Path(video_dir) / _video_filename(args.task, run_id, seed_base, ep_i))
            _write_mp4(vp, subsample(video_frames[i], args.video_frame_stride))
        results.append({"seed": int(sd), "success": successes[i], "steps": steps[i] or args.max_env_steps, "wall_s": time.time() - t0})
    return results


def main(args: Args) -> None:
    # Shared hard-fail eval fidelity guards (PR3): refuse the login node up front.
    from robofactory.utils.eval_guards import assert_not_login_node, assert_shader_pack_default
    assert_not_login_node()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always-on recording: --video-dir only chooses location; default to <out_dir>/videos.
    if not args.video_dir:
        args.video_dir = str(out_dir / "videos")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cam_map = _resolve_camera_mapping(args.camera_mapping)
    active_dim = args.active_dim or args.num_arms * 8

    env_kwargs = dict(
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        # Match data-gen shader_pack ("default"). ManiSkill default is "minimal" which clears
        # the framebuffer to pure black; data-gen used "default" which has a gray clear, so
        # without this kwarg head_camera_global above the table geometry diverges from train
        # by RMSE 0.53 in [-1,1] space. See memory/feedback_sapien_shader_pack_eval_mismatch.md.
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
    # Single-arm tasks (PickMeat) get a flat Box action space; multi-arm tasks
    # get a Dict action space keyed by f"<prefix>-<i>".
    is_dict_action = hasattr(env.action_space, "spaces")
    if is_dict_action:
        action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]
    else:
        if args.num_arms != 1:
            raise RuntimeError(
                f"env.action_space is Box but --num-arms={args.num_arms}; expected Dict for multi-arm"
            )
        action_prefix = "panda"  # placeholder; flat path uses key 'panda-0' internally
    print(f"obs_prefix='{args.robot_uid}' action_prefix='{action_prefix}' is_dict_action={is_dict_action}", flush=True)

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = dict(policy.get_server_metadata() or {})
    print(f"Server metadata: {server_metadata}", flush=True)
    # PR2 server-identity handshake: refuse to run (exit 3) if the served config or
    # action dim is not what this client was told to expect. Centralised model action
    # dim is num_arms*8. Missing metadata (legacy server) is un-verifiable -> proceeds.
    from robofactory.utils.server_identity import assert_server_identity_or_exit
    assert_server_identity_or_exit(
        server_metadata,
        args.expect_config or None,
        expect_action_dim=(args.num_arms * 8),
        label=f"port {args.port}",
    )
    print(f"num_arms={args.num_arms} active_dim={active_dim} cam_map={cam_map}")

    run_id = _resolve_run_id(args.run_id)

    with WandbRun(
        enabled=args.wandb,
        project=args.wandb_project,
        job_type="eval",
        name=f"eval_cent_{args.task}_run{run_id}",
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
        config=dataclasses.asdict(args),
    ) as wandb_run, VideoRecorder(
        args.video_dir, max_recorded=10**9, all_seeds=True,
    ) as videos:
        # S1 collapse probe (metric-only; never blocks eval). Opaque server
        # path: feature_rank/var unavailable, only MSE ratios are logged.
        try:
            import warnings as _warnings
            from robofactory.utils.preflight_collapse import probe_collapse_pi05_loaded_policy
            _calib = '/iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz'
            _ref_shape = (224, 224, 3)
            def _build_obs_for_probe(img_chw, qpos):
                # Tile/truncate qpos to active_dim so the server's state head sees
                # the expected length regardless of calibration P.
                state = np.resize(np.asarray(qpos, dtype=np.float32), active_dim).astype(np.float32)
                hwc_u8 = (np.clip(np.moveaxis(img_chw, 0, -1), 0, 1) * 255.0).astype(np.uint8)
                out = {"state": state, "prompt": args.prompt}
                for slot in IMAGE_SLOTS:
                    out[slot] = hwc_u8 if cam_map.get(slot) is not None else np.zeros(_ref_shape, dtype=np.uint8)
                return out
            _rep = probe_collapse_pi05_loaded_policy(
                policy, _calib,
                build_obs_dict=_build_obs_for_probe,
                image_slots=IMAGE_SLOTS, proprio_key="state", max_episodes=8,
            )
            wandb_run.log_raw(_rep.to_wandb_payload())
            _r = _rep.image_to_baseline_ratio
            if _r < 1.5:
                _warnings.warn(f"[collapse] mse_zero_image/baseline = {_r:.2f} < 1.5 - image input may be ignored")
            print(f"[collapse-probe] {_rep.summary()}", flush=True)
        except Exception as _e:
            import traceback as _tb
            print(f"[collapse-probe] skipped: {type(_e).__name__}: {_e!r}", file=_sys.stderr)
            _tb.print_exc(file=_sys.stderr)
        results: list[dict] = []
        episode_global_idx = 0
        if args.num_envs > 1:
            # Vectorised path: one batch of size num_envs, all seeds run in lockstep.
            all_ep_seeds = [seed * 100_000 + ep_i for seed in seeds for ep_i in range(args.num_episodes)]
            for batch_start in range(0, len(all_ep_seeds), args.num_envs):
                batch = all_ep_seeds[batch_start : batch_start + args.num_envs]
                if len(batch) < args.num_envs:
                    # Pad with last seed to fill batch (vectorised env requires fixed N)
                    batch = batch + [batch[-1]] * (args.num_envs - len(batch))
                    pad_count = batch_start + args.num_envs - len(all_ep_seeds)
                else:
                    pad_count = 0
                try:
                    rs = run_batch_episodes(env, policy, args, cam_map, active_dim, batch, action_prefix, args.video_dir)
                except Exception as e:  # noqa: BLE001
                    rs = [{"seed": int(s), "success": False, "steps": -1, "error": repr(e)} for s in batch]
                if pad_count:
                    rs = rs[: args.num_envs - pad_count]
                for r in rs:
                    results.append(r)
                    print(f"[seed={r['seed']}] success={r['success']} steps={r['steps']} wall_s={r.get('wall_s', -1):.1f}")
                    wandb_run.log_episode(
                        episode_global_idx,
                        seed=int(r["seed"]),
                        success=int(bool(r["success"])),
                        steps=int(r["steps"]),
                        wall_s=float(r.get("wall_s", -1)),
                    )
                    episode_global_idx += 1
        else:
            for seed_idx, seed in enumerate(seeds):
                for ep_i in range(args.num_episodes):
                    ep_seed = seed * 100_000 + ep_i
                    # Always record every episode (args.video_dir defaulted in main()).
                    video_path = str(Path(args.video_dir) / _video_filename(args.task, run_id, seed, ep_i))
                    try:
                        r = run_episode(env, policy, args, cam_map, active_dim, ep_seed, action_prefix, video_path, is_dict_action=is_dict_action)
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

        from robofactory.utils.eval_guards import shader_mismatch_override_active
        out_file = out_dir / f"eval_{args.task}_{int(time.time())}.json"
        out_file.write_text(json.dumps({
            "args": dataclasses.asdict(args),
            "shader_mismatch_override": shader_mismatch_override_active(),
            "server_metadata": {f"port_{args.port}": server_metadata},
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
