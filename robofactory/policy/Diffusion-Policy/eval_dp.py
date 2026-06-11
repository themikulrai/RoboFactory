import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import torch
import os

import hydra
from pathlib import Path
from collections import defaultdict, deque
from robofactory.tasks import *
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver


import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from robofactory.utils.wrappers.record import RecordEpisodeMA

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "${CONFIG_DIR}/robocasa/take_photo.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use."""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Shader pack."""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """Auto-pause sim viewer on load."""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for the simulator. Can be a single int or a list of ints (e.g. -s 10000 10001 10002)."""

    data_num: int = 100
    """Number of demos used to train the ckpt (encoded in checkpoint dir name)."""

    checkpoint_num: int = 300
    """Training epoch of the checkpoint to load."""

    max_steps: int = 250
    """Outer-loop iterations per episode (each iteration dispatches 6 actions × TOPP-expanded env steps)."""

    jsonl_path: Optional[str] = None
    """Path to append per-episode JSON lines; if None a timestamped file under /iris/u/mikulrai/logs/ is created."""

    wandb: bool = False
    """Enable W&B logging to project 'diffusion-robofactory' job_type='eval'."""

    wandb_tags: str = "eval,baseline,single-dp"
    """Comma-separated W&B tags."""

    wandb_project: str = "diffusion-robofactory"
    """W&B project name. Override to keep eval runs in the same project as the matching train run (e.g. PM-DP)."""

    wandb_name: Optional[str] = None
    """W&B run display name. If None, defaults to eval_single_{env_id}_ckpt{N}_{ts}. Override to match train-run names (e.g. 'Eval PM WC DP')."""

    obs_cam_family: str = "workspace"
    """Which scene camera supplies 'head_cam' to the policy: 'workspace' uses sensor_data['head_camera'] (the table-mounted scene cam); 'wristcam' uses sensor_data['hand_camera'] (the robot-mounted wrist cam). Must match training-data camera family."""

    include_global: bool = False
    """If True, additionally feed 'head_cam_global' to the policy. Required for models trained with default_task_wristcam (2-cam). PM scene has no head_camera_global sensor — falls back to head_camera to mirror train-side parse_h5_to_zarr_unified.py:_global_cam_path."""

    img_height: Optional[int] = None
    """Optional resize height before feeding to the policy. None = keep env-native resolution (typical for paper-data PickMeat: 240)."""

    img_width: Optional[int] = None
    """Optional resize width before feeding to the policy. None = keep env-native resolution (typical for paper-data PickMeat: 320)."""

    ckpt_path: Optional[str] = None
    """Override the default checkpoint path. If set, takes precedence over data_num/checkpoint_num lookup."""

    video_max: int = 3
    """Number of seeds to record video for (first N). Default capped at 3 to keep disk under control."""

    video_all: bool = False
    """If true, record video for every seed (overrides --video-max)."""


def get_policy(checkpoint, output_dir, device):
    ckpt_full = checkpoint if os.path.isabs(checkpoint) else './' + checkpoint
    payload = torch.load(open(ckpt_full, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    return policy


class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, ckpt_path: Optional[str] = None):
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
        else:
            self.ckpt_path = f'checkpoints/{task_name}_{data_num}/{checkpoint_num}.ckpt'
        self.policy = get_policy(self.ckpt_path, None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)

    def get_last_obs(self):
        return self.runner.obs[-1]


import cv2 as _cv2

def _rgb_chw(rgb_tensor, img_h: Optional[int] = None, img_w: Optional[int] = None):
    t = rgb_tensor.squeeze(0)
    arr = t.cpu().numpy() if hasattr(t, 'numpy') else np.asarray(t)
    if img_h is not None and img_w is not None and (arr.shape[0] != img_h or arr.shape[1] != img_w):
        arr = _cv2.resize(arr, (img_w, img_h), interpolation=_cv2.INTER_AREA)
    return np.moveaxis(arr, -1, 0).astype(np.float32) / 255.0


_SINGLE_AGENT_CAM = {"workspace": "head_camera", "wristcam": "hand_camera"}


def _global_frame_for_guard(observation):
    """Best-effort HxWx3 uint8-ish global frame for shader_bg_guard. Prefer
    head_camera_global; fall back to head_camera (PM scene has no global cam)."""
    sd = observation.get('sensor_data', {}) if isinstance(observation, dict) else {}
    for key in ('head_camera_global', 'head_camera'):
        cam = sd.get(key)
        if isinstance(cam, dict) and 'rgb' in cam:
            rgb = cam['rgb']
            rgb = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
            while rgb.ndim > 3:
                rgb = rgb[0]
            return rgb
    return None


def get_model_input(observation, agent_pos, img_h: Optional[int] = None, img_w: Optional[int] = None, cam_family: str = "workspace", include_global: bool = False):
    sd = observation['sensor_data']
    per_agent_key = _SINGLE_AGENT_CAM.get(cam_family, "head_camera")
    if per_agent_key not in sd:
        raise KeyError(f"sensor_data missing {per_agent_key} (cam_family={cam_family}); available={list(sd.keys())}")
    out = dict(
        head_cam=_rgb_chw(sd[per_agent_key]['rgb'], img_h=img_h, img_w=img_w),
        agent_pos=agent_pos,
    )
    if include_global:
        # PM scene has no head_camera_global — fall back to head_camera, mirroring
        # the train-side fallback in parse_h5_to_zarr_unified.py:_global_cam_path.
        if 'head_camera_global' in sd:
            global_key = 'head_camera_global'
        elif 'head_camera' in sd:
            global_key = 'head_camera'
        else:
            raise KeyError(f"sensor_data missing head_camera_global (no head_camera fallback); available={list(sd.keys())}")
        out['head_cam_global'] = _rgb_chw(sd[global_key]['rgb'], img_h=img_h, img_w=img_w)
    return out


def run_episode(env, planner, dp_model, seed, args, verbose, video_path: Optional[str] = None):
    """Run one PickMeat-style single-agent episode. Returns metrics dict."""
    import time as _t
    import random as _random
    torch.cuda.reset_peak_memory_stats()
    t_ep = _t.perf_counter()
    raw_obs, _ = env.reset(seed=seed)
    if env.action_space is not None:
        env.action_space.seed(seed)
    # PR5: seed the diffusion sampler RNG per episode so the same env seed on the same
    # node/GPU yields an identical trajectory. torch is NEVER seeded at eval otherwise →
    # diffusion sampling noise free-runs. Cross-GPU bitwise determinism is NOT promised.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % 2**32)
    _random.seed(seed)
    # Hard-fail (once/process) if the first global frame is black-skied (PR3).
    from robofactory.utils.eval_guards import shader_bg_guard
    shader_bg_guard(_global_frame_for_guard(raw_obs))
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    # Reset policy obs deque + planner gripper state
    dp_model.runner.reset_obs()
    try:
        from robofactory.planner.motionplanner import OPEN
        planner.gripper_state = OPEN
    except Exception:
        pass

    # Seed planner state with the initial proprioception
    initial_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].cpu().numpy()
    initial_qpos = np.append(initial_qpos, planner.gripper_state)
    obs_dict = get_model_input(raw_obs, initial_qpos, img_h=args.img_height, img_w=args.img_width, cam_family=args.obs_cam_family, include_global=args.include_global)
    dp_model.update_obs(obs_dict)

    infer_times_ms = []
    cnt = 0
    success = False
    info = {}
    _video_frames = []
    while True:
        if verbose:
            print("Iteration:", cnt)
        cnt += 1
        if cnt > args.max_steps:
            break
        t0 = _t.perf_counter()
        action_list = dp_model.get_action()
        infer_times_ms.append((_t.perf_counter() - t0) * 1000.0)
        for i in range(6):
            now_action = action_list[i]
            raw_obs = env.get_obs()
            if i == 0:
                current_qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].cpu().numpy()
            else:
                current_qpos = action_list[i - 1][:-1]
            path = np.vstack((current_qpos, now_action[:-1]))
            try:
                times, position, right_vel, acc, duration = planner.planner[0].TOPP(path, 0.05, verbose=True)
            except Exception:
                # TOPP fails on degenerate paths; execute target directly for one step
                observation, reward, terminated, truncated, info = env.step(now_action)
                if video_path is not None:
                    _f = observation['sensor_data']['head_camera']['rgb'][0]
                    if hasattr(_f, 'cpu'): _f = _f.cpu().numpy()
                    _video_frames.append(_f.astype(np.uint8))
                obs_dict = get_model_input(observation, now_action, img_h=args.img_height, img_w=args.img_width, cam_family=args.obs_cam_family, include_global=args.include_global)
                dp_model.update_obs(obs_dict)
                continue
            n_step = position.shape[0]
            if n_step == 0:
                continue
            gripper_state = now_action[-1]
            for j in range(n_step):
                true_action = np.hstack([position[j], gripper_state])
                observation, reward, terminated, truncated, info = env.step(true_action)
                if video_path is not None:
                    _f = observation['sensor_data']['head_camera']['rgb'][0]
                    if hasattr(_f, 'cpu'): _f = _f.cpu().numpy()
                    _video_frames.append(_f.astype(np.uint8))
                if verbose:
                    env.render_human()
            obs_dict = get_model_input(observation, true_action, img_h=args.img_height, img_w=args.img_width, cam_family=args.obs_cam_family, include_global=args.include_global)
            dp_model.update_obs(obs_dict)
        if verbose:
            print("info", info)
        if args.render_mode is not None:
            env.render()
        if info.get('success', False) == True:
            success = True
            break

    if video_path is not None and _video_frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        h, w = _video_frames[0].shape[:2]
        vw = _cv2.VideoWriter(video_path, _cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
        for _f in _video_frames:
            vw.write(_cv2.cvtColor(_f, _cv2.COLOR_RGB2BGR))
        vw.release()

    wallclock = _t.perf_counter() - t_ep
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1e6
    infer_ms_mean = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0
    return dict(
        seed=int(seed),
        success=int(success),
        steps=int(cnt - 1),
        wallclock_s=round(wallclock, 3),
        infer_ms_mean=round(infer_ms_mean, 2),
        vram_peak_mb=round(vram_peak_mb, 1),
    )


def main(args: Args):
    import time, json, subprocess, socket
    from datetime import datetime
    # Shared hard-fail eval fidelity guards (PR3): refuse the login node up front.
    from robofactory.utils.eval_guards import assert_not_login_node, assert_shader_pack_default
    assert_not_login_node()
    np.set_printoptions(suppress=True, precision=5)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    seeds = list(args.seed) if args.seed is not None else [10000]
    np.random.seed(seeds[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one.")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'
    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=False,  # training data was collected with shadow=False (env default)
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    assert_shader_pack_default(env_kwargs)
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    record_root = args.record_dir.format(env_id=env_id) + f'/eval_{ts}_data{args.data_num}_ckpt{args.checkpoint_num}'
    env = RecordEpisodeMA(env, record_root, info_on_video=False, save_trajectory=False, save_video=False, max_steps_per_video=30000)

    raw_obs, _ = env.reset(seed=seeds[0])
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
    )

    dp_model = DP(env_id, args.checkpoint_num, args.data_num, ckpt_path=args.ckpt_path)
    print(f"Loaded single-agent DP policy. VRAM now: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Provenance + sinks
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/iris/u/mikulrai/projects/RoboFactory').decode().strip()
    except Exception:
        git_sha = 'unknown'
    jsonl_path = args.jsonl_path or f'/iris/u/mikulrai/logs/eval_{env_id}_ckpt{args.checkpoint_num}_{ts}.jsonl'
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    from robofactory.utils.eval_guards import shader_mismatch_override_active
    manifest = dict(
        task=env_id, scene_config=args.config,
        data_num=args.data_num, checkpoint_num=args.checkpoint_num,
        ckpt_path=dp_model.ckpt_path,
        max_steps=args.max_steps, n_seeds=len(seeds), seeds=seeds,
        sim_backend=args.sim_backend, obs_mode=args.obs_mode,
        img_height=args.img_height, img_width=args.img_width,
        git_sha=git_sha, host=socket.gethostname(),
        shader_mismatch_override=shader_mismatch_override_active(),
        start_utc=ts, record_root=record_root, jsonl_path=jsonl_path,
    )
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({'kind': 'manifest', **manifest}) + '\n')
    print('MANIFEST:', json.dumps(manifest, indent=2), flush=True)

    from policy._shared.eval_context import WandbRun, VideoRecorder

    with WandbRun(
        enabled=args.wandb,
        project=args.wandb_project,
        job_type='eval',
        name=args.wandb_name or f'eval_single_{env_id}_ckpt{args.checkpoint_num}_{ts}',
        group=f'eval_single_{env_id}_ckpt{args.checkpoint_num}',
        tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
        config=manifest,
    ) as wandb_run, VideoRecorder(
        record_root, max_recorded=args.video_max, all_seeds=args.video_all,
    ) as videos:
        # S1 collapse probe (metric-only; never blocks eval).
        try:
            import warnings as _warnings
            from robofactory.utils.preflight_collapse import probe_collapse_with_loaded_policy
            _calib = '/iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz'
            collapse = probe_collapse_with_loaded_policy(dp_model.policy, _calib)
            wandb_run.log_raw(collapse.to_wandb_payload())
            _ratio = collapse.image_to_baseline_ratio
            if _ratio < 1.5:
                _warnings.warn(f"[collapse] mse_zero_image/baseline = {_ratio:.2f} < 1.5 - image input may be ignored")
            print(f"[collapse-probe] {collapse.summary()}", flush=True)
        except Exception as _e:
            import traceback as _tb
            print(f"[collapse-probe] skipped: {type(_e).__name__}: {_e!r}", file=sys.stderr)
            _tb.print_exc(file=sys.stderr)
        results = []
        for idx, seed in enumerate(seeds):
            video_path = videos.video_path_for(idx, seed)
            metrics = run_episode(env, planner, dp_model, seed, args, verbose, video_path=video_path)
            metrics['episode_idx'] = idx
            results.append(metrics)
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps({'kind': 'episode', **metrics}) + '\n')
            wandb_run.log_episode(
                idx,
                success=metrics['success'],
                steps=metrics['steps'],
                wallclock_s=metrics['wallclock_s'],
                vram_peak_mb=metrics['vram_peak_mb'],
                seed=metrics['seed'],
                infer_ms=metrics['infer_ms_mean'],
            )
            n_succ = sum(r['success'] for r in results)
            print(f"[seed {seed}] success={metrics['success']} steps={metrics['steps']} wallclock={metrics['wallclock_s']}s vram_mb={metrics['vram_peak_mb']} | running SR {n_succ}/{len(results)} = {100.0*n_succ/len(results):.2f}%", flush=True)

        env.close()

        n_total = len(results)
        n_succ = sum(r['success'] for r in results)
        sr = n_succ / n_total if n_total else 0.0
        from math import sqrt
        ci = 1.96 * sqrt(max(sr * (1 - sr), 1e-9) / max(n_total, 1))
        steps_succ = [r['steps'] for r in results if r['success']]
        mean_steps_succ = float(np.mean(steps_succ)) if steps_succ else float('nan')
        summary = dict(
            n_total=n_total, n_success=n_succ, success_rate=sr, ci95=ci,
            mean_steps_on_success=mean_steps_succ,
            mean_episode_wallclock_s=float(np.mean([r['wallclock_s'] for r in results])) if results else 0.0,
        )
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps({'kind': 'summary', **summary}) + '\n')
        print('SUMMARY:', json.dumps(summary, indent=2), flush=True)
        wandb_run.log_summary(**summary)
        print('success' if sr > 0 else 'failed')


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
