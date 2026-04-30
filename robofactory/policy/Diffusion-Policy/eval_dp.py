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

    img_height: Optional[int] = None
    """Optional resize height before feeding to the policy. None = keep env-native resolution (typical for paper-data PickMeat: 240)."""

    img_width: Optional[int] = None
    """Optional resize width before feeding to the policy. None = keep env-native resolution (typical for paper-data PickMeat: 320)."""

    ckpt_path: Optional[str] = None
    """Override the default checkpoint path. If set, takes precedence over data_num/checkpoint_num lookup."""


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


def get_model_input(observation, agent_pos, img_h: Optional[int] = None, img_w: Optional[int] = None):
    sd = observation['sensor_data']
    if 'head_camera' not in sd:
        raise KeyError(f"sensor_data missing head_camera; available={list(sd.keys())}")
    return dict(
        head_cam=_rgb_chw(sd['head_camera']['rgb'], img_h=img_h, img_w=img_w),
        agent_pos=agent_pos,
    )


def run_episode(env, planner, dp_model, seed, args, verbose, video_path: Optional[str] = None):
    """Run one PickMeat-style single-agent episode. Returns metrics dict."""
    import time as _t
    torch.cuda.reset_peak_memory_stats()
    t_ep = _t.perf_counter()
    raw_obs, _ = env.reset(seed=seed)
    if env.action_space is not None:
        env.action_space.seed(seed)
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
    obs_dict = get_model_input(raw_obs, initial_qpos, img_h=args.img_height, img_w=args.img_width)
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
                obs_dict = get_model_input(observation, now_action, img_h=args.img_height, img_w=args.img_width)
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
            obs_dict = get_model_input(observation, true_action, img_h=args.img_height, img_w=args.img_width)
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
    manifest = dict(
        task=env_id, scene_config=args.config,
        data_num=args.data_num, checkpoint_num=args.checkpoint_num,
        ckpt_path=dp_model.ckpt_path,
        max_steps=args.max_steps, n_seeds=len(seeds), seeds=seeds,
        sim_backend=args.sim_backend, obs_mode=args.obs_mode,
        img_height=args.img_height, img_width=args.img_width,
        git_sha=git_sha, host=socket.gethostname(),
        start_utc=ts, record_root=record_root, jsonl_path=jsonl_path,
    )
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({'kind': 'manifest', **manifest}) + '\n')
    print('MANIFEST:', json.dumps(manifest, indent=2), flush=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project='diffusion-robofactory', job_type='eval',
            name=f'eval_single_{env_id}_ckpt{args.checkpoint_num}_{ts}',
            group=f'eval_single_{env_id}_ckpt{args.checkpoint_num}',
            tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
            config=manifest,
        )

    results = []
    for idx, seed in enumerate(seeds):
        video_path = os.path.join(record_root, f'seed{seed:07d}.mp4')
        metrics = run_episode(env, planner, dp_model, seed, args, verbose, video_path=video_path)
        metrics['episode_idx'] = idx
        results.append(metrics)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps({'kind': 'episode', **metrics}) + '\n')
        if wandb_run is not None:
            import wandb
            wandb.log({
                'episode/success': metrics['success'],
                'episode/steps': metrics['steps'],
                'episode/wallclock_s': metrics['wallclock_s'],
                'episode/vram_peak_mb': metrics['vram_peak_mb'],
                'episode/seed': metrics['seed'],
                'episode/infer_ms': metrics['infer_ms_mean'],
            })
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
    if wandb_run is not None:
        import wandb
        wandb.log({f'summary/{k}': v for k, v in summary.items()})
        wandb_run.finish()
    print('success' if sr > 0 else 'failed')


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
