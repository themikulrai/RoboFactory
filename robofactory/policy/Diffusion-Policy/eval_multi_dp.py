import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import torch  
import os

import hydra
from pathlib import Path
from collections import deque, defaultdict
from robofactory.tasks import *
import robofactory.agents  # register panda_wristcam_multi
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
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

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
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = './testvideo/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    data_num: int = 100
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint"""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 250
    """Maximum number of steps to run the simulation"""

    ckpt_suffix: str = ""
    """Suffix inside decent-DP ckpt dir name, e.g. 'd2_wristcam' -> checkpoints/{task}_agent{id}_d2_wristcam_{data_num}/. Empty = stock 'Agent{id}_{data_num}' path."""

    jsonl_path: Optional[str] = None
    """Path to append per-episode JSON lines; if None a timestamped file under /iris/u/mikulrai/logs/ is created."""

    wandb: bool = False
    """Enable W&B logging to project 'diffusion-robofactory' job_type='eval'."""

    wandb_tags: str = "eval,baseline,decentralised-dp"
    """Comma-separated W&B tags."""

    obs_cam_family: str = "workspace"
    """Which cameras supply 'head_cam' to the policy: 'workspace' (scene-mounted head_camera_agent{i}) or 'wristcam' (robot-mounted hand_camera_{i}). d2_wristcam ckpts require 'wristcam'."""

    include_global: bool = True
    """Whether to include head_cam_global in the model input. Dataset 1 (default_task.yaml) models were NOT trained with head_cam_global — set --no-include-global for those."""

    img_height: int = 224
    """Height to resize camera frames before feeding to the policy. Dataset 1 (default_task.yaml) uses 240; Dataset 2 (wristcam) uses 224."""

    img_width: int = 224
    """Width to resize camera frames before feeding to the policy. Dataset 1 (default_task.yaml) uses 320; Dataset 2 (wristcam) uses 224."""

def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open('./'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy


class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, id: int = 0, ckpt_suffix: str = ""):
        if ckpt_suffix:
            ckpt_dir = f'checkpoints/{task_name}_agent{id}_{ckpt_suffix}_{data_num}'
        else:
            ckpt_dir = f'checkpoints/{task_name}_Agent{id}_{data_num}'
        self.policy = get_policy(f'{ckpt_dir}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)
        self.ckpt_path = f'{ckpt_dir}/{checkpoint_num}.ckpt'

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

import cv2 as _cv2

def _rgb_chw(rgb_tensor, img_h: int = 224, img_w: int = 224):
    t = rgb_tensor.squeeze(0)
    arr = t.cpu().numpy() if hasattr(t, 'numpy') else np.asarray(t)
    # arr is HxWx3 uint8; cv2.resize takes (width, height)
    if arr.shape[0] != img_h or arr.shape[1] != img_w:
        arr = _cv2.resize(arr, (img_w, img_h), interpolation=_cv2.INTER_AREA)
    return np.moveaxis(arr, -1, 0).astype(np.float32) / 255.0


_CAM_TPL = {"workspace": "head_camera_agent{i}", "wristcam": "hand_camera_{i}"}


def get_model_input(observation, agent_pos, agent_id, include_global: bool = True, cam_family: str = "workspace", img_h: int = 224, img_w: int = 224):
    sd = observation['sensor_data']
    per_agent_key = _CAM_TPL[cam_family].format(i=agent_id)
    if per_agent_key not in sd:
        raise KeyError(f"sensor_data missing {per_agent_key}; available={list(sd.keys())}")
    out = dict(
        head_cam = _rgb_chw(sd[per_agent_key]['rgb'], img_h=img_h, img_w=img_w),
        agent_pos = agent_pos,
    )
    if include_global:
        if 'head_camera_global' not in sd:
            raise KeyError(f"sensor_data missing head_camera_global; available={list(sd.keys())}")
        out['head_cam_global'] = _rgb_chw(sd['head_camera_global']['rgb'], img_h=img_h, img_w=img_w)
    return out

def run_episode(env, planner, dp_models, agent_num, seed, args, verbose, agent_prefix='panda', action_prefix='panda', video_path: str = None):
    """Run one episode. Returns dict(success, steps, wallclock_s, infer_ms_per_arm, vram_peak_mb)."""
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
    # Reset DP runners' obs deque and planner gripper state
    for m in dp_models:
        m.runner.reset_obs()
    try:
        from robofactory.planner.motionplanner import OPEN
        planner.gripper_state = [OPEN] * agent_num
    except Exception:
        pass
    # Seed planner state
    for id in range(agent_num):
        initial_qpos = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[id])
        obs_dict = get_model_input(raw_obs, initial_qpos, id, include_global=args.include_global, cam_family=args.obs_cam_family, img_h=args.img_height, img_w=args.img_width)
        dp_models[id].update_obs(obs_dict)

    infer_times_ms = [[] for _ in range(agent_num)]
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
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for id in range(agent_num):
            t0 = _t.perf_counter()
            action_list = dp_models[id].get_action()
            infer_times_ms[id].append((_t.perf_counter() - t0) * 1000.0)
            for i in range(6):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.05, verbose=True)
                except Exception as e:
                    # TOPP fails on near-zero or degenerate paths; fall back to executing
                    # the policy target directly (1 env step) rather than freezing in place.
                    action_dict[f'{action_prefix}-{id}'].append(now_action)
                    action_step_dict[f'{action_prefix}-{id}'].append(1)
                    continue
                n_step = position.shape[0]
                action_step_dict[f'{action_prefix}-{id}'].append(n_step)
                gripper_state = now_action[-1]
                if n_step == 0:
                    action_dict[f'{action_prefix}-{id}'].append(now_action)
                for j in range(n_step):
                    true_action = np.hstack([position[j], gripper_state])
                    action_dict[f'{action_prefix}-{id}'].append(true_action)

        start_idx = [0 for _ in range(agent_num)]
        for i in range(6):
            max_step = 0
            for id in range(agent_num):
                max_step = max(max_step, action_step_dict[f'{action_prefix}-{id}'][i])
            for j in range(max_step):
                true_action = dict()
                for id in range(agent_num):
                    now_step = min(j, action_step_dict[f'{action_prefix}-{id}'][i] - 1)
                    true_action[f'{action_prefix}-{id}'] = action_dict[f'{action_prefix}-{id}'][start_idx[id] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                if video_path is not None:
                    _gframe = observation['sensor_data'].get('head_camera_global', {}).get('rgb')
                    if _gframe is not None:
                        _f = _gframe[0]
                        if hasattr(_f, 'cpu'): _f = _f.cpu().numpy()
                        _video_frames.append(_f.astype(np.uint8))
                if verbose:
                    env.render_human()
            if verbose:
                print(true_action)
                print("max_step", max_step)
            for id in range(agent_num):
                start_idx[id] += action_step_dict[f'{action_prefix}-{id}'][i]
                if action_step_dict[f'{action_prefix}-{id}'][i] == 0:
                    continue
                obs_dict = get_model_input(observation, true_action[f'{action_prefix}-{id}'], id, include_global=args.include_global, cam_family=args.obs_cam_family, img_h=args.img_height, img_w=args.img_width)
                dp_models[id].update_obs(obs_dict)
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
    infer_ms_mean = [float(np.mean(v)) if v else 0.0 for v in infer_times_ms]
    return dict(
        seed=int(seed),
        success=int(success),
        steps=int(cnt - 1),
        wallclock_s=round(wallclock, 3),
        infer_ms_mean_per_arm=[round(x, 2) for x in infer_ms_mean],
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
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
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
        enable_shadow=True,
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
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=True
    )

    # Load decentralised DP policies once (reused across all seeds)
    agent_num = planner.agent_num
    dp_models = []
    for i in range(agent_num):
        dp_models.append(DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix))
    print(f"Loaded {agent_num} decentralised DP policies. VRAM now: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # agent_prefix: used for obs dict keys (e.g. raw_obs['agent']['panda_wristcam_multi-0'])
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    # action_prefix: used for env.step() dict keys — ManiSkill uses URDF body name ('panda'),
    # NOT the registered agent uid, so derive it from the actual action space keys.
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    print(f"agent_prefix='{agent_prefix}' action_prefix='{action_prefix}'", flush=True)

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
        ckpt_suffix=args.ckpt_suffix,
        ckpt_paths=[m.ckpt_path for m in dp_models],
        max_steps=args.max_steps, n_seeds=len(seeds), seeds=seeds,
        sim_backend=args.sim_backend, obs_mode=args.obs_mode,
        git_sha=git_sha, host=socket.gethostname(),
        start_utc=ts, record_root=record_root, jsonl_path=jsonl_path,
    )
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({'kind': 'manifest', **manifest}) + '\n')
    print('MANIFEST:', json.dumps(manifest, indent=2), flush=True)

    # Optional W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project='diffusion-robofactory', job_type='eval',
            name=f'eval_decent_{env_id}_ckpt{args.checkpoint_num}_{ts}',
            group=f'eval_decent_{env_id}_ckpt{args.checkpoint_num}',
            tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
            config=manifest,
        )

    # Seed loop (reuses env + policies)
    results = []
    for idx, seed in enumerate(seeds):
        video_path = os.path.join(record_root, f'seed{seed:07d}_global.mp4')
        metrics = run_episode(env, planner, dp_models, agent_num, seed, args, verbose, agent_prefix=agent_prefix, action_prefix=action_prefix, video_path=video_path)
        metrics['episode_idx'] = idx
        results.append(metrics)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps({'kind': 'episode', **metrics}) + '\n')
        if wandb_run is not None:
            wandb.log({
                'episode/success': metrics['success'],
                'episode/steps': metrics['steps'],
                'episode/wallclock_s': metrics['wallclock_s'],
                'episode/vram_peak_mb': metrics['vram_peak_mb'],
                'episode/seed': metrics['seed'],
                **{f'episode/infer_ms_arm{i}': v for i, v in enumerate(metrics['infer_ms_mean_per_arm'])},
            })
        n_succ = sum(r['success'] for r in results)
        print(f"[seed {seed}] success={metrics['success']} steps={metrics['steps']} wallclock={metrics['wallclock_s']}s vram_mb={metrics['vram_peak_mb']} | running SR {n_succ}/{len(results)} = {100.0*n_succ/len(results):.2f}%", flush=True)

    env.close()

    # Aggregate
    n_total = len(results)
    n_succ = sum(r['success'] for r in results)
    sr = n_succ / n_total if n_total else 0.0
    # Wilson 95% CI half-width approximation via normal
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
        wandb.log({f'summary/{k}': v for k, v in summary.items()})
        wandb_run.finish()
    # Preserve legacy stdout marker for eval_multi.sh compatibility (last line parse)
    print('success' if sr > 0 else 'failed')

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
