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
from robofactory.utils.success_persistence import probe_sustained, SUSTAIN_K  # PR7
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
    """Seed(s) for random actions and simulator. Final env seed(s) passed straight to env.reset.
    DEPRECATED alias for --env-seeds; prefer --seed-pool (PR4)."""

    seed_pool: str = ""
    """PR4: name of a frozen seed pool (robofactory.utils.eval_seeds), e.g. canonical_env_60.
    When set, overrides --seed/--env-seeds. Both DP and pi0.5 resolve the SAME final env seeds
    from a pool name -> true seed pairing (an env seed is just an int to env.reset)."""

    env_seeds: str = ""
    """PR4: ad-hoc comma/space list of FINAL env seeds (recorded as pool 'adhoc'). Overrides --seed."""

    allow_train_seeds: bool = False
    """PR4: permit datagen seeds 0..182 (recorded in the result manifest)."""

    data_num: int = 100
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 300
    """The number of training epoch of the checkpoint"""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 250
    """DEPRECATED (PR7): this used to count CHUNK iterations (each ~variable env steps via
    TOPP) and could overrun the 500-step TimeLimit. Kept as a launcher-compat alias: if set
    away from its default while --max-env-steps is left at default, it is mapped onto the
    ENV-step budget. Prefer --max-env-steps."""

    max_env_steps: int = 400
    """PR7: per-episode ENV-step budget (unified across all drivers: cap = 400 env steps,
    recorded in the manifest). The success-persistence probe shares this budget. The episode
    loop now counts ENV steps (not chunk iterations) and checks info["success"] every env step."""

    ckpt_suffix: str = ""
    """Suffix inside decent-DP ckpt dir name, e.g. 'd2_wristcam' -> checkpoints/{task}_agent{id}_d2_wristcam_{data_num}/. Empty = stock 'Agent{id}_{data_num}' path."""

    jsonl_path: Optional[str] = None
    """Path to append per-episode JSON lines; if None a timestamped file under /iris/u/mikulrai/logs/ is created."""

    wandb: bool = False
    """Enable W&B logging to project 'diffusion-robofactory' job_type='eval'."""

    wandb_tags: str = "eval,baseline,decentralised-dp"
    """Comma-separated W&B tags."""

    wandb_project: str = "diffusion-robofactory"
    """W&B project name. Override to keep eval runs in the same project as the matching train runs (e.g. PM-DP, 2SC-DP)."""

    wandb_name: Optional[str] = None
    """W&B run display name. If None, defaults to eval_decent_{env_id}_ckpt{N}_{ts}. Override to match train-run names (e.g. 'Eval 2SC WS DP Decent A0')."""

    obs_cam_family: str = "workspace"
    """Which cameras supply 'head_cam' to the policy: 'workspace' (scene-mounted head_camera_agent{i}) or 'wristcam' (robot-mounted hand_camera_{i}). d2_wristcam ckpts require 'wristcam'."""

    include_global: bool = True
    """Whether to include head_cam_global in the model input. Dataset 1 (default_task.yaml) models were NOT trained with head_cam_global — set --no-include-global for those."""

    img_height: int = 224
    """Height to resize camera frames before feeding to the policy. Dataset 1 (default_task.yaml) uses 240; Dataset 2 (wristcam) uses 224."""

    img_width: int = 224
    """Width to resize camera frames before feeding to the policy. Dataset 1 (default_task.yaml) uses 320; Dataset 2 (wristcam) uses 224."""

    video_dir: Optional[str] = None
    """Directory to save tiled multi-view eval videos. Recording is ALWAYS on (every seed).
    If None, defaults to <record_root>/videos. Use this only to CHOOSE the location."""

    video_frame_stride: int = 2
    """Subsample factor applied to recorded frames before writing the mp4 (frames[::stride],
    last frame always kept). Default 2 halves file size; set to 1 to keep every frame."""

    video_max: int = 3
    """DEPRECATED/ignored: recording is now always-on for every seed. Kept for launcher compat."""

    video_all: bool = False
    """DEPRECATED/ignored: every seed always records. Kept so --video-all still parses."""

    gripper_snap: bool = False
    """Diagnostic: snap predicted gripper command (dim 7) to sign() so DP's MSE-averaged soft values become hard {-1, +1}. Tests whether mode-averaging on the near-binary gripper signal is the eval-failure root cause."""

    gripper_source: str = "action"
    """Where the proprio (agent_pos) channel comes from at eval. 'action' (legacy): dim 7 is the DP's commanded gripper in {-1, +1} (matches state=action zarrs). 'qpos': dim 7 is the env's observed finger-left width in meters (matches state=qpos zarrs from parse_h5_to_zarr_unified.py --state-source qpos). MUST match the zarr the ckpt was trained on."""

    max_chunk_actions: int = 6
    """How many of the DP-predicted actions to execute per chunk before re-observing (default 6 matches paper n_action_steps). Set to 1 to force re-observe after each predicted action — tests H4 (TOPP-amplified train/eval temporal mismatch)."""

    skip_collapse_probe: bool = False
    """Skip the S1 encoder-collapse probe between env.make and the first per-seed env.reset. Use when the probe interferes with downstream env state (observed for WC-trained ckpts on orion: probe predict_action with shape-mismatched obs corrupts PhysX init)."""

    save_trajectory: bool = False
    """PR9: capture per-episode env_states + actions + proprio (qpos) to an h5 under
    --trajectory-root for future self-training. RGB is NOT recorded (re-renderable from
    env_states with shader_pack=default). The recorded actions are the ABSOLUTE joint
    targets stepped into the env, directly consumable by
    parse_h5_to_zarr_unified.py --state-source qpos. Default off."""

    trajectory_root: Optional[str] = None
    """PR9: root dir for --save-trajectory h5s. Defaults to /iris/u/mikulrai/data/eval_trajs/
    (symlinked, never the project tree) or $RF_EVAL_TRAJ_ROOT."""

def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    ckpt_full = checkpoint if os.path.isabs(checkpoint) else './' + checkpoint
    payload = torch.load(open(ckpt_full, 'rb'), pickle_module=dill)
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


from policy._shared.multiview_video import tile_views, ordered_unique, subsample


_CAM_TPL = {"workspace": "head_camera_agent{i}", "wristcam": "hand_camera_{i}"}


_SINGLE_AGENT_CAM_FALLBACK = {"workspace": "head_camera", "wristcam": "hand_camera"}


def derive_view_sensors(observation, agent_num, include_global: bool, cam_family: str):
    """Ordered, deduped list of sensor names the policy actually receives as image input.

    Mirrors get_model_input()'s key resolution exactly: per-agent head_cam for each
    agent (with single-agent suffix fallback), then head_camera_global if include_global
    (with head_camera fallback). This is the UNION of cameras the robots collectively saw.
    """
    sd = observation['sensor_data']
    names = []
    for agent_id in range(agent_num):
        per_agent_key = _CAM_TPL[cam_family].format(i=agent_id)
        if per_agent_key not in sd:
            sa_fallback = _SINGLE_AGENT_CAM_FALLBACK.get(cam_family)
            if sa_fallback and sa_fallback in sd:
                per_agent_key = sa_fallback
        names.append(per_agent_key)
    if include_global:
        if 'head_camera_global' in sd:
            names.append('head_camera_global')
        elif 'head_camera' in sd:
            names.append('head_camera')
    return ordered_unique(names)


def _extract_view_uint8(observation, sensor_name):
    """Pull sensor_name's RGB from obs as an HWC uint8 numpy frame (env idx 0)."""
    frame = observation['sensor_data'][sensor_name]['rgb'][0]
    if hasattr(frame, 'cpu'):
        frame = frame.cpu().numpy()
    return np.asarray(frame).astype(np.uint8)


def _global_frame_for_guard(observation):
    """Best-effort HWC global frame for shader_bg_guard. Prefer head_camera_global;
    fall back to head_camera (single-arm scenes have no global cam)."""
    sd = observation.get('sensor_data', {}) if isinstance(observation, dict) else {}
    for key in ('head_camera_global', 'head_camera'):
        if key in sd:
            try:
                return _extract_view_uint8(observation, key)
            except Exception:
                return None
    return None


def get_model_input(observation, agent_pos, agent_id, include_global: bool = True, cam_family: str = "workspace", img_h: int = 224, img_w: int = 224):
    sd = observation['sensor_data']
    per_agent_key = _CAM_TPL[cam_family].format(i=agent_id)
    # Single-agent tasks (e.g. PickMeat) drop the agent-id suffix. Workspace family
    # falls back to 'head_camera', wristcam family to 'hand_camera' — mirrors the
    # train-side zarr convention in parse_h5_to_zarr_unified.py:_camera_key.
    if per_agent_key not in sd:
        sa_fallback = _SINGLE_AGENT_CAM_FALLBACK.get(cam_family)
        if sa_fallback and sa_fallback in sd:
            per_agent_key = sa_fallback
    if per_agent_key not in sd:
        raise KeyError(f"sensor_data missing {per_agent_key}; available={list(sd.keys())}")
    out = dict(
        head_cam = _rgb_chw(sd[per_agent_key]['rgb'], img_h=img_h, img_w=img_w),
        agent_pos = agent_pos,
    )
    if include_global:
        # Multi-arm scenes expose head_camera_global; PM (single-arm) does not — train
        # side falls back to head_camera (parse_h5_to_zarr_unified.py:_global_cam_path).
        if 'head_camera_global' in sd:
            global_key = 'head_camera_global'
        elif 'head_camera' in sd:
            global_key = 'head_camera'
        else:
            raise KeyError(f"sensor_data missing head_camera_global (no head_camera fallback); available={list(sd.keys())}")
        out['head_cam_global'] = _rgb_chw(sd[global_key]['rgb'], img_h=img_h, img_w=img_w)
    return out

def run_episode(env, planner, dp_models, agent_num, seed, args, verbose, agent_prefix='panda', action_prefix='panda', video_path: str = None, view_sensors=None):
    """Run one episode. Returns dict(success, steps, wallclock_s, infer_ms_per_arm, vram_peak_mb)."""
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
        if args.gripper_source == "qpos":
            qpos_full = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0).cpu().numpy()
            initial_qpos = qpos_full[:8].astype(np.float32, copy=False)
        else:
            initial_qpos = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
            initial_qpos = np.append(initial_qpos, planner.gripper_state[id])
        obs_dict = get_model_input(raw_obs, initial_qpos, id, include_global=args.include_global, cam_family=args.obs_cam_family, img_h=args.img_height, img_w=args.img_width)
        dp_models[id].update_obs(obs_dict)

    infer_times_ms = [[] for _ in range(agent_num)]
    # PR7: ENV-step accounting (unified budget unit) + per-env-step success sampling.
    # Pre-PR7 this loop counted CHUNK iterations (`cnt`) for both the budget and the
    # success check, which (a) could run far past the 500-step TimeLimit and (b) only
    # sampled info["success"] once per chunk — a structural pi0.5-favoring bias on the
    # instantaneous LB criterion (audit_A3 §1). Now: cap on ENV steps, success checked
    # every env step, truncated respected, steps = env steps.
    env_steps = 0
    success = False
    info = {}
    observation = raw_obs
    last_true_action = None  # last per-arm action dict actually stepped (for the hold probe)
    sustained_info = None
    _video_frames = []

    def _record_frame(_obs):
        if video_path is not None and view_sensors:
            _video_frames.append(tile_views([_extract_view_uint8(_obs, s) for s in view_sensors]))

    while not success:
        if env_steps >= args.max_env_steps:
            break
        if verbose:
            print("env_steps:", env_steps)
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for id in range(agent_num):
            t0 = _t.perf_counter()
            action_list = dp_models[id].get_action()
            infer_times_ms[id].append((_t.perf_counter() - t0) * 1000.0)
            if args.gripper_snap:
                _g = action_list[:, -1]
                action_list[:, -1] = np.where(_g >= 0, 1.0, -1.0)
            for i in range(args.max_chunk_actions):
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
        _budget_hit = False
        for i in range(args.max_chunk_actions):
            max_step = 0
            for id in range(agent_num):
                max_step = max(max_step, action_step_dict[f'{action_prefix}-{id}'][i])
            for j in range(max_step):
                true_action = dict()
                for id in range(agent_num):
                    now_step = min(j, action_step_dict[f'{action_prefix}-{id}'][i] - 1)
                    true_action[f'{action_prefix}-{id}'] = action_dict[f'{action_prefix}-{id}'][start_idx[id] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                env_steps += 1
                last_true_action = true_action
                _record_frame(observation)
                if verbose:
                    env.render_human()
                # PR7: per-ENV-step success check (parity with the pi0.5 drivers, which
                # check info["success"] every env step). On the FIRST success, stop the
                # policy loop and run the persistence probe below.
                _succ = info.get('success', False)
                if hasattr(_succ, 'item'):
                    _succ = _succ.item()
                if bool(_succ):
                    success = True
                    break
                _trunc = bool(np.asarray(truncated).reshape(-1)[0]) if truncated is not None else False
                if _trunc or env_steps >= args.max_env_steps:
                    _budget_hit = True
                    break
            if success or _budget_hit:
                break
            if verbose:
                print(true_action)
                print("max_step", max_step)
            for id in range(agent_num):
                start_idx[id] += action_step_dict[f'{action_prefix}-{id}'][i]
                if action_step_dict[f'{action_prefix}-{id}'][i] == 0:
                    continue
                if args.gripper_source == "qpos":
                    qpos_full = observation['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0).cpu().numpy()
                    agent_pos = qpos_full[:8].astype(np.float32, copy=False)
                else:
                    agent_pos = true_action[f'{action_prefix}-{id}']
                obs_dict = get_model_input(observation, agent_pos, id, include_global=args.include_global, cam_family=args.obs_cam_family, img_h=args.img_height, img_w=args.img_width)
                dp_models[id].update_obs(obs_dict)
        if verbose:
            print("info", info)
        if args.render_mode is not None:
            env.render()

    # PR7: persistence probe — on a first success run K hold-qpos env steps (within the
    # shared budget) and record success_sustained_10. A hold action commands each arm to
    # stay at its current qpos with its last commanded gripper. Headline (success_first)
    # is never silently redefined.
    if success and last_true_action is not None:
        def _hold():
            act = {}
            for id in range(agent_num):
                q7 = observation['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy().astype(np.float32)
                grip = float(np.asarray(last_true_action[f'{action_prefix}-{id}']).reshape(-1)[-1])
                act[f'{action_prefix}-{id}'] = np.hstack([q7, grip]).astype(np.float32)
            return act

        def _step(act):
            nonlocal observation, env_steps
            observation, _r, _term, _trunc, _info = env.step(act)
            env_steps += 1
            _record_frame(observation)
            s = _info.get('success', False)
            if hasattr(s, 'item'):
                s = s.item()
            return bool(s), bool(_term), bool(_trunc)

        sustained_info = probe_sustained(_hold, _step, k=SUSTAIN_K, budget_left=args.max_env_steps - env_steps)

    if video_path is not None and _video_frames:
        _video_frames = subsample(_video_frames, args.video_frame_stride)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        h, w = _video_frames[0].shape[:2]
        vw = _cv2.VideoWriter(video_path, _cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
        for _f in _video_frames:
            vw.write(_cv2.cvtColor(_f, _cv2.COLOR_RGB2BGR))
        vw.release()
    wallclock = _t.perf_counter() - t_ep
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1e6
    infer_ms_mean = [float(np.mean(v)) if v else 0.0 for v in infer_times_ms]
    out = dict(
        seed=int(seed),
        success=int(success),
        success_first=int(success),  # PR7: explicit headline
        steps=int(env_steps),        # PR7: ENV steps (unified budget unit)
        wallclock_s=round(wallclock, 3),
        infer_ms_mean_per_arm=[round(x, 2) for x in infer_ms_mean],
        vram_peak_mb=round(vram_peak_mb, 1),
    )
    if sustained_info is not None:
        out["success_sustained_10"] = bool(sustained_info["sustained"])
        out["sustained_info"] = sustained_info
    elif success:
        out["success_sustained_10"] = None  # success at last budget step, no room to probe
    return out


def main(args: Args):
    import time, json, subprocess, socket
    from datetime import datetime
    # Shared hard-fail eval fidelity guards (PR3): refuse the login node up front.
    from robofactory.utils.eval_guards import assert_not_login_node, assert_shader_pack_default
    assert_not_login_node()
    np.set_printoptions(suppress=True, precision=5)
    verbose = not args.quiet
    # PR7: launcher-compat. --max-steps was the old CHUNK-iteration budget; the loop now
    # counts ENV steps. If a launcher overrode --max-steps but left --max-env-steps at its
    # default, honor the launcher's intent as an ENV-step cap so old scripts don't silently
    # explode the budget. Explicit --max-env-steps always wins.
    if args.max_steps != 250 and args.max_env_steps == 400:
        print(f"[eval_multi_dp] PR7: mapping legacy --max-steps {args.max_steps} -> --max-env-steps "
              f"(env-step budget); pass --max-env-steps to override.", flush=True)
        args.max_env_steps = args.max_steps
    # PR4: resolve FINAL env seeds via the single source of truth. --seed-pool/--env-seeds
    # take precedence over the legacy --seed. DP and pi0.5 resolve the same env seeds from a
    # pool name -> true seed pairing. No transform: each seed is passed straight to env.reset.
    if args.seed_pool or args.env_seeds:
        from robofactory.utils.eval_seeds import resolve_seeds
        seeds, seed_provenance = resolve_seeds(
            pool=args.seed_pool or None,
            env_seeds=args.env_seeds or None,
            allow_train=args.allow_train_seeds,
        )
    else:
        if isinstance(args.seed, int):
            args.seed = [args.seed]
        seeds = list(args.seed) if args.seed is not None else [10000]
        from robofactory.utils.eval_seeds import resolve_seeds
        # Record provenance for the legacy --seed path too (recorded as pool 'adhoc').
        seeds, seed_provenance = resolve_seeds(
            env_seeds=",".join(str(s) for s in seeds),
            allow_train=args.allow_train_seeds,
        )
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
        enable_shadow=False,  # training data was collected with shadow=False (env default)
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    assert_shader_pack_default(env_kwargs)
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    record_root = args.record_dir.format(env_id=env_id) + f'/eval_{ts}_data{args.data_num}_ckpt{args.checkpoint_num}'
    # PR9: when --save-trajectory is set, wrap with the self-training capture config
    # (env_states + actions + qpos, NO rgb, symlinked root). The DP runners step ABSOLUTE
    # joint targets, so the recorded actions are absolute — no converter-side delta fix.
    traj_h5_path = None
    if args.save_trajectory:
        from robofactory.utils.eval_trajectory import trajectory_output_dir, wrap_record_trajectory
        _traj_label = f'eval_multi_dp_{env_id}_data{args.data_num}_ckpt{args.checkpoint_num}_{ts}'
        _traj_dir = trajectory_output_dir(args.trajectory_root, _traj_label)
        env, traj_h5_path = wrap_record_trajectory(env, _traj_dir, trajectory_name='trajectory')
        print(f"[eval_multi_dp] PR9 --save-trajectory -> {traj_h5_path}", flush=True)
    else:
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
    print(f"[eval_multi_dp] agent_num={agent_num}, loading DP policies...", flush=True)
    dp_models = []
    for i in range(agent_num):
        import time as _t_load
        _ts = _t_load.perf_counter()
        dp_models.append(DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix))
        print(f"[eval_multi_dp] loaded DP agent {i} in {_t_load.perf_counter()-_ts:.1f}s", flush=True)
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

    # Derive the ordered, deduped list of cameras the policy actually sees (= union of
    # per-agent head_cams across agents + head_camera_global if include_global). Recorded
    # videos tile exactly these views. raw_obs is the post-reset obs from above.
    view_sensors = derive_view_sensors(raw_obs, agent_num, args.include_global, args.obs_cam_family)
    print(f"[eval_multi_dp] tiled video view_sensors={view_sensors}", flush=True)

    # ALWAYS record every seed. Default the video dir to a real path under record_root.
    video_dir = args.video_dir or os.path.join(record_root, 'videos')

    # Provenance + sinks
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/iris/u/mikulrai/projects/RoboFactory').decode().strip()
    except Exception:
        git_sha = 'unknown'
    jsonl_path = args.jsonl_path or f'/iris/u/mikulrai/logs/eval_{env_id}_ckpt{args.checkpoint_num}_{ts}.jsonl'
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    from robofactory.utils.eval_guards import shader_mismatch_override_active
    # PR6: full provenance (eval_protocol v2) — both-repo git sha+dirty, GPU, shader/shadow,
    # seed pool+sha, ckpt path+md5, chunk config. DP has no --prompt (no language input).
    from robofactory.utils.eval_validity import build_provenance
    _ckpt_paths = [m.ckpt_path for m in dp_models]
    provenance = build_provenance(
        shader_pack=args.shader,
        enable_shadow=False,
        sim_backend=args.sim_backend,
        seed_provenance=seed_provenance,
        prompts=None,  # DP is not language-conditioned
        max_env_steps=args.max_env_steps,  # PR7: ENV-step budget (was chunk-iteration max_steps)
        chunk_config={"max_chunk_actions": args.max_chunk_actions, "gripper_source": args.gripper_source},
        ckpt_paths=_ckpt_paths,
        zarr_repo_id=args.ckpt_suffix or None,
    )
    manifest = dict(
        task=env_id, scene_config=args.config,
        data_num=args.data_num, checkpoint_num=args.checkpoint_num,
        ckpt_suffix=args.ckpt_suffix,
        gripper_source=args.gripper_source,
        ckpt_paths=_ckpt_paths,
        max_steps=args.max_steps, max_env_steps=args.max_env_steps,  # PR7: env-step cap
        n_seeds=len(seeds), seeds=seeds,
        seed_provenance=seed_provenance,  # PR4: pool name + sha + allow_train
        sim_backend=args.sim_backend, obs_mode=args.obs_mode,
        git_sha=git_sha, host=socket.gethostname(),
        shader_mismatch_override=shader_mismatch_override_active(),
        start_utc=ts, record_root=record_root, jsonl_path=jsonl_path,
        view_sensors=view_sensors, video_dir=video_dir,
        video_frame_stride=args.video_frame_stride,
        save_trajectory=args.save_trajectory, trajectory_h5=traj_h5_path,  # PR9
        provenance=provenance,  # PR6: eval_protocol v2
    )
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({'kind': 'manifest', **manifest}) + '\n')
    print('MANIFEST:', json.dumps(manifest, indent=2), flush=True)

    from policy._shared.eval_context import WandbRun, VideoRecorder

    with WandbRun(
        enabled=args.wandb,
        project=args.wandb_project,
        job_type='eval',
        name=args.wandb_name or f'eval_decent_{env_id}_ckpt{args.checkpoint_num}_{ts}',
        group=f'eval_decent_{env_id}_ckpt{args.checkpoint_num}',
        tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
        config=manifest,
    ) as wandb_run, VideoRecorder(
        video_dir, all_seeds=True,
    ) as videos:
        # S1 collapse probe (metric-only; never blocks eval). Probe each
        # per-arm policy under namespaces collapse/arm{i}/*.
        if args.skip_collapse_probe:
            print("[collapse-probe] disabled via --skip-collapse-probe", flush=True)
        else:
            try:
                import warnings as _warnings
                from robofactory.utils.preflight_collapse import probe_collapse_with_loaded_policy
                _calib = '/iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz'
                for _i, _dpm in enumerate(dp_models):
                    _rep = probe_collapse_with_loaded_policy(_dpm.policy, _calib)
                    wandb_run.log_raw(_rep.to_wandb_payload(prefix=f'collapse/arm{_i}'))
                    _r = _rep.image_to_baseline_ratio
                    if _r < 1.5:
                        _warnings.warn(f"[collapse arm{_i}] mse_zero_image/baseline = {_r:.2f} < 1.5 - image input may be ignored")
                    print(f"[collapse-probe arm{_i}] {_rep.summary()}", flush=True)
            except Exception as _e:
                import traceback as _tb
                print(f"[collapse-probe] skipped: {type(_e).__name__}: {_e!r}", file=sys.stderr)
                _tb.print_exc(file=sys.stderr)
        # Seed loop (reuses env + policies)
        results = []
        for idx, seed in enumerate(seeds):
            video_path = videos.video_path_for(idx, seed, suffix='_tiled')
            try:
                metrics = run_episode(env, planner, dp_models, agent_num, seed, args, verbose, agent_prefix=agent_prefix, action_prefix=action_prefix, video_path=video_path, view_sensors=view_sensors)
            except Exception as _e:  # PR6: a per-episode crash records steps=-1+error so the
                # >5% validity guard can fire instead of taking the whole run down silently.
                metrics = dict(seed=int(seed), success=0, steps=-1, wallclock_s=0.0,
                               infer_ms_mean_per_arm=[], vram_peak_mb=0.0, error=repr(_e))
            metrics['episode_idx'] = idx
            # PR9: record where this episode's trajectory lands (h5 path + traj group id).
            # RecordEpisodeMA flushes on the NEXT reset / env.close(); group ids are
            # assigned in episode order, so episode `idx` is traj_{idx} in the h5.
            if traj_h5_path is not None:
                metrics['trajectory_path'] = traj_h5_path
                metrics['trajectory_group'] = f'traj_{idx}'
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
                **{f'infer_ms_arm{i}': v for i, v in enumerate(metrics['infer_ms_mean_per_arm'])},
            )
            n_succ = sum(r['success'] for r in results)
            print(f"[seed {seed}] success={metrics['success']} steps={metrics['steps']} wallclock={metrics['wallclock_s']}s vram_mb={metrics['vram_peak_mb']} | running SR {n_succ}/{len(results)} = {100.0*n_succ/len(results):.2f}%", flush=True)

        env.close()

        # Aggregate
        n_total = len(results)
        n_succ = sum(r['success'] for r in results)
        sr = n_succ / n_total if n_total else 0.0
        # PR7: persistence — success_sustained_10 alongside success_first (== success).
        n_sustained = sum(1 for r in results if r.get('success_sustained_10') is True)
        sr_sustained = n_sustained / n_total if n_total else 0.0
        # Wilson 95% CI half-width approximation via normal
        from math import sqrt
        ci = 1.96 * sqrt(max(sr * (1 - sr), 1e-9) / max(n_total, 1))
        steps_succ = [r['steps'] for r in results if r['success']]
        mean_steps_succ = float(np.mean(steps_succ)) if steps_succ else float('nan')
        # PR6: validity guard — if >5% episodes are invalid (steps==-1 or error)
        # the whole run is invalid; the JSONL is renamed *_INVALID.jsonl + exit 2.
        from robofactory.utils.eval_validity import classify_validity, finalize_validity
        validity = classify_validity(results)
        summary = dict(
            n_total=n_total, n_success=n_succ, success_rate=sr, ci95=ci,
            success_first_rate=sr,  # PR7: explicit headline
            n_success_sustained_10=n_sustained, success_sustained_10_rate=sr_sustained,  # PR7
            mean_steps_on_success=mean_steps_succ,
            mean_episode_wallclock_s=float(np.mean([r['wallclock_s'] for r in results])) if results else 0.0,
            valid=validity['valid'], invalid_reason=validity['invalid_reason'],
        )
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps({'kind': 'summary', **summary}) + '\n')
            f.write(json.dumps({'kind': 'validity', **validity}) + '\n')
        print('SUMMARY:', json.dumps(summary, indent=2), flush=True)
        wandb_run.log_summary(**summary)
        # Preserve legacy stdout marker for eval_multi.sh compatibility (last line parse)
        print('success' if sr > 0 else 'failed')
        # PR6: rename JSONL -> *_INVALID.jsonl, tag wandb 'invalid', exit 2 if invalid.
        finalize_validity(validity, jsonl_path, wandb_run=wandb_run)

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
