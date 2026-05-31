"""Fixed Tier C / Fixed Tier E for the state=qpos ckpt.

Mode 'proprio' (Fixed Tier C):
  - Override agent_pos with zarr `data/state[ep_start + zarr_step]` every obs update
  - Image stays LIVE from env
  - Isolates: does correct (training-distribution) proprio rescue qpos ckpt under live image?

Mode 'full' (Fixed Tier E):
  - Override agent_pos with zarr state[ep_start + zarr_step]
  - Override head_cam and head_cam_global from zarr at every obs update
  - Env still steps under policy actions
  - Tests: with perfect training-distribution inputs forced into obs, what happens?

Both modes use TOPP-skipped action execution (matches Tier K's speed).
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml
import zarr
import gymnasium as gym

from robofactory.tasks import *  # noqa
import robofactory.agents  # noqa
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from eval_multi_dp import DP


def _zarr_image(arr_chw_uint8):
    """zarr stores (3, H, W) uint8. Policy expects (3, H, W) float32 in [0, 1]."""
    if arr_chw_uint8.ndim == 3 and arr_chw_uint8.shape[0] in (1, 3):
        return arr_chw_uint8.astype(np.float32) / 255.0
    raise ValueError(f'unexpected zarr image shape {arr_chw_uint8.shape}')


def _live_image(rgb_tensor, img_h=224, img_w=224):
    """env rgb is HxWx3 uint8. Convert to (3, H, W) float32 / 255."""
    import cv2
    t = rgb_tensor.squeeze(0)
    arr = t.cpu().numpy() if hasattr(t, 'numpy') else np.asarray(t)
    if arr.shape[0] != img_h or arr.shape[1] != img_w:
        arr = cv2.resize(arr, (img_w, img_h), interpolation=cv2.INTER_AREA)
    return np.moveaxis(arr, -1, 0).astype(np.float32) / 255.0


def build_obs(observation, arm_id, zarr_info, zarr_step, mode, agent_prefix):
    """Build obs_dict for the policy.

    mode='proprio': proprio from zarr, image live from env
    mode='full':    proprio from zarr, image from zarr
    """
    z = zarr_info['zarrs'][arm_id]
    start = zarr_info['ep_start'][arm_id]
    Tmax = zarr_info['ep_T'][arm_id]
    idx = start + min(zarr_step, Tmax - 1)

    agent_pos = np.asarray(z['data/state'][idx]).astype(np.float32)

    if mode == 'full':
        head_cam = _zarr_image(np.asarray(z['data/head_camera'][idx]))
        head_cam_global = _zarr_image(np.asarray(z['data/head_camera_global'][idx]))
    else:
        sd = observation['sensor_data']
        per_agent_key = f'head_camera_agent{arm_id}'
        if per_agent_key not in sd:
            per_agent_key = 'head_camera'
        head_cam = _live_image(sd[per_agent_key]['rgb'])
        head_cam_global = _live_image(sd['head_camera_global']['rgb'])

    return dict(head_cam=head_cam, head_cam_global=head_cam_global, agent_pos=agent_pos)


def run_episode(env, dp_models, agent_num, seed, args, zarr_info,
                agent_prefix='panda', action_prefix='panda', video_path=None):
    raw_obs, _ = env.reset(seed=int(seed))
    if env.action_space is not None:
        env.action_space.seed(int(seed))
    for m in dp_models:
        m.runner.reset_obs()
    try:
        from robofactory.planner.motionplanner import OPEN
        # we don't use planner here but keep state consistent
        _ = OPEN
    except Exception:
        pass

    zarr_step = 0
    # Build initial obs at zarr_step=0
    for arm_id in range(agent_num):
        obs_dict = build_obs(raw_obs, arm_id, zarr_info, zarr_step, args.mode, agent_prefix)
        dp_models[arm_id].update_obs(obs_dict)

    cnt = 0
    success = False
    info = {}
    _video_frames = []
    observation = raw_obs
    while True:
        cnt += 1
        if cnt > args.max_steps:
            break
        # Predict per-arm chunks (TOPP-skipped)
        per_arm_chunks = []
        for arm_id in range(agent_num):
            chunk = dp_models[arm_id].get_action()  # (n_action_steps, D)
            per_arm_chunks.append(chunk)

        # Step env with chunks[0..max_chunk_actions-1]
        for i in range(args.max_chunk_actions):
            true_action = {}
            for arm_id in range(agent_num):
                true_action[f'{action_prefix}-{arm_id}'] = per_arm_chunks[arm_id][i]
            observation, reward, terminated, truncated, info = env.step(true_action)
            if video_path is not None:
                _gframe = observation['sensor_data'].get('head_camera_global', {}).get('rgb')
                if _gframe is not None:
                    _f = _gframe[0]
                    if hasattr(_f, 'cpu'):
                        _f = _f.cpu().numpy()
                    _video_frames.append(_f.astype(np.uint8))
            zarr_step += 1  # advance zarr index per env step
            if info.get('success', False) is True:
                break

        if info.get('success', False) is True:
            success = True
            break

        # Build next obs (proprio always from zarr; image per mode)
        for arm_id in range(agent_num):
            obs_dict = build_obs(observation, arm_id, zarr_info, zarr_step, args.mode, agent_prefix)
            dp_models[arm_id].update_obs(obs_dict)

        if cnt % 20 == 0:
            print(f'  step {cnt}/{args.max_steps}  success={info.get("success", False)}', flush=True)

    if video_path is not None and _video_frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        try:
            import imageio
            imageio.mimwrite(video_path, _video_frames, fps=20, codec='libx264', quality=8)
            print(f'  wrote video: {video_path} ({len(_video_frames)} frames)', flush=True)
        except Exception as e:
            print(f'  video write failed: {e}', flush=True)

    return dict(seed=int(seed), success=bool(success), steps=int(cnt - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', choices=['proprio', 'full'], required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=2000)
    ap.add_argument('--ckpt-suffix', default='workspace_overfit1qpos')
    ap.add_argument('--zarr-tpl',
                    default='/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_overfit1qpos_agent{i}_150.zarr')
    ap.add_argument('--ep-idx', type=int, default=13)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max-steps', type=int, default=80)  # 80 chunks × 6 = 480 env steps, demo is 257
    ap.add_argument('--max-chunk-actions', type=int, default=6)
    ap.add_argument('--robot-uids', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--video-dir', default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    env_kwargs = dict(
        config=args.config, obs_mode='rgb', control_mode='pd_joint_pos',
        render_mode='sensors',
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=1, sim_backend='gpu', enable_shadow=False,
    )
    if args.robot_uids:
        env_kwargs['robot_uids'] = tuple(args.robot_uids.split(','))
    env = gym.make(env_id, **env_kwargs)
    raw_obs, _ = env.reset(seed=int(args.seeds[0]))
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    agent_num = len(env.unwrapped.agent.agents)

    # Pre-open zarrs
    zarrs = []
    ep_start, ep_T = [], []
    for arm_id in range(agent_num):
        zpath = args.zarr_tpl.format(i=arm_id)
        z = zarr.open(zpath, mode='r')
        zarrs.append(z)
        ends = np.asarray(z['meta/episode_ends'])
        s = int(0 if args.ep_idx == 0 else ends[args.ep_idx - 1])
        e = int(ends[args.ep_idx])
        ep_start.append(s)
        ep_T.append(e - s)
        print(f'arm{arm_id}: zarr={zpath} ep={args.ep_idx} start={s} T={e-s}', flush=True)
    zarr_info = dict(zarrs=zarrs, ep_start=ep_start, ep_T=ep_T)

    dp_models = [DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix)
                 for i in range(agent_num)]
    print(f'mode={args.mode}  env={env_id}  agent_num={agent_num}  ckpt={args.ckpt_suffix}', flush=True)

    per_seed, n_succ = [], 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    for idx, seed in enumerate(args.seeds):
        video_path = None
        if args.video_dir:
            video_path = os.path.join(args.video_dir, f'rollout_seed{seed}_retry{idx}.mp4')
        m = run_episode(env, dp_models, agent_num, seed, args, zarr_info,
                         agent_prefix=agent_prefix, action_prefix=action_prefix,
                         video_path=video_path)
        if video_path:
            m['video'] = video_path
        n_succ += int(m['success'])
        per_seed.append(m)
        print(json.dumps(m), flush=True)

    summary = dict(env_id=env_id, mode=args.mode, ckpt_suffix=args.ckpt_suffix,
                   checkpoint_num=args.checkpoint_num,
                   n_seeds=len(args.seeds), n_succ=n_succ,
                   sr=round(n_succ / max(1, len(args.seeds)), 4),
                   per_seed=per_seed)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== SUMMARY === SR={summary["sr"]} ({n_succ}/{len(args.seeds)})')
    env.close()


if __name__ == '__main__':
    main()
