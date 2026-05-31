"""Tier C — Live env, but force agent_pos = recorded zarr action[step_idx].

For each per-arm policy, every time `update_obs` would normally pass a live qpos
(or the just-executed action) as `agent_pos`, we instead pass the recorded
training action at the matching step. This is the most direct test of the
overfit hypothesis: if the model memorised f(image, action) → action_chunk and
the eval failure is the eval-time proprio distribution shift, then teacher-
forcing proprio with the recorded actions should recover the demo.

Builds on eval_multi_dp_qpos_swap.py — same TOPP / per-arm wiring, same DP
runners; only the source of `agent_pos` differs (recorded action[step_idx],
clamped to last step at end-of-episode).
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os
import socket
import subprocess
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml
import zarr
import gymnasium as gym

from robofactory.tasks import *  # registers env ids
import robofactory.agents  # noqa: F401
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from eval_multi_dp import DP, get_model_input


def load_zarr_episode_actions(zarr_path: str, ep_idx: int):
    """Return action array for one episode: (T, 8) float32."""
    root = zarr.open(zarr_path, mode='r')
    action = np.asarray(root['data']['action'])             # (sum_T, 8)
    ep_ends = np.asarray(root['meta']['episode_ends']).astype(np.int64)
    s = int(ep_ends[ep_idx - 1]) if ep_idx > 0 else 0
    e = int(ep_ends[ep_idx])
    return action[s:e].astype(np.float32, copy=False)


def forced_proprio_at(recorded: np.ndarray, t: int) -> np.ndarray:
    """Recorded action at step t, clamped to the last available step."""
    T = recorded.shape[0]
    return recorded[min(t, T - 1)].astype(np.float32, copy=False)


def run_episode_force_proprio(env, planner, dp_models, agent_num, seed, args,
                              recorded_per_arm,
                              agent_prefix='panda', action_prefix='panda',
                              video_path: str = None):
    """Closed-loop rollout with agent_pos = recorded action[step_idx]."""
    import cv2 as _cv2
    raw_obs, _ = env.reset(seed=int(seed))
    if env.action_space is not None:
        env.action_space.seed(int(seed))

    for m in dp_models:
        m.runner.reset_obs()
    try:
        from robofactory.planner.motionplanner import OPEN
        planner.gripper_state = [OPEN] * agent_num
    except Exception:
        pass

    # Per-arm step index into recorded action stream
    arm_t = [0 for _ in range(agent_num)]
    chunk_l2 = [[] for _ in range(agent_num)]   # per-step L2 between predicted chunk[0] and recorded action[t]

    # Seed planner state with FIRST obs (proprio forced from recorded[0])
    for aid in range(agent_num):
        ap = forced_proprio_at(recorded_per_arm[aid], arm_t[aid])
        obs_dict = get_model_input(
            raw_obs, ap, aid,
            include_global=args.include_global,
            cam_family=args.obs_cam_family,
            img_h=args.img_height, img_w=args.img_width,
        )
        dp_models[aid].update_obs(obs_dict)

    cnt = 0
    success = False
    info = {}
    _video_frames = []
    while True:
        cnt += 1
        if cnt > args.max_steps:
            break

        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for aid in range(agent_num):
            action_list = dp_models[aid].get_action()
            # Diagnostic: compare predicted chunk[0] to recorded action at the matching step
            rec_now = forced_proprio_at(recorded_per_arm[aid], arm_t[aid])
            chunk_l2[aid].append(float(np.linalg.norm(action_list[0] - rec_now)))
            if args.gripper_snap:
                _g = action_list[:, -1]
                action_list[:, -1] = np.where(_g >= 0, 1.0, -1.0)
            for i in range(args.max_chunk_actions):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'{agent_prefix}-{aid}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[aid].TOPP(path, 0.05, verbose=True)
                except Exception:
                    action_dict[f'{action_prefix}-{aid}'].append(now_action)
                    action_step_dict[f'{action_prefix}-{aid}'].append(1)
                    continue
                n_step = position.shape[0]
                action_step_dict[f'{action_prefix}-{aid}'].append(n_step)
                gripper_state = now_action[-1]
                if n_step == 0:
                    action_dict[f'{action_prefix}-{aid}'].append(now_action)
                for j in range(n_step):
                    true_action = np.hstack([position[j], gripper_state])
                    action_dict[f'{action_prefix}-{aid}'].append(true_action)

        start_idx = [0 for _ in range(agent_num)]
        observation = None
        for i in range(args.max_chunk_actions):
            max_step = 0
            for aid in range(agent_num):
                max_step = max(max_step, action_step_dict[f'{action_prefix}-{aid}'][i])
            for j in range(max_step):
                true_action = dict()
                for aid in range(agent_num):
                    now_step = min(j, action_step_dict[f'{action_prefix}-{aid}'][i] - 1)
                    true_action[f'{action_prefix}-{aid}'] = action_dict[f'{action_prefix}-{aid}'][start_idx[aid] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                if video_path is not None:
                    _gframe = observation['sensor_data'].get('head_camera_global', {}).get('rgb')
                    if _gframe is not None:
                        _f = _gframe[0]
                        if hasattr(_f, 'cpu'):
                            _f = _f.cpu().numpy()
                        _video_frames.append(_f.astype(np.uint8))
            for aid in range(agent_num):
                start_idx[aid] += action_step_dict[f'{action_prefix}-{aid}'][i]
                if action_step_dict[f'{action_prefix}-{aid}'][i] == 0:
                    continue
                # FORCED PROPRIO: agent_pos = recorded action at the next step
                arm_t[aid] += 1
                ap = forced_proprio_at(recorded_per_arm[aid], arm_t[aid])
                obs_dict = get_model_input(
                    observation, ap, aid,
                    include_global=args.include_global,
                    cam_family=args.obs_cam_family,
                    img_h=args.img_height, img_w=args.img_width,
                )
                dp_models[aid].update_obs(obs_dict)
        if info.get('success', False) is True:
            success = True
            break

    if video_path is not None and _video_frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        h, w = _video_frames[0].shape[:2]
        vw = _cv2.VideoWriter(video_path, _cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
        for _f in _video_frames:
            vw.write(_cv2.cvtColor(_f, _cv2.COLOR_RGB2BGR))
        vw.release()

    return dict(
        seed=int(seed),
        success=bool(success),
        steps=int(cnt - 1),
        chunk_l2_mean_per_arm=[float(np.mean(v)) if v else 0.0 for v in chunk_l2],
        chunk_l2_max_per_arm=[float(np.max(v)) if v else 0.0 for v in chunk_l2],
        arm_t_final=arm_t,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=2000)
    ap.add_argument('--ckpt-suffix', default='workspace_overfit1')
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=224)
    ap.add_argument('--img-width', type=int, default=224)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max-steps', type=int, default=250)
    ap.add_argument('--max-chunk-actions', type=int, default=6)
    ap.add_argument('--gripper-snap', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--zarr-paths', nargs='+', required=True,
                    help='Per-arm zarrs (in arm order). Recorded actions for `--ep-idx` are loaded from each.')
    ap.add_argument('--ep-idx', type=int, default=13,
                    help='Which episode in the zarr to teacher-force from (default = the overfit-selected idx 13).')
    ap.add_argument('--robot-uids', default=None,
                    help='Comma-separated. Defaults to env-config robot_uids if None.')
    ap.add_argument('--video-out', default=None)
    args = ap.parse_args()

    np.set_printoptions(suppress=True, precision=5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    with open(args.config, 'r') as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    recorded_per_arm = []
    for zp in args.zarr_paths:
        rec = load_zarr_episode_actions(zp, args.ep_idx)
        recorded_per_arm.append(rec)
        print(f'loaded recorded actions: {zp}  ep={args.ep_idx}  shape={rec.shape}', flush=True)

    env_kwargs = dict(
        config=args.config,
        obs_mode='rgb',
        control_mode='pd_joint_pos',
        render_mode='sensors',
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=1,
        sim_backend='gpu',
        enable_shadow=False,
    )
    if args.robot_uids:
        env_kwargs['robot_uids'] = tuple(args.robot_uids.split(','))
    env = gym.make(env_id, **env_kwargs)

    raw_obs, _ = env.reset(seed=int(args.seeds[0]))
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True,
    )
    agent_num = planner.agent_num
    if len(args.zarr_paths) != agent_num:
        env.close()
        raise ValueError(f'--zarr-paths gave {len(args.zarr_paths)} entries; env has {agent_num} arms.')

    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    print(f"env_id={env_id} agent_prefix='{agent_prefix}' action_prefix='{action_prefix}' n_arms={agent_num}", flush=True)

    dp_models = [DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix)
                 for i in range(agent_num)]
    print(f'Loaded {agent_num} ckpts.', flush=True)

    per_seed = []
    n_succ = 0
    for seed in args.seeds:
        m = run_episode_force_proprio(
            env, planner, dp_models, agent_num, seed, args,
            recorded_per_arm,
            agent_prefix=agent_prefix, action_prefix=action_prefix,
            video_path=args.video_out,
        )
        n_succ += int(m['success'])
        per_seed.append(m)
        print(json.dumps(m), flush=True)

    summary = dict(
        env_id=env_id,
        ckpt_suffix=args.ckpt_suffix,
        checkpoint_num=args.checkpoint_num,
        ep_idx=args.ep_idx,
        n_seeds=len(args.seeds),
        n_succ=n_succ,
        sr=round(n_succ / max(1, len(args.seeds)), 4),
        per_seed=per_seed,
    )
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== SUMMARY ===\nSR = {summary["sr"]} ({n_succ}/{len(args.seeds)})')
    env.close()


if __name__ == '__main__':
    main()
