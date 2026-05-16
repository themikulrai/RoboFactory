"""Simple test: feed the state=action ckpt agent_pos[7] = +1.0 (commanded gripper)
at step 0 instead of planner.gripper_state (=0.04 finger width). For the
state=action ckpt this is the one-line fix from rc-init-grip.
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
import gymnasium as gym

from robofactory.tasks import *  # noqa
import robofactory.agents  # noqa
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from eval_multi_dp import DP, get_model_input


def run_episode(env, planner, dp_models, agent_num, seed, args,
                agent_prefix='panda', action_prefix='panda',
                video_path: str = None,
                initial_gripper: float = 1.0):
    """Like the default eval, but pass initial_gripper as the agent_pos[7] at step 0."""
    _video_frames = []
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

    # Step 0: feed `initial_gripper` (1.0 = commanded open) as agent_pos[7]
    for id in range(agent_num):
        initial_qpos = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
        agent_pos = np.append(initial_qpos, initial_gripper)
        obs_dict = get_model_input(raw_obs, agent_pos, id,
                                    include_global=args.include_global,
                                    cam_family=args.obs_cam_family,
                                    img_h=args.img_height, img_w=args.img_width)
        dp_models[id].update_obs(obs_dict)

    cnt = 0
    success = False
    info = {}
    while True:
        cnt += 1
        if cnt > args.max_steps:
            break
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for id in range(agent_num):
            action_list = dp_models[id].get_action()
            for i in range(args.max_chunk_actions):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, _v, _a, _d = planner.planner[id].TOPP(path, 0.05, verbose=False)
                except Exception:
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
        observation = None
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
                if video_path is not None:
                    _gframe = observation['sensor_data'].get('head_camera_global', {}).get('rgb')
                    if _gframe is not None:
                        _f = _gframe[0]
                        if hasattr(_f, 'cpu'):
                            _f = _f.cpu().numpy()
                        _video_frames.append(_f.astype(np.uint8))
            for id in range(agent_num):
                start_idx[id] += action_step_dict[f'{action_prefix}-{id}'][i]
                if action_step_dict[f'{action_prefix}-{id}'][i] == 0:
                    continue
                # Default behavior (matches eval_multi_dp.py): use true_action as agent_pos.
                # For state=action ckpt, this is the right semantics (commanded actions).
                obs_dict = get_model_input(observation, true_action[f'{action_prefix}-{id}'], id,
                                            include_global=args.include_global,
                                            cam_family=args.obs_cam_family,
                                            img_h=args.img_height, img_w=args.img_width)
                dp_models[id].update_obs(obs_dict)
        if info.get('success', False) is True:
            success = True
            break
        if cnt % 20 == 0:
            print(f'  step {cnt}/{args.max_steps}  success={info.get("success", False)}', flush=True)

    if video_path is not None and _video_frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        try:
            import imageio
            imageio.mimwrite(video_path, _video_frames, fps=20, codec='libx264', quality=8)
            print(f'  wrote video: {video_path} ({len(_video_frames)} frames)', flush=True)
        except Exception as _e:
            print(f'  video write failed: {_e}', flush=True)
    return dict(seed=int(seed), success=bool(success), steps=int(cnt - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=2000)
    ap.add_argument('--ckpt-suffix', default='workspace_overfit1')
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=224)
    ap.add_argument('--img-width', type=int, default=224)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max-steps', type=int, default=250)
    ap.add_argument('--max-chunk-actions', type=int, default=6)
    ap.add_argument('--robot-uids', default=None)
    ap.add_argument('--initial-gripper', type=float, default=1.0,
                    help='Value to use for agent_pos[7] at step 0. 1.0 = commanded open '
                         '(matches state=action ckpt training), 0.04 = finger width (default broken).')
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
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False, print_env_info=False, is_multi_agent=True,
    )
    agent_num = planner.agent_num
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    dp_models = [DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix)
                 for i in range(agent_num)]
    print(f'env={env_id} ckpt={args.ckpt_suffix}  initial_gripper={args.initial_gripper}', flush=True)

    per_seed, n_succ = [], 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    for idx, seed in enumerate(args.seeds):
        video_path = None
        if args.video_dir:
            video_path = os.path.join(args.video_dir, f'rollout_seed{seed}_retry{idx}.mp4')
        m = run_episode(env, planner, dp_models, agent_num, seed, args,
                         agent_prefix=agent_prefix, action_prefix=action_prefix,
                         video_path=video_path, initial_gripper=args.initial_gripper)
        if video_path:
            m['video'] = video_path
        n_succ += int(m['success'])
        per_seed.append(m)
        print(json.dumps(m), flush=True)

    summary = dict(env_id=env_id, ckpt_suffix=args.ckpt_suffix, checkpoint_num=args.checkpoint_num,
                   initial_gripper=args.initial_gripper,
                   n_seeds=len(args.seeds), n_succ=n_succ,
                   sr=round(n_succ / max(1, len(args.seeds)), 4), per_seed=per_seed)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== SUMMARY === SR={summary["sr"]} ({n_succ}/{len(args.seeds)})')
    env.close()


if __name__ == '__main__':
    main()
