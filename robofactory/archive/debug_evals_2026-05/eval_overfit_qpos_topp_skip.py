"""Tier G — qpos ckpt eval with TOPP entirely skipped.

H3a showed env.step(recorded_action) replays cleanly. Tier E showed the
policy predicts ~recorded actions but still fails. Suspect: TOPP/chunked
open-loop. This test removes TOPP entirely — env.step(policy.action[0])
each iteration, like H3a but with the policy instead of the recording.
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os
import time as _t

import numpy as np
import torch
import yaml
import gymnasium as gym

from robofactory.tasks import *  # noqa
import robofactory.agents  # noqa

from eval_multi_dp import DP, get_model_input


def _qpos_agent_pos(raw_obs, arm_id, agent_prefix):
    """8-D agent_pos: [arm_qpos(7), left_finger_width(1)] — matches zarr."""
    qp = raw_obs['agent'][f'{agent_prefix}-{arm_id}']['qpos'].squeeze(0).cpu().numpy()
    return np.concatenate([qp[:7], qp[7:8]], axis=0).astype(np.float32)


def run_episode_topp_skip(env, dp_models, agent_num, seed, args,
                           agent_prefix='panda', action_prefix='panda',
                           video_path: str = None,
                           proprio_source: str = 'action'):
    """proprio_source='action' (legacy/buggy) or 'qpos' (correct for state=qpos ckpt)."""
    _video_frames = []
    raw_obs, _ = env.reset(seed=int(seed))
    if env.action_space is not None:
        env.action_space.seed(int(seed))
    for m in dp_models:
        m.runner.reset_obs()

    # Initial obs
    for id in range(agent_num):
        if proprio_source == 'qpos':
            agent_pos = _qpos_agent_pos(raw_obs, id, agent_prefix)
        else:  # 'action' (legacy)
            initial_qpos = raw_obs['agent'][f'{agent_prefix}-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
            gripper_init = 1.0
            agent_pos = np.append(initial_qpos, gripper_init)
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
        true_action = dict()
        for id in range(agent_num):
            # Get the next predicted action (chunk[0])
            action_list = dp_models[id].get_action()
            true_action[f'{action_prefix}-{id}'] = action_list[0]
        observation, reward, terminated, truncated, info = env.step(true_action)
        if video_path is not None:
            _gframe = observation['sensor_data'].get('head_camera_global', {}).get('rgb')
            if _gframe is not None:
                _f = _gframe[0]
                if hasattr(_f, 'cpu'):
                    _f = _f.cpu().numpy()
                _video_frames.append(_f.astype(np.uint8))
        # Update obs: for state=qpos ckpt, feed actual qpos from env (with finger width as dim 7).
        # For state=action ckpt (legacy), feed the just-commanded action.
        for id in range(agent_num):
            if proprio_source == 'qpos':
                agent_pos = _qpos_agent_pos(observation, id, agent_prefix)
            else:
                agent_pos = true_action[f'{action_prefix}-{id}']
            obs_dict = get_model_input(observation, agent_pos, id,
                                        include_global=args.include_global,
                                        cam_family=args.obs_cam_family,
                                        img_h=args.img_height, img_w=args.img_width)
            dp_models[id].update_obs(obs_dict)
        if info.get('success', False) is True:
            success = True
            break
        if cnt % 20 == 0:
            print(f'  step {cnt}/{args.max_steps} success={info.get("success", False)}', flush=True)
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
    ap.add_argument('--ckpt-suffix', default='workspace_overfit1qpos')
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=224)
    ap.add_argument('--img-width', type=int, default=224)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--robot-uids', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--video-dir', default=None)
    ap.add_argument('--proprio-source', choices=('action', 'qpos'), default='qpos',
                    help="'qpos' = feed env's qpos with finger width (correct for state=qpos ckpt). "
                         "'action' = legacy/buggy: feed just-commanded action.")
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
    env.reset(seed=int(args.seeds[0]))

    agent_num = len(env.unwrapped.agent.agents)
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    dp_models = [DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix)
                 for i in range(agent_num)]
    print(f'env={env_id} agent_num={agent_num} prefix={agent_prefix}/{action_prefix} ckpt={args.ckpt_suffix}',
          flush=True)

    per_seed, n_succ = [], 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    for idx, seed in enumerate(args.seeds):
        t0 = _t.perf_counter()
        video_path = None
        if args.video_dir:
            video_path = os.path.join(args.video_dir, f'rollout_seed{seed}_retry{idx}.mp4')
        m = run_episode_topp_skip(env, dp_models, agent_num, seed, args,
                                   agent_prefix=agent_prefix, action_prefix=action_prefix,
                                   video_path=video_path, proprio_source=args.proprio_source)
        if video_path:
            m['video'] = video_path
        m['wall_s'] = round(_t.perf_counter() - t0, 1)
        n_succ += int(m['success'])
        per_seed.append(m)
        print(json.dumps(m), flush=True)

    summary = dict(env_id=env_id, ckpt_suffix=args.ckpt_suffix, checkpoint_num=args.checkpoint_num,
                   n_seeds=len(args.seeds), n_succ=n_succ,
                   sr=round(n_succ / max(1, len(args.seeds)), 4), per_seed=per_seed)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== SUMMARY === SR={summary["sr"]} ({n_succ}/{len(args.seeds)})')
    env.close()


if __name__ == '__main__':
    main()
