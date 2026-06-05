"""Tier E — Live env, but force BOTH agent_pos AND head_cam/head_cam_global from zarr.

The model gets training-distribution obs at every step. Env still steps under
the policy's actions (real dynamics). If success > 0, confirms that the model
memorised the (image, state=action) → action mapping and the eval failure is
entirely a train/eval observation distribution mismatch.

Differs from eval_multi_dp_force_proprio.py: also substitutes head_cam (and
head_cam_global when present) with the recorded zarr images at step_idx.
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


def load_zarr_episode(zarr_path: str, ep_idx: int, include_global: bool):
    root = zarr.open(zarr_path, mode='r')
    ee = np.asarray(root['meta']['episode_ends']).astype(np.int64)
    s = int(ee[ep_idx - 1]) if ep_idx > 0 else 0
    e = int(ee[ep_idx])
    out = dict(
        action=np.asarray(root['data']['action'][s:e], dtype=np.float32),
        state=np.asarray(root['data']['state'][s:e], dtype=np.float32),
        head_camera=np.asarray(root['data']['head_camera'][s:e], dtype=np.uint8),  # (T, 3, 224, 224)
    )
    if include_global and 'head_camera_global' in root['data']:
        out['head_camera_global'] = np.asarray(root['data']['head_camera_global'][s:e], dtype=np.uint8)
    return out


def clamp(idx, T):
    return min(idx, T - 1)


def build_obs_full(rec, t):
    """Construct training-distribution obs dict from recorded zarr arrays."""
    t = clamp(t, rec['head_camera'].shape[0])
    obs = dict(
        head_cam=(rec['head_camera'][t].astype(np.float32) / 255.0),     # (3, 224, 224) [0,1]
        agent_pos=rec['state'][t].astype(np.float32),                    # (8,)
    )
    if 'head_camera_global' in rec:
        obs['head_cam_global'] = (rec['head_camera_global'][t].astype(np.float32) / 255.0)
    return obs


def run_episode_force_full(env, planner, dp_models, agent_num, seed, args,
                            recorded_per_arm,
                            action_prefix='panda', agent_prefix='panda'):
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

    arm_t = [0] * agent_num
    chunk_l2 = [[] for _ in range(agent_num)]

    for aid in range(agent_num):
        dp_models[aid].update_obs(build_obs_full(recorded_per_arm[aid], arm_t[aid]))

    cnt = 0
    success = False
    info = {}
    while True:
        cnt += 1
        if cnt > args.max_steps:
            break
        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for aid in range(agent_num):
            action_list = dp_models[aid].get_action()
            rec_a = recorded_per_arm[aid]['action'][clamp(arm_t[aid], recorded_per_arm[aid]['action'].shape[0])]
            chunk_l2[aid].append(float(np.linalg.norm(action_list[0] - rec_a)))
            for i in range(args.max_chunk_actions):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'{agent_prefix}-{aid}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, _v, _a, _d = planner.planner[aid].TOPP(path, 0.05, verbose=True)
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

        start_idx = [0] * agent_num
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
            for aid in range(agent_num):
                start_idx[aid] += action_step_dict[f'{action_prefix}-{aid}'][i]
                if action_step_dict[f'{action_prefix}-{aid}'][i] == 0:
                    continue
                arm_t[aid] += 1
                dp_models[aid].update_obs(build_obs_full(recorded_per_arm[aid], arm_t[aid]))
        if info.get('success', False) is True:
            success = True
            break

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
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max-steps', type=int, default=250)
    ap.add_argument('--max-chunk-actions', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--zarr-paths', nargs='+', required=True)
    ap.add_argument('--ep-idx', type=int, default=13)
    ap.add_argument('--robot-uids', default=None)
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    args = ap.parse_args()

    np.set_printoptions(suppress=True, precision=5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    with open(args.config) as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    recorded = [load_zarr_episode(zp, args.ep_idx, args.include_global) for zp in args.zarr_paths]
    for arm, zp in enumerate(args.zarr_paths):
        print(f'loaded zarr ep {args.ep_idx} for arm {arm}: T={recorded[arm]["action"].shape[0]}  '
              f'has_global={"head_camera_global" in recorded[arm]}', flush=True)

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
        debug=False, vis=False,
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
    print(f'Loaded {agent_num} ckpts. agent_prefix={agent_prefix} action_prefix={action_prefix}', flush=True)

    per_seed, n_succ = [], 0
    for seed in args.seeds:
        m = run_episode_force_full(env, planner, dp_models, agent_num, seed, args, recorded,
                                    action_prefix=action_prefix, agent_prefix=agent_prefix)
        n_succ += int(m['success'])
        per_seed.append(m)
        print(json.dumps(m), flush=True)

    summary = dict(env_id=env_id, ckpt_suffix=args.ckpt_suffix, ep_idx=args.ep_idx,
                   n_seeds=len(args.seeds), n_succ=n_succ, sr=round(n_succ / max(1, len(args.seeds)), 4),
                   per_seed=per_seed)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== SUMMARY === SR={summary["sr"]} ({n_succ}/{len(args.seeds)})')
    env.close()


if __name__ == '__main__':
    main()
