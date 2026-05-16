"""Compare zarr-stored obs (training) vs env.get_obs at the same IID seed (eval).

Goal: detect offline/online obs distribution shift. If they're identical -> bug
is elsewhere. If they differ visibly -> rendering or state-extraction mismatch
is the smoking gun.
"""
import os
import sys
import argparse
import numpy as np
import zarr
import h5py
import cv2

sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')
os.chdir('/iris/u/mikulrai/projects/RoboFactory/robofactory')

import gymnasium as gym
import robofactory  # noqa
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr-a0', default='/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_decent_agent0_150.zarr')
    parser.add_argument('--zarr-a1', default='/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_decent_agent1_150.zarr')
    parser.add_argument('--demo-idx', type=int, default=13)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--config', default='configs/table/two_robots_stack_cube.yaml')
    parser.add_argument('--out-dir', default='/iris/u/mikulrai/projects/RoboFactory/robofactory/eval_video/obs_diff_seed17_demo13')
    parser.add_argument('--robot-uids', default='panda,panda')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Read zarr's stored obs at demo_idx t=0
    z0 = zarr.open(args.zarr_a0, mode='r')
    z1 = zarr.open(args.zarr_a1, mode='r')
    ep_ends = z0['meta']['episode_ends'][:]
    t0 = 0 if args.demo_idx == 0 else int(ep_ends[args.demo_idx - 1])
    print(f"Zarr demo_idx={args.demo_idx} starts at flat index t0={t0}")

    zarr_head_a0 = z0['data']['head_camera'][t0]   # (3, 224, 224) uint8
    zarr_head_a1 = z1['data']['head_camera'][t0]
    zarr_global = z0['data']['head_camera_global'][t0]   # shared across arms
    zarr_state_a0 = z0['data']['state'][t0]  # 8-dim
    zarr_state_a1 = z1['data']['state'][t0]
    zarr_action_a0 = z0['data']['action'][t0]
    zarr_action_a1 = z1['data']['action'][t0]

    print(f"  zarr state_a0  = {zarr_state_a0}")
    print(f"  zarr action_a0 = {zarr_action_a0}")
    print(f"  state == action? {np.allclose(zarr_state_a0, zarr_action_a0)}")

    # 2) Build env, reset at the IID seed
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'

    robot_uids = tuple(args.robot_uids.split(','))
    # Match eval_multi_dp.py env construction EXACTLY (incl. shader_pack=default + enable_shadow=False)
    shader = 'default'
    env = gym.make(
        env_id,
        config=args.config,
        obs_mode='rgb',
        reward_mode='dense',
        control_mode=None,
        render_mode='sensors',
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
        viewer_camera_configs=dict(shader_pack=shader),
        sim_backend='gpu',
        enable_shadow=False,
        parallel_in_single_scene=False,
        robot_uids=robot_uids,
        num_envs=1,
    )

    raw_obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    print(f"\nEnv built with robots {robot_uids}. agent keys: {list(raw_obs['agent'].keys())}")

    # Construct expected agent_prefix
    agent_keys = list(raw_obs['agent'].keys())
    p0_key = agent_keys[0]
    p1_key = agent_keys[1]

    # 3) Read env head cams + qpos
    sd = raw_obs['sensor_data']
    env_head_a0_hwc = sd['head_camera_agent0']['rgb'].squeeze(0).cpu().numpy() if hasattr(sd['head_camera_agent0']['rgb'], 'cpu') else sd['head_camera_agent0']['rgb'][0]
    env_head_a1_hwc = sd['head_camera_agent1']['rgb'].squeeze(0).cpu().numpy() if hasattr(sd['head_camera_agent1']['rgb'], 'cpu') else sd['head_camera_agent1']['rgb'][0]
    env_global_hwc = sd['head_camera_global']['rgb'].squeeze(0).cpu().numpy() if hasattr(sd['head_camera_global']['rgb'], 'cpu') else sd['head_camera_global']['rgb'][0]

    env_head_a0 = cv2.resize(env_head_a0_hwc, (224, 224), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    env_head_a1 = cv2.resize(env_head_a1_hwc, (224, 224), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    env_global  = cv2.resize(env_global_hwc, (224, 224), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

    env_qpos_p0 = raw_obs['agent'][p0_key]['qpos'].squeeze(0)
    env_qpos_p1 = raw_obs['agent'][p1_key]['qpos'].squeeze(0)
    if hasattr(env_qpos_p0, 'cpu'):
        env_qpos_p0 = env_qpos_p0.cpu().numpy()
        env_qpos_p1 = env_qpos_p1.cpu().numpy()
    env_state_p0 = np.append(env_qpos_p0[:-2], 1.0)  # match eval's gripper_state OPEN = 1.0
    env_state_p1 = np.append(env_qpos_p1[:-2], 1.0)

    print(f"\n  env state_a0 (qpos[:-2]+1.0) = {env_state_p0}")
    print(f"  zarr state_a0                = {zarr_state_a0}")
    print(f"  state diff a0                = {env_state_p0 - zarr_state_a0}")
    print(f"  state diff a0 (abs max)      = {np.abs(env_state_p0 - zarr_state_a0).max():.6f}")

    print(f"\n  env state_a1 (qpos[:-2]+1.0) = {env_state_p1}")
    print(f"  zarr state_a1                = {zarr_state_a1}")
    print(f"  state diff a1                = {env_state_p1 - zarr_state_a1}")
    print(f"  state diff a1 (abs max)      = {np.abs(env_state_p1 - zarr_state_a1).max():.6f}")

    # 4) Compare images
    def mse(a, b):
        return float(np.mean((a.astype(np.int32) - b.astype(np.int32)) ** 2))

    print(f"\n  head_cam_a0 MSE (uint8):     {mse(env_head_a0, zarr_head_a0):.2f}")
    print(f"  head_cam_a1 MSE (uint8):     {mse(env_head_a1, zarr_head_a1):.2f}")
    print(f"  head_cam_global MSE (uint8): {mse(env_global, zarr_global):.2f}")
    print(f"  (note: MSE ~0-50 is essentially identical; >500 is noticeable; >2000 is visibly different)")

    # 5) Save images side by side for visual inspection
    def save_pair(name, env_chw, zarr_chw):
        env_hwc = env_chw.transpose(1, 2, 0)
        zarr_hwc = zarr_chw.transpose(1, 2, 0)
        # Note: zarr was stored RGB (per convert pipeline using cv2 imagery which is BGR).
        # We want to display both as-is to see visual diff.
        h, w, _ = env_hwc.shape
        canvas = np.zeros((h, 2 * w + 8, 3), dtype=np.uint8)
        canvas[:, :w, :] = env_hwc
        canvas[:, w + 8:, :] = zarr_hwc
        cv2.putText(canvas, 'ENV (online)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, 'ZARR (offline)', (w + 18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_path = os.path.join(args.out_dir, f'{name}_side_by_side.png')
        cv2.imwrite(out_path, canvas[..., ::-1])  # cv2 wants BGR
        print(f"  wrote {out_path}")

    save_pair('head_cam_a0', env_head_a0, zarr_head_a0)
    save_pair('head_cam_a1', env_head_a1, zarr_head_a1)
    save_pair('head_cam_global', env_global, zarr_global)

    env.close()
    print(f"\nDONE. Inspect images in {args.out_dir}/")


if __name__ == '__main__':
    main()
