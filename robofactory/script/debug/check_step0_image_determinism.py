"""Quick check: is env.reset(seed=17) image byte-identical to the zarr's
recorded image at step 0 of episode 13? If not, there's a step-0 image
distribution shift entirely independent of policy actions.
"""
from __future__ import annotations

import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os

import numpy as np
import yaml
import zarr
import gymnasium as gym

from robofactory.tasks import *  # registers env ids
import robofactory.agents  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/table/two_robots_stack_cube.yaml')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--zarr-a0', default='/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_overfit1_agent0_150.zarr')
    ap.add_argument('--zarr-a1', default='/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_overfit1_agent1_150.zarr')
    ap.add_argument('--ep-idx', type=int, default=13)
    ap.add_argument('--out-json', default='/iris/u/mikulrai/runs/diagnostics/step0_image_determinism/summary.json')
    args = ap.parse_args()

    with open(args.config) as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

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
        robot_uids=('panda_wristcam_multi', 'panda_wristcam_multi'),
    )
    env = gym.make(env_id, **env_kwargs)
    raw_obs, _ = env.reset(seed=args.seed)

    out = dict(env_id=env_id, seed=args.seed, ep_idx=args.ep_idx, arms={})

    for arm, zp in enumerate([args.zarr_a0, args.zarr_a1]):
        zr = zarr.open(zp, mode='r')
        ee = np.asarray(zr['meta/episode_ends']).astype(np.int64)
        s = int(ee[args.ep_idx - 1]) if args.ep_idx > 0 else 0
        # zarr per-arm head_camera is at zarr['data/head_camera'][s] (training-time first frame)
        zarr_img = np.asarray(zr['data/head_camera'][s])      # (3, 224, 224) uint8
        zarr_glob = np.asarray(zr['data/head_camera_global'][s])
        # env image: same agent's per-arm camera, resized to 224x224
        # mirror what eval_multi_dp.get_model_input does with cv2.resize area interp
        import cv2 as _cv2
        sd = raw_obs['sensor_data']
        cam_key = f'head_camera_agent{arm}'
        glob_key = 'head_camera_global'
        live = sd[cam_key]['rgb'].squeeze(0)
        live = live.cpu().numpy() if hasattr(live, 'cpu') else np.asarray(live)
        if live.shape[:2] != (224, 224):
            live = _cv2.resize(live, (224, 224), interpolation=_cv2.INTER_AREA)
        live_chw = np.moveaxis(live, -1, 0).astype(np.uint8)
        live_g = sd[glob_key]['rgb'].squeeze(0)
        live_g = live_g.cpu().numpy() if hasattr(live_g, 'cpu') else np.asarray(live_g)
        if live_g.shape[:2] != (224, 224):
            live_g = _cv2.resize(live_g, (224, 224), interpolation=_cv2.INTER_AREA)
        live_g_chw = np.moveaxis(live_g, -1, 0).astype(np.uint8)

        diff_loc = np.abs(zarr_img.astype(np.int32) - live_chw.astype(np.int32))
        diff_glob = np.abs(zarr_glob.astype(np.int32) - live_g_chw.astype(np.int32))
        out['arms'][arm] = dict(
            local=dict(
                shape=list(zarr_img.shape),
                bytes_equal=int(np.array_equal(zarr_img, live_chw)),
                mean_abs_diff=float(diff_loc.mean()),
                max_abs_diff=int(diff_loc.max()),
                p99_abs_diff=float(np.percentile(diff_loc, 99)),
            ),
            global_=dict(
                shape=list(zarr_glob.shape),
                bytes_equal=int(np.array_equal(zarr_glob, live_g_chw)),
                mean_abs_diff=float(diff_glob.mean()),
                max_abs_diff=int(diff_glob.max()),
                p99_abs_diff=float(np.percentile(diff_glob, 99)),
            ),
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    env.close()


if __name__ == '__main__':
    main()
