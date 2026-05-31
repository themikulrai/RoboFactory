"""Tier P — image-distribution audit.

Compare:
  (1) zarr image at traj_13, step 0..4 (what training saw)
  (2) live env image at env.reset(seed=17), step 0..4 with sim_backend='gpu'
  (3) live env image at env.reset(seed=17), step 0..4 with sim_backend='cpu'  (matches H5 datagen)

Saves per-pixel diff stats + PNG triptychs to /iris/u/mikulrai/runs/diagnostics/overfit_h3_tierP_image/.
If gpu and zarr differ but cpu and zarr match, we've found the smoking gun.
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os

import numpy as np
import zarr
import yaml
import gymnasium as gym
from PIL import Image

from robofactory.tasks import *  # noqa
import robofactory.agents  # noqa


def render_env(sim_backend, config_path, seed, n_steps, env_id, robot_uids, agent_prefix='panda-0'):
    """Reset env, capture first n_steps frames for head_camera_0/global. No action stepping."""
    env_kwargs = dict(
        config=config_path, obs_mode='rgb', control_mode='pd_joint_pos',
        render_mode='sensors',
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=1, sim_backend=sim_backend, enable_shadow=False,
    )
    if robot_uids:
        env_kwargs['robot_uids'] = tuple(robot_uids.split(','))
    env = gym.make(env_id, **env_kwargs)
    raw, _ = env.reset(seed=int(seed))
    cam_names = list(raw['sensor_data'].keys())
    frames = {n: [] for n in cam_names}
    def _grab():
        obs = env.get_obs()
        for cn in cam_names:
            rgb = obs['sensor_data'][cn].get('rgb')
            if rgb is None:
                continue
            f = rgb[0]
            if hasattr(f, 'cpu'):
                f = f.cpu().numpy()
            frames[cn].append(f.astype(np.uint8))
    _grab()
    env.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--zarr-path', required=True)
    ap.add_argument('--ep-idx', type=int, default=13)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--robot-uids', default=None)
    ap.add_argument('--n-frames', type=int, default=1)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config) as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    # 1) zarr image
    root = zarr.open(args.zarr_path, mode='r')
    ends = np.asarray(root['meta/episode_ends'])
    start = int(0 if args.ep_idx == 0 else ends[args.ep_idx - 1])
    print(f'zarr keys under data: {list(root["data"].keys())}', flush=True)
    zarr_imgs = {}
    for k in ['head_camera', 'head_camera_global']:
        if f'data/{k}' in root:
            arr = np.asarray(root[f'data/{k}'][start])  # zarr stores CHW
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            zarr_imgs[k] = arr
    print(f'zarr_imgs keys: {list(zarr_imgs.keys())}, shape: {[v.shape for v in zarr_imgs.values()]}', flush=True)

    # 2) GPU env  --  3) CPU env
    summary = {'env_id': env_id, 'seed': args.seed, 'ep_idx': args.ep_idx, 'zarr': args.zarr_path}
    for backend in ['gpu', 'cpu']:
        try:
            env_imgs = render_env(backend, args.config, args.seed, args.n_frames, env_id, args.robot_uids)
        except Exception as e:
            print(f'backend={backend} FAILED: {type(e).__name__}: {e}', flush=True)
            import traceback; traceback.print_exc()
            summary[backend] = {'error': f'{type(e).__name__}: {e}'}
            continue
        backend_stats = {}
        for cam, frames in env_imgs.items():
            if not frames:
                continue
            env_img = frames[0]  # (H,W,3) uint8
            # Resize zarr image down to match env's native res for fair compare? Actually zarr was already 224x224.
            # Take env's first frame, resize to 224x224 to match zarr if needed.
            if cam in zarr_imgs:
                z = zarr_imgs[cam]
                # Convert env_img to 224x224 to match zarr
                import cv2
                env_resized = cv2.resize(env_img, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_AREA)
                diff = np.abs(env_resized.astype(np.float32) - z.astype(np.float32))
                stats = {
                    'env_mean_rgb': env_resized.mean(axis=(0, 1)).round(2).tolist(),
                    'zarr_mean_rgb': z.mean(axis=(0, 1)).round(2).tolist(),
                    'mean_abs_diff': float(diff.mean()),
                    'max_abs_diff': float(diff.max()),
                    'env_shape': env_resized.shape,
                    'zarr_shape': z.shape,
                }
                backend_stats[cam] = stats
                # Save side-by-side
                triptych = np.concatenate([z, env_resized, np.clip(diff * 5, 0, 255).astype(np.uint8)], axis=1)
                Image.fromarray(triptych).save(os.path.join(args.out_dir, f'{backend}_{cam}_zarr_env_diff.png'))
                print(f'backend={backend} cam={cam}: mean_abs_diff={stats["mean_abs_diff"]:.2f} env_mean={stats["env_mean_rgb"]} zarr_mean={stats["zarr_mean_rgb"]}', flush=True)
        summary[backend] = backend_stats

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n=== SUMMARY at {args.out_dir}/summary.json ===')


if __name__ == '__main__':
    main()
