"""Replay a multi-arm H5 trajectory step-by-step on the eval env, save MP4.

This is the 2SC-specific extension of replay_h5.py: 2SC stores actions as
`actions/panda-0 (T,8)` and `actions/panda-1 (T,8)` and the env expects a
dict `{'panda-0': a0, 'panda-1': a1}` per env.step (mirrors eval_multi_dp.py).

Records `head_camera_global` RGB at every env step into an MP4.

Usage:
  python script/debug/replay_2sc_h5_video.py \
    --h5 /iris/u/mikulrai/data/RoboFactory/h5_data/TwoRobotsStackCube-rf.h5 \
    --traj 13 --seed 17 \
    --config configs/table/two_robots_stack_cube.yaml \
    --out-mp4 /iris/u/mikulrai/runs/replay/2sc_seed17_traj13.mp4
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import os
import json
import h5py
import yaml
import numpy as np
import gymnasium as gym
import imageio.v2 as imageio

from robofactory.tasks import *  # register envs


def _to_np(x):
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x).squeeze()


def _grab_frame(obs, key='head_camera_global'):
    sd = obs.get('sensor_data', {})
    cam = sd.get(key)
    if cam is None:
        # fallback: try first available camera
        for k, v in sd.items():
            if isinstance(v, dict) and 'rgb' in v:
                cam = v
                break
    if cam is None or 'rgb' not in cam:
        return None
    f = cam['rgb']
    if hasattr(f, 'cpu'):
        f = f.cpu().numpy()
    f = np.asarray(f)
    if f.ndim == 4:           # (B,H,W,3)
        f = f[0]
    return f.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--traj', type=int, required=True)
    ap.add_argument('--seed', type=int, default=None,
                    help='env reset seed (defaults to sidecar episode_seed)')
    ap.add_argument('--out-mp4', required=True)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--obs-mode', default='rgb')
    ap.add_argument('--control-mode', default='pd_joint_pos')
    ap.add_argument('--sim-backend', default='cpu')
    ap.add_argument('--cam', default='head_camera_global')
    args = ap.parse_args()

    # Resolve seed via sidecar if not given
    sc = args.h5.replace('.h5', '.json')
    if args.seed is None and os.path.exists(sc):
        with open(sc) as fp:
            sd = json.load(fp)
        for ep in sd.get('episodes', []):
            if ep.get('episode_id') == args.traj:
                args.seed = ep.get('episode_seed', ep.get('reset_kwargs', {}).get('seed'))
                break
    if args.seed is None:
        raise SystemExit('seed not provided and not found in sidecar')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'

    env = gym.make(
        env_id, config=args.config, obs_mode=args.obs_mode,
        control_mode=args.control_mode, render_mode='sensors',
        num_envs=1, sim_backend=args.sim_backend, enable_shadow=False,
        sensor_configs=dict(shader_pack='default'),
    )

    # Pull actions
    with h5py.File(args.h5, 'r') as f:
        traj = f[f'traj_{args.traj}']
        keys = sorted(k for k in traj['actions'].keys() if k.startswith('panda-'))
        agents = len(keys)
        per_arm = [np.asarray(traj['actions'][k], dtype=np.float32) for k in keys]
        ep_len = min(a.shape[0] for a in per_arm)
        gt_success = bool(np.asarray(traj['success'])[-1])
        print(f'traj={args.traj} seed={args.seed} agents={agents} ep_len={ep_len} h5_success={gt_success}', flush=True)

    obs, _ = env.reset(seed=int(args.seed))
    os.makedirs(os.path.dirname(args.out_mp4), exist_ok=True)
    writer = imageio.get_writer(args.out_mp4, fps=args.fps, codec='libx264', quality=8)

    # frame at reset
    f0 = _grab_frame(obs, args.cam)
    if f0 is None:
        print(f'WARN: camera {args.cam} not in sensor_data; using fallback', flush=True)
    if f0 is not None:
        writer.append_data(f0)

    succ = False
    first_succ_step = None
    for t in range(ep_len):
        action_dict = {keys[i]: per_arm[i][t] for i in range(agents)}
        obs, reward, terminated, truncated, info = env.step(action_dict)
        frame = _grab_frame(obs, args.cam)
        if frame is not None:
            writer.append_data(frame)
        s = info.get('success', False)
        if hasattr(s, 'item'):
            s = s.item()
        elif hasattr(s, '__len__'):
            s = bool(np.asarray(s).any())
        if s and not succ:
            succ = True
            first_succ_step = t
        if (t + 1) % 25 == 0:
            print(f'  step {t+1}/{ep_len}  success={succ}', flush=True)

    writer.close()
    env.close()

    summary = dict(
        h5=args.h5, traj=args.traj, seed=int(args.seed),
        ep_len=ep_len, h5_success=gt_success,
        replay_success=bool(succ), first_success_step=first_succ_step,
        out_mp4=args.out_mp4,
    )
    out_json = args.out_mp4 + '.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== summary ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
