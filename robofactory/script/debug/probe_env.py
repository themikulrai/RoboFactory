"""T1 / T7 / T9 — Dump env contract from env.reset(seed).

Usage:
  python script/debug/probe_env.py --config configs/table/pick_meat.yaml --seed 10000
  python script/debug/probe_env.py --config configs/table/pick_meat.yaml --seed-list 0,1,156,10000 \
    --out-csv /tmp/pm_spawn.csv

Per seed prints/dumps:
  - sensor_data keys, image (min/max/mean/dtype/shape) for head_camera (or per-agent variants)
  - per-agent qpos[:7] and finger qpos[7:9]
  - actor (object) names + (x,y,z) world positions + quat
  - scene/agent uids
Saves first head-camera RGB for the first seed to /tmp/eval_frame_{seed}.png.

Use --out-csv to write spawn positions (one row per seed × actor) for T9.
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import os
import yaml
import numpy as np
import gymnasium as gym
from PIL import Image

from robofactory.tasks import *  # register envs
from mani_skill.envs.sapien_env import BaseEnv


def _to_np(x):
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x)


def probe(env, seed, save_image_path=None, csv_writer=None, label=''):
    raw_obs, info = env.reset(seed=seed)
    if env.action_space is not None:
        env.action_space.seed(seed)
    print(f"\n=== seed={seed} ({label}) ===")

    # 1) sensor_data keys + image stats
    sd = raw_obs.get('sensor_data') or {}
    print(f"sensor_data keys: {list(sd.keys())}")
    cam_keys = list(sd.keys())
    first_img = None
    for k in cam_keys:
        rgb = sd[k].get('rgb')
        if rgb is None:
            continue
        rgb_np = _to_np(rgb)
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        if first_img is None and 'global' not in k:
            first_img = (k, rgb_np)
        print(
            f"  {k}.rgb: shape={rgb_np.shape} dtype={rgb_np.dtype}"
            f" min={int(rgb_np.min())} max={int(rgb_np.max())}"
            f" mean(rgb)={rgb_np.reshape(-1, rgb_np.shape[-1]).mean(axis=0).round(2).tolist()}"
        )

    # 2) per-agent qpos
    agent = raw_obs.get('agent', {}) or {}
    if 'qpos' in agent:
        q = _to_np(agent['qpos']).squeeze(0)
        print(f"agent.qpos shape={q.shape} arm[:7]={q[:7].round(4).tolist()} fingers[7:]={q[7:].round(4).tolist()}")
    else:
        for ak, av in agent.items():
            if isinstance(av, dict) and 'qpos' in av:
                q = _to_np(av['qpos']).squeeze(0)
                print(f"agent.{ak}.qpos shape={q.shape} arm[:7]={q[:7].round(4).tolist()} fingers[7:]={q[7:].round(4).tolist()}")

    # 3) actor (object) positions
    rows = []
    actors = getattr(env.unwrapped, 'actors', None) or {}
    if isinstance(actors, dict) and len(actors) > 0:
        print(f"actors ({len(actors)}):")
        for name, actor in actors.items():
            try:
                pose = actor.pose
                p = _to_np(pose.p).squeeze()
                q = _to_np(pose.q).squeeze()
                print(f"  {name:20s}  xyz={p.round(4).tolist()}  quat={q.round(4).tolist()}")
                rows.append((seed, name, p[0], p[1], p[2], q[0], q[1], q[2], q[3]))
            except Exception as e:
                print(f"  {name}: <pose unavailable> ({type(e).__name__})")
    else:
        # fallback: walk scene actors
        scene = env.unwrapped.scene
        try:
            for a in scene.actors.values():
                p = _to_np(a.pose.p).squeeze()
                print(f"  {a.name:20s}  xyz={p.round(4).tolist()}")
                rows.append((seed, a.name, float(p[0]), float(p[1]), float(p[2]), 0, 0, 0, 0))
        except Exception:
            pass

    # 4) save first frame
    if save_image_path is not None and first_img is not None:
        k, rgb_np = first_img
        Image.fromarray(rgb_np.astype(np.uint8)).save(save_image_path)
        print(f"saved first head-camera frame ({k}) → {save_image_path}")

    if csv_writer is not None:
        for row in rows:
            csv_writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--seed-list', type=str, default=None,
                    help='comma-separated, e.g. "0,1,2,156,10000"')
    ap.add_argument('--seed-label-list', type=str, default=None,
                    help='comma-separated labels matching --seed-list, e.g. "indist-success,indist-fail,…"')
    ap.add_argument('--out-csv', type=str, default=None)
    ap.add_argument('--save-frame-dir', type=str, default='/tmp')
    ap.add_argument('--obs-mode', default='rgb')
    ap.add_argument('--control-mode', default='pd_joint_pos')
    ap.add_argument('--sim-backend', default='cpu')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'

    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode='sensors',
        num_envs=1,
        sim_backend=args.sim_backend,
        enable_shadow=False,  # match training data collection default
    )
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(',')]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [10000]

    labels = args.seed_label_list.split(',') if args.seed_label_list else [''] * len(seeds)

    csv_f = None
    csv_writer = None
    if args.out_csv:
        import csv
        csv_f = open(args.out_csv, 'w', newline='')
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['seed', 'actor', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])

    for s, lbl in zip(seeds, labels):
        save_path = None
        if args.save_frame_dir:
            os.makedirs(args.save_frame_dir, exist_ok=True)
            save_path = os.path.join(args.save_frame_dir, f'eval_frame_seed{s}.png')
        probe(env, s, save_image_path=save_path, csv_writer=csv_writer, label=lbl)

    if csv_f:
        csv_f.close()
        print(f"\nWrote spawn CSV → {args.out_csv}")
    env.close()


if __name__ == '__main__':
    main()
