"""Probe TSC cube positions for training episodes and eval seeds.

Training: reset env with JSON episode seeds, read cube xyz.
Eval:     reset env with seeds 10000-10029, read cube xyz.

Also reconstructs training cube positions from TSC h5 images using color detection:
- cubeA = blue  [0.047, 0.165, 0.627]
- cubeB = green [0, 1, 0]
- cubeC = red   [1, 0, 0]

Usage:
    python script/debug/probe_tsc_spawn.py
"""
import sys, os
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

import numpy as np
import h5py
import json
import gymnasium as gym
from robofactory.tasks import *

H5_PATH   = '/iris/u/mikulrai/data/RoboFactory/hf_download/ThreeRobotsStackCube/ThreeRobotsStackCube.h5'
JSON_PATH = '/iris/u/mikulrai/data/RoboFactory/hf_download/ThreeRobotsStackCube/ThreeRobotsStackCube.json'
CFG       = 'configs/table/three_robots_stack_cube.yaml'

# ---------------------------------------------------------------------------
# 1. Load episode seeds from JSON
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    jd = json.load(f)
episode_seeds = [ep['episode_seed'] for ep in jd['episodes']]
print(f"TSC training: {len(episode_seeds)} episodes, seeds {episode_seeds[:10]} ...")

# ---------------------------------------------------------------------------
# 2. Reconstruct training cube positions from images (color detection)
# ---------------------------------------------------------------------------
print("\n=== TRAINING cube positions (image-based reconstruction) ===")
with h5py.File(H5_PATH, 'r') as f:
    # Camera params from traj_0
    K = np.array(f['traj_0/obs/sensor_param/head_camera_global/intrinsic_cv'][0])
    cam2world_gl = np.array(f['traj_0/obs/sensor_param/head_camera_global/cam2world_gl'][0])
    print(f"Global cam position: {cam2world_gl[:3,3].round(4)}")
    print(f"Intrinsic K:\n{K.round(2)}")

def pixel_to_world_xy(u, v, K, cam2world_gl, z_target=0.02):
    """Backproject pixel (u,v) to world (x,y) at z=z_target."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    d_gl = np.array([x_c, -y_c, -1.0])
    R = cam2world_gl[:3, :3]
    d_world = R @ d_gl
    cam_pos = cam2world_gl[:3, 3]
    if abs(d_world[2]) < 1e-9:
        return None
    t = (z_target - cam_pos[2]) / d_world[2]
    if t < 0:
        return None
    world_pt = cam_pos + t * d_world
    return world_pt[:2]

def detect_cube_by_color(frame_rgb, target_rgb, tol=60):
    """Find centroid of pixels close to target_rgb. frame_rgb: (H,W,3) uint8."""
    diff = np.abs(frame_rgb.astype(float) - np.array(target_rgb, dtype=float))
    mask = diff.max(axis=-1) < tol
    if mask.sum() < 5:
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())

# Cube colors in 0-255
CUBE_COLORS = {
    'cubeA': [int(0.047*255), int(0.165*255), int(0.627*255)],  # blue
    'cubeB': [0, 255, 0],   # green
    'cubeC': [255, 0, 0],   # red
}

train_positions = {k: [] for k in CUBE_COLORS}
n_show = 10
print(f"\n{'seed':>6}  {'A_x':>7} {'A_y':>7}  {'B_x':>7} {'B_y':>7}  {'C_x':>7} {'C_y':>7}")
print("-" * 62)
with h5py.File(H5_PATH, 'r') as f:
    for ep_idx, seed in enumerate(episode_seeds):
        traj = f[f'traj_{ep_idx}']
        frame = np.array(traj['obs/sensor_data/head_camera_global/rgb'][0])  # (H,W,3) uint8
        pos = {}
        for cube, color in CUBE_COLORS.items():
            pix = detect_cube_by_color(frame, color, tol=60)
            if pix is not None:
                world_xy = pixel_to_world_xy(pix[0], pix[1], K, cam2world_gl, z_target=0.02)
                pos[cube] = world_xy
            else:
                pos[cube] = None
            if world_xy is not None:
                train_positions[cube].append(world_xy)
        if ep_idx < n_show:
            def fmt(p): return f"{p[0]:7.4f} {p[1]:7.4f}" if p is not None else "  None    None"
            print(f"{seed:>6}  {fmt(pos.get('cubeA'))}  {fmt(pos.get('cubeB'))}  {fmt(pos.get('cubeC'))}")

print(f"\nTraining positions summary (image-based):")
for cube, positions in train_positions.items():
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        print(f"  {cube}: n={len(positions)}  x=[{min(xs):.4f},{max(xs):.4f}] mean={np.mean(xs):.4f}  y=[{min(ys):.4f},{max(ys):.4f}] mean={np.mean(ys):.4f}")

# ---------------------------------------------------------------------------
# 3. Probe eval cube positions by running the env
# ---------------------------------------------------------------------------
print("\n=== EVAL cube positions (env.reset) ===")
eval_seeds = list(range(10000, 10030))

env = gym.make('ThreeRobotsStackCube-rf', config=CFG, obs_mode='state',
               control_mode='pd_joint_pos', render_mode='sensors',
               num_envs=1, sim_backend='cpu', enable_shadow=False,
               sensor_configs=dict(shader_pack='default'))

def cube_xyz(env, name):
    for a in env.unwrapped.scene.actors.values():
        if a.name.lower() == name.lower():
            p = a.pose.p
            if hasattr(p, 'cpu'): p = p.cpu().numpy()
            return np.asarray(p).flatten()[:3]
    return None

eval_cube_positions = {}
print(f"{'seed':>6}  {'A_x':>7} {'A_y':>7}  {'B_x':>7} {'B_y':>7}  {'C_x':>7} {'C_y':>7}")
print("-" * 62)
for seed in eval_seeds:
    env.reset(seed=seed)
    a = cube_xyz(env, 'cubeA')
    b = cube_xyz(env, 'cubeB')
    c = cube_xyz(env, 'cubeC')
    eval_cube_positions[seed] = {'cubeA': a, 'cubeB': b, 'cubeC': c}
    def fmt3(p): return f"{p[0]:7.4f} {p[1]:7.4f}" if p is not None else "    -       -  "
    print(f"{seed:>6}  {fmt3(a)}  {fmt3(b)}  {fmt3(c)}")

env.close()

print("\nEval positions summary:")
for cube in ['cubeA', 'cubeB', 'cubeC']:
    xs = [v[cube][0] for v in eval_cube_positions.values() if v[cube] is not None]
    ys = [v[cube][1] for v in eval_cube_positions.values() if v[cube] is not None]
    print(f"  {cube}: x=[{min(xs):.4f},{max(xs):.4f}] mean={np.mean(xs):.4f}  y=[{min(ys):.4f},{max(ys):.4f}] mean={np.mean(ys):.4f}")
