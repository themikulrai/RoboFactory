"""Reconstruct training-time meat xy positions from PM zarr images.

Method: pixel-variance trick.
1. Load all 150 first frames from zarr (background = constant; meat = moving).
2. Compute per-pixel variance across 150 episodes to locate the meat region.
3. For each episode: diff = |frame - mean_frame| => high-diff pixels = meat.
4. Compute centroid of high-diff pixels in image coordinates.
5. Back-project to 3D using h5 camera intrinsic/extrinsic (OpenCV convention).
6. Output training xy distribution and compare with eval seeds.

Usage:
    python script/debug/reconstruct_train_spawns_pm.py
"""
import sys, os
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

import numpy as np
import zarr
import h5py
import json

ZARR_PATH = '/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150.zarr'
H5_PATH   = '/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5'
JSON_PATH = '/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.json'

# ---------------------------------------------------------------------------
# 1. Load all first frames from zarr
# ---------------------------------------------------------------------------
z = zarr.open(ZARR_PATH, 'r')
episode_ends = z['meta/episode_ends'][:]
ep_starts = np.concatenate([[0], episode_ends[:-1]])
n_ep = len(ep_starts)

print(f"Loading {n_ep} first frames from zarr ...")
head_cam = z['data/head_camera']  # (total_T, C, H, W) uint8 [0,255]
first_frames = np.stack([head_cam[s] for s in ep_starts])  # (150, C, H, W) uint8
# Convert to HWC float [0,255]
first_frames_hwc = np.moveaxis(first_frames, 1, -1).astype(np.float32)  # (150, H, W, 3)
H, W = first_frames_hwc.shape[1], first_frames_hwc.shape[2]
print(f"  frames shape: {first_frames_hwc.shape}  dtype={first_frames_hwc.dtype}")

# ---------------------------------------------------------------------------
# 2. Compute per-pixel variance to locate meat region
# ---------------------------------------------------------------------------
mean_frame = first_frames_hwc.mean(axis=0)          # (H, W, 3)
variance_frame = first_frames_hwc.var(axis=0)        # (H, W, 3)
total_variance = variance_frame.sum(axis=-1)          # (H, W)

# Threshold: pixels with variance > 80th percentile are "variable" = meat region
threshold = np.percentile(total_variance, 80)
meat_mask_global = total_variance > threshold
print(f"  variance threshold={threshold:.2f}  meat pixels={meat_mask_global.sum()}")

# ---------------------------------------------------------------------------
# 3. Load camera parameters from h5 (traj_0, frame 0)
# ---------------------------------------------------------------------------
with h5py.File(H5_PATH, 'r') as f:
    # intrinsic_cv: (T, 3, 3), use frame 0
    K = np.array(f['traj_0/obs/sensor_param/head_camera/intrinsic_cv'][0])  # (3,3)
    # cam2world_gl: (T, 4, 4) — OpenGL camera-to-world
    cam2world_gl = np.array(f['traj_0/obs/sensor_param/head_camera/cam2world_gl'][0])  # (4,4)

print(f"\nCamera intrinsic K:\n{K.round(2)}")
print(f"cam2world_gl[:3,3] (camera position): {cam2world_gl[:3,3].round(4)}")

# ---------------------------------------------------------------------------
# 4. Backprojection helper: pixel (u,v) -> world XY at z=0 table plane
# ---------------------------------------------------------------------------
def pixel_to_world_xy(u, v, K, cam2world_gl):
    """Project pixel (u,v) to world (x,y) at z=0 using OpenGL cam2world."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    # Ray in OpenCV camera space: (x_c, y_c, 1)
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    # OpenGL convention: Y is flipped relative to OpenCV
    # cam2world_gl transforms from GL camera coords to world
    # GL camera: +X right, +Y up, -Z forward
    # OpenCV camera: +X right, +Y down, +Z forward
    # So GL coords from OpenCV coords: x_gl = x_c, y_gl = -y_c, z_gl = -1
    d_gl = np.array([x_c, -y_c, -1.0])
    # World direction
    R = cam2world_gl[:3, :3]  # rotation (cam GL -> world)
    d_world = R @ d_gl
    # Camera position in world
    cam_pos = cam2world_gl[:3, 3]
    # Intersect with z=0 plane: cam_pos + t*d_world => z=0 => t = -cam_pos[2] / d_world[2]
    if abs(d_world[2]) < 1e-9:
        return None
    t = -cam_pos[2] / d_world[2]
    if t < 0:
        return None
    world_pt = cam_pos + t * d_world
    return world_pt[:2]  # (x, y)

# ---------------------------------------------------------------------------
# 5. For each episode, find meat centroid via diff from mean frame
# ---------------------------------------------------------------------------
print("\n=== TRAINING meat positions (reconstructed from images) ===")
training_positions = []
with h5py.File(H5_PATH, 'r') as f:
    episode_seeds = []
    with open(JSON_PATH) as jf:
        episodes_meta = json.load(jf)['episodes']
    episode_seeds = [ep['episode_seed'] for ep in episodes_meta]

for ep_idx in range(n_ep):
    frame = first_frames_hwc[ep_idx]  # (H, W, 3)
    diff = np.abs(frame - mean_frame).sum(axis=-1)  # (H, W)
    # Use global meat mask to restrict to meat region
    diff_masked = diff * meat_mask_global
    # Additional threshold: must be above 20 to avoid noise
    ep_meat_mask = diff_masked > 20
    if ep_meat_mask.sum() < 10:
        # Try lower threshold
        ep_meat_mask = diff_masked > diff_masked.max() * 0.3
    if ep_meat_mask.sum() == 0:
        print(f"  ep {ep_idx}: NO meat pixels detected!")
        training_positions.append(None)
        continue
    # Compute centroid in pixel coordinates
    ys, xs = np.where(ep_meat_mask)
    u_center = xs.mean()
    v_center = ys.mean()
    # Back-project to world
    world_xy = pixel_to_world_xy(u_center, v_center, K, cam2world_gl)
    if world_xy is None:
        training_positions.append(None)
        continue
    training_positions.append((float(world_xy[0]), float(world_xy[1])))

# Print results
valid = [(seed, pos) for seed, pos in zip(episode_seeds, training_positions) if pos is not None]
print(f"\n{'seed':>6}  {'meat_x':>8}  {'meat_y':>8}")
print("-" * 28)
for seed, pos in valid[:30]:
    print(f"{seed:>6}  {pos[0]:>8.4f}  {pos[1]:>8.4f}")
if len(valid) > 30:
    print(f"  ... ({len(valid)} total)")

xs_train = [p[1][0] for p in valid]
ys_train = [p[1][1] for p in valid]
print(f"\nTRAINING spawn range:   x=[{min(xs_train):.4f}, {max(xs_train):.4f}]  y=[{min(ys_train):.4f}, {max(ys_train):.4f}]")
print(f"TRAINING spawn mean:    x={np.mean(xs_train):.4f}  y={np.mean(ys_train):.4f}")
print(f"TRAINING spawn std:     x={np.std(xs_train):.4f}  y={np.std(ys_train):.4f}")

# ---------------------------------------------------------------------------
# 6. Compare with eval success centroid
# ---------------------------------------------------------------------------
success_centroid = (0.063, 0.041)
print(f"\nEval success centroid:  x={success_centroid[0]:.4f}  y={success_centroid[1]:.4f}")
print(f"Training mean vs success centroid offset: dx={np.mean(xs_train)-success_centroid[0]:.4f}  dy={np.mean(ys_train)-success_centroid[1]:.4f}")

# What fraction of training positions fall within 3.3cm of success centroid?
dists = [np.sqrt((x - success_centroid[0])**2 + (y - success_centroid[1])**2) for x,y in zip(xs_train, ys_train)]
frac_in_band = np.mean(np.array(dists) < 0.033)
print(f"Training positions within 3.3cm of success centroid: {frac_in_band:.1%} ({sum(d < 0.033 for d in dists)}/{len(dists)})")

# Histogramming in 5cm buckets
print("\nTraining x distribution (4cm buckets):")
for lo in np.arange(-0.06, 0.18, 0.04):
    count = sum(lo <= x < lo+0.04 for x in xs_train)
    print(f"  x=[{lo:.2f},{lo+0.04:.2f}): {'*'*count} ({count})")
print("Training y distribution (4cm buckets):")
for lo in np.arange(-0.06, 0.18, 0.04):
    count = sum(lo <= y < lo+0.04 for y in ys_train)
    print(f"  y=[{lo:.2f},{lo+0.04:.2f}): {'*'*count} ({count})")
