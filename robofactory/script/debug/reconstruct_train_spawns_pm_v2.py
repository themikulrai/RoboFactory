"""T14 v2: Reconstruct training meat xy positions from h5 tcp_pose at grasp time.

Idea: PickMeat trajectories end with the gripper holding the meat above the table.
The tcp_pose just before the lift (e.g., when tcp z first reaches its minimum)
is approximately the meat (x, y) location. This is far more robust than image
reconstruction.

Algorithm: for each traj, find the step with the LOWEST tcp z value
(grasp-down moment). Record (tcp_x, tcp_y) at that step.

Usage:
    python script/debug/reconstruct_train_spawns_pm_v2.py
"""
import sys
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')

import numpy as np
import h5py
import json

H5_PATH   = '/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5'
JSON_PATH = '/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.json'

with open(JSON_PATH) as f:
    jd = json.load(f)
episode_seeds = [ep['episode_seed'] for ep in jd['episodes']]
print(f"PM training: {len(episode_seeds)} episodes; seeds[:10] = {episode_seeds[:10]}")

train_xy = []
print(f"\n{'seed':>6}  {'meat_x':>8}  {'meat_y':>8}  {'meat_z':>8}  {'grasp_step':>10}  {'success':>8}")
print("-" * 60)

with h5py.File(H5_PATH, 'r') as f:
    for ep_idx, seed in enumerate(episode_seeds):
        traj = f[f'traj_{ep_idx}']
        tcp = np.array(traj['obs/extra/tcp_pose'])  # (T, 7)
        success = np.array(traj['success'])         # (T-1,) per-step success bool
        # Find the step where tcp reaches minimum z (grasp moment)
        # Restrict to first half to avoid lift phase confusion
        T = tcp.shape[0]
        first_half = tcp[:T//2 + 5]
        z = first_half[:, 2]
        grasp_idx = int(np.argmin(z))
        meat_xy = tcp[grasp_idx, :3]  # (x, y, z) at grasp
        train_xy.append((float(meat_xy[0]), float(meat_xy[1])))
        succ = bool(success[-1]) if len(success) > 0 else False
        if ep_idx < 30:
            print(f"{seed:>6}  {meat_xy[0]:>8.4f}  {meat_xy[1]:>8.4f}  {meat_xy[2]:>8.4f}  {grasp_idx:>10d}  {str(succ):>8}")

xs = np.array([p[0] for p in train_xy])
ys = np.array([p[1] for p in train_xy])
print(f"\nTRAINING meat-grasp xy summary (n={len(train_xy)}):")
print(f"  x: range=[{xs.min():.4f}, {xs.max():.4f}]  mean={xs.mean():.4f}  std={xs.std():.4f}")
print(f"  y: range=[{ys.min():.4f}, {ys.max():.4f}]  mean={ys.mean():.4f}  std={ys.std():.4f}")

# Compare with eval success centroid
success_centroid = np.array([0.063, 0.041])
print(f"\nEval success centroid:  x={success_centroid[0]:.4f}  y={success_centroid[1]:.4f}")
print(f"Offset (train mean - centroid): dx={xs.mean()-success_centroid[0]:.4f}  dy={ys.mean()-success_centroid[1]:.4f}")

# Fraction of training positions within distance d of centroid
for d in [0.033, 0.050, 0.080, 0.120]:
    dists = np.sqrt((xs - success_centroid[0])**2 + (ys - success_centroid[1])**2)
    frac = float((dists < d).mean())
    print(f"  Training within {d*100:.1f}cm of centroid: {frac:.1%} ({int((dists<d).sum())}/{len(xs)})")

print("\nTraining x distribution (4cm buckets):")
for lo in np.arange(-0.10, 0.20, 0.04):
    count = int(((xs >= lo) & (xs < lo+0.04)).sum())
    bar = '*' * min(count, 50)
    print(f"  x=[{lo:.2f},{lo+0.04:.2f}): {bar} ({count})")
print("\nTraining y distribution (4cm buckets):")
for lo in np.arange(-0.10, 0.20, 0.04):
    count = int(((ys >= lo) & (ys < lo+0.04)).sum())
    bar = '*' * min(count, 50)
    print(f"  y=[{lo:.2f},{lo+0.04:.2f}): {bar} ({count})")

# 2D scatter (text-based)
print("\n2D heatmap (rows=y, cols=x, each cell=count of training positions in 2cm bin):")
xbin_lo, xbin_hi, ybin_lo, ybin_hi, dx = -0.10, 0.20, -0.10, 0.20, 0.02
nx = int((xbin_hi - xbin_lo) / dx) + 1
ny = int((ybin_hi - ybin_lo) / dx) + 1
grid = np.zeros((ny, nx), dtype=int)
for x, y in zip(xs, ys):
    ix = int((x - xbin_lo) / dx)
    iy = int((y - ybin_lo) / dx)
    if 0 <= ix < nx and 0 <= iy < ny:
        grid[iy, ix] += 1
# Print column header
print("    y\\x", "  ".join(f"{xbin_lo + i*dx:+.2f}" for i in range(nx)))
for j in range(ny):
    row_label = f"{ybin_lo + j*dx:+.2f}"
    cells = "  ".join(f"{grid[j,i]:>5d}" for i in range(nx))
    print(f"  {row_label}", cells)

# Mark eval success centroid bin
ix_c = int((success_centroid[0] - xbin_lo) / dx)
iy_c = int((success_centroid[1] - ybin_lo) / dx)
print(f"\nEval success centroid bin: (col={ix_c} -> x={xbin_lo+ix_c*dx:.2f}, row={iy_c} -> y={ybin_lo+iy_c*dx:.2f})  count={grid[iy_c,ix_c]}")
