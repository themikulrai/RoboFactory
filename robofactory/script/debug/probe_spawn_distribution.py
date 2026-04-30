"""Quick scan: for a range of seeds, print meat xy + whether within training spawn range."""
import sys, os
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

import argparse, numpy as np, h5py
import gymnasium as gym
from robofactory.tasks import *

def meat_xyz(env):
    for a in env.unwrapped.scene.actors.values():
        if 'meat' in a.name.lower():
            p = a.pose.p
            if hasattr(p, 'cpu'): p = p.cpu().numpy()
            return np.asarray(p).flatten()[:3]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/table/pick_meat.yaml')
    ap.add_argument('--h5', default='/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(10000, 10016)))
    args = ap.parse_args()

    # Print training distribution
    print("=== TRAINING spawn distribution (from h5) ===")
    with h5py.File(args.h5, 'r') as f:
        n_traj = sum(1 for k in f if k.startswith('traj_'))
        train_xy = []
        for i in range(min(n_traj, 150)):
            # Can't read env_states (PM has none), but we can try to infer from
            # first frame content. Instead just report action stats.
            pass
        print(f"  n_traj={n_traj} (no env_states in PM h5; spawn pos not stored)")

    print("\n=== EVAL spawn distribution ===")
    env = gym.make('PickMeat-rf', config=args.config, obs_mode='state',
                   control_mode='pd_joint_pos', render_mode='sensors',
                   num_envs=1, sim_backend='cpu', enable_shadow=False,
                   sensor_configs=dict(shader_pack='default'))

    results = []
    for seed in args.seeds:
        env.reset(seed=seed)
        xy = meat_xyz(env)
        if xy is not None:
            results.append((seed, float(xy[0]), float(xy[1])))

    env.close()

    print(f"{'seed':>8}  {'meat_x':>8}  {'meat_y':>8}")
    print("-" * 30)
    for seed, x, y in results:
        print(f"{seed:>8}  {x:>8.4f}  {y:>8.4f}")

    xs = [r[1] for r in results]
    ys = [r[2] for r in results]
    print(f"\nRange: x=[{min(xs):.4f}, {max(xs):.4f}]  y=[{min(ys):.4f}, {max(ys):.4f}]")
    print(f"Mean: x={np.mean(xs):.4f}  y={np.mean(ys):.4f}")

if __name__ == '__main__':
    main()
