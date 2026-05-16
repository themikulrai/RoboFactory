"""Map demo idx -> generating seed by matching cubeA init pose.

The planner.run() loops seeds and only saves successful demos, so traj_0 != seed_0.
This script iterates seeds, resets the env, and compares cubeA.t=0 position to the
demo's recorded env_states[0] cubeA position to identify the actual seed.
"""
import os
import sys
import argparse
import h5py
import yaml
import numpy as np

sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
os.chdir('/iris/u/mikulrai/projects/RoboFactory/robofactory')

import gymnasium as gym
import robofactory  # noqa: register envs


def get_cube_pos(env, name='cubeA'):
    """Read xyz position of the named actor from the unwrapped env."""
    base = env.unwrapped
    # Try the actor dict first
    if hasattr(base, 'actors') and name in base.actors:
        pose = base.actors[name].pose
    elif hasattr(base, name):
        pose = getattr(base, name).pose
    else:
        # Fall back: iterate all actors
        for k, v in base.actors.items() if hasattr(base, 'actors') else []:
            if k == name:
                pose = v.pose
                break
        else:
            return None
    p = pose.p
    if hasattr(p, 'cpu'):
        p = p.cpu().numpy()
    if p.ndim == 2:
        p = p[0]
    return np.asarray(p, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='/iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/TwoRobotsStackCube/TwoRobotsStackCube.h5')
    parser.add_argument('--config', default='configs/table/two_robots_stack_cube.yaml')
    parser.add_argument('--demos', nargs='+', type=int, default=[0])
    parser.add_argument('--max-seed', type=int, default=300)
    parser.add_argument('--tol', type=float, default=1e-3)
    args = parser.parse_args()

    # Load target cubeA positions
    targets = {}
    with h5py.File(args.h5, 'r') as f:
        for di in args.demos:
            key = f'traj_{di}'
            pos = np.asarray(f[key]['env_states']['actors']['cubeA'][0][:3], dtype=np.float64)
            targets[di] = pos
            print(f'  target traj_{di} cubeA t=0: {pos}')

    # Build env (mirror eval_multi_dp.py env construction)
    print('Constructing env...')
    with open(args.config, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    env_id = cfg_yaml['task_name'] + '-rf'
    env = gym.make(
        env_id,
        config=args.config,
        obs_mode='state',
        robot_uids=('panda_wristcam_multi', 'panda_wristcam_multi'),
        num_envs=1,
        sim_backend='cpu',
        enable_shadow=False,
    )

    matches = {}
    for s in range(args.max_seed):
        env.reset(seed=s)
        cp = get_cube_pos(env, 'cubeA')
        if cp is None:
            print(f'  could not read cubeA pos at seed {s}; aborting')
            break
        for di, tgt in targets.items():
            if np.allclose(cp[:3], tgt, atol=args.tol):
                matches[di] = s
                print(f'  MATCH: traj_{di} -> seed {s}  (cubeA pos {cp[:3]})')
        if len(matches) == len(targets):
            break

    env.close()
    print(f'\nFinal: {matches}')


if __name__ == '__main__':
    main()
