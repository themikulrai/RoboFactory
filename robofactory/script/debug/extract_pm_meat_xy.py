"""Extract PickMeat 'meat' actor xy at a fixed list of seeds.

Used as the positive-control label for the H2 encoder linear probe: the existing
PM probe NPZ saved features but no meat positions (probe_decent_dp_features.py
only dumps cubeA/B/C). This script builds the PM env, env.reset(seed) for each
seed in the audit set, and writes meat.pose.p[:3] to a small NPZ.
"""
import argparse
import sys
sys.path.append('./')
import numpy as np
import yaml
import gymnasium as gym

from robofactory.tasks import *  # noqa: registers env ids
import robofactory.agents  # noqa


def meat_xyz(env):
    for a in env.unwrapped.scene.actors.values():
        if a.name.lower() == 'meat':
            p = a.pose.p
            if hasattr(p, 'cpu'):
                p = p.cpu().numpy()
            return np.asarray(p).flatten()[:3]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    env = gym.make(
        env_id,
        config=args.config,
        obs_mode='rgb',
        control_mode='pd_joint_pos',
        render_mode='sensors',
        num_envs=1,
        sim_backend='cpu',
        enable_shadow=False,
        sensor_configs=dict(shader_pack='default'),
    )
    print(f'env_id={env_id}', flush=True)

    seeds = list(args.seeds)
    meat = np.full((len(seeds), 3), np.nan, dtype=np.float32)
    for si, s in enumerate(seeds):
        env.reset(seed=int(s))
        p = meat_xyz(env)
        if p is None:
            print(f'  seed {s}: meat actor NOT FOUND', flush=True)
            continue
        meat[si] = p
        print(f'  seed {s}: meat xyz = {p.round(4).tolist()}', flush=True)

    np.savez(args.out, seeds=np.asarray(seeds, dtype=np.int64), meat_xyz=meat)
    print(f'wrote {args.out}', flush=True)
    print(f'meat xy std across {len(seeds)} seeds: '
          f'x={np.nanstd(meat[:,0]):.4f}  y={np.nanstd(meat[:,1]):.4f}  z={np.nanstd(meat[:,2]):.4f}',
          flush=True)


if __name__ == '__main__':
    main()
