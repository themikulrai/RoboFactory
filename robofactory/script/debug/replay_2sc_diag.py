"""Diagnostic replay: dumps cube positions + TCP trajectory mismatch.

Hypothesis under test: at env.reset(seed=17), are the cube positions the same as
the recording-time placement? And do the arms actually follow the recorded TCP
trajectory when fed the recorded actions, or do they diverge over time?

Outputs JSON with:
- cubes_at_reset_run1 / run2: cube xyz immediately after env.reset(seed=17) in
  two back-to-back resets (in the SAME process). If these differ, scene seeding
  is non-deterministic.
- tcp_diff_per_step: per-step L2 between recorded obs/extra/{left,right}_arm_tcp
  and live env TCP, both arms.
- cubes_at_end: cube xyz after the full 254-step replay.
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

from robofactory.tasks import *  # register envs


def _to_np(x):
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x).squeeze()


def _actor_xyz(env, name_match):
    try:
        for a in env.unwrapped.scene.actors.values():
            if name_match.lower() in a.name.lower():
                return _to_np(a.pose.p).tolist()
    except Exception:
        pass
    return None


def _tcp(env, arm_idx):
    try:
        ag = env.unwrapped.agent
        agents = ag.agents if hasattr(ag, 'agents') else [ag]
        if arm_idx >= len(agents):
            return None
        return _to_np(agents[arm_idx].tcp.pose.p).tolist()
    except Exception:
        return None


def reset_and_get_cubes(env, seed):
    env.reset(seed=seed)
    return dict(
        cubeA=_actor_xyz(env, 'cubeA'),
        cubeB=_actor_xyz(env, 'cubeB'),
        goal=_actor_xyz(env, 'goal'),
        tcp_left=_tcp(env, 0),
        tcp_right=_tcp(env, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--traj', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--sim-backend', default='gpu')
    ap.add_argument('--robot-uids', default='panda_wristcam_multi,panda_wristcam_multi')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'
    robot_uids = tuple(args.robot_uids.split(','))
    env = gym.make(env_id, config=args.config, obs_mode='rgb',
                   control_mode='pd_joint_pos', render_mode='sensors',
                   num_envs=1, sim_backend=args.sim_backend, enable_shadow=False,
                   sensor_configs=dict(shader_pack='default'),
                   robot_uids=robot_uids)

    # 1) Two back-to-back resets with the same seed
    cubes_run1 = reset_and_get_cubes(env, args.seed)
    cubes_run2 = reset_and_get_cubes(env, args.seed)
    print('reset 1:', json.dumps(cubes_run1, indent=2), flush=True)
    print('reset 2:', json.dumps(cubes_run2, indent=2), flush=True)

    # 2) Replay with recorded actions, compare TCP to recorded TCP
    with h5py.File(args.h5, 'r') as f:
        tr = f[f'traj_{args.traj}']
        p0 = np.asarray(tr['actions']['panda-0'], dtype=np.float32)  # (T, 8)
        p1 = np.asarray(tr['actions']['panda-1'], dtype=np.float32)
        T = min(p0.shape[0], p1.shape[0])
        rec_left = np.asarray(tr['obs']['extra']['left_arm_tcp'], dtype=np.float32)   # (T+1, 7) [xyz qwxyz]
        rec_right = np.asarray(tr['obs']['extra']['right_arm_tcp'], dtype=np.float32)
        h5_success = bool(np.asarray(tr['success'])[-1])

    env.reset(seed=args.seed)
    cubes_at_start = dict(
        cubeA=_actor_xyz(env, 'cubeA'),
        cubeB=_actor_xyz(env, 'cubeB'),
        goal=_actor_xyz(env, 'goal'),
    )
    tcp_diffs = []   # list of dicts per step
    cubes_per_step = []  # to track if cube drifts
    succ = False
    first_succ_step = None
    for t in range(T):
        env.step({'panda-0': p0[t], 'panda-1': p1[t]})
        live_left = _tcp(env, 0)
        live_right = _tcp(env, 1)
        rec_l_xyz = rec_left[t + 1, :3].tolist() if rec_left.shape[1] >= 3 else None
        rec_r_xyz = rec_right[t + 1, :3].tolist() if rec_right.shape[1] >= 3 else None
        d_l = float(np.linalg.norm(np.array(live_left) - np.array(rec_l_xyz))) if (live_left and rec_l_xyz) else None
        d_r = float(np.linalg.norm(np.array(live_right) - np.array(rec_r_xyz))) if (live_right and rec_r_xyz) else None
        tcp_diffs.append({'step': t, 'd_left_m': d_l, 'd_right_m': d_r})
        if t in (0, 10, 30, 60, 100, 150, 200, T - 1):
            cubes_per_step.append({'step': t,
                                   'cubeA': _actor_xyz(env, 'cubeA'),
                                   'cubeB': _actor_xyz(env, 'cubeB')})

    cubes_at_end = dict(
        cubeA=_actor_xyz(env, 'cubeA'),
        cubeB=_actor_xyz(env, 'cubeB'),
        goal=_actor_xyz(env, 'goal'),
    )

    # Summary stats on TCP drift
    arr_l = np.array([r['d_left_m'] for r in tcp_diffs if r['d_left_m'] is not None])
    arr_r = np.array([r['d_right_m'] for r in tcp_diffs if r['d_right_m'] is not None])
    summary = dict(
        h5=args.h5, traj=args.traj, seed=int(args.seed),
        ep_len=T, h5_success=h5_success,
        cubes_at_reset_run1=cubes_run1,
        cubes_at_reset_run2=cubes_run2,
        cubes_at_start_replay=cubes_at_start,
        cubes_at_end=cubes_at_end,
        cubes_per_step=cubes_per_step,
        tcp_drift_left_m=dict(mean=float(arr_l.mean()), max=float(arr_l.max()),
                              p50=float(np.median(arr_l)), p95=float(np.percentile(arr_l, 95))),
        tcp_drift_right_m=dict(mean=float(arr_r.mean()), max=float(arr_r.max()),
                               p50=float(np.median(arr_r)), p95=float(np.percentile(arr_r, 95))),
        sim_backend=args.sim_backend,
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    # also a slim CSV of per-step drift for plotting
    csv_path = args.out_json.replace('.json', '_tcp_diffs.csv')
    with open(csv_path, 'w') as f:
        f.write('step,d_left_m,d_right_m\n')
        for r in tcp_diffs:
            f.write(f"{r['step']},{r['d_left_m']},{r['d_right_m']}\n")

    print('\n=== summary ===')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cubes_per_step'}, indent=2))
    env.close()


if __name__ == '__main__':
    main()
