"""T8 — Replay h5 actions on the eval env. Does the env succeed when fed paper actions?

If success: the env is sound; bug is in the policy/obs-conditioning side (T1-T5).
If failure: env has drifted from data-recording env (different mani_skill / physics / scene).

Usage:
  python script/debug/replay_h5.py \
    --h5 /iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5 \
    --traj 0 1 2 \
    --config configs/table/pick_meat.yaml
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import h5py
import yaml
import numpy as np
import gymnasium as gym

from robofactory.tasks import *  # register envs


def _to_np(x):
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x).squeeze()


def _meat_xy(env):
    actors = getattr(env.unwrapped, 'actors', None) or {}
    if isinstance(actors, dict):
        for name, actor in actors.items():
            if 'meat' in name.lower():
                p = _to_np(actor.pose.p)
                return float(p[0]), float(p[1]), float(p[2])
    try:
        for a in env.unwrapped.scene.actors.values():
            if 'meat' in a.name.lower():
                p = _to_np(a.pose.p)
                return float(p[0]), float(p[1]), float(p[2])
    except Exception:
        pass
    return None


def _tcp_xy(env):
    try:
        tcp = env.unwrapped.agent.tcp
        p = _to_np(tcp.pose.p)
        return float(p[0]), float(p[1]), float(p[2])
    except Exception:
        return None


def replay_one(env, traj_id, h5_path, sidecar_seed=None):
    with h5py.File(h5_path, 'r') as f:
        traj = f[f'traj_{traj_id}']
        actions = traj['actions'][:]              # (T-1, 8)
        seed = sidecar_seed if sidecar_seed is not None else int(traj.attrs.get('episode_seed', traj_id))
        gt_success = bool(traj['success'][-1])
        ep_len = int(actions.shape[0])

    obs, _ = env.reset(seed=seed)
    meat0 = _meat_xy(env)
    tcp0 = _tcp_xy(env)
    success = False
    fail_step = None
    tcp_grasp = None  # tcp xyz at the step paper data first commands gripper close
    grasp_step = None
    last_meat = meat0
    for t in range(ep_len):
        a = np.asarray(actions[t]).astype(np.float32)
        # detect first close-command in paper actions
        if grasp_step is None and a[-1] < 0.0:
            grasp_step = t
            tcp_grasp = _tcp_xy(env)
        obs, reward, terminated, truncated, info = env.step(a)
        last_meat = _meat_xy(env)
        succ = info.get('success', False)
        if hasattr(succ, 'item'):
            succ = succ.item()
        elif hasattr(succ, '__len__'):
            succ = bool(np.asarray(succ).any())
        if succ:
            success = True
            fail_step = t
            break
    tcp_end = _tcp_xy(env)

    # dist (meat at reset) vs (tcp at the moment paper closes gripper)
    grasp_miss = None
    if meat0 is not None and tcp_grasp is not None:
        dx, dy = tcp_grasp[0] - meat0[0], tcp_grasp[1] - meat0[1]
        grasp_miss = (dx * dx + dy * dy) ** 0.5

    return dict(
        traj=traj_id, seed=seed, ep_len=ep_len, h5_says_success=gt_success,
        replay_success=int(success), step_succeeded=fail_step,
        meat_xy=meat0, tcp_xyz0=tcp0, grasp_step=grasp_step,
        tcp_xyz_at_grasp=tcp_grasp, grasp_miss=grasp_miss,
        tcp_xyz_end=tcp_end, meat_xy_end=last_meat,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--traj', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--obs-mode', default='state')
    ap.add_argument('--control-mode', default='pd_joint_pos')
    ap.add_argument('--sim-backend', default='cpu')
    ap.add_argument('--enable-shadow', action='store_true', default=False)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'
    env = gym.make(env_id, config=args.config, obs_mode=args.obs_mode,
                   control_mode=args.control_mode, render_mode='sensors',
                   num_envs=1, sim_backend=args.sim_backend, enable_shadow=args.enable_shadow)

    # parse sidecar if present
    import json, os as _os
    sidecar_seeds = {}
    sc = args.h5.replace('.h5', '.json')
    if _os.path.exists(sc):
        with open(sc) as f:
            sd = json.load(f)
        for ep in sd.get('episodes', []):
            sidecar_seeds[ep['episode_id']] = ep.get('episode_seed', ep.get('reset_kwargs', {}).get('seed'))

    rows = []
    for tid in args.traj:
        r = replay_one(env, tid, args.h5, sidecar_seed=sidecar_seeds.get(tid))
        print(f"traj={tid:3d} seed={r['seed']:5d} ep_len={r['ep_len']:3d} "
              f"replay_success={r['replay_success']} step_succ={r['step_succeeded']}")
        print(f"    meat0_xy={r['meat_xy']}  tcp0_xyz={r['tcp_xyz0']}")
        print(f"    grasp_step={r['grasp_step']}  tcp@grasp={r['tcp_xyz_at_grasp']}  "
              f"grasp_miss={r['grasp_miss']:.4f}m" if r['grasp_miss'] is not None else
              f"    grasp_step={r['grasp_step']}  (no grasp commanded)")
        print(f"    tcp_end={r['tcp_xyz_end']}  meat_end_xy={r['meat_xy_end']}")
        rows.append(r)

    env.close()
    n = len(rows)
    nsucc = sum(r['replay_success'] for r in rows)
    print(f"\n=== T8 summary: {nsucc}/{n} replays succeeded ===")
    if nsucc == 0:
        print("DECISION: env has drifted from data-recording env. Mode A is *also* env-side.")
    elif nsucc < n:
        print("DECISION: partial success. Some seed-conventions match; investigate per-traj.")
    else:
        print("DECISION: env is sound. Mode A is purely on the policy/obs side (T1-T5).")


if __name__ == '__main__':
    main()
