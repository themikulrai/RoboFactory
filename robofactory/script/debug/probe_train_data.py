"""T1 / T7 — Dump training-data contract from a paper-data h5.

Usage:
  python script/debug/probe_train_data.py \
    --h5 /iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5 --traj 0
  # or for TSC:
  python script/debug/probe_train_data.py \
    --h5 /iris/u/mikulrai/data/RoboFactory/hf_download/ThreeRobotsStackCube/ThreeRobotsStackCube.h5 \
    --traj 0 --multi-agent

Prints camera image stats + action stats per dim + initial qpos +
first env_state slice (object positions) + recorded episode_seed if any.
Saves first head-camera frame for visual T1 comparison.
"""
import argparse
import os
import h5py
import numpy as np
from PIL import Image


def _walk(h5g, prefix=''):
    out = []
    for k in h5g.keys():
        full = f"{prefix}/{k}" if prefix else k
        v = h5g[k]
        if isinstance(v, h5py.Dataset):
            out.append((full, v.shape, str(v.dtype)))
        else:
            out.extend(_walk(v, full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--traj', type=int, default=0)
    ap.add_argument('--multi-agent', action='store_true')
    ap.add_argument('--save-frame-dir', default='/tmp')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        traj_keys = sorted(f.keys())
        print(f"=== {args.h5} ===")
        print(f"#trajs: {len(traj_keys)}  first 5: {traj_keys[:5]}")
        tname = f'traj_{args.traj}'
        traj = f[tname]

        # 1) episode_seed (if recorded)
        attrs = dict(traj.attrs)
        print(f"\n--- traj_{args.traj} attrs ---")
        for k, v in attrs.items():
            print(f"  {k}: {v}")

        # 2) walk structure
        print(f"\n--- traj_{args.traj} datasets (top 30) ---")
        for path, shape, dtype in _walk(traj)[:30]:
            print(f"  {path:60s}  {shape}  {dtype}")

        # 3) camera image stats
        print("\n--- camera image stats (frame 0) ---")
        cam_paths = []
        if args.multi_agent:
            for i in range(3):
                cam_paths.append((f'panda-{i} head_cam', f'obs/sensor_data/head_camera_agent{i}/rgb'))
            cam_paths.append(('global head_cam', 'obs/sensor_data/head_camera_global/rgb'))
        else:
            cam_paths.append(('head_cam', 'obs/sensor_data/head_camera/rgb'))

        for label, path in cam_paths:
            if path not in traj:
                # alt locations
                for alt in [path.replace('obs/', ''),
                            path.replace('sensor_data/', 'sensors/')]:
                    if alt in traj:
                        path = alt
                        break
            if path not in traj:
                print(f"  {label}: <missing> ({path})")
                continue
            arr = traj[path]
            f0 = arr[0]
            print(f"  {label}: shape={arr.shape} dtype={arr.dtype}"
                  f" frame0_min={int(f0.min())} max={int(f0.max())}"
                  f" mean_per_channel={f0.reshape(-1, f0.shape[-1]).mean(axis=0).round(2).tolist()}")

            # save first frame for visual diff
            os.makedirs(args.save_frame_dir, exist_ok=True)
            out_png = os.path.join(args.save_frame_dir, f'train_frame_traj{args.traj}_{label.replace(" ", "_")}.png')
            Image.fromarray(f0.astype(np.uint8)).save(out_png)
            print(f"    saved → {out_png}")

        # 4) action stats
        print("\n--- action stats ---")
        action_keys = []
        if args.multi_agent:
            for i in range(3):
                k = f'actions/panda-{i}'
                if k in traj:
                    action_keys.append((f'panda-{i}', k))
        else:
            for k in ('actions', 'action'):
                if k in traj:
                    action_keys.append(('action', k)); break
            if not action_keys:
                # try sub-keyed
                if 'actions' in traj:
                    for sk in traj['actions'].keys():
                        action_keys.append((f'actions/{sk}', f'actions/{sk}'))

        for label, k in action_keys:
            a = traj[k][:]
            print(f"  {label}: shape={a.shape} dtype={a.dtype}")
            for d in range(a.shape[-1]):
                col = a[..., d].reshape(-1)
                print(f"    dim {d}: min={col.min():.4f} max={col.max():.4f} "
                      f"mean={col.mean():.4f} std={col.std():.4f}")
            print(f"    action[0]: {a[0].round(4).tolist()}")

        # 5) qpos at t=0
        print("\n--- proprio at t=0 ---")
        qpos_paths = []
        if args.multi_agent:
            for i in range(3):
                qpos_paths.append((f'panda-{i}', f'obs/agent/panda-{i}/qpos'))
        else:
            qpos_paths.append(('agent', 'obs/agent/qpos'))
        for label, p in qpos_paths:
            if p in traj:
                q0 = traj[p][0]
                if q0.ndim > 1:
                    q0 = q0.squeeze()
                print(f"  {label}: shape={traj[p].shape} qpos[0]={q0.round(4).tolist()}")
            else:
                print(f"  {label}: <missing> ({p})")

        # 6) env_state at t=0 (object spawn)
        if 'env_states' in traj:
            es = traj['env_states'][0]
            print(f"\n--- env_states[0] shape={traj['env_states'].shape} (first 30 floats) ---")
            print(f"  {np.asarray(es).flatten()[:30].round(4).tolist()}")
        elif 'env_init_state' in traj:
            print(f"\n--- env_init_state shape={traj['env_init_state'].shape} ---")
            print(f"  {np.asarray(traj['env_init_state']).flatten()[:30].round(4).tolist()}")
        else:
            print("\n--- no env_states / env_init_state recorded ---")


if __name__ == '__main__':
    main()
