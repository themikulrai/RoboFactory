"""
Unified HDF5 -> zarr converter for diffusion-policy training.

Produces either:
  * joint zarr   (centralised; one policy over all N arms)
  * per-agent zarr (decentralised; one policy per arm)

from either dataset family:
  * workspace cams: obs/sensor_data/head_camera_agent{i}/rgb        (D1; also in D2)
  * wristcam      : obs/sensor_data/hand_camera_{i}/rgb             (D2)

Optionally appends a single global camera (obs/sensor_data/head_camera_global/rgb).
All RGB is resized (OpenCV area interp) to 224x224 uint8.

Output zarr schema
------------------
joint mode:
    data/head_camera_0 ... head_camera_{N-1}   (T, 3, 224, 224) uint8
    data/head_camera_global (optional)          (T, 3, 224, 224) uint8
    data/state                                  (T, 8*N) float32  (=action[t])
    data/action                                 (T, 8*N) float32
    meta/episode_ends                           (n_eps,) int64

per_agent mode (matches legacy RobotImageDataset schema + optional global):
    data/head_camera                            (T, 3, 224, 224) uint8
    data/head_camera_global (optional)          (T, 3, 224, 224) uint8
    data/state, data/action, data/tcp_action    (T, 8) float32
    meta/episode_ends                           (n_eps,) int64
"""
import argparse
import os
import shutil

import cv2
import h5py
import numpy as np
import zarr


CAM_KEY = {
    "workspace": "head_camera_agent{i}",
    "wristcam":  "hand_camera_{i}",
}


def _resize_batch(rgb_thwc: np.ndarray, size: int) -> np.ndarray:
    """Resize (T, H, W, 3) -> (T, 3, size, size) uint8."""
    T, H, W, _ = rgb_thwc.shape
    if H == size and W == size:
        out = rgb_thwc
    else:
        out = np.empty((T, size, size, 3), dtype=np.uint8)
        for t in range(T):
            out[t] = cv2.resize(rgb_thwc[t], (size, size), interpolation=cv2.INTER_AREA)
    return np.moveaxis(out, -1, 1).astype(np.uint8, copy=False)


def _parse_joint(f, n_agents, cam_family, include_global, resize, load_num):
    traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:load_num]
    per_cam = [[] for _ in range(n_agents)]
    glob = []
    actions = []
    ep_ends = []
    total = 0
    tpl = CAM_KEY[cam_family]
    for ep_idx, tk in enumerate(traj_keys):
        tr = f[tk]
        acts = [np.asarray(tr[f"actions/panda-{i}"], dtype=np.float32) for i in range(n_agents)]
        T = acts[0].shape[0]
        if not all(a.shape == (T, 8) for a in acts):
            raise ValueError(f"{tk}: action shapes mismatch {[a.shape for a in acts]}")
        joint = np.concatenate(acts, axis=1)  # (T, 8N)
        actions.append(joint)
        for i in range(n_agents):
            rgb = np.asarray(tr[f"obs/sensor_data/{tpl.format(i=i)}/rgb"])
            if rgb.shape[0] < T:
                raise ValueError(f"{tk}: cam{i} has {rgb.shape[0]} frames, need >= {T}")
            per_cam[i].append(_resize_batch(rgb[:T], resize))
        if include_global:
            gr = np.asarray(tr["obs/sensor_data/head_camera_global/rgb"])
            glob.append(_resize_batch(gr[:T], resize))
        total += T
        ep_ends.append(total)
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(traj_keys) - 1:
            print(f"  ep {ep_idx+1}/{len(traj_keys)}  total_steps={total}", flush=True)

    action_arr = np.concatenate(actions, axis=0)
    state_arr = action_arr.copy()
    cam_arrs = [np.concatenate(c, axis=0) for c in per_cam]
    glob_arr = np.concatenate(glob, axis=0) if include_global else None
    return {
        "cams": cam_arrs, "global": glob_arr,
        "state": state_arr, "action": action_arr,
        "episode_ends": np.asarray(ep_ends, dtype=np.int64),
        "mode": "joint",
    }


def _parse_per_agent(f, agent_id, cam_family, include_global, resize, load_num):
    traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:load_num]
    head_chunks = []
    glob_chunks = []
    act_chunks = []
    ep_ends = []
    total = 0
    cam_key = CAM_KEY[cam_family].format(i=agent_id)
    for ep_idx, tk in enumerate(traj_keys):
        tr = f[tk]
        act = np.asarray(tr[f"actions/panda-{agent_id}"], dtype=np.float32)
        T = act.shape[0]
        rgb = np.asarray(tr[f"obs/sensor_data/{cam_key}/rgb"])
        if rgb.shape[0] < T:
            raise ValueError(f"{tk}: cam {cam_key} has {rgb.shape[0]} frames, need >= {T}")
        head_chunks.append(_resize_batch(rgb[:T], resize))
        act_chunks.append(act)
        if include_global:
            gr = np.asarray(tr["obs/sensor_data/head_camera_global/rgb"])
            glob_chunks.append(_resize_batch(gr[:T], resize))
        total += T
        ep_ends.append(total)
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(traj_keys) - 1:
            print(f"  ep {ep_idx+1}/{len(traj_keys)}  total_steps={total}", flush=True)

    head_arr = np.concatenate(head_chunks, axis=0)
    glob_arr = np.concatenate(glob_chunks, axis=0) if include_global else None
    action_arr = np.concatenate(act_chunks, axis=0)
    state_arr = action_arr.copy()
    return {
        "head": head_arr, "global": glob_arr,
        "state": state_arr, "action": action_arr, "tcp_action": action_arr.copy(),
        "episode_ends": np.asarray(ep_ends, dtype=np.int64),
        "mode": "per_agent",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-path", required=True)
    ap.add_argument("--out-zarr", required=True)
    ap.add_argument("--mode", choices=["joint", "per_agent"], required=True)
    ap.add_argument("--n-agents", type=int, default=None,
                    help="required for --mode joint")
    ap.add_argument("--agent-id", type=int, default=None,
                    help="required for --mode per_agent")
    ap.add_argument("--camera-family", choices=["workspace", "wristcam"], required=True)
    ap.add_argument("--include-global", action="store_true")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--load-num", type=int, default=150)
    args = ap.parse_args()

    if args.mode == "joint" and args.n_agents is None:
        ap.error("--n-agents is required for --mode joint")
    if args.mode == "per_agent" and args.agent_id is None:
        ap.error("--agent-id is required for --mode per_agent")

    if os.path.exists(args.out_zarr):
        shutil.rmtree(args.out_zarr)

    print(f"[unified-converter] reading {args.h5_path}")
    print(f"[unified-converter] mode={args.mode} family={args.camera_family} "
          f"global={args.include_global} resize={args.resize}")
    print(f"[unified-converter] writing {args.out_zarr}")

    with h5py.File(args.h5_path, "r") as f:
        if args.mode == "joint":
            out = _parse_joint(f, args.n_agents, args.camera_family,
                               args.include_global, args.resize, args.load_num)
        else:
            out = _parse_per_agent(f, args.agent_id, args.camera_family,
                                   args.include_global, args.resize, args.load_num)

    root = zarr.group(args.out_zarr)
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    def _save(name, arr, chunk_first=100):
        data.create_dataset(name, data=arr,
                            chunks=(chunk_first, *arr.shape[1:]),
                            dtype=arr.dtype, overwrite=True, compressor=compressor)

    if out["mode"] == "joint":
        for i, arr in enumerate(out["cams"]):
            _save(f"head_camera_{i}", arr)
        if out["global"] is not None:
            _save("head_camera_global", out["global"])
    else:
        _save("head_camera", out["head"])
        if out["global"] is not None:
            _save("head_camera_global", out["global"])
        _save("tcp_action", out["tcp_action"])

    _save("action", out["action"])
    _save("state", out["state"])
    meta.create_dataset("episode_ends", data=out["episode_ends"],
                        dtype="int64", overwrite=True, compressor=compressor)

    total = int(out["episode_ends"][-1]) if len(out["episode_ends"]) else 0
    print(f"[unified-converter] done. n_eps={len(out['episode_ends'])} "
          f"total_steps={total} action_dim={out['action'].shape[1]}")


if __name__ == "__main__":
    main()
