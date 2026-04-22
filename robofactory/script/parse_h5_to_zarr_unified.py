"""
Unified HDF5 -> zarr converter for diffusion-policy training.

Streams episode data directly into zarr arrays so peak RAM stays bounded by
one episode (~150 MB for LP 4-arm+global). This lets multiple parsers run in
parallel on a 64 GB login node without OOM.

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

COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)


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


def _measure_total_steps(f, traj_keys, n_agents):
    total = 0
    for tk in traj_keys:
        T = f[tk][f"actions/panda-{0}"].shape[0]
        total += T
    return total


def _create_zarr(out_path, schema, total_steps, resize):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    root = zarr.group(out_path)
    data = root.create_group("data")
    meta = root.create_group("meta")

    def _cam(name):
        data.create_dataset(
            name, shape=(total_steps, 3, resize, resize), dtype="uint8",
            chunks=(100, 3, resize, resize), compressor=COMPRESSOR, overwrite=True,
        )

    if schema["mode"] == "joint":
        N = schema["n_agents"]
        for i in range(N):
            _cam(f"head_camera_{i}")
        if schema["include_global"]:
            _cam("head_camera_global")
        dim = 8 * N
        data.create_dataset("action", shape=(total_steps, dim), dtype="float32",
                            chunks=(100, dim), compressor=COMPRESSOR, overwrite=True)
        data.create_dataset("state",  shape=(total_steps, dim), dtype="float32",
                            chunks=(100, dim), compressor=COMPRESSOR, overwrite=True)
    else:
        _cam("head_camera")
        if schema["include_global"]:
            _cam("head_camera_global")
        data.create_dataset("action", shape=(total_steps, 8), dtype="float32",
                            chunks=(100, 8), compressor=COMPRESSOR, overwrite=True)
        data.create_dataset("state",  shape=(total_steps, 8), dtype="float32",
                            chunks=(100, 8), compressor=COMPRESSOR, overwrite=True)
        data.create_dataset("tcp_action", shape=(total_steps, 8), dtype="float32",
                            chunks=(100, 8), compressor=COMPRESSOR, overwrite=True)
    return root, data, meta


def _stream_joint(f, traj_keys, data, n_agents, cam_family, include_global, resize, total_steps):
    tpl = CAM_KEY[cam_family]
    cursor = 0
    episode_ends = []
    for ep_idx, tk in enumerate(traj_keys):
        tr = f[tk]
        acts = [np.asarray(tr[f"actions/panda-{i}"], dtype=np.float32) for i in range(n_agents)]
        T = acts[0].shape[0]
        if not all(a.shape == (T, 8) for a in acts):
            raise ValueError(f"{tk}: action shapes {[a.shape for a in acts]}")
        joint = np.concatenate(acts, axis=1)  # (T, 8N)
        sl = slice(cursor, cursor + T)
        data["action"][sl] = joint
        data["state"][sl] = joint
        for i in range(n_agents):
            rgb = np.asarray(tr[f"obs/sensor_data/{tpl.format(i=i)}/rgb"])
            if rgb.shape[0] < T:
                raise ValueError(f"{tk}: cam{i} {rgb.shape[0]} < T={T}")
            data[f"head_camera_{i}"][sl] = _resize_batch(rgb[:T], resize)
        if include_global:
            gr = np.asarray(tr["obs/sensor_data/head_camera_global/rgb"])
            data["head_camera_global"][sl] = _resize_batch(gr[:T], resize)
        cursor += T
        episode_ends.append(cursor)
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(traj_keys) - 1:
            print(f"  ep {ep_idx+1}/{len(traj_keys)}  total_steps={cursor}", flush=True)
    assert cursor == total_steps, f"cursor {cursor} != total {total_steps}"
    return np.asarray(episode_ends, dtype=np.int64)


def _stream_per_agent(f, traj_keys, data, agent_id, cam_family, include_global, resize, total_steps):
    tpl = CAM_KEY[cam_family]
    cam_key = tpl.format(i=agent_id)
    cursor = 0
    episode_ends = []
    for ep_idx, tk in enumerate(traj_keys):
        tr = f[tk]
        act = np.asarray(tr[f"actions/panda-{agent_id}"], dtype=np.float32)
        T = act.shape[0]
        rgb = np.asarray(tr[f"obs/sensor_data/{cam_key}/rgb"])
        if rgb.shape[0] < T:
            raise ValueError(f"{tk}: cam {cam_key} {rgb.shape[0]} < T={T}")
        sl = slice(cursor, cursor + T)
        data["action"][sl] = act
        data["state"][sl] = act
        data["tcp_action"][sl] = act
        data["head_camera"][sl] = _resize_batch(rgb[:T], resize)
        if include_global:
            gr = np.asarray(tr["obs/sensor_data/head_camera_global/rgb"])
            data["head_camera_global"][sl] = _resize_batch(gr[:T], resize)
        cursor += T
        episode_ends.append(cursor)
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(traj_keys) - 1:
            print(f"  ep {ep_idx+1}/{len(traj_keys)}  total_steps={cursor}", flush=True)
    assert cursor == total_steps, f"cursor {cursor} != total {total_steps}"
    return np.asarray(episode_ends, dtype=np.int64)


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

    print(f"[unified-converter] reading {args.h5_path}")
    print(f"[unified-converter] mode={args.mode} family={args.camera_family} "
          f"global={args.include_global} resize={args.resize}")
    print(f"[unified-converter] writing {args.out_zarr}")

    schema = {
        "mode": args.mode,
        "n_agents": args.n_agents,
        "include_global": args.include_global,
    }

    with h5py.File(args.h5_path, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:args.load_num]
        probe_n = args.n_agents if args.mode == "joint" else 1
        total_steps = _measure_total_steps(f, traj_keys, probe_n)
        print(f"[unified-converter] total_steps={total_steps} over {len(traj_keys)} eps")

        root, data, meta = _create_zarr(args.out_zarr, schema, total_steps, args.resize)

        if args.mode == "joint":
            ep_ends = _stream_joint(
                f, traj_keys, data, args.n_agents, args.camera_family,
                args.include_global, args.resize, total_steps)
        else:
            ep_ends = _stream_per_agent(
                f, traj_keys, data, args.agent_id, args.camera_family,
                args.include_global, args.resize, total_steps)

    meta.create_dataset("episode_ends", data=ep_ends, dtype="int64",
                        chunks=(len(ep_ends),), compressor=COMPRESSOR, overwrite=True)

    action_dim = 8 * args.n_agents if args.mode == "joint" else 8
    print(f"[unified-converter] done. n_eps={len(ep_ends)} "
          f"total_steps={total_steps} action_dim={action_dim}")


if __name__ == "__main__":
    main()
