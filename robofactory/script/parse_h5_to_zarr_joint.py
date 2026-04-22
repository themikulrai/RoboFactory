"""
HDF5 -> joint zarr converter for centralised diffusion-policy training.

Input : data/h5_data/{TASK}-rf.h5 (the raw RoboFactory dump: obs/agent/panda-{i}/qpos,
        obs/sensor_data/head_camera_agent{i}/rgb, actions/panda-{i}).
Output: data/zarr_data/{TASK}-rf_joint_{N}.zarr with

    data/head_camera_0 .. head_camera_{N-1}   (T, 3, H, W) uint8
    data/state                                (T, 8*N)      float32  -- prev joint action
    data/action                               (T, 8*N)      float32
    meta/episode_ends                         (n_eps,)      int64

The "state" convention mirrors parse_pkl_to_zarr_dp.py: state[t] = action[t]
(current commanded joint-pos target). Each trajectory contributes T = len(actions)
steps; the leading obs frame is dropped to align obs[t] with action[t].
"""
import argparse
import os
import shutil

import h5py
import numpy as np
import zarr


def qpos_to_cmd8(qpos_arm: np.ndarray) -> np.ndarray:
    # qpos is (T, 9) = 7 arm + 2 finger. The policy speaks in 8d (7 arm + 1 gripper).
    # Existing code uses the *previous action* as state, so we do not actually need this
    # at dataset build time -- state is already 8d from actions/panda-{i}. This helper
    # is kept for completeness if you ever want to condition on proprio instead.
    arm = qpos_arm[:, :-2]
    gripper = np.ones((qpos_arm.shape[0], 1), dtype=arm.dtype)
    return np.concatenate([arm, gripper], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="e.g. TakePhoto-rf, LongPipelineDelivery-rf")
    parser.add_argument("--h5-dir", default="data/h5_data")
    parser.add_argument("--out-dir", default="data/zarr_data")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--load-num", type=int, default=150,
                        help="cap on number of trajectories to include")
    args = parser.parse_args()

    h5_path = os.path.join(args.h5_dir, f"{args.task}.h5")
    save_dir = os.path.join(args.out_dir, f"{args.task}_joint_{args.load_num}.zarr")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    print(f"[joint-converter] reading {h5_path}")
    print(f"[joint-converter] writing {save_dir} with n_agents={args.n_agents}")

    head_buffers = [[] for _ in range(args.n_agents)]
    action_buffer = []
    state_buffer = []
    episode_ends = []
    total = 0

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        traj_keys = traj_keys[: args.load_num]
        for ep_idx, tk in enumerate(traj_keys):
            tr = f[tk]
            # per-arm actions: (T, 8); obs cams: (T+1, H, W, 3). Drop last obs frame to align.
            acts = [np.asarray(tr[f"actions/panda-{i}"], dtype=np.float32)
                    for i in range(args.n_agents)]
            T = acts[0].shape[0]
            if not all(a.shape == (T, 8) for a in acts):
                raise ValueError(f"{tk}: action shapes mismatch {[a.shape for a in acts]}")

            joint_action = np.concatenate(acts, axis=1)  # (T, 8*N)
            action_buffer.append(joint_action)
            # state = current commanded action (same as existing per-arm pipeline)
            state_buffer.append(joint_action.copy())

            for i in range(args.n_agents):
                rgb = np.asarray(tr[f"obs/sensor_data/head_camera_agent{i}/rgb"])  # (T+1, H, W, 3)
                if rgb.shape[0] < T:
                    raise ValueError(f"{tk}: cam{i} has {rgb.shape[0]} frames, need >= {T}")
                rgb = rgb[:T]  # align to action timesteps
                rgb = np.moveaxis(rgb, -1, 1)  # (T, 3, H, W)
                head_buffers[i].append(rgb)

            total += T
            episode_ends.append(total)
            if (ep_idx + 1) % 10 == 0 or ep_idx == len(traj_keys) - 1:
                print(f"  ep {ep_idx + 1}/{len(traj_keys)}  total_steps={total}", flush=True)

    # concatenate episodes
    action_arr = np.concatenate(action_buffer, axis=0)          # (total, 8N)
    state_arr = np.concatenate(state_buffer, axis=0)            # (total, 8N)
    head_arrs = [np.concatenate(b, axis=0) for b in head_buffers]  # each (total, 3, H, W)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    root = zarr.group(save_dir)
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    for i, arr in enumerate(head_arrs):
        data.create_dataset(f"head_camera_{i}", data=arr,
                            chunks=(100, *arr.shape[1:]),
                            dtype=arr.dtype, overwrite=True, compressor=compressor)
    data.create_dataset("action", data=action_arr,
                        chunks=(100, action_arr.shape[1]),
                        dtype="float32", overwrite=True, compressor=compressor)
    data.create_dataset("state", data=state_arr,
                        chunks=(100, state_arr.shape[1]),
                        dtype="float32", overwrite=True, compressor=compressor)
    meta.create_dataset("episode_ends", data=episode_ends_arr,
                        dtype="int64", overwrite=True, compressor=compressor)

    print(f"[joint-converter] done. n_eps={len(episode_ends_arr)} "
          f"total_steps={total} action_dim={action_arr.shape[1]}")


if __name__ == "__main__":
    main()
