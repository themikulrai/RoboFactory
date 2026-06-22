"""Probe DP overfit-ep0 ckpts: replay-error at t∈{0,24,49,73,97} + determinism (3x call).

Compares against Pi0.5 probe-A signature so we can settle whether the 0.003 rad floor
is flow-matching-specific or universal.

Run from /iris/u/mikulrai/projects/RoboFactory/robofactory/ with the RoboFactory env.
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, "./policy/Diffusion-Policy")
from eval_multi_dp import get_policy
from diffusion_policy.env_runner.dp_runner import DPRunner

H5 = "/iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/LiftBarrier/LiftBarrier.h5"
CKPT = {
    0: "/iris/u/mikulrai/checkpoints/RoboFactory/3aa8b94ca354/5000.ckpt",
    1: "/iris/u/mikulrai/checkpoints/RoboFactory/c0917d6d7836/5000.ckpt",
}
TIMESTEPS = [0, 24, 49, 73, 97]
N_OBS = 3


def _gripper(q):
    return float((q[7] + q[8]) / 2.0)


def load_ep0():
    """Return dicts of per-arm timeseries: state[T,8], action[T,8], wrist_hwc[T,H,W,3], global_hwc[T,H,W,3]."""
    with h5py.File(H5, "r") as f:
        tr = f["traj_0"]
        out = {}
        q0 = np.asarray(tr["obs/agent/panda_wristcam_multi-0/qpos"])
        q1 = np.asarray(tr["obs/agent/panda_wristcam_multi-1/qpos"])
        for arm, q in [(0, q0), (1, q1)]:
            grip = ((q[:, 7] + q[:, 8]) / 2.0).astype(np.float32)
            out[arm] = {
                "state": np.concatenate([q[:, :7].astype(np.float32), grip[:, None]], axis=1),
            }
        out[0]["action"] = np.asarray(tr["actions/panda-0"]).astype(np.float32)
        out[1]["action"] = np.asarray(tr["actions/panda-1"]).astype(np.float32)
        out[0]["wrist"] = np.asarray(tr["obs/sensor_data/hand_camera_0/rgb"])
        out[1]["wrist"] = np.asarray(tr["obs/sensor_data/hand_camera_1/rgb"])
        out["global"] = np.asarray(tr["obs/sensor_data/head_camera_global/rgb"])
    return out


def to_chw_norm(hwc_u8, h=224, w=224):
    """uint8 HxWx3 → float32 3xHxW normalized to [0,1], resized to (h,w) (matches DP _rgb_chw)."""
    import cv2
    if hwc_u8.shape[0] != h or hwc_u8.shape[1] != w:
        hwc_u8 = cv2.resize(hwc_u8, (w, h), interpolation=cv2.INTER_AREA)
    return (hwc_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)


def build_obs_window(data, arm, t):
    """Build a 3-step deque of obs ending at t (clamps below 0 → repeat t=0)."""
    runner = DPRunner(output_dir=None, n_obs_steps=N_OBS, n_action_steps=8)
    for dt in range(-(N_OBS - 1), 1):
        s = max(0, t + dt)
        obs = {
            "head_cam": to_chw_norm(data[arm]["wrist"][s]),
            "head_cam_global": to_chw_norm(data["global"][s]),
            "agent_pos": data[arm]["state"][s].astype(np.float32),
        }
        runner.update_obs(obs)
    return runner


def main():
    data = load_ep0()
    rows = []

    for arm in (0, 1):
        print(f"\n=== arm{arm} ===", flush=True)
        policy = get_policy(CKPT[arm], output_dir=None, device="cuda:0")
        # Determinism probe: 3 calls at t=0
        runner = build_obs_window(data, arm, 0)
        chunks = []
        for k in range(3):
            chunks.append(runner.get_action(policy))
        max01 = float(np.max(np.abs(chunks[0] - chunks[1])))
        max02 = float(np.max(np.abs(chunks[0] - chunks[2])))
        rmse01 = float(np.sqrt(np.mean((chunks[0] - chunks[1]) ** 2)))
        print(f"determinism arm{arm}: max|c0-c1|={max01:.6f} max|c0-c2|={max02:.6f} RMSE c0-c1={rmse01:.6f}")
        rows.append({"probe": "determinism", "arm": arm, "max01": max01, "max02": max02, "rmse01": rmse01,
                     "chunk0_t0": chunks[0][0].tolist()})

        # Replay-error at each t∈TIMESTEPS
        for t in TIMESTEPS:
            runner = build_obs_window(data, arm, t)
            chunk = runner.get_action(policy)  # (8, 8)
            pred = chunk[0]
            demo = data[arm]["action"][t]
            err = pred - demo
            joint_rmse = float(np.sqrt(np.mean(err[:7] ** 2)))
            grip_err = float(err[7])
            print(f"t={t:3d} arm{arm}: joint_rmse={joint_rmse:.5f} grip_err={grip_err:+.4f}  pred={pred} demo={demo}")
            rows.append({"probe": "replay", "arm": arm, "t": t, "joint_rmse_rad": joint_rmse,
                         "grip_err": grip_err, "pred": pred.tolist(), "demo": demo.tolist()})

    out = pathlib.Path("/iris/u/mikulrai/projects/openpi/tools/probe_dp_overfit_ep0_5000.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote: {out}", flush=True)


if __name__ == "__main__":
    main()
