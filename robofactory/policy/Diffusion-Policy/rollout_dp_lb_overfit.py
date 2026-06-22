"""DP rollout @ seed=0 (ep0 IC) for LB-wc-decent. Direct ckpt paths, shader_pack='default'.

Run from /iris/u/mikulrai/projects/RoboFactory/robofactory/ with the RoboFactory env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import sapien  # noqa: F401
import torch  # noqa: F401

from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from robofactory.tasks import *  # noqa: F401, F403
import robofactory.agents  # noqa: F401

sys.path.insert(0, "./policy/Diffusion-Policy")
from eval_multi_dp import get_policy
from diffusion_policy.env_runner.dp_runner import DPRunner


def to_chw(rgb_hwc_u8, h=224, w=224):
    if rgb_hwc_u8.shape[0] != h or rgb_hwc_u8.shape[1] != w:
        rgb_hwc_u8 = cv2.resize(rgb_hwc_u8, (w, h), interpolation=cv2.INTER_AREA)
    return (rgb_hwc_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _extract(obs, cam):
    img = obs["sensor_data"][cam]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _state(obs, robot_uid, i):
    q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
    grip = float((q[7] + q[8]) / 2.0)
    return np.concatenate([q[:7].astype(np.float32), np.array([grip], dtype=np.float32)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt0", required=True)
    p.add_argument("--ckpt1", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--video-out", required=True)
    p.add_argument("--config", default="/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml")
    p.add_argument("--task", default="LiftBarrier-rf")
    p.add_argument("--robot-uid", default="panda_wristcam_multi")
    args = p.parse_args()

    env = gym.make(
        args.task,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="auto",
        robot_uids=tuple([args.robot_uid, args.robot_uid]),
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]

    policies = [get_policy(args.ckpt0, None, "cuda:0"),
                get_policy(args.ckpt1, None, "cuda:0")]
    runners = [DPRunner(output_dir=None, n_obs_steps=3, n_action_steps=8) for _ in range(2)]

    obs, _ = env.reset(seed=args.seed)
    head = _extract(obs, "head_camera_global")
    print(f"[shader-check] head[:10,:10] mean={head[:10,:10].mean():.1f} (>0 means non-black)")
    frames = [head]

    chunks = [None, None]
    chunk_idxs = [0, 0]
    chunk_lens = [0, 0]
    success = False
    for step in range(args.max_steps):
        # build obs for each arm
        for i in range(2):
            wrist = _extract(obs, f"hand_camera_{i}")
            head_g = _extract(obs, "head_camera_global")
            obs_i = {
                "head_cam": to_chw(wrist),
                "head_cam_global": to_chw(head_g),
                "agent_pos": _state(obs, args.robot_uid, i),
            }
            runners[i].update_obs(obs_i)

        for i in range(2):
            if chunks[i] is None or chunk_idxs[i] >= chunk_lens[i]:
                chunks[i] = runners[i].get_action(policies[i])
                chunk_lens[i] = chunks[i].shape[0]
                chunk_idxs[i] = 0
                if step == 0 and i == 0:
                    print(f"[chunk-shape] arm0 chunk shape: {chunks[i].shape}")

        action_dict = {}
        for i in range(2):
            a = chunks[i][chunk_idxs[i]]
            action_dict[f"{action_prefix}-{i}"] = a.astype(np.float32)
            chunk_idxs[i] += 1

        obs, _, term, trunc, info = env.step(action_dict)
        frames.append(_extract(obs, "head_camera_global"))
        s = info.get("success", False)
        success = bool(s.item() if hasattr(s, "item") else s)
        if success or term or trunc:
            break

    print(f"[result] success={success} steps={step + 1}")

    # write mp4 (mp4v then transcode to h264 + faststart for browser compat)
    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    tmp = args.video_out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", args.video_out], check=True)
    Path(tmp).unlink()
    print(f"[video] wrote {args.video_out} (success={success}, steps={step + 1})")


if __name__ == "__main__":
    main()
