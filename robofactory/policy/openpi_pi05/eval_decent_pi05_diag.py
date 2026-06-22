"""Diagnostic variant of eval_decent_pi05.py with two probe modes (combinable).

Probes B & C from the overfit diagnostic plan:
  --inject-h5-images: at each rollout step, replace live-rendered images with the
      images stored in the LB H5 at min(step, 97). State still comes from the live env.
      → isolates render-drift from dynamics-drift.
  --log-pred-vs-demo PATH: at each step, also log per-arm predicted-action vs the
      demo's recorded action at min(step, 97). Writes JSONL.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import h5py
import numpy as np
import sapien  # noqa: F401
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robofactory.tasks import *  # noqa: F401, F403
import robofactory.agents  # noqa: F401

H5 = "/iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/LiftBarrier/LiftBarrier.h5"
IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")


@dataclasses.dataclass
class Args:
    task: str = "LiftBarrier-rf"
    config: str = "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml"
    host: str = "127.0.0.1"
    ports: str = "8000,8001"
    seed: int = 0
    max_env_steps: int = 400
    replan_after: int = 8
    prompt: str = "lift the steel barrier using two robot arms"
    sim_backend: str = "auto"
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = "panda_wristcam_multi,panda_wristcam_multi"
    video_path: str = ""
    inject_h5_images: bool = False
    log_pred_vs_demo: str = ""  # JSONL path
    num_arms: int = 2


def _gripper(q):
    return float((q[7] + q[8]) / 2.0)


def _build_state(obs, num_arms, robot_uid):
    parts = []
    for i in range(num_arms):
        q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
        parts.append(q[:7].astype(np.float32))
        parts.append(np.array([_gripper(q)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _extract_image(obs, cam):
    img = obs["sensor_data"][cam]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _cur_qpos(obs, num_arms, robot_uid):
    return [
        np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()[:7].astype(np.float32)
        for i in range(num_arms)
    ]


class H5DemoStore:
    """Lazy-load H5 traj_0 images + actions once."""

    def __init__(self):
        with h5py.File(H5, "r") as f:
            t = f["traj_0"]
            self.head = np.asarray(t["obs/sensor_data/head_camera_global/rgb"])  # (T, H, W, 3)
            self.hand0 = np.asarray(t["obs/sensor_data/hand_camera_0/rgb"])
            self.hand1 = np.asarray(t["obs/sensor_data/hand_camera_1/rgb"])
            self.a0 = np.asarray(t["actions/panda-0"]).astype(np.float32)  # (T, 8) abs targets
            self.a1 = np.asarray(t["actions/panda-1"]).astype(np.float32)
            self.q0 = np.asarray(t["obs/agent/panda_wristcam_multi-0/qpos"]).astype(np.float32)
            self.q1 = np.asarray(t["obs/agent/panda_wristcam_multi-1/qpos"]).astype(np.float32)
        self.T = self.a0.shape[0]
        print(f"[H5] loaded traj_0: T={self.T}")

    def step_index(self, t):
        return min(t, self.T - 1)

    def images_at(self, t):
        i = self.step_index(t)
        return self.head[i], self.hand0[i], self.hand1[i]

    def demo_action_at(self, t):
        i = self.step_index(t)
        return self.a0[i], self.a1[i]

    def demo_qpos_at(self, t):
        i = self.step_index(t)
        return self.q0[i], self.q1[i]


def _build_obs_dict(obs, args, store, step, cam_map):
    state = _build_state(obs, args.num_arms, args.robot_uid)
    out = {"state": state, "prompt": args.prompt}
    if args.inject_h5_images:
        h, h0, h1 = store.images_at(step)
        out["base_0_rgb_raw"] = h.astype(np.uint8)
        out["left_wrist_0_rgb_raw"] = h0.astype(np.uint8)
        out["right_wrist_0_rgb_raw"] = h1.astype(np.uint8)
        out["extra_0_rgb_raw"] = np.zeros_like(h, dtype=np.uint8)
    else:
        ref = None
        for slot, cam in cam_map.items():
            if cam is None:
                continue
            img = _extract_image(obs, cam)
            out[slot] = img
            if ref is None:
                ref = img.shape
        if ref is None:
            ref = (224, 224, 3)
        for slot in IMAGE_SLOTS:
            if slot not in out:
                out[slot] = np.zeros(ref, dtype=np.uint8)
    return out


def _write_mp4(path, frames):
    import cv2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main(args: Args):
    cam_map = {
        "base_0_rgb_raw": "head_camera_global",
        "left_wrist_0_rgb_raw": "hand_camera_0",
        "right_wrist_0_rgb_raw": "hand_camera_1",
        "extra_0_rgb_raw": None,
    }
    ports = [int(p) for p in args.ports.split(",")]
    env = gym.make(
        args.task,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids=tuple(args.robot_uids_csv.split(",")),
        # Match data-gen shader_pack ("default") — see memory/feedback_sapien_shader_pack_eval_mismatch.md.
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]
    policies = [WebsocketClientPolicy(host=args.host, port=p) for p in ports]
    print(f"action_prefix={action_prefix} ports={ports} inject_h5={args.inject_h5_images} log_pred={bool(args.log_pred_vs_demo)}")

    store = H5DemoStore()
    obs, _ = env.reset(seed=args.seed)

    # Sanity: compare live qpos at t=0 vs H5
    live_state = _build_state(obs, args.num_arms, args.robot_uid)
    h5_q0, h5_q1 = store.demo_qpos_at(0)
    print(f"[t=0 state] live arm0: {live_state[:8]}")
    print(f"           H5 arm0:   {np.concatenate([h5_q0[:7], [_gripper(h5_q0)]])}")
    print(f"           live arm1: {live_state[8:]}")
    print(f"           H5 arm1:   {np.concatenate([h5_q1[:7], [_gripper(h5_q1)]])}")

    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    frames = []
    log_fp = open(args.log_pred_vs_demo, "w") if args.log_pred_vs_demo else None

    success = False
    for step in range(args.max_env_steps):
        if args.video_path:
            frames.append(_extract_image(obs, "head_camera_global"))

        obs_dict = None
        replanned = [False] * args.num_arms
        for i in range(args.num_arms):
            if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
                if obs_dict is None:
                    obs_dict = _build_obs_dict(obs, args, store, step, cam_map)
                chunks[i] = np.asarray(policies[i].infer(obs_dict)["actions"])
                chunk_idxs[i] = 0
                replanned[i] = True

        cur_q = _cur_qpos(obs, args.num_arms, args.robot_uid)
        action_dict = {}
        per_arm_pred_abs = []
        per_arm_err = []
        for i in range(args.num_arms):
            step_i = chunks[i][chunk_idxs[i]]
            delta = step_i[:7]
            grip = step_i[7]
            target = np.concatenate([cur_q[i] + delta, np.array([grip], dtype=np.float32)])
            action_dict[f"{action_prefix}-{i}"] = target.astype(np.float32)
            chunk_idxs[i] += 1
            per_arm_pred_abs.append(target.tolist())

        if log_fp is not None:
            demo_a0, demo_a1 = store.demo_action_at(step)
            demo_acts = [demo_a0, demo_a1]
            for i in range(args.num_arms):
                err = np.asarray(per_arm_pred_abs[i]) - demo_acts[i]
                per_arm_err.append({
                    "joint_rmse_rad": float(np.sqrt(np.mean(err[:7] ** 2))),
                    "grip_err": float(err[7]),
                    "max_abs_joint_err": float(np.max(np.abs(err[:7]))),
                })
            row = {
                "step": int(step),
                "h5_step_used": int(store.step_index(step)),
                "live_qpos": [c.tolist() for c in cur_q],
                "h5_qpos_at_step": [store.demo_qpos_at(step)[i][:7].tolist() for i in range(args.num_arms)],
                "pred_abs": per_arm_pred_abs,
                "demo_abs": [demo_acts[i].tolist() for i in range(args.num_arms)],
                "err": per_arm_err,
                "replanned": replanned,
            }
            log_fp.write(json.dumps(row) + "\n")
            log_fp.flush()

        obs, _, term, trunc, info = env.step(action_dict)
        s = info.get("success", False)
        success = bool(s.item() if hasattr(s, "item") else s)
        if success or term or trunc:
            break

    print(f"[result] success={success} steps={step + 1}")
    if log_fp is not None:
        log_fp.close()
    if args.video_path and frames:
        _write_mp4(args.video_path, frames)
        print(f"[video] wrote {args.video_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
