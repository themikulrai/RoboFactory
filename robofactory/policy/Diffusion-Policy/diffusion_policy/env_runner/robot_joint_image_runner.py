"""
Env runner for the centralised (joint) diffusion policy.

One policy consumes obs from all N arms (separate RGB keys `head_camera_0..head_camera_{N-1}`,
plus an optional `head_camera_global`, plus a single concatenated `agent_pos` of
length 8*N) and emits a (horizon, 8*N) action chunk that is split back into N
per-arm actions and dispatched via `panda-{i}` in the env step call.

All RGB streams are resized to `resize x resize` to match the training zarr schema
(default 224).
"""
import os
import traceback
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import torch
import tqdm

try:
    import wandb
except Exception:
    wandb = None


CAM_KEY_TPL = {
    "workspace": "head_camera_agent{i}",
    "wristcam":  "hand_camera_{i}",
}


class RobotJointImageRunner:
    def __init__(
        self,
        output_dir: str,
        env_id: str = "ThreeRobotsStackCube-rf",
        config_path: Optional[str] = None,
        n_agents: int = 3,
        include_global: bool = True,
        camera_family: str = "wristcam",
        resize: int = 224,
        data_num: int = 150,
        n_eval_episodes: int = 10,
        n_action_exec: int = 6,
        max_episode_steps: int = 1500,
        n_obs_steps: int = 3,
        n_action_steps: int = 8,
        test_start_seed: int = 100000,
        record_video_episodes: Optional[List[int]] = None,
        fps: int = 20,
        device: str = "cuda:0",
        tqdm_interval_sec: float = 5.0,
        robot_uids: Optional[tuple] = None,
    ):
        self.output_dir = output_dir
        self.env_id = env_id
        self.config_path = config_path
        self.robot_uids = robot_uids
        self.n_agents = n_agents
        self.include_global = include_global
        assert camera_family in CAM_KEY_TPL, f"bad camera_family={camera_family}"
        self.camera_family = camera_family
        self.resize = int(resize)
        self.data_num = data_num
        self.n_eval_episodes = n_eval_episodes
        self.n_action_exec = n_action_exec
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.test_start_seed = test_start_seed
        self.record_video_episodes = set(record_video_episodes or [0, 5])
        self.fps = fps
        self.device = device
        self.tqdm_interval_sec = tqdm_interval_sec

    # ------------------- env plumbing -------------------

    def _make_env(self):
        import gymnasium as gym
        import robofactory.tasks  # noqa: F401  (register_env side effect)

        cfg_path = self.config_path
        if cfg_path is None:
            from robofactory import CONFIG_DIR
            default_cfg = {
                "TakePhoto-rf": ("robocasa", "take_photo.yaml"),
                "LongPipelineDelivery-rf": ("robocasa", "long_pipeline_delivery.yaml"),
                "ThreeRobotsStackCube-rf": ("robocasa", "three_robots_stack_cube.yaml"),
            }
            if self.env_id not in default_cfg:
                raise ValueError(
                    f"No default config for env_id={self.env_id}; set task.env_runner.config_path"
                )
            cfg_path = os.path.join(CONFIG_DIR, *default_cfg[self.env_id])

        env_kwargs = dict(
            config=cfg_path,
            obs_mode="rgb",
            reward_mode="dense",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack="default"),
            human_render_camera_configs=dict(shader_pack="default"),
            viewer_camera_configs=dict(shader_pack="default"),
            num_envs=1,
            sim_backend="cpu",
            enable_shadow=True,
        )
        if self.robot_uids is not None:
            env_kwargs["robot_uids"] = self.robot_uids
        return gym.make(self.env_id, **env_kwargs)

    # ------------------- obs extraction -------------------

    def _resize_rgb(self, rgb_hwc: np.ndarray) -> np.ndarray:
        if rgb_hwc.shape[0] == self.resize and rgb_hwc.shape[1] == self.resize:
            return rgb_hwc
        return cv2.resize(rgb_hwc, (self.resize, self.resize), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _agent_prefix(raw_obs: dict) -> str:
        """Infer agent key prefix from obs dict (e.g. 'panda' or 'panda_wristcam_multi')."""
        keys = list(raw_obs["agent"].keys())
        # keys look like 'panda-0', 'panda_wristcam_multi-0', etc.
        return keys[0].rsplit("-", 1)[0]

    @staticmethod
    def _initial_agent_pos_one(raw_obs, agent_id: int) -> np.ndarray:
        prefix = RobotJointImageRunner._agent_prefix(raw_obs)
        qpos = raw_obs["agent"][f"{prefix}-{agent_id}"]["qpos"]
        if hasattr(qpos, "squeeze"):
            qpos = qpos.squeeze(0)
        qpos_np = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
        arm = qpos_np[:-2]
        gripper = np.array([1.0], dtype=arm.dtype)
        return np.concatenate([arm, gripper]).astype(np.float32)

    def _pull_rgb(self, raw_obs, key: str) -> np.ndarray:
        rgb = raw_obs["sensor_data"][key]["rgb"]
        if hasattr(rgb, "squeeze"):
            rgb = rgb.squeeze(0)
        rgb_np = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        rgb_np = self._resize_rgb(rgb_np)
        return np.moveaxis(rgb_np, -1, 0).astype(np.float32) / 255.0

    def _build_obs_dict(self, raw_obs, last_exec: Optional[np.ndarray]) -> dict:
        if last_exec is not None:
            agent_pos = last_exec.astype(np.float32)  # (8*N,)
        else:
            agent_pos = np.concatenate(
                [self._initial_agent_pos_one(raw_obs, i) for i in range(self.n_agents)]
            ).astype(np.float32)

        out = {"agent_pos": agent_pos}
        tpl = CAM_KEY_TPL[self.camera_family]
        for i in range(self.n_agents):
            out[f"head_camera_{i}"] = self._pull_rgb(raw_obs, tpl.format(i=i))
        if self.include_global:
            out["head_camera_global"] = self._pull_rgb(raw_obs, "head_camera_global")
        return out

    # ------------------- policy forward -------------------

    @staticmethod
    def _stack_last_n(deque_, n_steps):
        lst = list(deque_)
        first = lst[-1]
        out = np.zeros((n_steps,) + first.shape, dtype=first.dtype)
        start = -min(n_steps, len(lst))
        out[start:] = np.array(lst[start:])
        if n_steps > len(lst):
            out[:start] = out[start]
        return out

    def _predict_joint_action(self, policy, obs_history: deque) -> np.ndarray:
        keys = list(obs_history[0].keys())
        stacked = {k: self._stack_last_n([o[k] for o in obs_history], self.n_obs_steps)
                   for k in keys}
        device = next(policy.parameters()).device
        obs_in = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in stacked.items()}
        with torch.no_grad():
            out = policy.predict_action(obs_in)
        return out["action"].squeeze(0).detach().cpu().numpy()  # (n_action_steps, 8*N)

    # ------------------- rendering -------------------

    @staticmethod
    def _frame_from_obs(raw_obs: dict) -> Optional[np.ndarray]:
        try:
            rgb = raw_obs["sensor_data"]["head_camera_global"]["rgb"]
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            else:
                rgb = np.asarray(rgb)
            while rgb.ndim > 3:
                rgb = rgb[0]
            return rgb.astype(np.uint8)
        except Exception:
            return None

    @staticmethod
    def _check_success(info: dict) -> bool:
        if not isinstance(info, dict):
            return False
        succ = info.get("success", False)
        if hasattr(succ, "__len__"):
            try:
                return bool(np.asarray(succ).all())
            except Exception:
                return False
        try:
            return bool(succ)
        except Exception:
            return False

    # ------------------- main rollout -------------------

    def _rollout_single_episode(self, env, policy, seed: int, record_frames: bool) -> dict:
        obs_history: deque = deque(maxlen=self.n_obs_steps + 1)
        raw_obs, _ = env.reset(seed=seed)
        if env.action_space is not None:
            env.action_space.seed(seed)

        obs_history.append(self._build_obs_dict(raw_obs, last_exec=None))

        frames: List[np.ndarray] = []
        if record_frames:
            f0 = self._frame_from_obs(raw_obs)
            if f0 is not None:
                frames.append(f0)

        success = False
        steps = 0
        info: dict = {}
        action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]

        while steps < self.max_episode_steps:
            joint_action_chunk = self._predict_joint_action(policy, obs_history)
            per_arm = np.split(joint_action_chunk, self.n_agents, axis=1)
            last_joint = joint_action_chunk[0]

            for t in range(min(self.n_action_exec, self.n_action_steps)):
                action_dict = {
                    f"{action_prefix}-{i}": np.asarray(per_arm[i][t], dtype=np.float64)
                    for i in range(self.n_agents)
                }
                raw_obs, _, term, trunc, info = env.step(action_dict)
                steps += 1
                if record_frames:
                    fr = self._frame_from_obs(raw_obs)
                    if fr is not None:
                        frames.append(fr)
                last_joint = joint_action_chunk[t]
                success = self._check_success(info)
                if success or steps >= self.max_episode_steps:
                    break

            obs_history.append(self._build_obs_dict(raw_obs, last_exec=last_joint))
            if success:
                break

        return dict(success=bool(success), length=steps, frames=frames)

    def run(self, policy) -> dict:
        try:
            policy.eval()
            env = self._make_env()
        except Exception as e:
            traceback.print_exc()
            return {
                "test_mean_score": 0.0,
                "rollout/success_rate": 0.0,
                "rollout/mean_episode_length": 0.0,
                "rollout/n_eval_episodes": 0,
                "rollout/enabled": 0,
                "rollout/error": f"setup_failed:{type(e).__name__}",
            }

        successes: List[bool] = []
        lengths: List[int] = []
        video_frames = {}

        try:
            pbar = tqdm.tqdm(
                range(self.n_eval_episodes),
                desc="Eval rollouts (joint)",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            for ep_idx in pbar:
                record = ep_idx in self.record_video_episodes
                seed = self.test_start_seed + ep_idx
                try:
                    result = self._rollout_single_episode(
                        env, policy, seed=seed, record_frames=record
                    )
                except Exception as e:
                    traceback.print_exc()
                    result = dict(success=False, length=self.max_episode_steps, frames=[])
                    if ep_idx == 0:
                        try:
                            env.close()
                        except Exception:
                            pass
                        return {
                            "test_mean_score": 0.0,
                            "rollout/success_rate": 0.0,
                            "rollout/mean_episode_length": 0.0,
                            "rollout/n_eval_episodes": 0,
                            "rollout/enabled": 0,
                            "rollout/error": f"rollout_failed:{type(e).__name__}",
                        }
                successes.append(result["success"])
                lengths.append(result["length"])
                if record and result["frames"]:
                    video_frames[ep_idx] = result["frames"]
                pbar.set_postfix(success=sum(successes), total=len(successes))
        finally:
            try:
                env.close()
            except Exception:
                pass

        try:
            policy.train()
        except Exception:
            pass

        success_rate = float(np.mean(successes)) if successes else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        log = {
            "test_mean_score": success_rate,
            "rollout/success_rate": success_rate,
            "rollout/mean_episode_length": mean_len,
            "rollout/n_eval_episodes": len(successes),
            "rollout/enabled": 1,
        }
        if wandb is not None:
            for ep_idx, frames in video_frames.items():
                if not frames:
                    continue
                try:
                    arr = np.stack(frames)
                    arr = np.transpose(arr, (0, 3, 1, 2))
                    log[f"rollout/video_ep{ep_idx}"] = wandb.Video(arr, fps=self.fps, format="mp4")
                except Exception:
                    traceback.print_exc()
        return log
