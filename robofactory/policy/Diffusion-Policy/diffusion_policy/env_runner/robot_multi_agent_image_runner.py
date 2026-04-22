import os
import glob
import traceback
from collections import defaultdict
from typing import Dict, List, Optional

import dill
import hydra
import numpy as np
import torch
import tqdm

try:
    import wandb
except Exception:
    wandb = None

from diffusion_policy.env_runner.dp_runner import DPRunner


class RobotMultiAgentImageRunner:
    """
    In-training env rollout for RoboFactory multi-arm tasks.

    Per-arm policies are trained independently. This runner uses the live
    `policy` passed to `.run()` for `current_agent_id` and loads peer-agent
    policies from disk (dill checkpoints). When any peer ckpt is missing
    (e.g. agents 0/1 trained before this one), `.run()` returns a no-op
    dict and skips env creation entirely -- training continues unaffected.

    Action execution is raw (not TOPP-interpolated): executes the first
    `n_action_exec` of the 8 predicted joint-pos targets per chunk before
    re-predicting. Matches data-collection fidelity less precisely than
    eval_multi_dp.py's TOPP sync, but is ~3-5x faster and provides a
    relative training signal. Ground-truth success rate should still come
    from `policy/Diffusion-Policy/eval_multi.sh` post-training.
    """

    def __init__(
        self,
        output_dir: str,
        env_id: str = "ThreeRobotsStackCube-rf",
        config_path: Optional[str] = None,
        current_agent_id: int = 0,
        n_agents: int = 3,
        data_num: int = 150,
        peer_checkpoint_num: int = 300,
        other_ckpt_search_roots: Optional[List[str]] = None,
        n_eval_episodes: int = 10,
        n_action_exec: int = 6,
        max_episode_steps: int = 800,
        n_obs_steps: int = 3,
        n_action_steps: int = 8,
        test_start_seed: int = 100000,
        record_video_episodes: Optional[List[int]] = None,
        fps: int = 20,
        device: str = "cuda:0",
        tqdm_interval_sec: float = 5.0,
    ):
        self.output_dir = output_dir
        self.env_id = env_id
        self.config_path = config_path
        self.current_agent_id = current_agent_id
        self.n_agents = n_agents
        self.data_num = data_num
        self.peer_checkpoint_num = peer_checkpoint_num
        self.other_ckpt_search_roots = other_ckpt_search_roots or ["data/outputs"]
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

        self._peer_policies_cache: Dict[int, object] = {}
        self._peer_ckpt_paths_cached: Optional[Dict[int, str]] = None

    def _resolve_peer_ckpts(self) -> Optional[Dict[int, str]]:
        task_name = self.env_id
        peer_ids = [i for i in range(self.n_agents) if i != self.current_agent_id]
        resolved: Dict[int, str] = {}
        for aid in peer_ids:
            pattern = f"{task_name}_Agent{aid}_{self.data_num}"
            candidates: List[str] = []
            for root in self.other_ckpt_search_roots:
                candidates.extend(
                    glob.glob(os.path.join(root, "**", "checkpoints", pattern, "*.ckpt"), recursive=True)
                )
            if not candidates:
                return None
            candidates.sort()
            preferred = [
                c for c in candidates
                if os.path.basename(c) == f"{self.peer_checkpoint_num}.ckpt"
            ]
            resolved[aid] = preferred[-1] if preferred else candidates[-1]
        return resolved

    def _load_policy_from_ckpt(self, ckpt_path: str):
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(torch.device(self.device))
        policy.eval()
        return policy

    def _make_env(self):
        import gymnasium as gym
        import robofactory.tasks  # register env side-effect (ThreeRobotsStackCube-rf)

        cfg_path = self.config_path
        if cfg_path is None:
            from robofactory import CONFIG_DIR
            cfg_path = os.path.join(CONFIG_DIR, "robocasa", "three_robots_stack_cube.yaml")

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
        return gym.make(self.env_id, **env_kwargs)

    @staticmethod
    def _initial_agent_pos(raw_obs, agent_id: int) -> np.ndarray:
        qpos = raw_obs["agent"][f"panda-{agent_id}"]["qpos"]
        if hasattr(qpos, "squeeze"):
            qpos = qpos.squeeze(0)
        qpos_np = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
        arm = qpos_np[:-2]
        gripper = np.array([1.0], dtype=arm.dtype)
        return np.concatenate([arm, gripper]).astype(np.float32)

    @staticmethod
    def _extract_obs_for_agent(raw_obs, agent_pos_8d: np.ndarray, agent_id: int) -> dict:
        camera_name = f"head_camera_agent{agent_id}"
        rgb = raw_obs["sensor_data"][camera_name]["rgb"]
        if hasattr(rgb, "squeeze"):
            rgb = rgb.squeeze(0)
        rgb_np = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        head_cam = np.moveaxis(rgb_np, -1, 0).astype(np.float32) / 255.0
        return dict(head_cam=head_cam, agent_pos=agent_pos_8d.astype(np.float32))

    @staticmethod
    def _render_frame(env) -> Optional[np.ndarray]:
        try:
            frame = env.render()
        except Exception:
            return None
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        else:
            frame = np.asarray(frame)
        while frame.ndim > 3:
            frame = frame[0]
        return frame.astype(np.uint8)

    def _rollout_single_episode(self, env, policies, dp_runners, seed: int, record_frames: bool) -> dict:
        for runner in dp_runners.values():
            runner.reset_obs()

        raw_obs, _ = env.reset(seed=seed)
        if env.action_space is not None:
            env.action_space.seed(seed)

        for aid in range(self.n_agents):
            init_pos = self._initial_agent_pos(raw_obs, aid)
            dp_runners[aid].update_obs(self._extract_obs_for_agent(raw_obs, init_pos, aid))

        frames: List[np.ndarray] = []
        if record_frames:
            f0 = self._render_frame(env)
            if f0 is not None:
                frames.append(f0)

        success = False
        steps = 0
        info: dict = {}

        while steps < self.max_episode_steps:
            per_agent_actions = {
                aid: dp_runners[aid].get_action(policies[aid]) for aid in range(self.n_agents)
            }
            last_exec = {aid: per_agent_actions[aid][0] for aid in range(self.n_agents)}
            for t in range(min(self.n_action_exec, self.n_action_steps)):
                action_dict = {
                    f"panda-{aid}": np.asarray(per_agent_actions[aid][t], dtype=np.float64)
                    for aid in range(self.n_agents)
                }
                raw_obs, _, term, trunc, info = env.step(action_dict)
                steps += 1
                if record_frames:
                    f = self._render_frame(env)
                    if f is not None:
                        frames.append(f)
                for aid in range(self.n_agents):
                    last_exec[aid] = per_agent_actions[aid][t]
                success = self._check_success(info)
                if success or steps >= self.max_episode_steps:
                    break
            for aid in range(self.n_agents):
                dp_runners[aid].update_obs(
                    self._extract_obs_for_agent(raw_obs, np.asarray(last_exec[aid], dtype=np.float32), aid)
                )
            if success:
                break

        return dict(success=bool(success), length=steps, frames=frames)

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

    def run(self, policy) -> dict:
        peer_paths = self._resolve_peer_ckpts()
        if peer_paths is None:
            return {
                "test_mean_score": 0.0,
                "rollout/success_rate": 0.0,
                "rollout/mean_episode_length": 0.0,
                "rollout/n_eval_episodes": 0,
                "rollout/enabled": 0,
                "rollout/error": "peer_ckpts_missing",
            }

        try:
            policies: Dict[int, object] = {self.current_agent_id: policy}
            for aid, path in peer_paths.items():
                if aid not in self._peer_policies_cache or self._peer_ckpt_paths_cached != peer_paths:
                    self._peer_policies_cache[aid] = self._load_policy_from_ckpt(path)
            self._peer_ckpt_paths_cached = peer_paths
            policies.update(self._peer_policies_cache)

            for aid, pol in policies.items():
                pol.eval()

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

        dp_runners = {
            aid: DPRunner(
                output_dir=None,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
            )
            for aid in range(self.n_agents)
        }

        successes: List[bool] = []
        lengths: List[int] = []
        video_frames: Dict[int, List[np.ndarray]] = {}

        try:
            pbar = tqdm.tqdm(
                range(self.n_eval_episodes),
                desc="Eval rollouts",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            for ep_idx in pbar:
                record = ep_idx in self.record_video_episodes
                seed = self.test_start_seed + ep_idx
                try:
                    result = self._rollout_single_episode(
                        env, policies, dp_runners, seed=seed, record_frames=record
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

        # restore training mode on the live policy; peer policies stay eval
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
                    arr = np.stack(frames)  # (T, H, W, 3) uint8
                    arr = np.transpose(arr, (0, 3, 1, 2))  # (T, 3, H, W)
                    log[f"rollout/video_ep{ep_idx}"] = wandb.Video(arr, fps=self.fps, format="mp4")
                except Exception:
                    traceback.print_exc()
        return log
