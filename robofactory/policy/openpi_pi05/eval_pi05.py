"""Evaluate a fine-tuned pi0.5 RoboFactory policy in the ManiSkill sim.

Architecture:
- Connects to a `scripts/serve_policy.py` instance over WebSocket (openpi_client).
- For each episode, resets the env, then loops:
    1. Build a state of shape (num_arms * 8,) from current qpos of all arms (matches the converter).
    2. Build the 4-image observation dict from --camera_mapping (slot -> sim camera name).
    3. Call the policy server -> action chunk shape (action_horizon, active_dim).
    4. Receding-horizon: execute first `replan_after` steps; for each step convert delta -> absolute
       joint target (target_q = current_q + delta) and pass {panda-i: [...7 joints, gripper]} to env.step.
    5. Re-plan if the chunk runs out before episode end.

Usage (3-arm default):
    python eval_pi05.py --task ThreeRobotsStackCube-rf --num-episodes 50 --seeds 0,1,2

Usage (4-arm):
    python eval_pi05.py \\
        --task LongPipelineDelivery-rf \\
        --num-arms 4 \\
        --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/long_pipeline_delivery.json \\
        --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/long_pipeline_delivery.yaml \\
        --max-env-steps 1800 \\
        --num-episodes 50 --seeds 0,1,2
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import sapien  # noqa: F401  (sapien import side effects)
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robofactory.tasks import *  # noqa: F401, F403  (registers env IDs with gym)


DEFAULT_PROMPT = "stack the three cubes using three robot arms"

DEFAULT_CAMERA_MAPPING = {
    "base_0_rgb_raw": "head_camera_global",
    "left_wrist_0_rgb_raw": "head_camera_agent0",
    "right_wrist_0_rgb_raw": "head_camera_agent1",
    "extra_0_rgb_raw": "head_camera_agent2",
}
IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")


@dataclasses.dataclass
class Args:
    task: str = "ThreeRobotsStackCube-rf"
    config: str = (
        "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml"
    )
    host: str = "127.0.0.1"
    port: int = 8000
    num_episodes: int = 50
    seeds: Annotated[str, tyro.conf.arg(help="comma-separated seed list")] = "0,1,2"
    max_env_steps: int = 1800
    replan_after: int = 8
    prompt: str = DEFAULT_PROMPT
    sim_backend: str = "auto"
    out_dir: str = "/iris/u/mikulrai/logs/eval_pi05"
    num_arms: int = 3
    active_dim: int = 0  # 0 => num_arms * 8
    camera_mapping: str = ""  # path to JSON; empty => DEFAULT_CAMERA_MAPPING
    robot_uid: str = "panda"  # agent key prefix, e.g. "panda_wristcam_multi" for D2
    robot_uids_csv: str = ""  # comma-separated UIDs for gym.make; empty = use default


def _gripper_from_qpos(qpos_step: np.ndarray) -> float:
    return float((qpos_step[7] + qpos_step[8]) / 2.0)


def _build_state(obs: dict, num_arms: int, robot_uid: str = "panda") -> np.ndarray:
    state_parts: list[np.ndarray] = []
    for i in range(num_arms):
        q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
        state_parts.append(q[:7].astype(np.float32))
        state_parts.append(np.array([_gripper_from_qpos(q)], dtype=np.float32))
    return np.concatenate(state_parts).astype(np.float32)


def _extract_image(obs: dict, cam_name: str) -> np.ndarray:
    img = obs["sensor_data"][cam_name]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _build_obs_dict(obs: dict, prompt: str, num_arms: int, cam_map: dict[str, str], robot_uid: str = "panda") -> dict:
    out: dict = {
        "state": _build_state(obs, num_arms, robot_uid),
        "prompt": prompt,
    }
    for slot in IMAGE_SLOTS:
        out[slot] = _extract_image(obs, cam_map[slot])
    return out


def _delta_to_absolute_action(
    chunk_step: np.ndarray, current_qpos_per_arm: list[np.ndarray], num_arms: int, robot_uid: str = "panda"
) -> dict:
    """chunk_step: (num_arms*8,) = per-arm [delta_joints(7), gripper(1)]."""
    out: dict[str, np.ndarray] = {}
    for i in range(num_arms):
        s = i * 8
        delta = chunk_step[s : s + 7]
        gripper = chunk_step[s + 7]
        target = np.concatenate([current_qpos_per_arm[i] + delta, np.array([gripper], dtype=np.float32)])
        out[f"{robot_uid}-{i}"] = target.astype(np.float32)
    return out


def _current_qpos_per_arm(obs: dict, num_arms: int, robot_uid: str = "panda") -> list[np.ndarray]:
    return [
        np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()[:7].astype(np.float32)
        for i in range(num_arms)
    ]


def _resolve_camera_mapping(path: str) -> dict[str, str]:
    if not path:
        return dict(DEFAULT_CAMERA_MAPPING)
    mapping = json.loads(Path(path).read_text())
    missing = set(IMAGE_SLOTS) - set(mapping.keys())
    if missing:
        raise ValueError(f"camera_mapping missing slots: {missing}")
    return {slot: mapping[slot] for slot in IMAGE_SLOTS}


def run_episode(env, policy, args: Args, cam_map: dict[str, str], active_dim: int, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    success = False
    t0 = time.time()
    chunk_idx: int = 10**9
    chunk: np.ndarray | None = None

    for step in range(args.max_env_steps):
        if chunk is None or chunk_idx >= args.replan_after:
            obs_dict = _build_obs_dict(obs, args.prompt, args.num_arms, cam_map, args.robot_uid)
            result = policy.infer(obs_dict)
            chunk = np.asarray(result["actions"])[:, :active_dim]  # (H, active_dim)
            chunk_idx = 0

        cur_qpos = _current_qpos_per_arm(obs, args.num_arms, args.robot_uid)
        action_dict = _delta_to_absolute_action(chunk[chunk_idx], cur_qpos, args.num_arms, args.robot_uid)
        chunk_idx += 1

        obs, reward, terminated, truncated, info = env.step(action_dict)
        succ_field = info.get("success", False)
        if hasattr(succ_field, "item"):
            succ_field = succ_field.item()
        success = bool(succ_field)
        if success or terminated or truncated:
            break

    return {"seed": seed, "success": success, "steps": step + 1, "wall_s": time.time() - t0}


def main(args: Args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cam_map = _resolve_camera_mapping(args.camera_mapping)
    active_dim = args.active_dim or args.num_arms * 8

    env_kwargs = dict(
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
    )
    if args.robot_uids_csv:
        env_kwargs["robot_uids"] = tuple(args.robot_uids_csv.split(","))
    env = gym.make(args.task, **env_kwargs)

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")
    print(f"num_arms={args.num_arms} active_dim={active_dim} cam_map={cam_map}")

    results: list[dict] = []
    for seed in seeds:
        for ep_i in range(args.num_episodes):
            ep_seed = seed * 100_000 + ep_i
            try:
                r = run_episode(env, policy, args, cam_map, active_dim, ep_seed)
            except Exception as e:  # noqa: BLE001
                r = {"seed": ep_seed, "success": False, "steps": -1, "error": repr(e)}
            results.append(r)
            print(
                f"[seed_base={seed} ep={ep_i:03d}] success={r['success']} "
                f"steps={r['steps']} wall_s={r.get('wall_s', -1):.1f}"
            )

    n = len(results)
    n_succ = sum(1 for r in results if r["success"])
    print(f"\n=== summary ===\nepisodes: {n}\nsuccess: {n_succ}/{n} ({100.0 * n_succ / n:.1f}%)")

    out_file = out_dir / f"eval_{args.task}_{int(time.time())}.json"
    out_file.write_text(json.dumps({"args": dataclasses.asdict(args), "results": results}, indent=2))
    print(f"Saved {out_file}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
