"""Decentralised pi0.5 eval: 3 per-arm policy servers, one per port.

Each server runs a per-arm LoRA model (pi05_robofactory_decent_wristcam_lora_finetune_arm{i}).
The server applies RoboFactoryDecentInputs internally, so this client sends the full
24-dim state + all 4 images (same format as centralised eval).  Each server returns
an 8-dim per-arm action chunk (delta joints + gripper).

Usage:
    # Start 3 servers first (e.g., in background via SLURM script), then:
    python eval_decent_pi05.py \\
        --seeds 10010,10011,10012 \\
        --max-env-steps 200 \\
        --robot-uid panda_wristcam_multi \\
        --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi" \\
        --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/three_robots_stack_cube_wristcam.json \\
        --num-episodes 1
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import sapien  # noqa: F401
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robofactory.tasks import *  # noqa: F401, F403


DEFAULT_PROMPT = "stack the three cubes using three robot arms"
DEFAULT_CAMERA_MAPPING = {
    "base_0_rgb_raw": "head_camera_global",
    "left_wrist_0_rgb_raw": "hand_camera_0",
    "right_wrist_0_rgb_raw": "hand_camera_1",
    "extra_0_rgb_raw": "hand_camera_2",
}
IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")


@dataclasses.dataclass
class Args:
    task: str = "ThreeRobotsStackCube-rf"
    config: str = (
        "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml"
    )
    host: str = "127.0.0.1"
    ports: Annotated[str, tyro.conf.arg(help="comma-separated ports, one per arm")] = "8000,8001,8002"
    num_episodes: int = 1
    seeds: Annotated[str, tyro.conf.arg(help="comma-separated seed list")] = "10010,10011,10012"
    max_env_steps: int = 200
    replan_after: int = 8
    prompt: str = DEFAULT_PROMPT
    sim_backend: str = "auto"
    out_dir: str = "/iris/u/mikulrai/logs/eval_pi05_decent"
    num_arms: int = 3
    camera_mapping: str = ""
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = ""


def _gripper_from_qpos(qpos_step: np.ndarray) -> float:
    return float((qpos_step[7] + qpos_step[8]) / 2.0)


def _build_state(obs: dict, num_arms: int, robot_uid: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i in range(num_arms):
        q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
        parts.append(q[:7].astype(np.float32))
        parts.append(np.array([_gripper_from_qpos(q)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _extract_image(obs: dict, cam_name: str) -> np.ndarray:
    img = obs["sensor_data"][cam_name]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _build_obs_dict(obs: dict, prompt: str, num_arms: int, cam_map: dict[str, str], robot_uid: str) -> dict:
    out: dict = {
        "state": _build_state(obs, num_arms, robot_uid),
        "prompt": prompt,
    }
    for slot in IMAGE_SLOTS:
        out[slot] = _extract_image(obs, cam_map[slot])
    return out


def _current_qpos_per_arm(obs: dict, num_arms: int, robot_uid: str) -> list[np.ndarray]:
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


def run_episode(env, policies: list, args: Args, cam_map: dict[str, str], seed: int) -> dict:
    """Run one episode with 3 per-arm policy servers."""
    obs, _ = env.reset(seed=seed)
    success = False
    t0 = time.time()

    chunks: list[np.ndarray | None] = [None] * args.num_arms
    chunk_idxs: list[int] = [args.replan_after] * args.num_arms

    for step in range(args.max_env_steps):
        # Replan arms whose chunk is exhausted
        obs_dict = None
        for i in range(args.num_arms):
            if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
                if obs_dict is None:
                    obs_dict = _build_obs_dict(obs, args.prompt, args.num_arms, cam_map, args.robot_uid)
                result = policies[i].infer(obs_dict)
                chunks[i] = np.asarray(result["actions"])  # (H, 8)
                chunk_idxs[i] = 0

        cur_qpos = _current_qpos_per_arm(obs, args.num_arms, args.robot_uid)
        action_dict: dict[str, np.ndarray] = {}
        for i in range(args.num_arms):
            step_i = chunks[i][chunk_idxs[i]]  # (8,)
            delta = step_i[:7]
            gripper = step_i[7]
            target = np.concatenate([cur_qpos[i] + delta, np.array([gripper], dtype=np.float32)])
            action_dict[f"{args.robot_uid}-{i}"] = target.astype(np.float32)
            chunk_idxs[i] += 1

        obs, _, terminated, truncated, info = env.step(action_dict)
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
    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    assert len(ports) == args.num_arms, f"Need {args.num_arms} ports, got {ports}"

    cam_map = _resolve_camera_mapping(args.camera_mapping)

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

    policies = [WebsocketClientPolicy(host=args.host, port=p) for p in ports]
    for i, p in enumerate(policies):
        print(f"[arm{i}] server metadata: {p.get_server_metadata()}")
    print(f"num_arms={args.num_arms} ports={ports} cam_map={cam_map}")

    results: list[dict] = []
    for seed in seeds:
        for ep_i in range(args.num_episodes):
            ep_seed = seed * 100_000 + ep_i
            try:
                r = run_episode(env, policies, args, cam_map, ep_seed)
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

    out_file = out_dir / f"eval_decent_{args.task}_{int(time.time())}.json"
    out_file.write_text(json.dumps({"args": dataclasses.asdict(args), "results": results}, indent=2))
    print(f"Saved {out_file}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
