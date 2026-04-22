"""Collect wrist-cam demos via motion planner — drop-in replacement for hf_download.

Output mirrors hf_download/<Task>/<Task>.h5 + .json layout so downstream code
can swap the root path with no other changes. New H5 contains all original head
cameras PLUS per-arm hand_camera_{i} at 224x224.

Usage (pilot — verify success rate before committing to 150):
    python scripts/feasibility_wrist_replay/run_wristcam_rollouts.py \\
        --task LongPipelineDelivery --num 20 \\
        --record-dir /iris/u/mikulrai/data/RoboFactory/hf_download_wristcam_pilot

Usage (full dataset):
    python scripts/feasibility_wrist_replay/run_wristcam_rollouts.py \\
        --task LongPipelineDelivery --num 150 \\
        --record-dir /iris/u/mikulrai/data/RoboFactory/hf_download_wristcam
"""
import os
os.environ.setdefault("SAPIEN_HEADLESS", "1")

import argparse
import json
import os.path as osp
import sys

import gymnasium as gym
from tqdm import tqdm

import robofactory  # registers envs + PandaWristCamMulti
from robofactory.planner.solutions import (
    solveTakePhoto,
    solveLongPipelineDelivery,
    solveThreeRobotsStackCube,
)
from robofactory.utils.wrappers.record import RecordEpisodeMA
from robofactory import CONFIG_DIR

HF_DOWNLOAD_ROOT = "/iris/u/mikulrai/data/RoboFactory/hf_download"

# (env_id, yaml_rel, solver, n_agents)
TASK_MAP = {
    "TakePhoto": (
        "TakePhoto-rf",
        "table/take_photo.yaml",
        solveTakePhoto,
        4,
    ),
    "LongPipelineDelivery": (
        "LongPipelineDelivery-rf",
        "table/long_pipeline_delivery.yaml",
        solveLongPipelineDelivery,
        4,
    ),
    "ThreeRobotsStackCube": (
        "ThreeRobotsStackCube-rf",
        "table/three_robots_stack_cube.yaml",
        solveThreeRobotsStackCube,
        3,
    ),
}


def _load_seeds(task_name, num):
    """Return a list of seeds to attempt. Uses old-dataset seeds when available
    so traj_i lines up with hf_download/traj_i for matched-state comparisons."""
    old_json = osp.join(HF_DOWNLOAD_ROOT, task_name, f"{task_name}.json")
    if osp.exists(old_json):
        episodes = json.load(open(old_json))["episodes"]
        seeds = [ep["episode_seed"] for ep in episodes]
        if len(seeds) >= num:
            print(f"  [seeds] using {num} seeds from {old_json}")
            return seeds[:num]
        print(f"  [seeds] old JSON has {len(seeds)} < {num}; appending sequential")
        seeds += list(range(max(seeds) + 1, max(seeds) + 1 + (num - len(seeds))))
        return seeds[:num]
    print(f"  [seeds] no old JSON found; using sequential 0..{num-1}")
    return list(range(num))


def run(task_name, num, record_dir, max_retries_per_seed=5):
    env_id, yaml_rel, solver, n_agents = TASK_MAP[task_name]
    config_path = osp.join(CONFIG_DIR, yaml_rel)

    # --- output path (mirrors hf_download layout) ---
    output_dir = osp.join(record_dir, task_name)
    out_h5 = osp.join(output_dir, f"{task_name}.h5")
    if osp.exists(out_h5) and osp.getsize(out_h5) > 1024:
        sys.exit(
            f"[ERROR] {out_h5} already exists and is non-empty.\n"
            f"Move it aside before running a fresh collection."
        )
    os.makedirs(output_dir, exist_ok=True)

    # --- build env ---
    env = gym.make(
        env_id,
        config=config_path,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="cpu",
        robot_uids=("panda_wristcam_multi",) * n_agents,
    )

    env = RecordEpisodeMA(
        env,
        output_dir=output_dir,
        trajectory_name=task_name,   # -> <task_name>.h5 + <task_name>.json
        save_video=False,            # H5 already stores every camera stream
        source_type="motionplanning",
        source_desc=(
            "wristcam drop-in dataset; BC-training scope only "
            "(robot_uids=panda_wristcam_multi, not panda)"
        ),
        video_fps=30,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,       # near-free; enables replay verification
        record_observation=True,
    )
    print(f"[wristcam] task={task_name}  env_id={env_id}  n_agents={n_agents}")
    print(f"[wristcam] output_h5={out_h5}")

    # --- seed list aligned with old dataset ---
    seeds = _load_seeds(task_name, num)

    passed = 0
    total_attempts = 0
    pbar = tqdm(total=num, desc=task_name)

    for seed in seeds:
        if passed >= num:
            break
        success_this_seed = False
        for attempt in range(max_retries_per_seed):
            total_attempts += 1
            res = solver(env, seed=seed, debug=False, vis=False)
            ok = res != -1 and bool(res[-1]["success"].item())
            if ok:
                env.flush_trajectory()
                passed += 1
                pbar.update(1)
                success_this_seed = True
                print(f"  seed {seed}: SUCCESS (attempt {attempt+1})")
                break
            env.flush_trajectory(save=False)
        if not success_this_seed:
            print(f"  seed {seed}: FAILED after {max_retries_per_seed} attempts — skipping")

    pbar.close()
    env.close()

    print()
    print(f"[wristcam] collected {passed}/{num} demos in {total_attempts} total sim runs")
    print(f"[wristcam] h5: {out_h5}")

    if passed < num:
        print(
            f"[WARN] only got {passed}/{num} — "
            f"raise --max-retries or investigate solver success rate"
        )
    assert passed == num, f"collection incomplete: {passed}/{num}"
    return out_h5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_MAP), required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument(
        "--record-dir",
        type=str,
        default="/iris/u/mikulrai/data/RoboFactory/hf_download_wristcam",
    )
    ap.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    args = ap.parse_args()
    run(args.task, args.num, args.record_dir, max_retries_per_seed=args.max_retries)
