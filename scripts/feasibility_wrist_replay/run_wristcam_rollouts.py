"""Collect wrist-cam demos via motion planner — drop-in replacement for hf_download.

Output mirrors hf_download/<Task>/<Task>.h5 + .json layout so downstream code
can swap the root path with no other changes. New H5 contains all original head
cameras PLUS per-arm hand_camera_{i} at 224x224.

Usage (pilot — verify success rate before committing to 150):
    python scripts/feasibility_wrist_replay/run_wristcam_rollouts.py \\
        --task LongPipelineDelivery --num 20 \\
        --record-dir /iris/u/mikulrai/datasets/multi_robot/RoboFactory/hf_download_wristcam_pilot

Usage (full dataset):
    python scripts/feasibility_wrist_replay/run_wristcam_rollouts.py \\
        --task LongPipelineDelivery --num 150 \\
        --record-dir /iris/u/mikulrai/datasets/multi_robot/RoboFactory/hf_download_wristcam
"""
import os
os.environ.setdefault("SAPIEN_HEADLESS", "1")

import argparse
import json
import os.path as osp
import signal
import sys

import gymnasium as gym
from tqdm import tqdm


class _SolverTimeout(Exception):
    """Raised by SIGALRM handler when a single solver attempt exceeds budget."""


def _solver_alarm_handler(signum, frame):
    raise _SolverTimeout()

import robofactory  # registers envs + PandaWristCamMulti
from robofactory.planner.solutions import (
    solveTakePhoto,
    solveLongPipelineDelivery,
    solveThreeRobotsStackCube,
    solvePickMeat,
    solveTwoRobotsStackCube,
    solveLiftBarrier,
    solvePlaceFood,
    solveStackCube,
)
from robofactory.utils.wrappers.record import RecordEpisodeMA
from robofactory import CONFIG_DIR

HF_DOWNLOAD_ROOT = "/iris/u/mikulrai/datasets/multi_robot/RoboFactory/hf_download"

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
    "PickMeat": (
        "PickMeat-rf",
        "table/pick_meat.yaml",
        solvePickMeat,
        1,
    ),
    "TwoRobotsStackCube": (
        "TwoRobotsStackCube-rf",
        "table/two_robots_stack_cube.yaml",
        solveTwoRobotsStackCube,
        2,
    ),
    "LiftBarrier": (
        "LiftBarrier-rf",
        "table/lift_barrier.yaml",
        solveLiftBarrier,
        2,
    ),
    "PlaceFood": (
        "PlaceFood-rf",
        "table/place_food.yaml",
        solvePlaceFood,
        2,
    ),
    "StackCube": (
        "StackCube-rf",
        "table/stack_cube.yaml",
        solveStackCube,
        1,
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


def run(task_name, num, record_dir, max_retries_per_seed=5,
        per_attempt_timeout=300, override_seeds=None):
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
    seeds = override_seeds if override_seeds is not None else _load_seeds(task_name, num)
    print(f"[wristcam] per_attempt_timeout={per_attempt_timeout}s "
          f"(SIGALRM aborts solver call; periodic _h5_file.flush() after each "
          f"successful episode persists link B-tree to disk)", flush=True)

    signal.signal(signal.SIGALRM, _solver_alarm_handler)

    passed = 0
    total_attempts = 0
    timeouts = 0
    pbar = tqdm(total=num, desc=task_name)

    for seed in seeds:
        if passed >= num:
            break
        success_this_seed = False
        timed_out_this_seed = False
        for attempt in range(max_retries_per_seed):
            total_attempts += 1
            signal.alarm(int(per_attempt_timeout))
            try:
                res = solver(env, seed=seed, debug=False, vis=False)
            except _SolverTimeout:
                timeouts += 1
                timed_out_this_seed = True
                print(f"  seed {seed}: TIMEOUT after {per_attempt_timeout}s "
                      f"(attempt {attempt+1}); skipping seed entirely "
                      f"(suspected unrecoverable hang)", flush=True)
                try:
                    env.flush_trajectory(save=False)
                except Exception as e:
                    print(f"    [WARN] could not clear in-progress trajectory "
                          f"buffer after timeout: {type(e).__name__}: {e}",
                          flush=True)
                break  # don't retry; treat as deterministic hang
            finally:
                signal.alarm(0)
            ok = res != -1 and bool(res[-1]["success"].item())
            if ok:
                env.flush_trajectory()
                # Persist the group-link B-tree to disk so a SIGKILL during
                # the next solver call leaves a valid (truncated) H5, not a
                # corrupted one with EOA=2048.
                try:
                    env._h5_file.flush()
                except AttributeError:
                    pass
                passed += 1
                pbar.update(1)
                success_this_seed = True
                print(f"  seed {seed}: SUCCESS (attempt {attempt+1})", flush=True)
                break
            env.flush_trajectory(save=False)
        if not success_this_seed and not timed_out_this_seed:
            print(f"  seed {seed}: FAILED after {max_retries_per_seed} attempts — skipping", flush=True)

    pbar.close()
    env.close()

    print()
    print(f"[wristcam] collected {passed}/{num} demos in {total_attempts} total sim runs "
          f"(timeouts: {timeouts})")
    print(f"[wristcam] h5: {out_h5}")

    if passed < num:
        print(
            f"[WARN] only got {passed}/{num} — "
            f"raise --max-retries or investigate solver success rate"
        )
        # Hard-fail only if the shortfall is severe (>5%); partial datasets
        # are still useful for downstream conversion + training.
        shortfall = (num - passed) / num
        if shortfall > 0.05:
            raise RuntimeError(
                f"collection severely incomplete: {passed}/{num} "
                f"({shortfall:.1%} short). Investigate before reusing."
            )
    return out_h5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_MAP), required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument(
        "--record-dir",
        type=str,
        default="/iris/u/mikulrai/datasets/multi_robot/RoboFactory/hf_download_wristcam",
    )
    ap.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    ap.add_argument("--per-attempt-timeout", type=int, default=300, dest="per_attempt_timeout",
                    help="seconds before SIGALRM aborts a single solver attempt; skip seed on timeout")
    ap.add_argument("--seeds-csv", type=str, default=None, dest="seeds_csv",
                    help="explicit comma-separated seed list; bypasses --num and legacy-JSON seeds")
    args = ap.parse_args()
    override_seeds = None
    if args.seeds_csv:
        override_seeds = [int(s) for s in args.seeds_csv.split(",") if s.strip()]
        num_eff = len(override_seeds)
    else:
        num_eff = args.num
    run(args.task, num_eff, args.record_dir,
        max_retries_per_seed=args.max_retries,
        per_attempt_timeout=args.per_attempt_timeout,
        override_seeds=override_seeds)
