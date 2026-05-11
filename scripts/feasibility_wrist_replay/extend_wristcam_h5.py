"""Append additional successful wristcam trajectories to an existing per-task H5.

Salvages an existing (task).h5 that ended short of 150 trajectories by collecting
extra demos starting from --start-seed (the seed budget that comes AFTER the
original collection's seed range) until --target-extra successes accumulate, then
merging them into the canonical H5 + JSON in place. The original 142 successful
trajectories are NEVER re-run.

Designed for the post-seedfix PlaceFood salvage (142 -> 150) but task-agnostic.

Usage:
    python extend_wristcam_h5.py \\
        --task PlaceFood \\
        --start-seed 153 --max-seed 300 --target-extra 8 --max-retries 5 \\
        --existing-h5 /iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/PlaceFood/PlaceFood.h5

The new H5 traj keys are appended as traj_<N>, traj_<N+1>, ... where N is the
existing trajectory count (NOT next_seed). JSON episodes get episode_id starting
at N as well; episode_seed records the actual solver seed used.
"""
import os
os.environ.setdefault("SAPIEN_HEADLESS", "1")

import argparse
import json
import os.path as osp
import shutil
import sys
import time
import traceback

import h5py


# Imported lazily inside run() so that --help and arg parsing do not require
# a working SAPIEN/Vulkan stack.


def _load_existing(existing_h5):
    """Return (h5_traj_keys_sorted, json_path, json_data). Refuse if empty."""
    if not osp.exists(existing_h5):
        sys.exit(f"[ERROR] existing H5 not found: {existing_h5}")
    json_path = existing_h5.replace(".h5", ".json")
    if not osp.exists(json_path):
        sys.exit(f"[ERROR] existing JSON sidecar not found: {json_path}")
    with h5py.File(existing_h5, "r") as f:
        traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
    if len(traj_keys) == 0:
        sys.exit(f"[ERROR] existing H5 has 0 trajectories — refusing to extend an empty file: {existing_h5}")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    if len(json_data.get("episodes", [])) != len(traj_keys):
        sys.exit(
            f"[ERROR] existing JSON episode count ({len(json_data['episodes'])}) "
            f"!= existing H5 traj count ({len(traj_keys)}). Refusing to merge."
        )
    return traj_keys, json_path, json_data


def _build_env(task_name):
    """Build the gym env + RecordEpisodeMA wrapper matching run_wristcam_rollouts.py exactly."""
    # Local imports so --help works without sapien.
    import gymnasium as gym
    import robofactory  # noqa: F401  registers envs + PandaWristCamMulti
    from robofactory import CONFIG_DIR
    from robofactory.planner.solutions import (
        solveTakePhoto,
        solveLongPipelineDelivery,
        solveThreeRobotsStackCube,
        solvePickMeat,
        solveTwoRobotsStackCube,
        solveLiftBarrier,
        solvePlaceFood,
    )
    from robofactory.utils.wrappers.record import RecordEpisodeMA

    TASK_MAP = {
        "TakePhoto":            ("TakePhoto-rf",            "table/take_photo.yaml",              solveTakePhoto,            4),
        "LongPipelineDelivery": ("LongPipelineDelivery-rf", "table/long_pipeline_delivery.yaml",  solveLongPipelineDelivery, 4),
        "ThreeRobotsStackCube": ("ThreeRobotsStackCube-rf", "table/three_robots_stack_cube.yaml", solveThreeRobotsStackCube, 3),
        "PickMeat":             ("PickMeat-rf",             "table/pick_meat.yaml",               solvePickMeat,             1),
        "TwoRobotsStackCube":   ("TwoRobotsStackCube-rf",   "table/two_robots_stack_cube.yaml",   solveTwoRobotsStackCube,   2),
        "LiftBarrier":          ("LiftBarrier-rf",          "table/lift_barrier.yaml",            solveLiftBarrier,          2),
        "PlaceFood":            ("PlaceFood-rf",            "table/place_food.yaml",              solvePlaceFood,            2),
    }
    if task_name not in TASK_MAP:
        sys.exit(f"[ERROR] unknown task {task_name!r}; choices = {list(TASK_MAP)}")

    env_id, yaml_rel, solver, n_agents = TASK_MAP[task_name]
    config_path = osp.join(CONFIG_DIR, yaml_rel)

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
    return env, solver, RecordEpisodeMA


def _collect_to_tmp(task_name, tmp_dir, start_seed, max_seed, target_extra, max_retries):
    """Run the solver across seeds in [start_seed, max_seed] until target_extra successes.

    Writes a fresh <task>_extend.h5 / .json under tmp_dir. Returns
    (tmp_h5_path, tmp_json_path, ordered_success_seeds).
    """
    env, solver, RecordEpisodeMA = _build_env(task_name)

    os.makedirs(tmp_dir, exist_ok=True)
    trajectory_name = f"{task_name}_extend"
    env = RecordEpisodeMA(
        env,
        output_dir=tmp_dir,
        trajectory_name=trajectory_name,
        save_video=False,
        source_type="motionplanning",
        source_desc=(
            "wristcam extend backfill; appended to existing dataset to reach target traj count "
            "(robot_uids=panda_wristcam_multi)"
        ),
        video_fps=30,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,
        record_observation=True,
    )

    success_seeds = []
    print(f"[extend] tmp_dir={tmp_dir}")
    print(f"[extend] sweeping seeds [{start_seed}..{max_seed}] target={target_extra} max_retries={max_retries}")
    sys.stdout.flush()

    for seed in range(start_seed, max_seed + 1):
        if len(success_seeds) >= target_extra:
            break
        success_this_seed = False
        for attempt in range(max_retries):
            try:
                res = solver(env, seed=seed, debug=False, vis=False)
            except Exception as e:
                print(f"  seed {seed}: solver EXCEPTION (attempt {attempt+1}): {e!r}")
                sys.stdout.flush()
                env.flush_trajectory(save=False)
                continue
            ok = res != -1 and bool(res[-1]["success"].item())
            if ok:
                env.flush_trajectory()
                success_seeds.append(seed)
                success_this_seed = True
                print(f"  seed {seed}: SUCCESS (attempt {attempt+1}) — collected {len(success_seeds)}/{target_extra}")
                sys.stdout.flush()
                break
            env.flush_trajectory(save=False)
        if not success_this_seed:
            print(f"  seed {seed}: FAILED after {max_retries} attempts — skipping")
            sys.stdout.flush()

    env.close()

    tmp_h5 = osp.join(tmp_dir, f"{trajectory_name}.h5")
    tmp_json = osp.join(tmp_dir, f"{trajectory_name}.json")
    if not osp.exists(tmp_h5) or not osp.exists(tmp_json):
        sys.exit(
            f"[ERROR] tmp outputs missing after collection:\n  h5={tmp_h5}\n  json={tmp_json}"
        )
    return tmp_h5, tmp_json, success_seeds


def _validate_schemas(existing_h5, tmp_h5):
    """Recursively collect dataset paths + dtypes; abort if they diverge between tmp[traj_0] and existing[last_traj]."""
    def collect(group, prefix=""):
        out = {}
        for k in group.keys():
            sub = group[k]
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(sub, h5py.Group):
                out.update(collect(sub, path))
            else:
                # Compare dtype + ndim, NOT shape[0] which is episode length.
                out[path] = (str(sub.dtype), sub.ndim, tuple(sub.shape[1:]))
        return out

    with h5py.File(existing_h5, "r") as f:
        last_key = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[-1]
        existing_schema = collect(f[last_key])
    with h5py.File(tmp_h5, "r") as f:
        first_key = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[0]
        tmp_schema = collect(f[first_key])

    only_existing = set(existing_schema) - set(tmp_schema)
    only_tmp = set(tmp_schema) - set(existing_schema)
    mismatched = {
        k: (existing_schema[k], tmp_schema[k])
        for k in set(existing_schema) & set(tmp_schema)
        if existing_schema[k] != tmp_schema[k]
    }
    if only_existing or only_tmp or mismatched:
        msg = ["[ERROR] H5 schema mismatch between existing and tmp:"]
        if only_existing:
            msg.append(f"  only in existing: {sorted(only_existing)[:20]}")
        if only_tmp:
            msg.append(f"  only in tmp:      {sorted(only_tmp)[:20]}")
        if mismatched:
            for k, (e, t) in list(mismatched.items())[:20]:
                msg.append(f"  mismatched dtype/shape@{k}: existing={e} tmp={t}")
        sys.exit("\n".join(msg))


def _merge(existing_h5, existing_json_path, existing_json_data, tmp_h5, tmp_json, success_seeds):
    """Copy tmp traj groups into existing H5, append matching JSON entries atomically."""
    existing_count = len(existing_json_data["episodes"])

    # Load tmp JSON sidecar to crib control_mode / reset_kwargs / elapsed_steps / success
    with open(tmp_json, "r") as f:
        tmp_json_data = json.load(f)
    tmp_episodes = tmp_json_data["episodes"]
    if len(tmp_episodes) != len(success_seeds):
        sys.exit(
            f"[ERROR] tmp JSON has {len(tmp_episodes)} episodes but {len(success_seeds)} successes were tallied"
        )

    # Sanity: confirm tmp h5 traj count == tmp json count
    with h5py.File(tmp_h5, "r") as ftmp:
        tmp_traj_keys_sorted = sorted(ftmp.keys(), key=lambda k: int(k.split("_")[1]))
        if len(tmp_traj_keys_sorted) != len(tmp_episodes):
            sys.exit(
                f"[ERROR] tmp H5 has {len(tmp_traj_keys_sorted)} groups but tmp JSON has {len(tmp_episodes)} episodes"
            )

    # Compute target keys + collision check (open existing readonly first)
    with h5py.File(existing_h5, "r") as fex:
        existing_keys = set(fex.keys())
    new_keys = [f"traj_{existing_count + i}" for i in range(len(tmp_traj_keys_sorted))]
    collisions = [k for k in new_keys if k in existing_keys]
    if collisions:
        sys.exit(f"[ERROR] target traj keys already exist in existing H5: {collisions[:10]}")

    # Cross-check that each tmp episode's recorded seed lines up with success_seeds order
    for i, (ep, seed) in enumerate(zip(tmp_episodes, success_seeds)):
        recorded_seed = ep.get("reset_kwargs", {}).get("seed")
        if recorded_seed != seed:
            print(
                f"[WARN] tmp episode {i} reset_kwargs.seed={recorded_seed} != tallied success_seeds[{i}]={seed}; "
                "trusting tmp JSON sidecar value"
            )

    # Append episodes to in-memory JSON (deepcopy so we don't mutate on dry-run paths)
    appended_episodes = []
    for i, ep in enumerate(tmp_episodes):
        new_ep = dict(ep)  # shallow copy
        new_ep["episode_id"] = existing_count + i
        # Keep episode_seed and reset_kwargs straight from the tmp sidecar — that's authoritative.
        appended_episodes.append(new_ep)
    new_json_data = dict(existing_json_data)
    new_json_data["episodes"] = list(existing_json_data["episodes"]) + appended_episodes

    # --- copy H5 groups (open existing in append mode) ---
    with h5py.File(tmp_h5, "r") as ftmp, h5py.File(existing_h5, "a") as fex:
        for src_key, dst_key in zip(tmp_traj_keys_sorted, new_keys):
            fex.copy(ftmp[src_key], fex, name=dst_key)
            print(f"  copied {src_key} -> {dst_key}")
            sys.stdout.flush()
        fex.flush()
        # Confirm H5 count == existing_count + new
        post_count = len(fex.keys())

    expected = existing_count + len(tmp_traj_keys_sorted)
    if post_count != expected:
        sys.exit(f"[ERROR] post-merge H5 has {post_count} groups, expected {expected}")

    # --- atomically replace JSON ---
    tmp_json_out = existing_json_path + ".new"
    with open(tmp_json_out, "w") as f:
        json.dump(new_json_data, f, indent=2)
    os.replace(tmp_json_out, existing_json_path)
    print(f"[extend] JSON replaced atomically: {existing_json_path}")

    # Final consistency check
    with h5py.File(existing_h5, "r") as fex:
        final_h5_count = len(fex.keys())
    with open(existing_json_path, "r") as f:
        final_json_count = len(json.load(f)["episodes"])
    if final_h5_count != final_json_count:
        sys.exit(
            f"[ERROR] post-merge mismatch: H5={final_h5_count} groups vs JSON={final_json_count} episodes"
        )
    print(f"[extend] FINAL: H5={final_h5_count} trajs, JSON={final_json_count} episodes (matched)")
    return final_h5_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--start-seed", type=int, required=True)
    ap.add_argument("--max-seed", type=int, required=True)
    ap.add_argument("--target-extra", type=int, required=True)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--existing-h5", type=str, required=True)
    args = ap.parse_args()

    if args.target_extra <= 0:
        sys.exit("[ERROR] --target-extra must be > 0")
    if args.max_seed < args.start_seed:
        sys.exit("[ERROR] --max-seed must be >= --start-seed")

    # 1) Read existing H5 + JSON. Refuse on empty / mismatched.
    existing_traj_keys, existing_json_path, existing_json_data = _load_existing(args.existing_h5)
    current_count = len(existing_traj_keys)
    next_episode_id = current_count
    print(f"[extend] existing H5: {args.existing_h5}")
    print(f"[extend] existing JSON: {existing_json_path}")
    print(f"[extend] current_count={current_count} next_episode_id={next_episode_id}")
    sys.stdout.flush()

    # 2) Collect to a TEMP dir.
    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    tmp_dir = f"/tmp/extend_{args.task}_{jobid}_{int(time.time())}"

    try:
        tmp_h5, tmp_json, success_seeds = _collect_to_tmp(
            task_name=args.task,
            tmp_dir=tmp_dir,
            start_seed=args.start_seed,
            max_seed=args.max_seed,
            target_extra=args.target_extra,
            max_retries=args.max_retries,
        )
    except Exception:
        traceback.print_exc()
        print(f"[extend] collection failed; tmp dir LEFT for inspection: {tmp_dir}")
        sys.exit(2)

    if len(success_seeds) != args.target_extra:
        print(
            f"[extend] only collected {len(success_seeds)}/{args.target_extra} successes "
            f"(seeds tried up to {args.max_seed}); aborting merge — tmp dir left for inspection: {tmp_dir}"
        )
        sys.exit(3)

    # 3) Validate schemas before touching the real H5.
    try:
        _validate_schemas(args.existing_h5, tmp_h5)
    except SystemExit:
        print(f"[extend] schema validation failed; tmp dir LEFT for inspection: {tmp_dir}")
        raise

    # 4) Merge (H5 copy + atomic JSON replace).
    try:
        _merge(
            existing_h5=args.existing_h5,
            existing_json_path=existing_json_path,
            existing_json_data=existing_json_data,
            tmp_h5=tmp_h5,
            tmp_json=tmp_json,
            success_seeds=success_seeds,
        )
    except Exception:
        traceback.print_exc()
        print(f"[extend] MERGE FAILED — tmp dir LEFT for inspection: {tmp_dir}")
        print(
            "[extend] WARNING: existing H5 may be partially mutated (H5 copy may have added some groups "
            "before the failure). Inspect both files before retrying."
        )
        sys.exit(4)

    # 5) On success, rm -rf tmp dir.
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[extend] tmp dir cleaned: {tmp_dir}")
    print("[extend] DONE.")


if __name__ == "__main__":
    main()
