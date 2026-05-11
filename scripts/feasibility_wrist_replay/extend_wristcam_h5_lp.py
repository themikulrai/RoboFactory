"""LP (4-arm) backfill — per-attempt timeout variant of extend_wristcam_h5.py.

Salvages an existing per-task wristcam H5 that ended short of the target trajectory
count by collecting extra demos starting from --start-seed. Differs from
extend_wristcam_h5.py in ONE place: each rollout attempt is bounded by a
per-attempt wall-clock timeout (default 600s) implemented via signal.SIGALRM in a
child collector process, with the parent respawning a fresh collector if the
child process itself hangs / dies (the SIGALRM-unrecoverable case for SAPIEN/Vulkan).

Designed for LongPipelineDelivery, where the data-gen job 15382995_3 deterministically
hung at seed 13 after capturing seeds 0-12. The bounded attempt + skip-on-timeout
behaviour lets us skip pathological seeds and march on.

Usage:
    python extend_wristcam_h5_lp.py \\
        --task LongPipelineDelivery \\
        --start-seed 13 --max-seed 400 --target-extra 137 \\
        --per-attempt-timeout 600 --max-retries 1 \\
        --existing-h5 /iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/LongPipelineDelivery/LongPipelineDelivery.h5
"""
import os
os.environ.setdefault("SAPIEN_HEADLESS", "1")

import argparse
import json
import multiprocessing as mp
import os.path as osp
import shutil
import signal
import sys
import time
import traceback

import h5py


# ------------------------------------------------------------------ helpers


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
        sys.exit(
            f"[ERROR] existing H5 has 0 trajectories — refusing to extend an empty file: {existing_h5}"
        )
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


# ----------------------------------------------------- child collector worker


class _AttemptTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _AttemptTimeout("per-attempt SIGALRM tripped")


def _child_collect(
    task_name,
    tmp_dir,
    start_seed,
    max_seed,
    target_extra,
    max_retries,
    per_attempt_timeout,
    progress_queue,
):
    """Runs in a child process. Streams (kind, payload) tuples back via progress_queue.

    kinds:
      'init_ok'        : env build complete
      'success'        : payload = (seed, attempt_idx)  -- traj flushed to tmp h5
      'fail'           : payload = (seed, attempt_idx, reason_str)
      'timeout'        : payload = (seed, attempt_idx)  -- SIGALRM tripped, traj discarded
      'seed_done'      : payload = (seed, ok_bool)      -- emitted after each seed
      'collection_done': payload = list[int] success_seeds
      'fatal'          : payload = traceback string
    """
    try:
        env, solver, RecordEpisodeMA = _build_env(task_name)
    except Exception:
        progress_queue.put(("fatal", traceback.format_exc()))
        return

    os.makedirs(tmp_dir, exist_ok=True)
    trajectory_name = f"{task_name}_extend"
    env = RecordEpisodeMA(
        env,
        output_dir=tmp_dir,
        trajectory_name=trajectory_name,
        save_video=False,
        source_type="motionplanning",
        source_desc=(
            "wristcam extend backfill (LP, per-attempt-timeout); appended to existing dataset "
            "to reach target traj count (robot_uids=panda_wristcam_multi)"
        ),
        video_fps=30,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,
        record_observation=True,
    )

    # Wire SIGALRM handler in this child only.
    signal.signal(signal.SIGALRM, _alarm_handler)

    progress_queue.put(("init_ok", None))

    success_seeds = []
    for seed in range(start_seed, max_seed + 1):
        if len(success_seeds) >= target_extra:
            break
        success_this_seed = False
        for attempt in range(max_retries):
            signal.alarm(int(per_attempt_timeout))
            try:
                res = solver(env, seed=seed, debug=False, vis=False)
                signal.alarm(0)  # disarm on normal return
            except _AttemptTimeout:
                signal.alarm(0)
                # Solver hung. Discard the in-flight trajectory; do NOT retry the same seed.
                try:
                    env.flush_trajectory(save=False)
                except Exception:
                    pass
                progress_queue.put(("timeout", (seed, attempt)))
                # Per requirements: hangs => single-attempt skip, break out.
                break
            except Exception as e:
                signal.alarm(0)
                try:
                    env.flush_trajectory(save=False)
                except Exception:
                    pass
                progress_queue.put(("fail", (seed, attempt, f"solver_exc: {e!r}")))
                continue

            ok = res != -1 and bool(res[-1]["success"].item())
            if ok:
                env.flush_trajectory()
                success_seeds.append(seed)
                success_this_seed = True
                progress_queue.put(("success", (seed, attempt)))
                break
            else:
                env.flush_trajectory(save=False)
                progress_queue.put(("fail", (seed, attempt, "solver_returned_unsuccessful")))
        progress_queue.put(("seed_done", (seed, success_this_seed)))

    try:
        env.close()
    except Exception:
        pass

    progress_queue.put(("collection_done", success_seeds))


# ----------------------------------------------------- parent supervisor


def _collect_to_tmp(
    task_name, tmp_dir, start_seed, max_seed, target_extra, max_retries,
    per_attempt_timeout, child_heartbeat_grace,
):
    """Spawn a child collector; respawn on hang/death; aggregate success seeds.

    Returns (tmp_h5_path, tmp_json_path, ordered_success_seeds).

    On respawn, the child resumes at last_seed_seen+1 (the seed that hung or that
    the child was working on when it died). The tmp_dir is shared across respawns
    so successful trajs accumulate into the same H5 (RecordEpisodeMA opens append).

    `child_heartbeat_grace` is how many extra seconds beyond per_attempt_timeout
    the parent waits for ANY message from the child before declaring it dead.
    Suggest per_attempt_timeout + 120s (covers env-init + flush slack).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    success_seeds = []
    next_seed = start_seed
    fail_counts = {}  # seed -> n_recorded_failures (timeouts + solver fails)

    ctx = mp.get_context("spawn")  # spawn avoids inheriting SAPIEN/CUDA state

    while len(success_seeds) < target_extra and next_seed <= max_seed:
        print(f"[parent] spawning child collector resuming at seed={next_seed} "
              f"(have {len(success_seeds)}/{target_extra})")
        sys.stdout.flush()
        q = ctx.Queue()
        p = ctx.Process(
            target=_child_collect,
            kwargs=dict(
                task_name=task_name,
                tmp_dir=tmp_dir,
                start_seed=next_seed,
                max_seed=max_seed,
                target_extra=target_extra - len(success_seeds),
                max_retries=max_retries,
                per_attempt_timeout=per_attempt_timeout,
                progress_queue=q,
            ),
        )
        p.start()

        last_event_t = time.time()
        last_seed_attempted = next_seed - 1  # nothing yet
        child_done = False
        watchdog_timeout = per_attempt_timeout + child_heartbeat_grace
        # Allow generous slack on the very first message (env init).
        first_msg_timeout = max(watchdog_timeout, 300)
        got_first_msg = False

        while True:
            try:
                kind, payload = q.get(timeout=5)
            except Exception:
                # queue empty — check timeouts + liveness
                idle = time.time() - last_event_t
                limit = first_msg_timeout if not got_first_msg else watchdog_timeout
                if idle > limit:
                    print(f"[parent] WATCHDOG: no child message for {idle:.0f}s "
                          f"(limit={limit:.0f}s) — terminating child")
                    sys.stdout.flush()
                    p.terminate()
                    p.join(timeout=10)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=5)
                    # If a seed was in flight, mark it as a fail/timeout and skip.
                    if last_seed_attempted >= next_seed:
                        skip_seed = last_seed_attempted
                    else:
                        skip_seed = next_seed
                    fail_counts[skip_seed] = fail_counts.get(skip_seed, 0) + 1
                    next_seed = skip_seed + 1
                    break  # respawn outer loop
                if not p.is_alive():
                    # Process exited without sending collection_done — treat as crash
                    print(f"[parent] child exited unexpectedly (exitcode={p.exitcode}); "
                          f"will respawn from seed {last_seed_attempted + 1}")
                    sys.stdout.flush()
                    next_seed = last_seed_attempted + 1
                    break
                continue

            got_first_msg = True
            last_event_t = time.time()

            if kind == "init_ok":
                print(f"[child] env initialised OK")
                sys.stdout.flush()
            elif kind == "success":
                seed, attempt = payload
                success_seeds.append(seed)
                last_seed_attempted = seed
                print(f"  seed {seed}: SUCCESS (attempt {attempt+1}) — collected "
                      f"{len(success_seeds)}/{target_extra}")
                sys.stdout.flush()
            elif kind == "fail":
                seed, attempt, reason = payload
                last_seed_attempted = seed
                fail_counts[seed] = fail_counts.get(seed, 0) + 1
                print(f"  seed {seed}: FAIL (attempt {attempt+1}) reason={reason}")
                sys.stdout.flush()
            elif kind == "timeout":
                seed, attempt = payload
                last_seed_attempted = seed
                fail_counts[seed] = fail_counts.get(seed, 0) + 1
                print(f"  seed {seed}: TIMEOUT (attempt {attempt+1}) — discarding & skipping")
                sys.stdout.flush()
            elif kind == "seed_done":
                seed, ok = payload
                last_seed_attempted = seed
                if not ok:
                    print(f"  seed {seed}: no success after all attempts")
                    sys.stdout.flush()
            elif kind == "collection_done":
                got_seeds = payload
                # `got_seeds` reflects the child's view (subset of success_seeds we just appended).
                child_done = True
                next_seed = (max(success_seeds + [next_seed - 1]) + 1)
                print(f"[child] collection_done; child reports {len(got_seeds)} successes")
                sys.stdout.flush()
                break
            elif kind == "fatal":
                print(f"[child] FATAL — child traceback:\n{payload}")
                sys.stdout.flush()
                # treat as crash, respawn from next seed
                next_seed = last_seed_attempted + 1
                break

        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)

        if child_done:
            break

    tmp_h5 = osp.join(tmp_dir, f"{task_name}_extend.h5")
    tmp_json = osp.join(tmp_dir, f"{task_name}_extend.json")
    if not osp.exists(tmp_h5) or not osp.exists(tmp_json):
        sys.exit(
            f"[ERROR] tmp outputs missing after collection:\n  h5={tmp_h5}\n  json={tmp_json}\n"
            f"  success_seeds={success_seeds}"
        )
    return tmp_h5, tmp_json, success_seeds


def _validate_schemas(existing_h5, tmp_h5):
    """Recursively collect dataset paths + dtypes; abort if they diverge."""
    def collect(group, prefix=""):
        out = {}
        for k in group.keys():
            sub = group[k]
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(sub, h5py.Group):
                out.update(collect(sub, path))
            else:
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

    # Explicit per-required-key sanity print for the LP 4-arm schema (no assertion;
    # the diff above is authoritative — this just makes the log obviously cover them).
    required = []
    for i in range(4):
        required += [
            f"actions/panda-{i}",
            f"obs/sensor_data/hand_camera_{i}/rgb",
            f"obs/sensor_data/head_camera_agent{i}/rgb",
        ]
    required.append("obs/sensor_data/head_camera_global/rgb")
    seen = sorted(set(tmp_schema) & set(existing_schema))
    missing_required = [r for r in required if r not in seen]
    if missing_required:
        print(f"[validate] NOTE: required-key list contained entries absent from BOTH H5s "
              f"(harmless if your task lacks those cameras): {missing_required}")
    else:
        print(f"[validate] all {len(required)} 4-arm cam/action keys match between existing & tmp")


def _merge(existing_h5, existing_json_path, existing_json_data, tmp_h5, tmp_json, success_seeds):
    """Copy tmp traj groups into existing H5, append matching JSON entries atomically."""
    existing_count = len(existing_json_data["episodes"])
    with open(tmp_json, "r") as f:
        tmp_json_data = json.load(f)
    tmp_episodes = tmp_json_data["episodes"]
    if len(tmp_episodes) != len(success_seeds):
        sys.exit(
            f"[ERROR] tmp JSON has {len(tmp_episodes)} episodes but {len(success_seeds)} successes were tallied"
        )

    with h5py.File(tmp_h5, "r") as ftmp:
        tmp_traj_keys_sorted = sorted(ftmp.keys(), key=lambda k: int(k.split("_")[1]))
        if len(tmp_traj_keys_sorted) != len(tmp_episodes):
            sys.exit(
                f"[ERROR] tmp H5 has {len(tmp_traj_keys_sorted)} groups but tmp JSON has {len(tmp_episodes)} episodes"
            )

    with h5py.File(existing_h5, "r") as fex:
        existing_keys = set(fex.keys())
    new_keys = [f"traj_{existing_count + i}" for i in range(len(tmp_traj_keys_sorted))]
    collisions = [k for k in new_keys if k in existing_keys]
    if collisions:
        sys.exit(f"[ERROR] target traj keys already exist in existing H5: {collisions[:10]}")

    for i, (ep, seed) in enumerate(zip(tmp_episodes, success_seeds)):
        recorded_seed = ep.get("reset_kwargs", {}).get("seed")
        if recorded_seed != seed:
            print(
                f"[WARN] tmp episode {i} reset_kwargs.seed={recorded_seed} != tallied success_seeds[{i}]={seed}; "
                "trusting tmp JSON sidecar value"
            )

    appended_episodes = []
    for i, ep in enumerate(tmp_episodes):
        new_ep = dict(ep)
        new_ep["episode_id"] = existing_count + i
        appended_episodes.append(new_ep)
    new_json_data = dict(existing_json_data)
    new_json_data["episodes"] = list(existing_json_data["episodes"]) + appended_episodes

    with h5py.File(tmp_h5, "r") as ftmp, h5py.File(existing_h5, "a") as fex:
        for src_key, dst_key in zip(tmp_traj_keys_sorted, new_keys):
            fex.copy(ftmp[src_key], fex, name=dst_key)
            print(f"  copied {src_key} -> {dst_key}")
            sys.stdout.flush()
        fex.flush()
        post_count = len(fex.keys())

    expected = existing_count + len(tmp_traj_keys_sorted)
    if post_count != expected:
        sys.exit(f"[ERROR] post-merge H5 has {post_count} groups, expected {expected}")

    tmp_json_out = existing_json_path + ".new"
    with open(tmp_json_out, "w") as f:
        json.dump(new_json_data, f, indent=2)
    os.replace(tmp_json_out, existing_json_path)
    print(f"[extend] JSON replaced atomically: {existing_json_path}")

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


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--start-seed", type=int, required=True)
    ap.add_argument("--max-seed", type=int, required=True)
    ap.add_argument("--target-extra", type=int, required=True)
    ap.add_argument("--max-retries", type=int, default=1,
                    help="attempts per seed (NOT counting timeouts — a timeout always single-shots)")
    ap.add_argument("--per-attempt-timeout", type=int, default=600,
                    help="seconds before SIGALRM aborts a single solver call (LP success ~4.4 min)")
    ap.add_argument("--child-heartbeat-grace", type=int, default=180,
                    help="extra seconds parent waits beyond per-attempt-timeout before declaring child dead")
    ap.add_argument("--existing-h5", type=str, required=True)
    ap.add_argument("--tmp-dir", type=str, default=None,
                    help="override scratch dir (default /tmp/extend_<task>_<jobid>_<ts>)")
    args = ap.parse_args()

    if args.target_extra <= 0:
        sys.exit("[ERROR] --target-extra must be > 0")
    if args.max_seed < args.start_seed:
        sys.exit("[ERROR] --max-seed must be >= --start-seed")

    existing_traj_keys, existing_json_path, existing_json_data = _load_existing(args.existing_h5)
    current_count = len(existing_traj_keys)
    print(f"[extend] existing H5: {args.existing_h5}")
    print(f"[extend] existing JSON: {existing_json_path}")
    print(f"[extend] current_count={current_count}")
    print(f"[extend] target_extra={args.target_extra}  start_seed={args.start_seed}  "
          f"max_seed={args.max_seed}  per_attempt_timeout={args.per_attempt_timeout}s")
    sys.stdout.flush()

    jobid = os.environ.get("SLURM_JOB_ID", "nojob")
    tmp_dir = args.tmp_dir or f"/tmp/extend_{args.task}_{jobid}_{int(time.time())}"

    try:
        tmp_h5, tmp_json, success_seeds = _collect_to_tmp(
            task_name=args.task,
            tmp_dir=tmp_dir,
            start_seed=args.start_seed,
            max_seed=args.max_seed,
            target_extra=args.target_extra,
            max_retries=args.max_retries,
            per_attempt_timeout=args.per_attempt_timeout,
            child_heartbeat_grace=args.child_heartbeat_grace,
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

    try:
        _validate_schemas(args.existing_h5, tmp_h5)
    except SystemExit:
        print(f"[extend] schema validation failed; tmp dir LEFT for inspection: {tmp_dir}")
        raise

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
            "[extend] WARNING: existing H5 may be partially mutated. Inspect both files before retrying."
        )
        sys.exit(4)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[extend] tmp dir cleaned: {tmp_dir}")
    print("[extend] DONE.")


if __name__ == "__main__":
    main()
