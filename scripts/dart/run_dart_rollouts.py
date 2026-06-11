"""DART (Disturbances for Augmenting Robot Trajectories) data generation.

Fork of scripts/feasibility_wrist_replay/run_wristcam_rollouts.py. The env build,
RecordEpisodeMA setup, seed loading, SIGALRM per-attempt timeout, and the
success/flush/verify-keep loop are KEPT IDENTICAL.

What DART adds: bounded joint-target noise is injected into the SCRIPTED
motion-planner mid-solve so the planner has to demonstrate RECOVERY from a
drifted arm pose. The noise itself is applied through ``env.unwrapped.step``
(the UNWRAPPED env) so the RecordEpisode buffer never sees it — only the clean
recovery path that the planner re-plans afterwards is recorded. Episodes that
still succeed are kept; the rest are discarded exactly as before.

Mechanism (verified against robofactory/planner/motionplanner.py):
  - ``PandaArmMotionPlanningSolver.move_to_pose_with_screw`` plans from the
    CURRENT qpos (line ~166). So if we perturb qpos *before* that call, the
    planner produces a genuine recovery trajectory back onto task.
  - We monkeypatch ``move_to_pose_with_screw``: with probability ``--p-inject``
    we first step the unwrapped env K times (K~U[k_min,k_max]) with a noisy
    joint target (sigma std-dev added to the 7 arm joints of the moving arm(s)
    only; gripper channel left clean), THEN call the original method.
  - ``self.env.step`` (wrapped) records; ``self.base_env.step`` /
    ``env.unwrapped.step`` does NOT. We assert ``base is self.env.unwrapped``.

Output mirrors hf_download/<Task>/<Task>.h5 + .json. A sidecar dart_meta.json
records the per-episode sigma so a downstream merge/analysis can recover which
trajectories were disturbed.

Usage (pilot):
    python scripts/dart/run_dart_rollouts.py \\
        --task LiftBarrier --num 5 --sigma 0.30 \\
        --record-dir /iris/u/mikulrai/data/RoboFactory/hf_download_dart/sigma0.30

One sigma per process (no inner sigma loop) — sweep sigma by launching one
process per value.
"""
import os
os.environ.setdefault("SAPIEN_HEADLESS", "1")

import argparse
import json
import os.path as osp
import signal
import sys

import gymnasium as gym
import numpy as np
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
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
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


# ----------------------------------------------------------------------------
# DART disturbance hook
# ----------------------------------------------------------------------------
# Captured by the monkeypatch and restored in a finally:. Module-level so the
# original is reachable from the closure that replaces the bound method.
_ORIG_MOVE = PandaArmMotionPlanningSolver.move_to_pose_with_screw

# When DART_CAPTURE_DISTURBANCES is non-empty it is treated as a list onto which
# every emitted disturbance action_dict is appended (used by the sim tests to
# verify no recorded action equals an emitted disturbance). Left None in prod.
_DISTURBANCE_SINK = None


def _inject_disturbance(self, rng, sigma, K, move_id):
    """Step the UNWRAPPED env K times with a noisy joint target so the arm
    drifts off-path. NOT recorded (unwrapped env => RecordEpisode buffer is not
    touched). Only the 7 arm joints of the moving arm(s) get noise; gripper
    channel stays clean.

    Mirrors follow_path's single-agent (flat action) vs multi-agent (dict)
    branching so the hook is general even though LiftBarrier is multi-agent.
    """
    move_id = move_id if isinstance(move_id, list) else [move_id]
    base = self.base_env
    assert base is self.env.unwrapped, (
        "disturbance MUST use unwrapped env (else noise becomes a training label)"
    )

    for _ in range(K):
        if not self.is_multi_agent:
            # single-agent: base.step takes a flat action [7 arm + gripper]
            q = self.robot[0].get_qpos()[0, :-2].cpu().numpy()
            if 0 in move_id:
                q = q + rng.normal(0.0, sigma, size=q.shape[0])  # drift arm only
            action = np.hstack([q, self.gripper_state[0]])  # gripper CLEAN
            if _DISTURBANCE_SINK is not None:
                _DISTURBANCE_SINK.append(np.asarray(action, dtype=np.float64).copy())
            base.step(action)  # UNWRAPPED -> not recorded
        else:
            action_dict = {}
            for aid in range(self.agent_num):
                q = self.robot[aid].get_qpos()[0, :-2].cpu().numpy()
                if aid in move_id:
                    q = q + rng.normal(0.0, sigma, size=q.shape[0])  # 7 joints
                action_dict[f"panda-{aid}"] = np.hstack([q, self.gripper_state[aid]])
            if _DISTURBANCE_SINK is not None:
                _DISTURBANCE_SINK.append(
                    {k: np.asarray(v, dtype=np.float64).copy() for k, v in action_dict.items()}
                )
            base.step(action_dict)  # UNWRAPPED -> not recorded


def _make_dart_move(rng, sigma, k_min, k_max, p_inject):
    """Build the replacement move_to_pose_with_screw. Keeps the original
    signature so existing solver code calls it unchanged."""

    def dart_move(self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1):
        # only inject on REAL moves (dry_run is a planning-only probe) and only
        # with probability p_inject; never perturb env RNG / seed reproducibility
        # (we use a dedicated Generator passed in here).
        if (not dry_run) and rng.random() < p_inject:
            K = int(rng.integers(k_min, k_max + 1))
            _inject_disturbance(self, rng, sigma, K, move_id)
        return _ORIG_MOVE(
            self, pose, dry_run=dry_run, refine_steps=refine_steps,
            move_id=move_id, jump=jump,
        )

    return dart_move


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


def run(task_name, num, record_dir, sigma, k_min=5, k_max=15, p_inject=0.5,
        dart_seed=0, max_retries_per_seed=5, per_attempt_timeout=300,
        override_seeds=None, save_video=False):
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

    # --- build env (IDENTICAL to wristcam fork) ---
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
        # With render_mode="sensors" the saved MP4 tiles the sensor cameras
        # (wrist + head views). The disturbance steps run on the UNWRAPPED env,
        # so they are NOT recorded into the video either — the clip shows a
        # small visual "jump" at the shove, then the clean recovery. That jump
        # is EXPECTED, not a bug. Off by default (H5 already stores every camera
        # stream); the pilot enables it for field-notes recovery clips.
        save_video=save_video,
        source_type="motionplanning",
        source_desc=(
            "DART OOD-recovery dataset; bounded joint-target noise injected "
            "mid-solve via unwrapped env (noise NOT recorded), only clean "
            "recovery kept (robot_uids=panda_wristcam_multi, not panda)"
        ),
        video_fps=30,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,       # near-free; enables replay verification
        record_observation=True,
    )
    print(f"[dart] task={task_name}  env_id={env_id}  n_agents={n_agents}")
    print(f"[dart] output_h5={out_h5}")
    print(f"[dart] sigma={sigma}  k=[{k_min},{k_max}]  p_inject={p_inject}  "
          f"dart_seed={dart_seed}")

    # Dedicated RNG so the disturbance draws NEVER perturb env RNG / the
    # reproducibility of the task seeds.
    rng = np.random.default_rng(dart_seed)

    # --- seed list aligned with old dataset ---
    seeds = override_seeds if override_seeds is not None else _load_seeds(task_name, num)
    print(f"[dart] per_attempt_timeout={per_attempt_timeout}s "
          f"(SIGALRM aborts solver call; periodic _h5_file.flush() after each "
          f"successful episode persists link B-tree to disk)", flush=True)

    signal.signal(signal.SIGALRM, _solver_alarm_handler)

    # episode_id -> sigma. episode_id is the RecordEpisodeMA counter
    # (env._episode_id), read right after a successful flush. The wrapper names
    # the just-saved H5 group "traj_{env._episode_id}", so this id is exactly
    # the H5/JSON traj index. _episode_id starts at -1; first save -> 0.
    episodes_meta = []

    # --- install the DART monkeypatch (restored in finally:) ---
    PandaArmMotionPlanningSolver.move_to_pose_with_screw = _make_dart_move(
        rng, sigma, k_min, k_max, p_inject
    )

    passed = 0
    total_attempts = 0
    timeouts = 0
    pbar = tqdm(total=num, desc=task_name)
    meta_path = osp.join(output_dir, "dart_meta.json")

    def _write_dart_meta():
        """Write the sidecar metadata. Called in a finally: so even a SIGALRM /
        partial run leaves a valid meta for the sweep figure. Records num,
        attempted, passed and the per-episode list (empty when passed==0).
        Never touches the wrapper's own <Task>.json."""
        dart_meta = {
            "task": task_name,
            "sigma": float(sigma),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "p_inject": float(p_inject),
            "dart_seed": int(dart_seed),
            "num": int(num),                   # requested episode count
            "attempted": int(total_attempts),  # total solver runs (incl retries)
            "passed": int(passed),
            "episodes": episodes_meta,         # [] when passed==0
        }
        with open(meta_path, "w") as f:
            json.dump(dart_meta, f, indent=2)

    try:
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
                    # episode_id of the just-saved traj == env._episode_id now.
                    episode_id = int(getattr(env, "_episode_id", passed))
                    episodes_meta.append({
                        "episode_id": episode_id,
                        "seed": int(seed),
                        "sigma": float(sigma),
                    })
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
                    print(f"  seed {seed}: SUCCESS (attempt {attempt+1}) "
                          f"-> traj_{episode_id}", flush=True)
                    break
                env.flush_trajectory(save=False)
            if not success_this_seed and not timed_out_this_seed:
                print(f"  seed {seed}: FAILED after {max_retries_per_seed} attempts — skipping", flush=True)
    finally:
        # ALWAYS restore the original method, even on exception.
        PandaArmMotionPlanningSolver.move_to_pose_with_screw = _ORIG_MOVE
        # ALWAYS write the sidecar metadata, even on SIGALRM / partial run, so
        # the sweep figure always has a valid keep-rate point for this sigma.
        try:
            _write_dart_meta()
        except Exception as e:
            print(f"[WARN] could not write dart_meta.json: "
                  f"{type(e).__name__}: {e}", flush=True)

    pbar.close()
    env.close()

    print()
    print(f"[dart] collected {passed}/{num} demos in {total_attempts} total sim runs "
          f"(timeouts: {timeouts})")
    print(f"[dart] h5: {out_h5}")
    print(f"[dart] meta: {meta_path}")

    # Low yield is NON-FATAL by design: at high sigma a low keep-rate IS the
    # signal we plot. Never raise on shortfall — just warn and return normally.
    if passed < num:
        keep_rate = (passed / num) if num else 0.0
        print(
            f"[WARN] kept {passed}/{num} demos (keep-rate {keep_rate:.1%}) — "
            f"this is EXPECTED to drop as sigma rises; that degradation is the "
            f"sweep signal, not an error. dart_meta.json written regardless.",
            flush=True,
        )
    return out_h5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_MAP), required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument(
        "--record-dir",
        type=str,
        default="/iris/u/mikulrai/data/RoboFactory/hf_download_dart",
    )
    ap.add_argument("--sigma", type=float, required=True,
                    help="std-dev of Gaussian joint-target noise added to the "
                         "7 arm joints of the moving arm(s) during injection")
    ap.add_argument("--k-min", type=int, default=5, dest="k_min",
                    help="min number of unwrapped noisy steps per injection")
    ap.add_argument("--k-max", type=int, default=15, dest="k_max",
                    help="max number of unwrapped noisy steps per injection")
    ap.add_argument("--p-inject", type=float, default=0.5, dest="p_inject",
                    help="probability of injecting before a given "
                         "move_to_pose_with_screw call")
    ap.add_argument("--dart-seed", type=int, default=0, dest="dart_seed",
                    help="seed for the dedicated disturbance RNG; does NOT "
                         "affect env/task seed reproducibility")
    ap.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    ap.add_argument("--per-attempt-timeout", type=int, default=300, dest="per_attempt_timeout",
                    help="seconds before SIGALRM aborts a single solver attempt; skip seed on timeout")
    ap.add_argument("--seeds-csv", type=str, default=None, dest="seeds_csv",
                    help="explicit comma-separated seed list; bypasses --num and legacy-JSON seeds")
    ap.add_argument("--save-video", action="store_true", dest="save_video", default=False,
                    help="also write per-episode MP4s (tiled sensor cameras) "
                         "under the record-dir; used for field-notes recovery clips")
    args = ap.parse_args()
    override_seeds = None
    if args.seeds_csv:
        override_seeds = [int(s) for s in args.seeds_csv.split(",") if s.strip()]
        num_eff = len(override_seeds)
    else:
        num_eff = args.num
    run(args.task, num_eff, args.record_dir,
        sigma=args.sigma,
        k_min=args.k_min,
        k_max=args.k_max,
        p_inject=args.p_inject,
        dart_seed=args.dart_seed,
        max_retries_per_seed=args.max_retries,
        per_attempt_timeout=args.per_attempt_timeout,
        override_seeds=override_seeds,
        save_video=args.save_video)
