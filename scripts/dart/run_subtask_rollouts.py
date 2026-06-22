"""Subtask-conditioned data generation (HL data aug).

Fork of ``scripts/dart/run_dart_rollouts.py``. The env build (shader_pack=default,
sim_backend=cpu, robot_uids=panda_wristcam_multi, RecordEpisodeMA flags), the seed
loop, the SIGALRM per-attempt timeout, the flush-on-success / verify-keep loop and
the ``dart_meta``-style sidecar json are KEPT.

What this REPLACES: the monolithic solver call. Instead, for each seed we build a
CONTRASTIVE set of per-arm programs via ``subtask_scenarios.sample(seed)`` and run
each variant through ``subtask_interpreter.run_program``. We keep ONLY the episodes
whose per-subtask success checks all pass, and we drop matched contrastive pairs
ATOMICALLY: if any member of a ``contrast_group_id`` fails its checks (or its plan
raises), the WHOLE group is discarded so the contrast stays balanced.

Each kept episode writes:
  * the H5 trajectory (RecordEpisodeMA, as before), and
  * an aligned ``subtask_stream`` (the per-arm (verb,target,text) labels) appended
    into a sibling ``<Task>_subtask_stream.h5`` keyed ``traj_{episode_id}`` (the
    SAME id the wrapper used). The stream length is asserted == len(actions) T.

DART / object-perturbation COMPOSES via ``boundary_hook``: the user plugs their
arm-noise + object-pose perturbation in here later (default None). It runs on the
UNWRAPPED env between primitives and is unrecorded; the next primitive re-plans
from the perturbed state so labels stay correct. ``--dart-sigma`` is accepted as a
PLACEHOLDER (wired to a no-op hook factory) so the CLI is forward-compatible.

HARD RULE: this script runs SAPIEN; do NOT run it on the login node (renders dark)
or without a GPU. Author-only here — launch on an iris-hi a40 compute node.

Usage (pilot, on a compute node):
    python scripts/dart/run_subtask_rollouts.py \\
        --task LiftBarrier --num 5 \\
        --record-dir /iris/u/mikulrai/data/RoboFactory/hf_download_subtask
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
    """Raised by the SIGALRM handler when a single attempt exceeds budget."""


def _solver_alarm_handler(signum, frame):
    raise _SolverTimeout()


import robofactory  # registers envs + PandaWristCamMulti  # noqa: E402
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver  # noqa: E402
from robofactory.planner import subtask_scenarios as scenarios  # noqa: E402
from robofactory.planner.subtask_interpreter import (  # noqa: E402
    SubtaskRecorder,
    run_program,
)
from robofactory.utils.wrappers.record import RecordEpisodeMA  # noqa: E402
from robofactory import CONFIG_DIR  # noqa: E402

HF_DOWNLOAD_ROOT = "/iris/u/mikulrai/data/RoboFactory/hf_download"

# (env_id, yaml_rel, n_agents, scenario_sampler). Only tasks with a scenario
# sampler are runnable here; LiftBarrier is Phase-1. (3SC is Phase-4.)
TASK_MAP = {
    "LiftBarrier": ("LiftBarrier-rf", "table/lift_barrier.yaml", 2, scenarios.sample),
}


# ----------------------------------------------------------------------------
# boundary_hook factory (DART / object-perturbation plug point)
# ----------------------------------------------------------------------------
def _make_boundary_hook(dart_sigma):
    """Return the optional boundary_hook, or None.

    PLACEHOLDER: the user is separately building the DART arm-noise + object-pose
    perturbation generator. When they wire it, it slots in HERE: a
    ``hook(unwrapped_env, state)`` called between primitives, stepping the
    UNWRAPPED env (unrecorded). The next primitive re-plans from the perturbed
    state so labels stay correct.

    Today, a non-zero ``dart_sigma`` only INSTALLS a no-op hook (so the wiring /
    call path is exercised) — it does NOT perturb anything. Returning None keeps
    the interpreter on its fast path (no hook calls).
    """
    if not dart_sigma:
        return None

    def _noop_hook(unwrapped_env, state):
        # Intentionally does nothing: real DART perturbation plugs in here.
        return None

    return _noop_hook


# ----------------------------------------------------------------------------
# planner build (mirrors solutions/lift_barrier.solve, minus the scripted moves)
# ----------------------------------------------------------------------------
def _build_planner(env, seed):
    """Reset env to ``seed`` and build a multi-agent planner (no scripted moves).

    Matches solutions/lift_barrier.solve: reset(seed), then a multi-agent
    PandaArmMotionPlanningSolver with per-agent base poses, vis off. Returns the
    planner. Frame-0 is fixed by the reset seed; the scenario sampler uses its OWN
    RNG so contrastive variants from the same seed share this exact reset state.
    """
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True,
    )
    return planner


def _run_one_variant(env, spec, max_steps, boundary_hook):
    """Reset to the spec's seed, build the program, run it, return the result.

    Returns (recorder, run_out) on a clean run, or (None, None) if planning or the
    rollout raised (treated as a failed variant -> its whole group is dropped).
    """
    planner = _build_planner(env, spec.seed)
    rec = SubtaskRecorder(num_arms=spec.num_arms)
    try:
        programs = spec.build(planner, env)
    except Exception as e:  # a plan failure (e.g. -1) drops this variant
        print(f"    [variant {spec.name}] BUILD/PLAN failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return None, None
    out = run_program(
        env, planner, programs, rec, max_steps=max_steps,
        boundary_hook=boundary_hook,
        control_mode=planner.control_mode,
    )
    return rec, out


def _group_all_passed(results):
    """Every member of a contrast group must have: a clean run, completed queues,
    and all_success True (every primitive with a success_check passed)."""
    for rec, out in results:
        if rec is None or out is None:
            return False
        if not out.get("completed", False):
            return False
        if not out.get("all_success", False):
            return False
    return True


def run(task_name, num, record_dir, variants=None, dart_sigma=0.0,
        max_steps=600, dart_seed=0,
        per_attempt_timeout=300, override_seeds=None, save_video=False):
    if task_name not in TASK_MAP:
        sys.exit(f"[ERROR] task {task_name!r} has no subtask scenario sampler "
                 f"(runnable: {sorted(TASK_MAP)}). 3SC is Phase-4.")
    env_id, yaml_rel, n_agents, sampler = TASK_MAP[task_name]
    config_path = osp.join(CONFIG_DIR, yaml_rel)

    # --- output paths (mirror hf_download layout) ---
    output_dir = osp.join(record_dir, task_name)
    out_h5 = osp.join(output_dir, f"{task_name}.h5")
    stream_h5_path = osp.join(output_dir, f"{task_name}_subtask_stream.h5")
    if osp.exists(out_h5) and osp.getsize(out_h5) > 1024:
        sys.exit(
            f"[ERROR] {out_h5} already exists and is non-empty.\n"
            f"Move it aside before running a fresh collection."
        )
    os.makedirs(output_dir, exist_ok=True)

    # --- build env (IDENTICAL flags to the DART fork) ---
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
        trajectory_name=task_name,
        save_video=save_video,
        source_type="motionplanning",
        source_desc=(
            "Subtask-conditioned dataset; per-arm primitives driven by a "
            "contrastive subtask scenario, labelled live (subtask_stream), only "
            "per-subtask-success episodes kept; matched contrastive pairs dropped "
            "atomically (robot_uids=panda_wristcam_multi)"
        ),
        video_fps=30,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,
        record_observation=True,
    )

    boundary_hook = _make_boundary_hook(dart_sigma)

    print(f"[subtask] task={task_name}  env_id={env_id}  n_agents={n_agents}")
    print(f"[subtask] output_h5={out_h5}")
    print(f"[subtask] subtask_stream_h5={stream_h5_path}")
    print(f"[subtask] variants_filter={variants}  dart_sigma={dart_sigma} "
          f"(hook={'on' if boundary_hook else 'off'})  max_steps={max_steps}")

    seeds = override_seeds if override_seeds is not None else _load_seeds(task_name, num)
    print(f"[subtask] per_attempt_timeout={per_attempt_timeout}s (SIGALRM per variant)",
          flush=True)
    signal.signal(signal.SIGALRM, _solver_alarm_handler)

    import h5py
    stream_h5 = h5py.File(stream_h5_path, "w")

    episodes_meta = []
    kept = 0           # episodes (variants) written
    kept_groups = 0    # contrast groups fully kept
    total_groups = 0
    pbar = tqdm(total=num, desc=task_name)
    meta_path = osp.join(output_dir, "subtask_meta.json")

    def _write_meta():
        meta = {
            "task": task_name,
            "variants_filter": variants,
            "dart_sigma": float(dart_sigma),
            "dart_seed": int(dart_seed),
            "num": int(num),
            "kept_episodes": int(kept),
            "kept_groups": int(kept_groups),
            "total_groups_attempted": int(total_groups),
            "episodes": episodes_meta,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    try:
        for seed in seeds:
            if kept >= num:
                break
            specs = sampler(int(seed))
            specs = scenarios.filter_variants(specs, variants)
            groups = scenarios.group_specs(specs)
            for gid, members in sorted(groups.items()):
                if kept >= num:
                    break
                total_groups += 1

                # --- PROBE: run every member to decide the group atomically ---
                # The wrapper holds ONE trajectory buffer, so we cannot keep two
                # members pending. We first PROBE all members (clearing the buffer
                # after each), make the atomic keep/drop decision, then RE-RUN the
                # kept members to flush clean trajectories. Re-runs are
                # deterministic: same reset seed + seed-derived sampler RNG.
                results = []  # [(recorder, run_out), ...] aligned with members
                timed_out = False
                for spec in members:
                    signal.alarm(int(per_attempt_timeout))
                    try:
                        rec, out = _run_one_variant(env, spec, max_steps, boundary_hook)
                    except _SolverTimeout:
                        print(f"  seed {seed} group {gid} variant {spec.name}: "
                              f"TIMEOUT after {per_attempt_timeout}s", flush=True)
                        timed_out = True
                        rec, out = None, None
                    finally:
                        signal.alarm(0)
                    results.append((rec, out))
                    # MUST clear the in-progress wrapper buffer after EVERY variant:
                    # only members that pass the group check are flushed below.
                    try:
                        env.flush_trajectory(save=False)
                    except Exception:
                        pass

                # --- atomic matched-pair decision ---
                if timed_out or not _group_all_passed(results):
                    print(f"  seed {seed} group {gid} "
                          f"[{members[0].family}]: DROPPED "
                          f"({[m.name for m in members]})", flush=True)
                    continue

                # --- group passed: RE-RUN each member to flush a CLEAN trajectory ---
                # (the buffer was cleared after each probe run so we can't keep the
                # buffered episode; re-running from the same seed reproduces it
                # deterministically because the sampler RNG is seed-derived.)
                group_ok = True
                pending = []  # [(episode_id, recorder, T)]
                for spec in members:
                    rec, out = _run_one_variant(env, spec, max_steps, boundary_hook)
                    if rec is None or out is None or not out.get("all_success", False):
                        group_ok = False
                        try:
                            env.flush_trajectory(save=False)
                        except Exception:
                            pass
                        break
                    T = out["steps"]
                    env.flush_trajectory()  # writes traj_{env._episode_id}
                    episode_id = int(getattr(env, "_episode_id", kept))
                    pending.append((episode_id, rec, T, spec))
                    try:
                        env._h5_file.flush()
                    except AttributeError:
                        pass

                if not group_ok:
                    print(f"  seed {seed} group {gid}: re-run diverged -> DROPPED",
                          flush=True)
                    continue

                # --- both flushed: now write the aligned subtask streams ---
                for episode_id, rec, T, spec in pending:
                    rec.flush(episode_id=episode_id, h5_group=stream_h5, expected_T=T)
                    stream_h5.flush()
                    episodes_meta.append({
                        "episode_id": episode_id,
                        "env_seed": int(seed),
                        "variant": spec.name,
                        "family": spec.family,
                        "contrast_group_id": int(spec.contrast_group_id),
                        "T": int(T),
                    })
                    kept += 1
                    pbar.update(1)
                kept_groups += 1
                print(f"  seed {seed} group {gid} [{members[0].family}]: KEPT "
                      f"{[m.name for m in members]} "
                      f"-> traj_{[p[0] for p in pending]}", flush=True)
    finally:
        try:
            stream_h5.close()
        except Exception:
            pass
        try:
            _write_meta()
        except Exception as e:
            print(f"[WARN] could not write subtask_meta.json: "
                  f"{type(e).__name__}: {e}", flush=True)

    pbar.close()
    env.close()

    print()
    print(f"[subtask] kept {kept} episodes in {kept_groups}/{total_groups} groups")
    print(f"[subtask] h5: {out_h5}")
    print(f"[subtask] subtask_stream: {stream_h5_path}")
    print(f"[subtask] meta: {meta_path}")
    return out_h5


def _load_seeds(task_name, num):
    """Seeds to attempt. Uses old-dataset seeds when available so traj indices line
    up with hf_download/traj_i for matched-state comparisons; else sequential.

    NOTE: each seed yields MULTIPLE episodes (one per kept contrastive variant), so
    we generate generously and stop once ``num`` episodes are kept.
    """
    old_json = osp.join(HF_DOWNLOAD_ROOT, task_name, f"{task_name}.json")
    if osp.exists(old_json):
        episodes = json.load(open(old_json))["episodes"]
        seeds = [ep["episode_seed"] for ep in episodes]
        if seeds:
            print(f"  [seeds] using {len(seeds)} seeds from {old_json}")
            return seeds
    print(f"  [seeds] no old JSON; using sequential 0..{max(num, 1) * 4 - 1}")
    return list(range(max(num, 1) * 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_MAP), default="LiftBarrier")
    ap.add_argument("--num", type=int, default=20,
                    help="target number of KEPT episodes (one per kept variant)")
    ap.add_argument(
        "--record-dir", type=str,
        default="/iris/u/mikulrai/data/RoboFactory/hf_download_subtask",
    )
    ap.add_argument("--variants", type=str, default=None,
                    help="comma-separated variant/family names to KEEP "
                         "(group-level filter; matched pairs stay intact). "
                         "e.g. 'a_approach,c_gate' or 'approach_stop'")
    ap.add_argument("--seeds", type=str, default=None, dest="seeds_csv",
                    help="explicit comma-separated env seed list; bypasses --num "
                         "seed selection (still stops once --num episodes kept)")
    ap.add_argument("--dart-sigma", type=float, default=0.0, dest="dart_sigma",
                    help="PLACEHOLDER for the user's DART perturbation strength; a "
                         "non-zero value installs a NO-OP boundary_hook (wiring "
                         "only). Real perturbation plugs into _make_boundary_hook.")
    ap.add_argument("--max-steps", type=int, default=600, dest="max_steps",
                    help="hard cap on env.steps per variant rollout")
    ap.add_argument("--per-attempt-timeout", type=int, default=300,
                    dest="per_attempt_timeout")
    ap.add_argument("--save-video", action="store_true", default=False,
                    dest="save_video")
    args = ap.parse_args()

    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    override_seeds = None
    if args.seeds_csv:
        override_seeds = [int(s) for s in args.seeds_csv.split(",") if s.strip()]

    run(args.task, args.num, args.record_dir,
        variants=variants,
        dart_sigma=args.dart_sigma,
        max_steps=args.max_steps,
        per_attempt_timeout=args.per_attempt_timeout,
        override_seeds=override_seeds,
        save_video=args.save_video)
