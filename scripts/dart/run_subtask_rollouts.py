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
UNWRAPPED env between primitives and is unrecorded. Because primitives are now
built JUST-IN-TIME (each ``QueuedRecipe`` is planned at the moment the arm enters
it, AFTER the boundary_hook for that transition fires), the hook MAY perturb both
arm joints AND object poses — the next recipe re-plans (fresh per-arm dry_run screw
plan) from the perturbed state, re-resolving moved actors' grasp poses, so labels
stay correct. ``--dart-sigma`` is accepted as a PLACEHOLDER (wired to a no-op hook
factory) so the CLI is forward-compatible.

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
from dataclasses import dataclass

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
from robofactory.planner import subtask_scenarios_tsc as scenarios_tsc  # noqa: E402
from robofactory.planner.subtask_interpreter import (  # noqa: E402
    SubtaskRecorder,
    run_program,
)
from robofactory.planner import dart_perturb  # noqa: E402
from robofactory.utils.wrappers.record import RecordEpisodeMA  # noqa: E402
from robofactory import CONFIG_DIR  # noqa: E402

HF_DOWNLOAD_ROOT = "/iris/u/mikulrai/data/RoboFactory/hf_download"

# (env_id, yaml_rel, n_agents, scenario_sampler). Only tasks with a scenario
# sampler are runnable here. LiftBarrier (2 arms) + ThreeRobotsStackCube (3 arms).
TASK_MAP = {
    "LiftBarrier": ("LiftBarrier-rf", "table/lift_barrier.yaml", 2, scenarios.sample),
    "ThreeRobotsStackCube": (
        "ThreeRobotsStackCube-rf", "table/three_robots_stack_cube.yaml", 3,
        scenarios_tsc.sample,
    ),
}


# ----------------------------------------------------------------------------
# DART config + REAL deterministic boundary_hook factory (per variant)
# ----------------------------------------------------------------------------
@dataclass
class DartCfg:
    """DART joint-disturbance knobs for the boundary_hook.

    ``sigma <= 0`` disables the hook entirely (clean slice). All other fields are
    forwarded to ``dart_perturb.inject_joint_disturbance`` or used by the per-variant
    hook to decide WHEN / WHICH arm to perturb.
    """
    sigma: float = 0.0
    floor: float = 0.15
    k_min: int = 5
    k_max: int = 15
    p_inject: float = 0.5
    dart_seed: int = 0


def _make_boundary_hook(dart_cfg, env_seed, variant_id):
    """Build a REAL, DETERMINISTIC per-variant boundary_hook (or None).

    Returns None if ``dart_cfg`` is falsy or ``dart_cfg.sigma <= 0`` (no hook ->
    interpreter fast path, no perturbation).

    The returned ``hook(unwrapped_env, state)`` is called by the interpreter BETWEEN
    primitives (after one completes, BEFORE the next is BUILT). It:

      * derives a FRESH ``np.random.default_rng`` seeded ONLY from
        ``SeedSequence([env_seed, variant_id, transition_counter, dart_seed])`` so the
        whole sequence of perturbations is a pure function of those four ints
        (reproducible) and NEVER touches the env reset RNG;
      * picks ONE arm to perturb from the arms that are about to ENTER a primitive
        this transition (``a.current is not None and a.ti == 0 and a.built is None``);
      * ALWAYS injects on the first transition (counter 0) and otherwise injects with
        probability ``p_inject``;
      * passes each arm's FROZEN grip (``state["arms"][i].frozen_grip``) so the carry-
        grip contract holds (the disturbance never changes a grasp's open/closed
        state), and forwards ``planner.control_mode`` so the action vector matches.

    The disturbance steps the UNWRAPPED env K times (K ~ U[k_min, k_max]); those
    steps are unrecorded, then the next primitive re-plans JIT from the drifted state.
    """
    if not dart_cfg or dart_cfg.sigma <= 0:
        return None

    counter = {"tc": 0}  # transition counter (closure state)

    def hook(unwrapped_env, state):
        tc = counter["tc"]
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(env_seed), int(variant_id), int(tc), int(dart_cfg.dart_seed)]
            )
        )
        counter["tc"] += 1

        planner = state["planner"]
        arms = state["arms"]
        # arms GENUINELY entering a new primitive this transition, as computed by the
        # interpreter (ti==0, not built, NOT blocked, qi changed). Use that exact set
        # so we never shove a blocked/gated arm. Fall back to the looser local
        # predicate only when the interpreter didn't annotate state (e.g. unit tests).
        moving = state.get("transitioning")
        if moving is None:
            moving = [
                i for i, a in arms.items()
                if a.current is not None and a.ti == 0 and a.built is None
            ]
        if not moving:
            return

        target_arm = int(rng.choice(np.array(sorted(moving))))
        # ALWAYS inject on the FIRST transition; otherwise with prob p_inject. The
        # arm choice consumed RNG first (so the decision draw is deterministic given
        # the seed), matching the documented order.
        inject = (tc == 0) or (rng.random() < dart_cfg.p_inject)
        if not inject:
            return

        K = int(rng.integers(dart_cfg.k_min, dart_cfg.k_max + 1))
        grips = [float(arms[i].frozen_grip) for i in range(len(arms))]
        # hold targets for NON-move arms = their last commanded setpoint (frozen_qpos),
        # so a concurrently moving arm keeps tracking during the K shove steps instead
        # of stalling at its lagging live qpos.
        hold_qpos = [np.asarray(arms[i].frozen_qpos, dtype=np.float64) for i in range(len(arms))]
        dart_perturb.inject_joint_disturbance(
            unwrapped_env,
            planner.robot,
            grips,
            [target_arm],
            rng,
            dart_cfg.sigma,
            K,
            dart_cfg.floor,
            hold_qpos=hold_qpos,
            control_mode=planner.control_mode,
        )

    return hook


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


def _run_one_variant(env, spec, max_steps, dart_cfg):
    """Reset to the spec's seed, build the program, run it, return the result.

    The DART boundary_hook is built PER VARIANT here from ``dart_cfg`` keyed on
    ``(spec.seed, spec.variant_id)`` so each variant's perturbation stream is a
    deterministic function of its identity (and never touches the env reset RNG).
    When ``dart_cfg`` is None / sigma<=0 the hook is None (clean slice).

    Returns (recorder, run_out) on a clean run, or (None, None) if planning or the
    rollout raised (treated as a failed variant -> its whole group is dropped).
    """
    planner = _build_planner(env, spec.seed)
    rec = SubtaskRecorder(num_arms=spec.num_arms)
    boundary_hook = _make_boundary_hook(dart_cfg, spec.seed, spec.variant_id)
    try:
        programs = spec.build(planner, env)
    except Exception as e:  # a plan failure (e.g. -1) drops this variant
        print(f"    [variant {spec.name}] BUILD/PLAN failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return None, None
    try:
        out = run_program(
            env, planner, programs, rec, max_steps=max_steps,
            boundary_hook=boundary_hook,
            control_mode=planner.control_mode,
        )
    except Exception as e:
        # JIT plan failures fire DURING the rollout, not at build (primitives plan
        # screw paths lazily from live pose). The most common is mplib TOPP's
        # "Fail to parameterize path". Honour this fn's contract -> (None, None) so
        # _process_group drops just this variant (and discards its open buffer);
        # an uncaught raise here would propagate up and kill the WHOLE shard.
        print(f"    [variant {spec.name}] ROLLOUT/PLAN failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return None, None
    return rec, out


def _env_success(info):
    """Coerce env-step ``info['success']`` to a python bool, robustly.

    ``info`` may carry a torch tensor / numpy array / scalar / None. Missing or
    None -> False. A (possibly batched) array is reduced to its first element.
    """
    if not info:
        return False
    s = info.get("success", None)
    if s is None:
        return False
    if hasattr(s, "detach"):  # torch tensor
        s = s.detach().cpu().numpy()
    try:
        return bool(np.asarray(s).reshape(-1)[0])
    except (ValueError, IndexError, TypeError):
        return False


def _member_passes(out):
    """Per-member KEEP predicate.

    A member is kept iff its primitive success checks all passed AND the episode
    actually reached the goal one of two ways:

      member passes  iff  out['all_success']  AND  (env_success(info) OR completed)

    WHY env-success OR completed (not ``completed`` alone): the LiftBarrier env
    AUTO-TERMINATES the moment the barrier clears z>0.15 (~step 69) — BEFORE the
    per-arm primitive queues empty — so ``completed`` is False even on a successful
    lift. Requiring ``completed`` alone therefore wrongly dropped EVERY success.
    We accept env-success OR completed instead. EVERY variant now ends with BOTH
    arms lifting (simultaneous / stagger_* — no truncated
    approach-stop variant), so the normal path is:
      * the env terminates on the barrier lift -> env_success True (the lift's own
        check_barrier_lifted may not even have RUN before the early terminate, which
        is fine — env_success confirms the lift). approach checks always run, so
        ``all_success`` has len(checked)>0 and reflects them.
      * if a variant's gating delays the lift past the queue draining without the
        env terminating, ``completed`` covers it (its approach checks passed).
      * a genuinely FAILED lift: the barrier is not raised, the env never reaches
        success, the program runs to completion, the lift's check_barrier_lifted RAN
        and returned False -> all_success False -> dropped.
    """
    if out is None:
        return False
    if not out.get("all_success", False):
        return False
    return _env_success(out.get("info")) or out.get("completed", False)


def _group_all_passed(results):
    """Every member of a contrast group must pass ``_member_passes`` (a clean run
    plus the env-success-or-completed keep criterion). One failure -> drop the
    whole matched group atomically (keeps the contrast balanced)."""
    for rec, out in results:
        if rec is None or out is None:
            return False
        if not _member_passes(out):
            return False
    return True


def _delete_episodes(episode_ids, traj_h5, stream_h5, json_data=None,
                     json_path=None, meta_list=None):
    """Delete already-written artifacts for ``episode_ids`` (a single-pass-delete
    rollback for an atomically-dropped contrast group).

    For each id we remove ``traj_{id}`` from BOTH the trajectory H5 (``traj_h5``)
    and the sibling subtask-stream H5 (``stream_h5``), drop the matching row from the
    RecordEpisode JSON sidecar (``json_data["episodes"]`` + re-dump to ``json_path``)
    so the dataset's episode index never references a deleted trajectory, and drop the
    matching rows from the in-memory subtask meta list (``meta_list``).

    NOTE: ``del`` only unlinks the H5 group; HDF5 does NOT reclaim the freed bytes
    (the file size is unchanged, the data is orphaned). This is intentional and safe:
    the loaders key off the JSON / present traj_ keys, so an orphaned blob is inert.

    Returns the number of trajectory groups actually unlinked.
    """
    ids = {int(e) for e in episode_ids}
    deleted = 0
    for eid in ids:
        key = f"traj_{eid}"
        for h5 in (traj_h5, stream_h5):
            if h5 is not None and key in h5:
                del h5[key]
                if h5 is traj_h5:
                    deleted += 1
    if json_data is not None and "episodes" in json_data:
        json_data["episodes"] = [
            ep for ep in json_data["episodes"]
            if int(ep.get("episode_id", -1)) not in ids
        ]
        if json_path is not None:
            try:
                from mani_skill.utils.io_utils import dump_json
                dump_json(json_path, json_data, indent=2)
            except Exception:
                pass
    if meta_list is not None:
        meta_list[:] = [
            row for row in meta_list if int(row.get("episode_id", -1)) not in ids
        ]
    return deleted


def _splice_suffix(yaml_rel, config_suffix):
    """Splice ``config_suffix`` before the .yaml extension.

    e.g. ``table/lift_barrier.yaml`` + ``_aug`` -> ``table/lift_barrier_aug.yaml``.
    Empty suffix -> unchanged.
    """
    if not config_suffix:
        return yaml_rel
    root, ext = osp.splitext(yaml_rel)
    return f"{root}{config_suffix}{ext}"


def _parse_mix(mix_str):
    """Parse a ``--mix`` spec like ``recovery=0.3,merged=0.4,clean=0.3`` into an
    ordered list of (slice_name, fraction). Empty / falsy -> []. Fractions are used
    to split the ``--num`` episode budget across slices. Unknown slice names raise.
    """
    if not mix_str:
        return []
    valid = {"recovery", "merged", "clean"}
    out = []
    for part in mix_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"[ERROR] bad --mix entry {part!r} (want name=frac)")
        name, frac = part.split("=", 1)
        name = name.strip()
        if name not in valid:
            sys.exit(f"[ERROR] unknown --mix slice {name!r} (valid: {sorted(valid)})")
        out.append((name, float(frac)))
    return out


def _split_budget(num, mix):
    """Split ``num`` episodes across mix slices by fraction (largest-remainder so
    they sum to exactly ``num``). Returns {slice_name: int_budget}."""
    total = sum(f for _, f in mix) or 1.0
    raw = {name: num * frac / total for name, frac in mix}
    floors = {name: int(v) for name, v in raw.items()}
    rem = num - sum(floors.values())
    # hand out the remainder to the largest fractional parts
    order = sorted(raw, key=lambda n: raw[n] - floors[n], reverse=True)
    for i in range(rem):
        floors[order[i % len(order)]] += 1
    return floors


def _write_member(stream_h5, episodes_meta, slice_tag, seed, episode_id, rec, T, spec):
    """Write one member's aligned subtask stream (keyed ``traj_{episode_id}``) + its
    meta row (tagged with ``slice_tag``). Shared by the atomic + per-member paths so
    they cannot drift."""
    rec.flush(episode_id=episode_id, h5_group=stream_h5, expected_T=T)
    stream_h5.flush()
    episodes_meta.append({
        "episode_id": episode_id,
        "env_seed": int(seed),
        "variant": spec.name,
        "family": spec.family,
        "contrast_group_id": int(spec.contrast_group_id),
        "T": int(T),
        "slice": slice_tag,
    })


def _process_group(env, members, max_steps, per_attempt_timeout, dart_cfg,
                   stream_h5, episodes_meta, slice_tag, seed, gid,
                   kept_so_far, per_member=False):
    """Run ONE contrast group.

    per_member=False (default): ATOMIC single-pass-with-delete. Each member runs ONCE;
    on pass its trajectory is flushed + remembered; on fail (or SIGALRM timeout) the
    buffer is discarded and, after the loop, EVERY already-written artifact for the group
    (trajectory H5 group, subtask-stream group, JSON row, in-memory meta row) is DELETED
    so the dataset never references a half-written group. On full success all members'
    aligned streams + meta rows are written. Preserves the guaranteed-complete matched
    contrast (used for the 'clean' slice and the no-mix 'default').

    per_member=True: each member that passes is kept INDEPENDENTLY -- its trajectory +
    aligned subtask stream + meta row are written immediately, and a failing member is
    discarded WITHOUT rolling back the others. Used for the perturbed 'merged' slice,
    where atomic-over-N collapses the yield (~p^N; e.g. 0.4^3 ~ 6%) -- the deterministic
    'clean' slice already supplies the matched contrast.

    Returns (kept_count, group_fully_passed).
    """
    written = []   # [(episode_id, recorder, T, spec)] with traj flushed this group
    group_ok = True
    for spec in members:
        signal.alarm(int(per_attempt_timeout))
        try:
            rec, out = _run_one_variant(env, spec, max_steps, dart_cfg)
        except _SolverTimeout:
            print(f"  [{slice_tag}] seed {seed} group {gid} variant {spec.name}: "
                  f"TIMEOUT after {per_attempt_timeout}s", flush=True)
            rec, out = None, None
        finally:
            signal.alarm(0)

        if rec is None or not _member_passes(out):
            group_ok = False
            try:
                env.flush_trajectory(save=False)  # discard this member's buffer
            except Exception:
                pass
            if per_member:
                print(f"  [{slice_tag}] seed {seed} group {gid} variant "
                      f"{spec.name}: DROPPED (per-member)", flush=True)
                continue  # keep the other members; no rollback
            break         # atomic: stop and roll back below

        T = out["steps"]
        env.flush_trajectory()  # writes traj_{env._episode_id}
        episode_id = int(getattr(env, "_episode_id", kept_so_far + len(written)))
        try:
            env._h5_file.flush()
        except AttributeError:
            pass
        written.append((episode_id, rec, T, spec))
        if per_member:
            # commit this variant NOW (independent keep); a later member failing will
            # NOT roll it back.
            _write_member(stream_h5, episodes_meta, slice_tag, seed,
                          episode_id, rec, T, spec)
            print(f"  [{slice_tag}] seed {seed} group {gid} variant {spec.name}: "
                  f"KEPT (per-member) -> traj_{episode_id}", flush=True)

    if not per_member and not group_ok:
        # ATOMIC rollback: delete everything this group already wrote (matched-pair drop).
        if written:
            _delete_episodes(
                [w[0] for w in written],
                getattr(env, "_h5_file", None),
                stream_h5,
                json_data=getattr(env, "_json_data", None),
                json_path=getattr(env, "_json_path", None),
                meta_list=episodes_meta,
            )
            try:
                env._h5_file.flush()
            except AttributeError:
                pass
        print(f"  [{slice_tag}] seed {seed} group {gid} "
              f"[{members[0].family}]: DROPPED "
              f"({[m.name for m in members]})", flush=True)
        return 0, False

    if not per_member:
        # ATOMIC: all members passed -> write aligned streams + meta rows now.
        for episode_id, rec, T, spec in written:
            _write_member(stream_h5, episodes_meta, slice_tag, seed,
                          episode_id, rec, T, spec)
        print(f"  [{slice_tag}] seed {seed} group {gid} [{members[0].family}]: KEPT "
              f"{[m.name for m in members]} -> traj_{[w[0] for w in written]}",
              flush=True)
        return len(written), True

    # PER-MEMBER: members already committed as we went.
    print(f"  [{slice_tag}] seed {seed} group {gid} [{members[0].family}]: "
          f"per-member kept {len(written)}/{len(members)} -> "
          f"traj_{[w[0] for w in written]}", flush=True)
    return len(written), group_ok


def run(task_name, num, record_dir, variants=None, dart_sigma=0.0,
        max_steps=600, dart_seed=0,
        per_attempt_timeout=300, override_seeds=None, save_video=False,
        floor=0.15, k_min=5, k_max=15, p_inject=0.5, config_suffix="", mix=""):
    if task_name not in TASK_MAP:
        sys.exit(f"[ERROR] task {task_name!r} has no subtask scenario sampler "
                 f"(runnable: {sorted(TASK_MAP)}). 3SC is Phase-4.")
    env_id, yaml_rel, n_agents, sampler = TASK_MAP[task_name]
    yaml_rel = _splice_suffix(yaml_rel, config_suffix)
    config_path = osp.join(CONFIG_DIR, yaml_rel)
    if not osp.exists(config_path):
        sys.exit(f"[ERROR] config {config_path} does not exist "
                 f"(config_suffix={config_suffix!r}).")
    mix_slices = _parse_mix(mix)
    # GUARD: the 'recovery'/'merged' slices are defined by an ACTIVE arm-disturbance
    # hook; with dart_sigma<=0 the hook is a no-op and those slices silently collapse
    # to the same object-noise-only recipe as 'clean' (but still tagged recovery/merged
    # in meta -> invisible mislabel). Fail loudly instead.
    if mix_slices and dart_sigma <= 0 and any(
        name in ("recovery", "merged") for name, _ in mix_slices
    ):
        sys.exit(
            "[ERROR] --mix includes recovery/merged (arm-disturbance slices) but "
            f"--dart-sigma={dart_sigma} <= 0 -> the hook is a no-op and those slices "
            "would be indistinguishable from 'clean'. Pass --dart-sigma>0."
        )

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
        # CRITICAL: do NOT let clean_trajectories() renumber surviving traj_{id}s to
        # contiguous 0..N at env.close(). We delete dropped groups mid-run (single-pass
        # -with-delete), leaving id gaps on PURPOSE; the sibling subtask_stream.h5 +
        # subtask_meta.json are keyed by those exact ids. A close-time renumber would
        # rewrite the traj H5 + JSON ids but NOT the stream/meta -> silent join-key
        # desync (every action paired with the wrong subtask). Keep ids stable.
        clean_on_close=False,
    )

    # --- DART config (a single object reused by every active slice) ---
    base_dart_cfg = DartCfg(
        sigma=float(dart_sigma), floor=float(floor),
        k_min=int(k_min), k_max=int(k_max), p_inject=float(p_inject),
        dart_seed=int(dart_seed),
    )

    # --- slice plan ---
    # No --mix: a SINGLE slice governed by --dart-sigma (today's behavior), tagged
    # "default". With --mix: three sub-runs into ONE combined dataset dir:
    #   recovery -> baseline-only specs, hook ON  (DART recovery data)
    #   merged   -> full contrast groups, hook ON (DART on the contrast)
    #   clean    -> full contrast groups, hook OFF (object-noise only; sigma forced 0)
    # Each slice gets a portion of the --num episode budget.
    if mix_slices:
        budgets = _split_budget(num, mix_slices)
        plan = []  # (slice_tag, baseline_only, dart_cfg, budget, per_member)
        for name, _frac in mix_slices:
            if name == "recovery":
                # baseline-only -> 1-member groups; atomic==per-member, keep atomic.
                plan.append(("recovery", True, base_dart_cfg, budgets[name], False))
            elif name == "merged":
                # PER-MEMBER keep: atomic-over-N collapses the yield once the arm-shove is
                # on (~p^N, e.g. 0.4^3~6%). Keep each passing variant independently; the
                # deterministic 'clean' slice supplies the guaranteed matched contrast.
                plan.append(("merged", False, base_dart_cfg, budgets[name], True))
            elif name == "clean":
                # object noise (aug config) stays ON; arm-disturbance hook OFF. ATOMIC so
                # the matched contrast (frame-0-identical A/B/C) is guaranteed complete.
                clean_cfg = DartCfg(
                    sigma=0.0, floor=base_dart_cfg.floor,
                    k_min=base_dart_cfg.k_min, k_max=base_dart_cfg.k_max,
                    p_inject=base_dart_cfg.p_inject, dart_seed=base_dart_cfg.dart_seed,
                )
                plan.append(("clean", False, clean_cfg, budgets[name], False))
    else:
        plan = [("default", False, base_dart_cfg, num, False)]

    print(f"[subtask] task={task_name}  env_id={env_id}  n_agents={n_agents}")
    print(f"[subtask] config={config_path} (suffix={config_suffix!r})")
    print(f"[subtask] output_h5={out_h5}")
    print(f"[subtask] subtask_stream_h5={stream_h5_path}")
    print(f"[subtask] variants_filter={variants}  dart_sigma={dart_sigma} "
          f"floor={floor} k=[{k_min},{k_max}] p_inject={p_inject}  max_steps={max_steps}")
    print(f"[subtask] slice plan: "
          f"{[(t, 'baseline-only' if bo else 'full', 'hookON' if c.sigma > 0 else 'hookOFF', 'per-member' if pm else 'atomic', b) for t, bo, c, b, pm in plan]}")

    seeds = override_seeds if override_seeds is not None else _load_seeds(task_name, num)
    print(f"[subtask] per_attempt_timeout={per_attempt_timeout}s (SIGALRM per variant)",
          flush=True)
    signal.signal(signal.SIGALRM, _solver_alarm_handler)

    import h5py
    stream_h5 = h5py.File(stream_h5_path, "w")

    episodes_meta = []
    kept = 0           # episodes (variants) written across ALL slices
    kept_groups = 0    # contrast groups fully kept
    total_groups = 0
    slice_kept = {}    # per-slice kept counts
    pbar = tqdm(total=num, desc=task_name)
    meta_path = osp.join(output_dir, "subtask_meta.json")

    def _write_meta():
        meta = {
            "task": task_name,
            "variants_filter": variants,
            "dart_sigma": float(dart_sigma),
            "dart_seed": int(dart_seed),
            "floor": float(floor),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "p_inject": float(p_inject),
            "config_suffix": config_suffix,
            "mix": mix,
            "num": int(num),
            "kept_episodes": int(kept),
            "kept_groups": int(kept_groups),
            "total_groups_attempted": int(total_groups),
            "slice_kept": dict(slice_kept),
            "episodes": episodes_meta,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    try:
        for slice_tag, baseline_only, dart_cfg, slice_budget, per_member in plan:
            this_slice_kept = 0
            print(f"[subtask] === slice {slice_tag!r}: budget={slice_budget} "
                  f"(baseline_only={baseline_only}, "
                  f"hook={'on' if dart_cfg.sigma > 0 else 'off'}) ===", flush=True)
            for seed in seeds:
                if this_slice_kept >= slice_budget:
                    break
                specs = sampler(int(seed))
                specs = scenarios.filter_variants(specs, variants)
                if baseline_only:
                    # recovery slice: keep ONLY the single baseline member so each
                    # group is a 1-member group (no contrast variants).
                    specs = [s for s in specs if s.family == "baseline"]
                groups = scenarios.group_specs(specs)
                for gid, members in sorted(groups.items()):
                    if this_slice_kept >= slice_budget:
                        break
                    total_groups += 1
                    added, ok = _process_group(
                        env, members, max_steps, per_attempt_timeout, dart_cfg,
                        stream_h5, episodes_meta, slice_tag, seed, gid, kept,
                        per_member=per_member,
                    )
                    # count kept episodes regardless of full-group success (per-member
                    # slices keep partial groups); kept_groups counts only complete groups.
                    if added:
                        kept += added
                        this_slice_kept += added
                        pbar.update(added)
                    if ok:
                        kept_groups += 1
            slice_kept[slice_tag] = this_slice_kept
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
    print(f"[subtask] kept {kept} episodes in {kept_groups}/{total_groups} groups "
          f"(per-slice: {slice_kept})")
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
                    help="DART per-joint disturbance stddev. >0 installs the REAL "
                         "deterministic per-variant boundary_hook (arm-joint noise on "
                         "the unwrapped env between primitives). 0 = no hook.")
    ap.add_argument("--inject-floor", type=float, default=0.15, dest="inject_floor",
                    help="minimum L2 norm of each DART joint offset (so a draw is "
                         "never trivially small). Default 0.15.")
    ap.add_argument("--k-min", type=int, default=5, dest="k_min",
                    help="min number of unwrapped steps the disturbance is held.")
    ap.add_argument("--k-max", type=int, default=15, dest="k_max",
                    help="max number of unwrapped steps the disturbance is held.")
    ap.add_argument("--p-inject", type=float, default=0.5, dest="p_inject",
                    help="probability of injecting a disturbance at a transition "
                         "(the FIRST transition of each variant always injects).")
    ap.add_argument("--config-suffix", type=str, default="", dest="config_suffix",
                    help="splice into the yaml stem, e.g. '_aug' maps "
                         "lift_barrier.yaml -> lift_barrier_aug.yaml (object noise).")
    ap.add_argument("--mix", type=str, default="", dest="mix",
                    help="combined dataset mix, e.g. "
                         "'recovery=0.3,merged=0.4,clean=0.3'. Splits --num across "
                         "three slices into ONE dataset dir (each meta row tagged with "
                         "its slice). Empty = single slice governed by --dart-sigma.")
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
        save_video=args.save_video,
        floor=args.inject_floor,
        k_min=args.k_min,
        k_max=args.k_max,
        p_inject=args.p_inject,
        config_suffix=args.config_suffix,
        mix=args.mix)
