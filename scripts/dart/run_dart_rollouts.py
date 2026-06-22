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
# The original follow_path; the dense "jitter" scheme wraps it (re-implements
# its step loop verbatim + an unwrapped nudge before each recorded step).
_ORIG_FOLLOW_PATH = PandaArmMotionPlanningSolver.follow_path

# When DART_CAPTURE_DISTURBANCES is non-empty it is treated as a list onto which
# every emitted disturbance action_dict is appended (used by the sim tests to
# verify no recorded action equals an emitted disturbance). Left None in prod.
_DISTURBANCE_SINK = None


def _inject_disturbance(self, rng, sigma, K, move_id, floor):
    """Step the UNWRAPPED env K times with a noisy joint target so the arm
    drifts off-path. NOT recorded (unwrapped env => RecordEpisode buffer is not
    touched). Only the 7 arm joints of the moving arm(s) get noise; gripper
    channel stays clean.

    Magnitude floor (user wants "random everything but never trivial"): for each
    moving arm we sample ONE random offset vector off ~ N(0, sigma, 7), and if
    its L2 norm falls below ``floor`` we scale it UP to the floor while keeping
    the random direction. The same q0+off target is then HELD for all K steps so
    the net joint displacement is ~= off (>= floor) with random direction AND
    magnitude. q0 (each moving arm's 7-joint qpos) is captured ONCE up front.

    Mirrors follow_path's single-agent (flat action) vs multi-agent (dict)
    branching so the hook is general even though LiftBarrier is multi-agent.
    """
    move_id = move_id if isinstance(move_id, list) else [move_id]
    base = self.base_env
    assert base is self.env.unwrapped, (
        "disturbance MUST use unwrapped env (else noise becomes a training label)"
    )

    def _offset(n):
        # random direction + magnitude; floored UP so it is never trivial.
        off = rng.normal(0.0, sigma, size=n)
        mag = float(np.linalg.norm(off))
        if mag < floor:
            off = off / (mag + 1e-9) * floor  # scale to floor, keep direction
        return off

    if not self.is_multi_agent:
        # single-agent: base.step takes a flat action [7 arm + gripper]
        q0 = self.robot[0].get_qpos()[0, :-2].cpu().numpy()  # captured ONCE
        target = q0.copy()
        if 0 in move_id:
            target = q0 + _offset(q0.shape[0])  # drift arm only
        action = np.hstack([target, self.gripper_state[0]])  # gripper CLEAN
        for _ in range(K):  # HOLD the same drifted target
            if _DISTURBANCE_SINK is not None:
                _DISTURBANCE_SINK.append(np.asarray(action, dtype=np.float64).copy())
            base.step(action)  # UNWRAPPED -> not recorded
    else:
        action_dict = {}
        for aid in range(self.agent_num):
            q0 = self.robot[aid].get_qpos()[0, :-2].cpu().numpy()  # captured ONCE
            target = q0.copy()
            if aid in move_id:
                target = q0 + _offset(q0.shape[0])  # 7 joints
            action_dict[f"panda-{aid}"] = np.hstack([target, self.gripper_state[aid]])
        for _ in range(K):  # HOLD the same drifted target
            if _DISTURBANCE_SINK is not None:
                _DISTURBANCE_SINK.append(
                    {k: np.asarray(v, dtype=np.float64).copy() for k, v in action_dict.items()}
                )
            base.step(action_dict)  # UNWRAPPED -> not recorded


def _estimate_n_moves(task_name):
    """Best-effort count of real move_to_pose_with_screw calls in a task's
    solver, used ONLY to tune the dense p_shove so expected events land near the
    [shove_min,shove_max] mid. Counts textual occurrences in the solver source
    (a static upper bound; loops/branches make it approximate — that's fine, it
    only sets a probability). Returns 0 if the source can't be inspected, which
    makes run() fall back to p_shove=1.0 (inject on every move)."""
    import inspect
    try:
        solver = TASK_MAP[task_name][2]
        src = inspect.getsource(solver)
        return src.count("move_to_pose_with_screw")
    except Exception:
        return 0


def _pick_single_arm(rng, move_id):
    """Pick ONE arm id at random from this move's move_id list. Dense scheme is
    single-arm by design (perturb one arm, not all), so both jitter and shove
    select exactly one eligible (moving) arm per event."""
    move_id = move_id if isinstance(move_id, list) else [move_id]
    if len(move_id) == 0:
        return None
    return int(rng.choice(np.asarray(move_id, dtype=np.int64)))


def _jitter_nudge(self, rng, jitter_sigma, arm_id, waypoint7):
    """ONE unwrapped env step that commands, for ONE chosen arm, a target of
    (that arm's CLEAN waypoint at this follow_path step + N(0, jitter_sigma, 7));
    gripper clean; all OTHER arms HOLD current qpos. NOT recorded (unwrapped env).
    Mirrors follow_path's single-agent (flat action) vs multi-agent (dict)
    branching + control_mode handling.

    BOUNDED, not a random walk: the nudge is centered on the PATH waypoint, not
    on the (already-drifted) current qpos, so the arm stays within ~sigma of the
    path every jittered step and the noise never accumulates. (The old
    qpos+noise form compounded over ~40% of steps and wandered the arm off-task
    -> 8% yield; this waypoint+noise form keeps yield high while still producing
    genuinely-perturbed recorded observations with clean-waypoint labels.)

    ``waypoint7`` is the 7-joint clean waypoint for ``arm_id`` at this step
    (result_group[move_idx]["position"][min(i, ...)][:7]); the caller supplies it.
    The recorded step (in the wrapper) still commands the clean waypoint, so the
    recorded action is unchanged -> only the NEXT recorded obs reflects a small,
    bounded drift."""
    base = self.base_env
    assert base is self.env.unwrapped, (
        "jitter MUST use unwrapped env (else noise becomes a training label)"
    )
    waypoint7 = np.asarray(waypoint7, dtype=np.float64)[:7]
    nudged = waypoint7 + rng.normal(0.0, jitter_sigma, size=waypoint7.shape[0])
    if not self.is_multi_agent:
        # single-agent: arm_id is always 0 (the only arm)
        if self.control_mode == "pd_joint_pos_vel":
            action = np.hstack([nudged, nudged * 0, self.gripper_state[0]])
        else:
            action = np.hstack([nudged, self.gripper_state[0]])  # gripper CLEAN
        if _DISTURBANCE_SINK is not None:
            _DISTURBANCE_SINK.append(np.asarray(action, dtype=np.float64).copy())
        base.step(action)  # UNWRAPPED -> not recorded
    else:
        action_dict = {}
        for aid in range(self.agent_num):
            if aid == arm_id:
                target = nudged  # waypoint + noise (bounded around the path)
            else:
                target = self.robot[aid].get_qpos()[0, :-2].cpu().numpy()  # HOLD
            if self.control_mode == "pd_joint_pos_vel":
                action_dict[f"panda-{aid}"] = np.hstack(
                    [target, target * 0, self.gripper_state[aid]])
            else:
                action_dict[f"panda-{aid}"] = np.hstack(
                    [target, self.gripper_state[aid]])
        if _DISTURBANCE_SINK is not None:
            _DISTURBANCE_SINK.append(
                {k: np.asarray(v, dtype=np.float64).copy() for k, v in action_dict.items()})
        base.step(action_dict)  # UNWRAPPED -> not recorded


def _make_dart_jitter_follow_path(rng, jitter_frac, jitter_sigma, ep_counter):
    """Replacement for follow_path implementing the dense JITTER mechanism.

    It re-implements follow_path's step loop VERBATIM (copied from
    motionplanner.py:101-146) and, before a randomly-chosen ~jitter_frac
    fraction of the recorded env.step calls, runs ONE unwrapped _jitter_nudge on
    ONE randomly-chosen moving arm (move_id). The nudge is centered on that arm's
    CLEAN WAYPOINT at this step (waypoint + N(0, jitter_sigma)), NOT on current
    qpos, so the drift is BOUNDED around the path and never compounds into a
    random walk off-task. The recorded step itself is UNCHANGED — it still
    issues self.env.step(clean waypoint). So the recorded action stays the clean
    waypoint while the next recorded obs shows a small, bounded drift.
    n_jitter_steps is tallied into ep_counter for the sidecar meta."""

    def dart_follow_path(self, result_group, move_id, refine_steps: int = 0):
        n_step = 0
        for i in range(len(result_group)):
            n_step = max(n_step, result_group[i]["position"].shape[0])
        # Multi-Agent check collision
        # if check_collision(self, result_group, move_id, n_step, jump):
        #     print("Collision detected")
        #     return False
        for i in range(n_step + refine_steps):
            # --- DART jitter: ONE unwrapped nudge before this recorded step,
            # with probability jitter_frac, on ONE random moving arm. The nudge
            # target is that arm's CLEAN waypoint at step i + N(0, jitter_sigma)
            # -> bounded around the path (no compounding random walk). NOT
            # recorded; the recorded self.env.step below is unchanged. ---
            if rng.random() < jitter_frac:
                arm_id = _pick_single_arm(rng, move_id)
                if arm_id is not None:
                    # the SAME waypoint the recorded step will command for arm_id
                    if not self.is_multi_agent:
                        waypoint7 = result_group[0]["position"][min(i, n_step - 1)]
                    else:
                        move_idx = move_id.index(arm_id)
                        self_step = result_group[move_idx]["position"].shape[0]
                        waypoint7 = result_group[move_idx]["position"][min(i, self_step - 1)]
                    _jitter_nudge(self, rng, jitter_sigma, arm_id, waypoint7)
                    ep_counter["n_jitter_steps"] += 1
            if not self.is_multi_agent:
                qpos = result_group[0]["position"][min(i, n_step - 1)]
                if self.control_mode == "pd_joint_pos_vel":
                    qvel = result_group[0]["velocity"][min(i, n_step - 1)]
                    action = np.hstack([qpos, qvel, self.gripper_state])
                else:
                    action = np.hstack([qpos, self.gripper_state])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = dict()
                for id in range(self.agent_num):
                    if id not in move_id:
                        qpos = self.robot[id].get_qpos()[0, :-2].cpu().numpy()
                        if self.control_mode == "pd_joint_pos":
                            action = np.hstack([qpos, self.gripper_state[id]])
                        else:
                            action = np.hstack([qpos, qpos * 0, self.gripper_state[id]])
                        action_dict[f"panda-{id}"] = action
                    else:
                        move_idx = move_id.index(id)
                        self_step = result_group[move_idx]["position"].shape[0]
                        qpos = result_group[move_idx]["position"][min(i, self_step - 1)]
                        if self.control_mode == "pd_joint_pos_vel":
                            qvel = result_group[move_idx]["velocity"][min(i, self_step - 1)]
                            action = np.hstack([qpos, qvel, self.gripper_state[id]])
                        else:
                            action = np.hstack([qpos, self.gripper_state[id]])
                        action_dict[f"panda-{id}"] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return True, obs, reward, terminated, truncated, info

    return dart_follow_path


def _make_dense_move(rng, sigma_min, sigma_max, K, floor, p_shove, ep_counter):
    """Build the replacement move_to_pose_with_screw for the DENSE scheme.

    SHOVE+RECOVER: on each REAL (not dry_run) move, with probability p_shove,
    shove ONE randomly-chosen arm from that move's move_id (single arm, random
    each event) with a held offset of intensity sigma ~ U(sigma_min, sigma_max)
    over K unwrapped steps, then the original move re-plans to target (the clean
    recovery is recorded). p_shove is tuned by run() so the expected event count
    over a typical episode lands at the mid of [shove-min, shove-max]. Tasks with
    few primitives (e.g. LiftBarrier ~2) simply get fewer events — acceptable."""

    def dense_move(self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1):
        if not dry_run:
            ep_counter["n_calls"] += 1
            if rng.random() < p_shove:
                sigma = float(rng.uniform(sigma_min, sigma_max))
                arm_id = _pick_single_arm(rng, move_id)
                if arm_id is not None:
                    _inject_disturbance(self, rng, sigma, K, [arm_id], floor)
                    ep_counter["n_shove_events"] += 1
        return _ORIG_MOVE(
            self, pose, dry_run=dry_run, refine_steps=refine_steps,
            move_id=move_id, jump=jump,
        )

    return dense_move


def _make_dart_move(rng, sigma, k_min, k_max, p_inject, floor, ep_counter):
    """Build the replacement move_to_pose_with_screw. Keeps the original
    signature so existing solver code calls it unchanged.

    ``ep_counter`` is a mutable dict {"n_calls", "n_injects"} owned by run() and
    reset to zero immediately BEFORE each solver call. It guarantees >=1
    injection per episode: the FIRST real (not dry_run) move always injects, so
    an episode can never be silently mislabeled as DART while being clean. Every
    later real move injects with probability p_inject — everything else stays
    random.
    """

    def dart_move(self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1):
        # only inject on REAL moves (dry_run is a planning-only probe); never
        # perturb env RNG / seed reproducibility (we use a dedicated Generator
        # passed in here).
        if not dry_run:
            # force-inject on the FIRST real move of the episode; afterwards
            # inject with probability p_inject. Guarantees n_injects >= 1.
            force = ep_counter["n_calls"] == 0
            ep_counter["n_calls"] += 1
            if force or rng.random() < p_inject:
                K = int(rng.integers(k_min, k_max + 1))
                _inject_disturbance(self, rng, sigma, K, move_id, floor)
                ep_counter["n_injects"] += 1
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
        override_seeds=None, save_video=False, inject_floor=0.15,
        config_suffix="", scheme="shove", jitter_frac=0.40, jitter_sigma=0.05,
        shove_min=4, shove_max=5, shove_k=10, shove_sigma_min=0.05,
        shove_sigma_max=0.09):
    env_id, yaml_rel, solver, n_agents = TASK_MAP[task_name]
    # --config-suffix lets data-gen select an alternate yaml (e.g.
    # table/lift_barrier_aug.yaml) WITHOUT touching eval: we splice the suffix in
    # before the .yaml extension. Empty suffix -> original path unchanged.
    config_rel = yaml_rel.replace(".yaml", f"{config_suffix}.yaml")
    config_path = osp.join(CONFIG_DIR, config_rel)

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
    print(f"[dart] config={config_rel}  (suffix={config_suffix!r})")
    print(f"[dart] scheme={scheme}")
    if scheme == "shove":
        print(f"[dart] sigma={sigma}  inject_floor={inject_floor}  "
              f"k=[{k_min},{k_max}]  p_inject={p_inject}  dart_seed={dart_seed}")
    else:  # dense
        print(f"[dart] jitter_frac={jitter_frac}  jitter_sigma={jitter_sigma}  "
              f"shove=[{shove_min},{shove_max}]  shove_k={shove_k}  "
              f"shove_sigma=[{shove_sigma_min},{shove_sigma_max}]  "
              f"inject_floor={inject_floor}  dart_seed={dart_seed}")

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

    # Episode-level counter shared with the monkeypatched methods. Reset to zero
    # immediately BEFORE each solver call (see the loop below). For "shove":
    # n_calls/n_injects (force-first-move => n_injects >= 1). For "dense":
    # n_calls/n_shove_events/n_jitter_steps (no force-first; events are
    # probabilistic so an episode can legitimately have 0 of either).
    ep_counter = {"n_calls": 0, "n_injects": 0,
                  "n_shove_events": 0, "n_jitter_steps": 0}

    # For "dense": tune p_shove so the EXPECTED number of shove events over a
    # typical episode lands at the mid of [shove_min, shove_max]. We don't know
    # the primitive count up front, so we estimate it from the solver's
    # move_to_pose_with_screw call count and clamp p_shove to [0, 1]. Tasks with
    # fewer primitives than the target simply get fewer events (e.g. LiftBarrier
    # has ~2 real moves so it caps below shove_min — acceptable/documented).
    n_moves_est = _estimate_n_moves(task_name)
    shove_target = (shove_min + shove_max) / 2.0
    p_shove = 1.0 if n_moves_est <= 0 else min(1.0, shove_target / n_moves_est)

    # --- install the DART monkeypatch(es) (restored in finally:) ---
    if scheme == "dense":
        PandaArmMotionPlanningSolver.move_to_pose_with_screw = _make_dense_move(
            rng, shove_sigma_min, shove_sigma_max, shove_k, inject_floor,
            p_shove, ep_counter
        )
        PandaArmMotionPlanningSolver.follow_path = _make_dart_jitter_follow_path(
            rng, jitter_frac, jitter_sigma, ep_counter
        )
        print(f"[dart] dense: n_moves_est={n_moves_est}  p_shove={p_shove:.3f} "
              f"(target ~{shove_target} events/episode)", flush=True)
    else:  # shove (EXISTING behavior, unchanged)
        PandaArmMotionPlanningSolver.move_to_pose_with_screw = _make_dart_move(
            rng, sigma, k_min, k_max, p_inject, inject_floor, ep_counter
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
            "scheme": str(scheme),
            "sigma": float(sigma),
            "inject_floor": float(inject_floor),
            "config_suffix": str(config_suffix),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "p_inject": float(p_inject),
            "dart_seed": int(dart_seed),
            # dense-scheme params (recorded regardless of scheme for provenance)
            "jitter_frac": float(jitter_frac),
            "jitter_sigma": float(jitter_sigma),
            "shove_min": int(shove_min),
            "shove_max": int(shove_max),
            "shove_k": int(shove_k),
            "shove_sigma_min": float(shove_sigma_min),
            "shove_sigma_max": float(shove_sigma_max),
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
                # Reset the episode-level injection counter so the patched move
                # force-injects on this episode's FIRST real move (>=1 guaranteed
                # in shove mode); also re-arms the dense per-episode tallies.
                ep_counter["n_calls"] = 0
                ep_counter["n_injects"] = 0
                ep_counter["n_shove_events"] = 0
                ep_counter["n_jitter_steps"] = 0
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
                    # n_injects is read from the shared counter for THIS episode
                    # (next attempt resets it). Guaranteed >= 1 by construction.
                    episodes_meta.append({
                        "episode_id": episode_id,
                        "env_seed": int(seed),
                        "seed": int(seed),     # back-compat alias
                        "sigma": float(sigma),
                        "n_injects": int(ep_counter["n_injects"]),
                        # dense-scheme per-episode tallies (0 in shove mode)
                        "n_jitter_steps": int(ep_counter["n_jitter_steps"]),
                        "n_shove_events": int(ep_counter["n_shove_events"]),
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
        # ALWAYS restore the original method(s), even on exception.
        PandaArmMotionPlanningSolver.move_to_pose_with_screw = _ORIG_MOVE
        PandaArmMotionPlanningSolver.follow_path = _ORIG_FOLLOW_PATH
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
    ap.add_argument("--scheme", choices=("shove", "dense"), default="shove",
                    help="'shove' (default) = EXISTING force-first-move + "
                         "probabilistic all-moving-arm shove behavior, unchanged. "
                         "'dense' = richly-but-gently perturbed recovery data: "
                         "per-step single-arm jitter (~jitter-frac of steps) PLUS "
                         "4-5 single-arm shove+replan events per episode")
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="[shove scheme] std-dev of Gaussian joint-target noise "
                         "added to the 7 arm joints of the moving arm(s) during "
                         "injection. Ignored by the dense scheme (which uses "
                         "--shove-sigma-min/--shove-sigma-max + --jitter-sigma)")
    ap.add_argument("--inject-floor", type=float, default=None, dest="inject_floor",
                    help="minimum L2 norm of the per-arm joint offset; offsets "
                         "drawn below this are scaled UP to the floor (random "
                         "direction kept) so a disturbance is never trivial. "
                         "Default 0.15 for shove, 0.0 for dense (dense sigma is "
                         "already small/explicit)")
    # --- dense-scheme params ---
    ap.add_argument("--jitter-frac", type=float, default=0.40, dest="jitter_frac",
                    help="[dense] fraction of recorded follow_path steps before "
                         "which one unwrapped single-arm jitter nudge fires")
    ap.add_argument("--jitter-sigma", type=float, default=0.05, dest="jitter_sigma",
                    help="[dense] std-dev of the per-step jitter nudge (7 arm "
                         "joints of one randomly-chosen moving arm)")
    ap.add_argument("--shove-min", type=int, default=4, dest="shove_min",
                    help="[dense] target min shove+replan events per episode")
    ap.add_argument("--shove-max", type=int, default=5, dest="shove_max",
                    help="[dense] target max shove+replan events per episode")
    ap.add_argument("--shove-k", type=int, default=10, dest="shove_k",
                    help="[dense] number of held unwrapped steps per shove event")
    ap.add_argument("--shove-sigma-min", type=float, default=0.05, dest="shove_sigma_min",
                    help="[dense] min of U(.,.) shove intensity (per event)")
    ap.add_argument("--shove-sigma-max", type=float, default=0.09, dest="shove_sigma_max",
                    help="[dense] max of U(.,.) shove intensity (per event)")
    ap.add_argument("--config-suffix", type=str, default="", dest="config_suffix",
                    help="suffix spliced before '.yaml' in the task config path "
                         "(e.g. '_aug' -> table/lift_barrier_aug.yaml) so data-gen "
                         "can pick an alternate scene WITHOUT touching eval")
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
    # inject_floor default depends on the scheme: 0.15 for shove (never-trivial
    # disturbances), 0.0 for dense (sigma is already small/explicit).
    inject_floor = args.inject_floor
    if inject_floor is None:
        inject_floor = 0.0 if args.scheme == "dense" else 0.15
    run(args.task, num_eff, args.record_dir,
        sigma=args.sigma,
        k_min=args.k_min,
        k_max=args.k_max,
        p_inject=args.p_inject,
        dart_seed=args.dart_seed,
        max_retries_per_seed=args.max_retries,
        per_attempt_timeout=args.per_attempt_timeout,
        override_seeds=override_seeds,
        save_video=args.save_video,
        inject_floor=inject_floor,
        config_suffix=args.config_suffix,
        scheme=args.scheme,
        jitter_frac=args.jitter_frac,
        jitter_sigma=args.jitter_sigma,
        shove_min=args.shove_min,
        shove_max=args.shove_max,
        shove_k=args.shove_k,
        shove_sigma_min=args.shove_sigma_min,
        shove_sigma_max=args.shove_sigma_max)
