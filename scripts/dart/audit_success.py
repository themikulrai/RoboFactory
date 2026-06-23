"""Success-check AUDIT driver: OLD env evaluate() vs. candidate success.

RUN ON A GPU COMPUTE NODE ONLY (shader_pack=default); never on the login node.
(SAPIEN renders ~32% darker on the login node and the node may lack a GPU.)

What this does
--------------
For a given task it produces N episodes, and for each episode records:
  * OLD success  = the env's own ``evaluate()['success']`` at the final step
    (the buggy / grasp-blind check we want to audit), and
  * NEW success  = the candidate in ``robofactory.tasks.success_candidates``
    (LiftBarrier strict = height AND both arms grasping; TSC fixed = cube-C grasp
    read from the MIDDLE arm) evaluated on the SAME final env state.

Two ways to get episodes
------------------------
  1. GENERATE  (default): drive fresh episodes through the existing
     ``scripts/dart/run_subtask_rollouts`` machinery (same env build flags,
     same scenario sampler, same per-arm program interpreter). We hook the OLD
     and NEW success at the *last* step of each kept variant.
  2. REPLAY    (``--replay <dir>``): re-step an existing RecordEpisodeMA H5
     (reset by stored seed, step the stored per-agent actions). At the final
     step we read OLD success from info and NEW from the candidate. This avoids
     re-planning and audits the data you already shipped.

Output
------
Prints a confusion summary:
    n_episodes, n_old_pass, n_new_pass,
    n_old_pass_new_fail (with seeds)   <- candidate is STRICTER (caught a FP)
    n_new_pass_old_fail (with seeds)   <- candidate is LOOSER  (should be ~0)
and writes the DISAGREEMENT episodes' overlay material to ``--out`` for human
review: an overlay video if frames were captured, else first+last frames as PNGs,
each annotated with the OLD/NEW verdicts.

IMPORTANCE OF __main__ GUARD: this module imports cleanly WITHOUT sapien/gym (so
it is importable + unit-testable on the login node). ``gym.make`` and all sim
imports happen ONLY inside the functions called from ``__main__``.

Usage (on a compute node):
    python scripts/dart/audit_success.py --task LiftBarrier --num 25 \\
        --out /iris/u/mikulrai/data/RoboFactory/audit/lift_barrier
    python scripts/dart/audit_success.py --task ThreeRobotsStackCube --num 25 \\
        --out /iris/u/mikulrai/data/RoboFactory/audit/tsc
    python scripts/dart/audit_success.py --task LiftBarrier --replay \\
        /iris/u/mikulrai/data/RoboFactory/hf_download/LiftBarrier \\
        --num 25 --out /iris/u/mikulrai/data/RoboFactory/audit/lb_replay
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# NOTE: deliberately NO sapien / gym / torch / numpy import at module top so this
# module imports cleanly on the login node for unit tests. Heavy imports live
# inside the functions that actually touch the sim.

# make scripts/dart + scripts importable when run as a file
_DART_DIR = osp.dirname(osp.abspath(__file__))
_SCRIPTS_DIR = osp.dirname(_DART_DIR)
for _p in (_DART_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Pure data structures + summary (no sim deps -> importable / unit-testable)
# ---------------------------------------------------------------------------
@dataclass
class EpisodeAudit:
    """One episode's audit row."""
    seed: int
    episode_id: Optional[int]
    variant: Optional[str]
    old_pass: bool
    new_pass: bool
    frames: list = field(default_factory=list)  # optional list of HxWx3 uint8

    @property
    def disagree(self) -> bool:
        return self.old_pass != self.new_pass


def summarize(rows: List[EpisodeAudit]) -> dict:
    """Build the confusion summary dict from audit rows (PURE)."""
    old_pass = [r for r in rows if r.old_pass]
    new_pass = [r for r in rows if r.new_pass]
    old_pass_new_fail = [r for r in rows if r.old_pass and not r.new_pass]
    new_pass_old_fail = [r for r in rows if r.new_pass and not r.old_pass]
    return {
        "n_episodes": len(rows),
        "n_old_pass": len(old_pass),
        "n_new_pass": len(new_pass),
        "n_old_pass_new_fail": len(old_pass_new_fail),
        "n_new_pass_old_fail": len(new_pass_old_fail),
        "old_pass_new_fail_seeds": [r.seed for r in old_pass_new_fail],
        "new_pass_old_fail_seeds": [r.seed for r in new_pass_old_fail],
    }


def format_summary(task_name: str, mode: str, summary: dict) -> str:
    """Render the confusion summary as a human-readable block (PURE)."""
    lines = [
        f"==== success-check AUDIT: {task_name}  (mode={mode}) ====",
        f"  n_episodes            = {summary['n_episodes']}",
        f"  n_old_pass            = {summary['n_old_pass']}",
        f"  n_new_pass            = {summary['n_new_pass']}",
        f"  n_old_pass_new_fail   = {summary['n_old_pass_new_fail']}  "
        f"(candidate STRICTER; OLD false-positives)",
        f"      seeds: {summary['old_pass_new_fail_seeds']}",
        f"  n_new_pass_old_fail   = {summary['n_new_pass_old_fail']}  "
        f"(candidate LOOSER than OLD; expect ~0)",
        f"      seeds: {summary['new_pass_old_fail_seeds']}",
        "=" * 56,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sim-side helpers (heavy imports INSIDE; only called from __main__)
# ---------------------------------------------------------------------------
def _as_bool(x) -> bool:
    """Coerce a torch/np/scalar (possibly batched) success flag to python bool."""
    import numpy as np

    if x is None:
        return False
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    try:
        return bool(np.asarray(x).reshape(-1)[0])
    except (ValueError, IndexError, TypeError):
        return False


def _candidate_for(task_name):
    """Look up the candidate success fn (imports the PURE candidate module)."""
    from robofactory.tasks.success_candidates import CANDIDATES

    if task_name not in CANDIDATES:
        sys.exit(f"[ERROR] no success candidate for task {task_name!r} "
                 f"(have: {sorted(CANDIDATES)})")
    return CANDIDATES[task_name]


def _capture_frame(env):
    """Try to grab an RGB frame from the env; return HxWx3 uint8 or None."""
    import numpy as np

    e = getattr(env, "unwrapped", env)
    try:
        img = e.render_rgb_array()
    except Exception:
        return None
    if img is None:
        return None
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    # batched (B,H,W,3) -> first env
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ----------------------------- GENERATE mode -------------------------------
def run_generate(task_name, num, out_dir, max_steps=600,
                 per_attempt_timeout=300, capture_frames=True) -> List[EpisodeAudit]:
    """Generate N episodes via run_subtask_rollouts machinery and audit each.

    We reuse the env build + planner + scenario sampler from
    ``run_subtask_rollouts`` but DO NOT write the dataset; we only need the final
    env state per kept variant to evaluate OLD vs. NEW success.
    """
    os.environ.setdefault("SAPIEN_HEADLESS", "1")
    import signal

    import gymnasium as gym

    import robofactory  # registers envs + PandaWristCamMulti  # noqa: F401
    from robofactory import CONFIG_DIR
    from robofactory.planner.subtask_interpreter import SubtaskRecorder, run_program
    import scripts.dart.run_subtask_rollouts as R

    cand = _candidate_for(task_name)
    env_id, yaml_rel, n_agents, sampler = R.TASK_MAP[task_name]
    config_path = osp.join(CONFIG_DIR, yaml_rel)

    env = gym.make(
        env_id,
        config=config_path,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="cpu",
        robot_uids=("panda_wristcam_multi",) * n_agents,
    )

    signal.signal(signal.SIGALRM, R._solver_alarm_handler)
    seeds = R._load_seeds(task_name, num)

    rows: List[EpisodeAudit] = []
    try:
        for seed in seeds:
            if len(rows) >= num:
                break
            specs = sampler(int(seed))
            import robofactory.planner.subtask_scenarios as scenarios
            groups = scenarios.group_specs(specs)
            for gid, members in sorted(groups.items()):
                if len(rows) >= num:
                    break
                for spec in members:
                    if len(rows) >= num:
                        break
                    signal.alarm(int(per_attempt_timeout))
                    try:
                        planner = R._build_planner(env, spec.seed)
                        rec = SubtaskRecorder(num_arms=spec.num_arms)
                        try:
                            programs = spec.build(planner, env)
                        except Exception as e:
                            print(f"  seed {seed} variant {spec.name}: BUILD failed "
                                  f"{type(e).__name__}: {e}", flush=True)
                            continue
                        out = run_program(
                            env, planner, programs, rec, max_steps=max_steps,
                            control_mode=planner.control_mode,
                        )
                    except R._SolverTimeout:
                        print(f"  seed {seed} variant {spec.name}: TIMEOUT", flush=True)
                        continue
                    finally:
                        signal.alarm(0)

                    info = out.get("info", {}) if out else {}
                    old_pass = _as_bool(info.get("success")) if info else \
                        _as_bool(env.unwrapped.evaluate().get("success"))
                    new_pass = _as_bool(cand(env))
                    frame = _capture_frame(env) if capture_frames else None
                    rows.append(EpisodeAudit(
                        seed=int(seed),
                        episode_id=None,
                        variant=spec.name,
                        old_pass=old_pass,
                        new_pass=new_pass,
                        frames=[frame] if frame is not None else [],
                    ))
                    print(f"  seed {seed} variant {spec.name}: "
                          f"OLD={old_pass} NEW={new_pass}"
                          f"{'  <-- DISAGREE' if old_pass != new_pass else ''}",
                          flush=True)
    finally:
        env.close()
    return rows


# ------------------------------ REPLAY mode --------------------------------
def run_replay(task_name, replay_dir, num, capture_frames=True) -> List[EpisodeAudit]:
    """Re-step an existing RecordEpisodeMA H5 and audit OLD vs. NEW at last step."""
    os.environ.setdefault("SAPIEN_HEADLESS", "1")
    import h5py
    import numpy as np
    import gymnasium as gym

    import robofactory  # registers envs  # noqa: F401
    from robofactory import CONFIG_DIR
    import scripts.dart.run_subtask_rollouts as R

    cand = _candidate_for(task_name)
    env_id, yaml_rel, n_agents, _ = R.TASK_MAP[task_name]
    config_path = osp.join(CONFIG_DIR, yaml_rel)

    h5_path = osp.join(replay_dir, f"{task_name}.h5")
    json_path = osp.join(replay_dir, f"{task_name}.json")
    if not osp.exists(h5_path):
        sys.exit(f"[ERROR] replay H5 not found: {h5_path}")
    meta = json.load(open(json_path)) if osp.exists(json_path) else None
    episodes = meta["episodes"][:num] if meta else None

    env = gym.make(
        env_id,
        config=config_path,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="cpu",
        robot_uids=("panda_wristcam_multi",) * n_agents,
    )
    agent_keys = [f"panda-{i}" for i in range(n_agents)]

    rows: List[EpisodeAudit] = []
    with h5py.File(h5_path, "r") as h5f:
        if episodes is None:
            keys = sorted(k for k in h5f.keys() if k.startswith("traj_"))[:num]
            episodes = [{"episode_id": int(k.split("_")[1]),
                         "episode_seed": int(k.split("_")[1])} for k in keys]
        for ep in episodes:
            if len(rows) >= num:
                break
            ep_id = ep["episode_id"]
            seed = ep.get("episode_seed", ep_id)
            traj = h5f[f"traj_{ep_id}"]
            actions = {k: np.asarray(traj["actions"][k][:]) for k in agent_keys}
            T = actions[agent_keys[0]].shape[0]
            env.reset(seed=int(seed))
            last_info = {}
            for t in range(T):
                act = {k: actions[k][t].astype(np.float32) for k in agent_keys}
                _o, _r, _term, trunc, last_info = env.step(act)
                if trunc:
                    break
            old_pass = _as_bool(last_info.get("success"))
            new_pass = _as_bool(cand(env))
            frame = _capture_frame(env) if capture_frames else None
            rows.append(EpisodeAudit(
                seed=int(seed),
                episode_id=int(ep_id),
                variant=None,
                old_pass=old_pass,
                new_pass=new_pass,
                frames=[frame] if frame is not None else [],
            ))
            print(f"  traj_{ep_id} (seed {seed}): OLD={old_pass} NEW={new_pass}"
                  f"{'  <-- DISAGREE' if old_pass != new_pass else ''}", flush=True)
    env.close()
    return rows


# --------------------------- disagreement output ---------------------------
def save_disagreements(rows: List[EpisodeAudit], out_dir: str):
    """Save first+last frames (annotated) for every disagreement episode."""
    import numpy as np

    disagree = [r for r in rows if r.disagree]
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for r in disagree:
        tag = f"seed{r.seed}"
        if r.episode_id is not None:
            tag += f"_traj{r.episode_id}"
        if r.variant:
            tag += f"_{r.variant}"
        tag += f"_OLD{int(r.old_pass)}_NEW{int(r.new_pass)}"
        if not r.frames:
            continue
        try:
            import imageio.v2 as imageio
        except Exception:
            imageio = None
        if imageio is None:
            # last-resort: dump as .npy so frames aren't lost
            np.save(osp.join(out_dir, tag + "_frames.npy"),
                    np.asarray(r.frames))
            saved.append(tag + "_frames.npy")
            continue
        first = r.frames[0]
        last = r.frames[-1]
        imageio.imwrite(osp.join(out_dir, tag + "_first.png"), first)
        imageio.imwrite(osp.join(out_dir, tag + "_last.png"), last)
        saved.append(tag)
        if len(r.frames) > 1:
            try:
                imageio.mimsave(osp.join(out_dir, tag + "_overlay.mp4"),
                                r.frames, fps=15)
            except Exception:
                pass
    # write the rows manifest (frames stripped) for the human reviewer
    manifest = [
        {"seed": r.seed, "episode_id": r.episode_id, "variant": r.variant,
         "old_pass": r.old_pass, "new_pass": r.new_pass, "disagree": r.disagree}
        for r in rows
    ]
    with open(osp.join(out_dir, "audit_rows.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="LiftBarrier",
                    choices=["LiftBarrier", "ThreeRobotsStackCube"])
    ap.add_argument("--num", type=int, default=25,
                    help="number of episodes to audit")
    ap.add_argument("--replay", type=str, default=None,
                    help="dir holding <Task>.h5 (+ optional .json) to re-step "
                         "instead of generating fresh episodes")
    ap.add_argument("--record-dir", type=str, default=None,
                    help="(reserved) record dir for generation; audit does not "
                         "write the dataset")
    ap.add_argument("--out", type=str, required=True,
                    help="dir to write disagreement frames/videos + manifest")
    ap.add_argument("--max-steps", type=int, default=600, dest="max_steps")
    ap.add_argument("--per-attempt-timeout", type=int, default=300,
                    dest="per_attempt_timeout")
    ap.add_argument("--no-frames", action="store_true", default=False,
                    help="skip RGB frame capture (faster; no disagreement videos)")
    args = ap.parse_args()

    capture = not args.no_frames
    if args.replay:
        mode = "replay"
        rows = run_replay(args.task, args.replay, args.num, capture_frames=capture)
    else:
        mode = "generate"
        rows = run_generate(args.task, args.num, args.record_dir,
                            max_steps=args.max_steps,
                            per_attempt_timeout=args.per_attempt_timeout,
                            capture_frames=capture)

    summary = summarize(rows)
    print()
    print(format_summary(args.task, mode, summary))
    saved = save_disagreements(rows, args.out)
    print(f"[audit] wrote {len(saved)} disagreement artifact set(s) + manifest "
          f"to {args.out}")


if __name__ == "__main__":
    main()
