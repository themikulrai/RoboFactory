"""Per-seed difficulty cascade driver for DART OOD-recovery data generation.

Repeatedly invokes ``run_dart_rollouts.py`` as a subprocess, one tier at a time,
in order T0 -> T1 -> T2 (hardest -> easiest). Each tier is run on the set of
candidate seeds that have NOT yet produced a kept episode; seeds that PASS in a
tier are removed from the pool, seeds that FAIL drop down to the next (easier)
tier. We stop once ``--target`` episodes have been kept (or seeds/tiers run out).

Each tier writes into its own record-dir (``<out-dir>/_tiers/<tier.name>``)
because ``run_dart_rollouts.py`` ``sys.exit()``s if its <Task>.h5 already exists
non-empty. After all tiers, the per-tier H5/JSON are MERGED into a single
``<out-dir>/<Task>/<Task>.h5`` (+ ``.json``) with ``traj_*`` renumbered
contiguously from 0, plus a ``cascade_meta.json`` recording which tier produced
each kept episode.

Pure subprocess + h5py + json on the driver side — no sim / mani_skill import,
so the merge logic runs anywhere (the GPU work happens inside the subprocess).

Usage:
    python scripts/dart/run_dart_cascade.py \\
        --task LiftBarrier \\
        --out-dir /iris/u/mikulrai/data/RoboFactory/hf_download_cascade/LB \\
        --target 4 --seed-pool 100:140 [--save-video]
"""
import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys

import h5py

_TRAJ_RE = re.compile(r"^traj_(\d+)$")

# ---------------------------------------------------------------------------
# Built-in default difficulty tiers (applied in order T0 -> T1 -> T2).
# All tiers also pass: --scheme dense --sigma 0.05 --inject-floor 0.0
#                      --max-retries 1
# ---------------------------------------------------------------------------
DEFAULT_TIERS = {
    "LiftBarrier": [
        {"name": "T0", "config_suffix": "_augwide", "jitter_frac": 0.40,
         "shove_min": 4, "shove_max": 5},
        {"name": "T1", "config_suffix": "_aug", "jitter_frac": 0.30,
         "shove_min": 3, "shove_max": 4},
        {"name": "T2", "config_suffix": "_aug", "jitter_frac": 0.20,
         "shove_min": 2, "shove_max": 3},
    ],
    # NOTE: no wide-object-noise (_aug) tier for TSC. The 3-arm stacking planner
    # HANGS on widely-displaced cubes -> _aug+aggressive yields 0% (confirmed
    # cascade-TSC 15973284 T0 = 0/25, all SIGALRM timeouts). TSC object-diversity
    # is therefore capped at _augmild (~1.6x canonical); start straight at the
    # mild tiers. LB keeps _augwide (it ate wide randomization at ~30%).
    "ThreeRobotsStackCube": [
        {"name": "T0", "config_suffix": "_augmild", "jitter_frac": 0.30,
         "shove_min": 3, "shove_max": 4},
        {"name": "T1", "config_suffix": "_augmild", "jitter_frac": 0.20,
         "shove_min": 2, "shove_max": 3},
    ],
}

DEFAULT_TIMEOUT = {"LiftBarrier": 60, "ThreeRobotsStackCube": 200}

_GEN_SCRIPT = osp.join(osp.dirname(osp.abspath(__file__)), "run_dart_rollouts.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sorted_traj_keys(h5):
    keys = [k for k in h5.keys() if _TRAJ_RE.match(k)]
    return sorted(keys, key=lambda k: int(_TRAJ_RE.match(k).group(1)))


def parse_seed_pool(spec):
    """Parse "START:END" (inclusive) or a comma list into a list of ints."""
    spec = spec.strip()
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        lo, hi = int(lo), int(hi)
        return list(range(lo, hi + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def _task_paths(record_dir, task):
    """(<rec>/<Task>, <rec>/<Task>/<Task>.h5, <...>.json, <...>/dart_meta.json)."""
    d = osp.join(record_dir, task)
    return d, osp.join(d, f"{task}.h5"), osp.join(d, f"{task}.json"), \
        osp.join(d, "dart_meta.json")


def build_gen_cmd(task, tier_dir, tier, per_attempt_timeout, seeds, save_video):
    """Build the run_dart_rollouts.py subprocess argv for one tier."""
    cmd = [
        sys.executable, _GEN_SCRIPT,
        "--task", task,
        "--record-dir", tier_dir,
        "--scheme", "dense",
        "--config-suffix", tier["config_suffix"],
        "--jitter-frac", str(tier["jitter_frac"]),
        "--shove-min", str(tier["shove_min"]),
        "--shove-max", str(tier["shove_max"]),
        "--sigma", "0.05",
        "--inject-floor", "0.0",
        "--max-retries", "1",
        "--per-attempt-timeout", str(per_attempt_timeout),
        "--seeds-csv", ",".join(str(s) for s in seeds),
    ]
    if save_video:
        cmd.append("--save-video")
    return cmd


def read_passed_seeds(tier_dir, task):
    """Read dart_meta.json -> [env_seed,...] for episodes that PASSED."""
    _, _, _, meta_path = _task_paths(tier_dir, task)
    if not osp.exists(meta_path):
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    seeds = []
    for ep in meta.get("episodes", []):
        s = ep.get("env_seed", ep.get("seed"))
        if s is not None:
            seeds.append(int(s))
    return seeds


# ---------------------------------------------------------------------------
# merge: concat per-tier H5/JSON into <out-dir>/<Task>/<Task>.h5 (+ .json)
# ---------------------------------------------------------------------------
def merge_tiers(task, out_dir, tier_records, target, save_video=False):
    """Concatenate every tier's H5/JSON into the final merged output.

    Arguments:
        tier_records: list of (tier_name, tier_dir, passed_seeds) in apply order.
    Returns the number of merged trajectories.
    """
    final_dir, out_h5, out_json_path, _ = _task_paths(out_dir, task)
    os.makedirs(final_dir, exist_ok=True)

    next_id = 0
    merged_episodes = []
    cascade_episodes = []  # [{traj_idx, tier, env_seed}]

    with h5py.File(out_h5, "w") as fout:
        for tier_name, tier_dir, _passed in tier_records:
            _, src_h5, src_json_path, _ = _task_paths(tier_dir, task)
            # A tier that produced 0 episodes may have no / empty H5 — skip it.
            if not osp.exists(src_h5) or osp.getsize(src_h5) <= 1024:
                continue

            src_json = None
            if osp.exists(src_json_path):
                with open(src_json_path) as f:
                    src_json = json.load(f)
            # episode_id -> episode dict, for json alignment by traj index
            ep_by_id = {}
            if src_json is not None:
                for ep in src_json.get("episodes", []):
                    if "episode_id" in ep:
                        ep_by_id[int(ep["episode_id"])] = ep

            with h5py.File(src_h5, "r") as fin:
                for k in _sorted_traj_keys(fin):
                    old_idx = int(_TRAJ_RE.match(k).group(1))
                    new_key = f"traj_{next_id}"
                    fin.copy(k, fout, name=new_key)

                    ep = ep_by_id.get(old_idx)
                    env_seed = None
                    if ep is not None:
                        new_ep = dict(ep)
                        new_ep["episode_id"] = next_id
                        merged_episodes.append(new_ep)
                        # surface the env seed if the generator recorded it
                        env_seed = new_ep.get("env_seed", new_ep.get("seed"))
                        rk = new_ep.get("reset_kwargs") or {}
                        if env_seed is None and isinstance(rk, dict):
                            env_seed = rk.get("seed")
                    cascade_episodes.append({
                        "traj_idx": next_id,
                        "tier": tier_name,
                        "env_seed": (int(env_seed) if env_seed is not None else None),
                    })
                    next_id += 1

    # ---- merged <Task>.json (preserve generator top-level keys) ----------
    out_json = {}
    # carry top-level keys from the FIRST tier json that has them
    for tier_name, tier_dir, _passed in tier_records:
        _, _, src_json_path, _ = _task_paths(tier_dir, task)
        if osp.exists(src_json_path):
            with open(src_json_path) as f:
                sj = json.load(f)
            for k, v in sj.items():
                if k == "episodes":
                    continue
                out_json.setdefault(k, v)
    out_json["episodes"] = merged_episodes
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    # ---- cascade_meta.json ----------------------------------------------
    cascade_meta = {
        "task": task,
        "target": target,
        "total_kept": next_id,
        "episodes": cascade_episodes,
    }
    with open(osp.join(final_dir, "cascade_meta.json"), "w") as f:
        json.dump(cascade_meta, f, indent=2)

    # ---- best-effort video copy -----------------------------------------
    # run_dart_rollouts.py -> RecordEpisodeMA names videos "<video_id>.mp4"
    # (a per-flush counter), NOT keyed to traj index, and failed attempts may
    # also flush. So we cannot reliably map a clip to a traj_idx; we copy each
    # tier's MP4s with a tier-prefixed name and record only the source path.
    # Never fail the merge over videos.
    if save_video:
        try:
            video_map = []
            vdir = osp.join(final_dir, "videos")
            os.makedirs(vdir, exist_ok=True)
            for tier_name, tier_dir, _passed in tier_records:
                src_task_dir, _, _, _ = _task_paths(tier_dir, task)
                if not osp.isdir(src_task_dir):
                    continue
                for fn in sorted(os.listdir(src_task_dir)):
                    if not fn.lower().endswith(".mp4"):
                        continue
                    src = osp.join(src_task_dir, fn)
                    dst_name = f"{tier_name}_{fn}"
                    shutil.copy2(src, osp.join(vdir, dst_name))
                    video_map.append({"file": dst_name, "tier": tier_name,
                                      "src": fn})
            if video_map:
                with open(osp.join(vdir, "video_map.json"), "w") as f:
                    json.dump(video_map, f, indent=2)
        except Exception as e:  # noqa: BLE001 - videos are non-essential
            print(f"[cascade] WARN: video copy failed ({type(e).__name__}: {e});"
                  " continuing without videos", flush=True)

    return next_id


# ---------------------------------------------------------------------------
# main cascade
# ---------------------------------------------------------------------------
def run_cascade(task, out_dir, target, seed_pool, tiers, per_attempt_timeout,
                save_video):
    remaining = list(seed_pool)
    kept = 0
    tier_records = []  # (name, tier_dir, passed_seeds)

    for tier in tiers:
        if kept >= target or not remaining:
            break
        tier_dir = osp.join(out_dir, "_tiers", tier["name"])
        cmd = build_gen_cmd(task, tier_dir, tier, per_attempt_timeout,
                            remaining, save_video)
        print(f"[cascade] tier {tier['name']}: launching on "
              f"{len(remaining)} seed(s) -> {tier_dir}", flush=True)
        print(f"[cascade]   cmd: {' '.join(cmd)}", flush=True)
        # pass through the current env (incl PYTHONPATH) so the subprocess
        # imports robofactory from the same worktree.
        subprocess.run(cmd, check=False, env=os.environ.copy())

        passed_seeds = read_passed_seeds(tier_dir, task)
        attempted = len(remaining)
        kept += len(passed_seeds)
        tier_records.append((tier["name"], tier_dir, passed_seeds))
        passed_set = set(passed_seeds)
        remaining = [s for s in remaining if s not in passed_set]
        print(f"[cascade] tier {tier['name']}: kept {len(passed_seeds)}/"
              f"{attempted}, cumulative {kept}/{target}", flush=True)

    total = merge_tiers(task, out_dir, tier_records, target, save_video)
    final_dir, out_h5, out_json_path, _ = _task_paths(out_dir, task)
    print(f"[cascade] DONE: merged {total} trajectories -> {out_h5}", flush=True)
    print(f"[cascade]   json    -> {out_json_path}", flush=True)
    print(f"[cascade]   cascade -> {osp.join(final_dir, 'cascade_meta.json')}",
          flush=True)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(DEFAULT_TIERS), required=True)
    ap.add_argument("--out-dir", required=True, dest="out_dir",
                    help="final merged output goes to <out-dir>/<Task>/...")
    ap.add_argument("--target", type=int, default=4,
                    help="stop once this many episodes are KEPT across tiers")
    ap.add_argument("--seed-pool", required=True, dest="seed_pool",
                    help='candidate env seeds: "START:END" (inclusive) or comma list')
    ap.add_argument("--save-video", action="store_true", dest="save_video",
                    default=False, help="pass through to the generator")
    ap.add_argument("--per-attempt-timeout", type=int, default=None,
                    dest="per_attempt_timeout",
                    help="seconds per solver attempt (default: 60 LB / 200 TSC)")
    ap.add_argument("--tiers-json", default=None, dest="tiers_json",
                    help="optional JSON file with a tier list overriding the "
                         "built-in default tiers for the task")
    args = ap.parse_args()

    if args.per_attempt_timeout is None:
        args.per_attempt_timeout = DEFAULT_TIMEOUT[args.task]

    if args.tiers_json:
        with open(args.tiers_json) as f:
            tiers = json.load(f)
    else:
        tiers = DEFAULT_TIERS[args.task]

    seed_pool = parse_seed_pool(args.seed_pool)
    print(f"[cascade] task={args.task} target={args.target} "
          f"pool={len(seed_pool)} seeds timeout={args.per_attempt_timeout}s "
          f"tiers={[t['name'] for t in tiers]}", flush=True)

    run_cascade(args.task, args.out_dir, args.target, seed_pool, tiers,
                args.per_attempt_timeout, args.save_video)


if __name__ == "__main__":
    main()
