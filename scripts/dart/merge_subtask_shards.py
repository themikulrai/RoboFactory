"""Merge parallel subtask-generation SHARDS into one combined dataset.

Data generation (``run_subtask_rollouts.py``) is embarrassingly parallel by seed: run
N shards on disjoint seed ranges, each writing its own dataset dir, then merge here.
Each shard numbers its ``traj_{episode_id}`` from 0, so naively concatenating would
collide; this script RE-NUMBERS every kept episode to a global contiguous id and copies
the trajectory H5 group, the subtask-stream H5 group, the RecordEpisodeMA JSON row, and
the subtask_meta row IN LOCKSTEP so the join key (episode_id) stays consistent across all
four artifacts -- the same invariant the converter relies on.

Pure h5py/json (no SAPIEN) -> safe to run on the login node.

Usage:
    python merge_subtask_shards.py --shards-root <ROOT>/shards --task LiftBarrier \\
        --out <ROOT>/LiftBarrier
    python merge_subtask_shards.py --selftest          # synthetic 2-shard round-trip
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def _shard_dirs(shards_root, task):
    # each shard wrote <shards_root>/shard_<k>/<task>/...
    dirs = sorted(
        glob.glob(os.path.join(shards_root, "shard_*", task)),
        key=lambda d: int(d.split("shard_")[1].split(os.sep)[0]),
    )
    return dirs


def merge(shard_dirs, task, out_dir, slice_filter=None):
    """Merge kept episodes from ``shard_dirs`` into ``out_dir``. If ``slice_filter``
    is set (e.g. "recovery"), only episodes whose meta ``slice`` matches are merged
    (used to build per-slice datasets for curriculum training)."""
    os.makedirs(out_dir, exist_ok=True)
    out_h5_path = os.path.join(out_dir, f"{task}.h5")
    out_stream_path = os.path.join(out_dir, f"{task}_subtask_stream.h5")
    out_json_path = os.path.join(out_dir, f"{task}.json")
    out_meta_path = os.path.join(out_dir, "subtask_meta.json")
    for p in (out_h5_path, out_stream_path):
        if os.path.exists(p):
            os.remove(p)

    combined_json = None
    combined_meta = None
    json_episodes = []
    meta_episodes = []
    kept_groups = 0
    total_groups = 0
    slice_kept = {}
    new_id = 0

    with h5py.File(out_h5_path, "w") as out_h5, h5py.File(out_stream_path, "w") as out_stream:
        for sd in shard_dirs:
            sh_h5 = os.path.join(sd, f"{task}.h5")
            sh_stream = os.path.join(sd, f"{task}_subtask_stream.h5")
            sh_json = os.path.join(sd, f"{task}.json")
            sh_meta = os.path.join(sd, "subtask_meta.json")
            if not (os.path.exists(sh_h5) and os.path.exists(sh_meta)):
                print(f"[merge] SKIP {sd} (missing h5/meta)")
                continue
            jd = _load_json(sh_json)
            md = _load_json(sh_meta)
            if combined_json is None:
                combined_json = {k: v for k, v in jd.items() if k != "episodes"}
                combined_meta = {k: v for k, v in md.items()
                                 if k not in ("episodes", "kept_episodes", "kept_groups",
                                              "total_groups_attempted", "slice_kept")}
            kept_groups += int(md.get("kept_groups", 0))
            total_groups += int(md.get("total_groups_attempted", 0))
            for k, v in md.get("slice_kept", {}).items():
                slice_kept[k] = slice_kept.get(k, 0) + int(v)

            # old_id -> RecordEpisodeMA json row (keyed by episode_id)
            json_by_id = {int(e["episode_id"]): e for e in jd.get("episodes", [])}

            with h5py.File(sh_h5, "r") as in_h5, h5py.File(sh_stream, "r") as in_stream:
                # iterate the kept set in meta order (authoritative for subtask data)
                for mrow in md.get("episodes", []):
                    if slice_filter is not None and mrow.get("slice") != slice_filter:
                        continue
                    old = int(mrow["episode_id"])
                    tk_old = f"traj_{old}"
                    if tk_old not in in_h5 or tk_old not in in_stream:
                        raise KeyError(f"{sd}: {tk_old} missing in h5/stream (join broken)")
                    if old not in json_by_id:
                        raise KeyError(f"{sd}: episode_id {old} missing in {task}.json")
                    tk_new = f"traj_{new_id}"
                    out_h5.copy(in_h5[tk_old], tk_new)
                    out_stream.copy(in_stream[tk_old], tk_new)
                    jrow = dict(json_by_id[old]); jrow["episode_id"] = new_id
                    json_episodes.append(jrow)
                    mr = dict(mrow); mr["episode_id"] = new_id
                    meta_episodes.append(mr)
                    new_id += 1

    combined_json = combined_json or {}
    combined_meta = combined_meta or {}
    combined_json["episodes"] = json_episodes
    combined_meta["episodes"] = meta_episodes
    combined_meta["kept_episodes"] = len(meta_episodes)
    combined_meta["kept_groups"] = kept_groups
    combined_meta["total_groups_attempted"] = total_groups
    combined_meta["slice_kept"] = slice_kept
    combined_meta["merged_from_shards"] = len(shard_dirs)
    with open(out_json_path, "w") as f:
        json.dump(combined_json, f, indent=2)
    with open(out_meta_path, "w") as f:
        json.dump(combined_meta, f, indent=2)

    # ---- verify join-key consistency ----
    with h5py.File(out_h5_path, "r") as f:
        h5_ids = set(f.keys())
    with h5py.File(out_stream_path, "r") as f:
        stream_ids = set(f.keys())
    json_ids = {f"traj_{e['episode_id']}" for e in json_episodes}
    meta_ids = {f"traj_{e['episode_id']}" for e in meta_episodes}
    ok = h5_ids == stream_ids == json_ids == meta_ids
    expect = {f"traj_{i}" for i in range(new_id)}
    contiguous = h5_ids == expect
    print(f"[merge] wrote {new_id} episodes to {out_dir}")
    print(f"[merge] per-slice: {slice_kept}")
    print(f"[merge] JOIN-KEY h5==stream==json==meta: {ok} | contiguous 0..{new_id-1}: {contiguous}")
    if not (ok and contiguous):
        raise SystemExit("[merge] FAILED join-key/contiguity check")
    return new_id


# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="mergetest_")
    task = "LiftBarrier"
    # build 2 fake shards, each 2 kept episodes with shard-local ids 0,1
    for k in range(2):
        sd = os.path.join(root, "shards", f"shard_{k}", task)
        os.makedirs(sd)
        with h5py.File(os.path.join(sd, f"{task}.h5"), "w") as h5, \
             h5py.File(os.path.join(sd, f"{task}_subtask_stream.h5"), "w") as st:
            for old in (0, 1):
                g = h5.create_group(f"traj_{old}")
                g.create_dataset("actions", data=np.full((5,), 100 * k + old, np.float32))
                sg = st.create_group(f"traj_{old}")
                sg.create_dataset("subtask_arm0_verb", data=np.array([k, old], np.int64))
        json.dump({"env_info": {"task": task}, "episodes": [
            {"episode_id": 0, "episode_seed": 250 * k + 0},
            {"episode_id": 1, "episode_seed": 250 * k + 1}]},
            open(os.path.join(sd, f"{task}.json"), "w"))
        json.dump({"task": task, "dart_sigma": 0.4, "kept_episodes": 2, "kept_groups": 2,
                   "total_groups_attempted": 5, "slice_kept": {"recovery": 1, "merged": 1},
                   "episodes": [
                       {"episode_id": 0, "env_seed": 250 * k, "variant": "simultaneous",
                        "family": "baseline", "contrast_group_id": 0, "T": 5, "slice": "recovery"},
                       {"episode_id": 1, "env_seed": 250 * k + 1, "variant": "stagger_a_leads",
                        "family": "stagger_a", "contrast_group_id": 0, "T": 5, "slice": "merged"}]},
                  open(os.path.join(sd, "subtask_meta.json"), "w"))
    out = os.path.join(root, task)
    n = merge(_shard_dirs(os.path.join(root, "shards"), task), task, out)
    assert n == 4, n
    # check a renumbered episode's data survived + matches its stream
    with h5py.File(os.path.join(out, f"{task}.h5")) as h5, \
         h5py.File(os.path.join(out, f"{task}_subtask_stream.h5")) as st:
        # traj_2 == shard1/traj_0 (value 100*1+0=100), stream verb [1,0]
        assert float(h5["traj_2/actions"][0]) == 100.0, float(h5["traj_2/actions"][0])
        assert list(st["traj_2/subtask_arm0_verb"][:]) == [1, 0]
    md = _load_json(os.path.join(out, "subtask_meta.json"))
    assert md["kept_episodes"] == 4 and md["slice_kept"] == {"recovery": 2, "merged": 2}, md["slice_kept"]
    assert [e["episode_id"] for e in md["episodes"]] == [0, 1, 2, 3]
    print("[selftest] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-root", help="dir containing shard_*/<task>/")
    ap.add_argument("--task", default="LiftBarrier")
    ap.add_argument("--out", help="combined output dir; with --by-slice, the parent "
                    "under which <slice>/<task>/ dirs are written")
    ap.add_argument("--by-slice", action="store_true",
                    help="emit 3 per-slice datasets (recovery/merged/clean) for curriculum "
                    "training instead of one combined dataset")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not (a.shards_root and a.out):
        ap.error("--shards-root and --out required (or --selftest)")
    dirs = _shard_dirs(a.shards_root, a.task)
    print(f"[merge] {len(dirs)} shards: {dirs}")
    if a.by_slice:
        for s in ("recovery", "merged", "clean"):
            print(f"\n[merge] === slice '{s}' ===")
            merge(dirs, a.task, os.path.join(a.out, s, a.task), slice_filter=s)
    else:
        merge(dirs, a.task, a.out)


if __name__ == "__main__":
    main()
