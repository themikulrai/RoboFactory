"""Merge N shard datasets (each <root>/shardK/<Task>/) into ONE dataset with
slices INTERLEAVED across traj_0..N-1 (no curriculum blocks), reindexing the
trajectory H5, the subtask-stream H5, and subtask_meta.json in lockstep.

Why interleave at merge time: the generator writes slices sequentially (recovery
block, merged block, clean block) PER shard. Concatenating shards naively would
leave slice-blocks. We re-emit episodes in a proportional round-robin so any
sequential read sees a mix of recovery/merged/clean.

Join-key safety: traj_{id} is the join key shared by the traj H5, the stream H5,
and meta. We reassign new ids and copy ALL THREE in the same order, so they can
never desync (the exact failure the generator's clean_on_close=False guards
against; here we own all three and rewrite them together).

Usage:
  python merge_shards.py --task LiftBarrier \
    --root <full200 dir> --out <merged dir> \
    --targets recovery=60,merged=60,clean=80
"""
import argparse, glob, json, os, os.path as osp, sys
import h5py
import numpy as np


def _parse_targets(s):
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def _interleave(by_slice, targets):
    """Proportional round-robin interleave. Returns an ordered list of
    (slice, shard_dir, old_key, ep_meta). Each step emits from the slice with the
    smallest (emitted+0.5)/target ratio that still has episodes left, so slices are
    spread evenly and each ends at its target count."""
    # trim each slice to its target (deterministic: keep first `target` in shard,
    # then env_seed, then old_id order so the selection is stable & reproducible).
    pools = {}
    for sl, eps in by_slice.items():
        eps_sorted = sorted(eps, key=lambda e: (e["_shard_idx"], int(e["env_seed"]), int(e["_old_id"])))
        tgt = targets.get(sl, len(eps_sorted))
        pools[sl] = eps_sorted[:tgt]
    emitted = {sl: 0 for sl in pools}
    order = []
    total = sum(min(len(pools[sl]), targets.get(sl, len(pools[sl]))) for sl in pools)
    for _ in range(total):
        # eligible slices that still have episodes
        cand = [sl for sl in pools if emitted[sl] < len(pools[sl])]
        if not cand:
            break
        # pick smallest fill ratio (ties -> larger target first for even spread)
        sl = min(cand, key=lambda s: ((emitted[s] + 0.5) / max(targets.get(s, 1), 1),
                                      -targets.get(s, 0)))
        ep = pools[sl][emitted[sl]]
        order.append((sl, ep["_shard_dir"], ep["_old_key"], ep))
        emitted[sl] += 1
    return order, emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--root", required=True, help="dir containing shard*/<Task>/")
    ap.add_argument("--out", required=True, help="merged output dir (gets <Task>/ inside)")
    ap.add_argument("--targets", required=True, help="recovery=60,merged=60,clean=80")
    args = ap.parse_args()
    task = args.task
    targets = _parse_targets(args.targets)

    shard_dirs = sorted(glob.glob(osp.join(args.root, "shard*", task)))
    if not shard_dirs:
        sys.exit(f"[merge] no shards found under {args.root}/shard*/{task}")
    print(f"[merge] {len(shard_dirs)} shards: {shard_dirs}")

    # ---- gather episode records from every shard meta ----
    by_slice = {}
    shard_meta = {}
    for si, sd in enumerate(shard_dirs):
        meta = json.load(open(osp.join(sd, "subtask_meta.json")))
        shard_meta[sd] = meta
        for ep in meta["episodes"]:
            rec = dict(ep)
            rec["_shard_dir"] = sd
            rec["_shard_idx"] = si
            rec["_old_id"] = ep["episode_id"]
            rec["_old_key"] = f"traj_{ep['episode_id']}"
            by_slice.setdefault(ep.get("slice", "?"), []).append(rec)
    avail = {sl: len(v) for sl, v in by_slice.items()}
    print(f"[merge] available per slice: {avail}  targets: {targets}")
    short = {sl: targets[sl] - avail.get(sl, 0) for sl in targets if avail.get(sl, 0) < targets[sl]}
    if short:
        print(f"[merge][WARN] UNDERSHOOT (need more generation): {short}")

    order, emitted = _interleave(by_slice, targets)
    print(f"[merge] emitting {len(order)} episodes; per slice: {emitted}")

    # ---- write merged dataset ----
    out_task = osp.join(args.out, task)
    os.makedirs(out_task, exist_ok=True)
    dst_h5 = h5py.File(osp.join(out_task, f"{task}.h5"), "w")
    dst_st = h5py.File(osp.join(out_task, f"{task}_subtask_stream.h5"), "w")

    # cache open source files
    src_h5 = {sd: h5py.File(osp.join(sd, f"{task}.h5"), "r", locking=False) for sd in shard_dirs}
    src_st = {sd: h5py.File(osp.join(sd, f"{task}_subtask_stream.h5"), "r", locking=False) for sd in shard_dirs}

    episodes_meta = []
    slice_kept = {}
    variant_kept = {}
    runlen = 1; maxrun = 1; prev = None
    for new_id, (sl, sd, old_key, ep) in enumerate(order):
        new_key = f"traj_{new_id}"
        src_h5[sd].copy(old_key, dst_h5, name=new_key)
        src_st[sd].copy(old_key, dst_st, name=new_key)
        row = {k: v for k, v in ep.items() if not k.startswith("_")}
        row["episode_id"] = new_id
        row["orig_shard"] = ep["_shard_idx"]
        row["orig_id"] = ep["_old_id"]
        episodes_meta.append(row)
        slice_kept[sl] = slice_kept.get(sl, 0) + 1
        var = ep.get("variant", "?")
        variant_kept[var] = variant_kept.get(var, 0) + 1
        if sl == prev:
            runlen += 1; maxrun = max(maxrun, runlen)
        else:
            runlen = 1
        prev = sl
    dst_h5.flush(); dst_st.flush()

    # ---- validation ----
    htraj = sorted(k for k in dst_h5 if k.startswith("traj_"))
    straj = sorted(k for k in dst_st if k.startswith("traj_"))
    N = len(order)
    problems = []
    if len(htraj) != N: problems.append(f"H5 trajs {len(htraj)} != {N}")
    if len(straj) != N: problems.append(f"stream trajs {len(straj)} != {N}")
    if set(htraj) != set(straj): problems.append("H5/stream key set mismatch")
    meta_ids = set(f"traj_{e['episode_id']}" for e in episodes_meta)
    if meta_ids != set(htraj): problems.append("meta/H5 id mismatch")
    for sl, tgt in targets.items():
        if slice_kept.get(sl, 0) != tgt:
            problems.append(f"slice {sl}: {slice_kept.get(sl,0)} != target {tgt}")
    # jitter audit + T consistency (sample)
    jit_clean = 0; jit_pert = 0
    for e in episodes_meta:
        nj = e.get("n_jitter_steps", 0)
        if e["slice"] == "clean" and nj != 0:
            problems.append(f"clean traj_{e['episode_id']} has jitter={nj}")
        if e["slice"] != "clean" and nj > 0:
            jit_pert += 1
        if e["slice"] == "clean":
            jit_clean += 1
    # T consistency: stream arm0 length vs H5 action length, on a sample of 10
    sample = np.linspace(0, N - 1, min(10, N)).astype(int)
    for i in sample:
        k = f"traj_{i}"
        st_T = dst_st[k]["subtask_arm0_text"].shape[0]
        # action length: try common layouts
        ah = dst_h5[k]
        a_len = None
        if "actions" in ah and isinstance(ah["actions"], h5py.Dataset):
            a_len = ah["actions"].shape[0]
        elif "actions" in ah:
            sub = list(ah["actions"].keys())
            a_len = ah["actions"][sub[0]].shape[0]
        if a_len is not None and st_T != a_len:
            problems.append(f"traj_{i}: stream T {st_T} != actions {a_len}")

    meta = {
        "task": task, "merged_from_shards": len(shard_dirs),
        "kept_episodes": N, "slice_kept": slice_kept, "variant_kept": variant_kept,
        "targets": targets, "max_consecutive_same_slice": maxrun,
        "jitter_clean_eps": jit_clean, "jitter_perturbed_eps": jit_pert,
        "interleaved": True, "episodes": episodes_meta,
    }
    json.dump(meta, open(osp.join(out_task, "subtask_meta.json"), "w"), indent=2)
    # Also emit <Task>.json (maniskill-style episodes list) — the openpi converter
    # reads this for its episode success-map (episode_id -> success). Every merged
    # episode passed the keep-criterion, so success=True for all.
    task_json = {
        "env_info": {"env_id": task, "merged": True},
        "source_type": "motionplanning",
        "source_desc": "merged+interleaved subtask DART dataset",
        "episodes": [
            {"episode_id": e["episode_id"], "episode_seed": int(e.get("env_seed", 0)),
             "elapsed_steps": int(e.get("T", 0)), "success": True,
             "reset_kwargs": {"options": {}, "seed": int(e.get("env_seed", 0))}}
            for e in episodes_meta
        ],
    }
    json.dump(task_json, open(osp.join(out_task, f"{task}.json"), "w"), indent=2)
    dst_h5.close(); dst_st.close()
    for f in list(src_h5.values()) + list(src_st.values()):
        f.close()

    print(f"\n[merge] WROTE {out_task}")
    print(f"[merge] N={N} slice_kept={slice_kept} variant_kept={variant_kept}")
    print(f"[merge] max consecutive same-slice run = {maxrun} (low = well interleaved)")
    print(f"[merge] jitter: clean eps={jit_clean} (all 0 jitter), perturbed eps with jitter={jit_pert}")
    # show the first 30 slice tags to eyeball the interleave
    head = "".join({"recovery": "R", "merged": "M", "clean": "C"}.get(e["slice"], "?")
                   for e in episodes_meta[:40])
    print(f"[merge] first-40 slice order: {head}")
    if problems:
        print("\n[merge][FAIL] " + "\n[merge][FAIL] ".join(problems))
        sys.exit(1)
    print("\n[merge] ALL VALIDATION PASSED ✓")


if __name__ == "__main__":
    main()
