"""Merge a scripted (clean) trajectory H5 with a DART (OOD-recovery) H5.

Concatenates all ``traj_*`` groups from the scripted file followed by all from
the DART file into a single output H5, renumbering the DART trajectories to
continue contiguously after the scripted ones (traj_0..traj_{N-1}). A sibling
output .json is written by concatenating both source .json ``episodes`` lists
with ``episode_id`` renumbered to match, preserving each episode's success flag
and reset_kwargs/seed. ``env_info`` is taken from the scripted json.

Pure h5py + json — no sim / mani_skill imports, so it runs anywhere.

Usage:
    python scripts/dart/merge_h5.py \\
        --scripted-h5 .../hf_download/LiftBarrier/LiftBarrier.h5 \\
        --dart-h5     .../hf_download_dart/sigma0.30/LiftBarrier/LiftBarrier.h5 \\
        --out-h5      .../hf_download_merged/LiftBarrier/LiftBarrier.h5
"""
import argparse
import json
import os
import os.path as osp
import re

import h5py

_TRAJ_RE = re.compile(r"^traj_(\d+)$")


def _sorted_traj_keys(h5):
    """Return traj group names sorted by their integer index."""
    keys = [k for k in h5.keys() if _TRAJ_RE.match(k)]
    return sorted(keys, key=lambda k: int(_TRAJ_RE.match(k).group(1)))


def _json_path_for(h5_path):
    """Sibling .json path for an .h5 path (foo.h5 -> foo.json)."""
    base, _ = osp.splitext(h5_path)
    return base + ".json"


def _load_json(h5_path):
    jp = _json_path_for(h5_path)
    if not osp.exists(jp):
        return None
    with open(jp) as f:
        return json.load(f)


def merge(scripted_h5, dart_h5, out_h5):
    """Merge scripted + dart H5/JSON into out_h5 (+ sibling json).

    Returns (n_total, out_json_path).
    """
    os.makedirs(osp.dirname(osp.abspath(out_h5)) or ".", exist_ok=True)

    scripted_json = _load_json(scripted_h5)
    dart_json = _load_json(dart_h5)

    next_id = 0
    merged_episodes = []

    with h5py.File(out_h5, "w") as fout:
        for src_path, src_json in ((scripted_h5, scripted_json), (dart_h5, dart_json)):
            with h5py.File(src_path, "r") as fin:
                src_keys = _sorted_traj_keys(fin)
                # map old traj index -> new contiguous id, for json alignment
                old_to_new = {}
                for k in src_keys:
                    old_idx = int(_TRAJ_RE.match(k).group(1))
                    new_key = f"traj_{next_id}"
                    # h5py.File.copy preserves the whole group subtree.
                    fin.copy(k, fout, name=new_key)
                    old_to_new[old_idx] = next_id
                    next_id += 1

            # build merged json episodes for this source, renumbering ids
            if src_json is not None:
                for ep in src_json.get("episodes", []):
                    old_idx = ep.get("episode_id")
                    if old_idx not in old_to_new:
                        # episode listed in json but no matching H5 group; skip
                        continue
                    new_ep = dict(ep)  # preserve success flag, reset_kwargs, seed
                    new_ep["episode_id"] = old_to_new[old_idx]
                    merged_episodes.append(new_ep)

    # assemble output json: env_info from scripted, episodes concatenated
    out_json = {}
    if scripted_json is not None:
        out_json["env_info"] = scripted_json.get("env_info", {})
        # carry over a few top-level keys if present (source_type, etc.)
        for k in ("source_type", "source_desc"):
            if k in scripted_json:
                out_json[k] = scripted_json[k]
    out_json["episodes"] = merged_episodes

    out_json_path = _json_path_for(out_h5)
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    return next_id, out_json_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted-h5", required=True, dest="scripted_h5")
    ap.add_argument("--dart-h5", required=True, dest="dart_h5")
    ap.add_argument("--out-h5", required=True, dest="out_h5")
    args = ap.parse_args()

    n_total, out_json_path = merge(args.scripted_h5, args.dart_h5, args.out_h5)
    print(f"[merge] wrote {n_total} trajectories -> {args.out_h5}")
    print(f"[merge] json -> {out_json_path}")
