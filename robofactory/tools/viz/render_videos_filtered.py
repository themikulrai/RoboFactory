"""Variant of render_videos.py that renders only an explicit allowlist of cameras.

Use case: a task's H5 has cameras (e.g. per-agent head cameras) we want to drop
from the visualization without modifying the source H5 or the canonical renderer.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from robofactory.tools.viz.render_videos import (
    DATA_ROOT,
    SITE_ROOT,
    _annotate_labels,
    _dedup_cameras,
    _load_actions,
    encode_video,
    save_thumb,
    tile_cameras,
    write_episodes_json,
)


def render_one_filtered(args: tuple[str, int, tuple[str, ...]]) -> dict:
    task, ep, allow = args
    allow_set = set(allow)
    h5_path = DATA_ROOT / task / f"{task}.h5"
    media_dir = SITE_ROOT / "media" / task
    video_path = media_dir / "videos" / f"ep_{ep:03d}.mp4"
    thumb_path = media_dir / "thumbs" / f"ep_{ep:03d}.jpg"
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f[f"traj_{ep}"]
            sensor = grp["obs/sensor_data"]
            cam_frames = {
                name: sensor[name]["rgb"][:]
                for name in sensor.keys()
                if name in allow_set
            }
            if not cam_frames:
                raise ValueError(f"no allowlisted cams present; have {list(sensor.keys())}")
            cam_frames, dup_groups = _dedup_cameras(cam_frames)
            tiled = tile_cameras(cam_frames)
            tiled = _annotate_labels(tiled, sorted(cam_frames.keys()), dup_groups)

            success = bool(grp["success"][:].any()) if "success" in grp else False
            actions = _load_actions(grp)
            length = int(actions.shape[0]) if actions is not None else tiled.shape[0]
            if actions is not None and actions.shape[0] > 1:
                mean_jerk = float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())
            else:
                mean_jerk = 0.0

        encode_video(tiled, video_path)
        save_thumb(tiled[0], thumb_path)
        row = {
            "task": task, "ep": ep,
            "success": success,
            "length": length,
            "mean_jerk": round(mean_jerk, 4),
            "n_cams": len(cam_frames),
            "video": f"media/{task}/videos/ep_{ep:03d}.mp4",
            "thumb": f"media/{task}/thumbs/ep_{ep:03d}.jpg",
        }
        if dup_groups:
            row["duplicate_cameras"] = dup_groups
        return row
    except Exception as e:
        return {"task": task, "ep": ep, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cameras", required=True, help="comma-separated allowlist")
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    allow = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    h5_path = DATA_ROOT / args.task / f"{args.task}.h5"
    with h5py.File(h5_path, "r") as f:
        all_eps = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("traj_"))
    eps = all_eps if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    jobs = [(args.task, ep, allow) for ep in eps]

    t0 = time.time()
    rows: list[dict] = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(render_one_filtered, jobs):
            tag = "OK" if "error" not in row else f"ERR ({row['error']})"
            print(f"[{args.task}] ep {row['ep']:3d}: {tag}", flush=True)
            rows.append(row)
    rows.sort(key=lambda r: r["ep"])
    write_episodes_json(args.task, rows)
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"[{args.task}] {n_ok}/{len(rows)} rendered in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
