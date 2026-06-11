"""Render single-camera mp4s + thumbs + episodes.json from a RoboFactory H5.

Lets you render just ONE camera (e.g. head_camera only, for a wristcam-equipped
H5 you want to view as if it were workspace-only). Mirrors render_videos.py
output paths so the viz site picks it up identically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from robofactory.tools.viz.render_videos import (
    encode_video,
    save_thumb,
    SITE_ROOT,
    _annotate_labels,
    _load_actions,
    _load_sidecar,
)


def render_one(h5_path: Path, ep: int, camera: str, out_task: str) -> dict:
    media_dir = SITE_ROOT / "media" / out_task
    video_path = media_dir / "videos" / f"ep_{ep:03d}.mp4"
    thumb_path = media_dir / "thumbs" / f"ep_{ep:03d}.jpg"
    try:
        with h5py.File(h5_path, "r") as f:
            ep_key = f"traj_{ep}"
            if ep_key not in f:
                raise KeyError(ep_key)
            grp = f[ep_key]
            sensor = grp["obs/sensor_data"]
            if camera not in sensor:
                raise KeyError(f"camera {camera!r} not in {list(sensor.keys())}")
            frames = sensor[camera]["rgb"][:]
            tiled = _annotate_labels(frames, [camera], [])

            success = bool(grp["success"][:].any()) if "success" in grp else False
            actions = _load_actions(grp)
            length = int(actions.shape[0]) if actions is not None else tiled.shape[0]
            if actions is not None and actions.shape[0] > 1:
                mean_jerk = float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())
            else:
                mean_jerk = 0.0

        encode_video(tiled, video_path)
        save_thumb(tiled[0], thumb_path)
        return {
            "task": out_task, "ep": ep,
            "success": success, "length": length,
            "mean_jerk": round(mean_jerk, 4), "n_cams": 1,
            "video": f"media/{out_task}/videos/ep_{ep:03d}.mp4",
            "thumb": f"media/{out_task}/thumbs/ep_{ep:03d}.jpg",
        }
    except Exception as e:
        return {"task": out_task, "ep": ep, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--camera", required=True, help="e.g. head_camera")
    ap.add_argument("--out-task", required=True, help="site/media/<out-task>/ subdir")
    args = ap.parse_args()

    h5_path = Path(args.h5)
    with h5py.File(h5_path, "r") as f:
        all_eps = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("traj_"))

    rows = []
    for ep in all_eps:
        row = render_one(h5_path, ep, args.camera, args.out_task)
        tag = "OK" if "error" not in row else "ERR"
        print(f"[{args.out_task}] ep {ep:3d}: {tag}")
        rows.append(row)

    out_json = SITE_ROOT / "media" / args.out_task / "episodes.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_json.exists():
        for r in json.loads(out_json.read_text()):
            existing[r["ep"]] = r
    for r in rows:
        existing[r["ep"]] = r
    merged = [existing[k] for k in sorted(existing)]
    out_json.write_text(json.dumps(merged, indent=2))
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"[{args.out_task}] {n_ok}/{len(rows)} episodes rendered")


if __name__ == "__main__":
    main()
