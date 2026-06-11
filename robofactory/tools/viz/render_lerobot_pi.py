"""LeRobot parquet -> 2x2-mosaic MP4 + first-frame JPG, per episode.

Renders the *exact* image inputs Pi0.5 sees at training time on the
RoboFactory LeRobot conversion (TwoRobotsStackCube / ThreeRobotsStackCube
in workspace and wristcam variants).

Tile layout matches the existing LiftBarrier_pi_ws mp4s:

    +------------+--------------+
    | base_0     | left_wrist_0 |
    +------------+--------------+
    | right_wrist_0 | extra_0   |
    +------------+--------------+

Each tile is 320x240 (resized from the 240x320x3 PNGs in parquet);
final mosaic is 640x480, libx264 + yuv420p + faststart, 30 fps.

Output:
  /iris/u/mikulrai/data/RoboFactory/site/media/<task>_pi_<cam>/{videos,thumbs}/ep_NNN.{mp4,jpg}

Run with the RoboFactory env interpreter:
  /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
      robofactory/tools/viz/render_lerobot_pi.py --task 2SC --cam ws --workers 4
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# Reuse encode_video + save_thumb from the existing renderer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_videos import encode_video, save_thumb  # noqa: E402

LEROBOT_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/lerobot")
SITE_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/site")
FPS = 30  # match LB's HTML5 convention (info.json says 20 but LB uses 30)

# Tile layout: order matters - 2x2 mosaic indexed row-major (TL, TR, BL, BR)
SLOT_ORDER = (
    "base_0_rgb_raw",       # TL
    "left_wrist_0_rgb_raw", # TR
    "right_wrist_0_rgb_raw",# BL
    "extra_0_rgb_raw",      # BR
)
TILE_H, TILE_W = 240, 320          # source PNG size from info.json
MOSAIC_H, MOSAIC_W = 2 * TILE_H, 2 * TILE_W  # 480x640

TASK_DATASETS = {
    ("2SC", "ws"): "robofactory_two_robots_stack_cube_workspace_seedfix_v1",
    ("2SC", "wc"): "robofactory_two_robots_stack_cube_wristcam_seedfix_v1",
    ("3SC", "ws"): "robofactory_three_robots_stack_cube_workspace_seedfix_v1",
    ("3SC", "wc"): "robofactory_three_robots_stack_cube_wristcam_seedfix_v1",
}
TASK_OUTPUT_NAME = {
    ("2SC", "ws"): "TwoRobotsStackCube_pi_ws",
    ("2SC", "wc"): "TwoRobotsStackCube_pi_wc",
    ("3SC", "ws"): "ThreeRobotsStackCube_pi_ws",
    ("3SC", "wc"): "ThreeRobotsStackCube_pi_wc",
}


def _decode_one(cell) -> np.ndarray:
    """Decode a single parquet image cell.

    LeRobot v2.1 stores images as {'bytes': PNG_bytes, 'path': None}.
    Returns (H, W, 3) uint8. If the cell is missing (None) returns black tile.
    """
    if cell is None:
        return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is None:
            return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    elif isinstance(cell, (bytes, bytearray)):
        b = bytes(cell)
        # Could also be raw 240*320*3=230400 bytes - handle that too
        if len(b) == TILE_H * TILE_W * 3:
            return np.frombuffer(b, dtype=np.uint8).reshape(TILE_H, TILE_W, 3)
    else:
        raise TypeError(f"unexpected image cell type: {type(cell).__name__}")

    img = Image.open(io.BytesIO(b))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    if arr.shape != (TILE_H, TILE_W, 3):
        # Resize defensively (e.g. wrist cams stored at 224x224 in HDF5 — but
        # the converter already resamples to 240x320 before writing parquet,
        # so this branch should rarely fire).
        img = img.resize((TILE_W, TILE_H), Image.LANCZOS)
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
    return arr


def _decode_column(table, col: str) -> np.ndarray:
    """Decode one image column from a parquet table -> (T, 240, 320, 3) uint8."""
    raw = table[col].to_pylist()  # list of dicts or None
    T = len(raw)
    out = np.empty((T, TILE_H, TILE_W, 3), dtype=np.uint8)
    for t, cell in enumerate(raw):
        out[t] = _decode_one(cell)
    return out


def _build_mosaic(table) -> np.ndarray:
    """Read 4 slot columns from a parquet table, return (T, 480, 640, 3) uint8.

    Missing columns (none of these datasets actually omit slots; they fill
    `extra_0_rgb_raw` with a tiny black PNG when unused) are rendered as
    black tiles.
    """
    cols_present = set(table.column_names)
    T = table.num_rows
    mosaic = np.zeros((T, MOSAIC_H, MOSAIC_W, 3), dtype=np.uint8)
    for i, slot in enumerate(SLOT_ORDER):
        r, c = divmod(i, 2)  # 2x2 grid
        if slot not in cols_present:
            continue  # leave that tile black
        frames = _decode_column(table, slot)
        mosaic[
            :,
            r * TILE_H:(r + 1) * TILE_H,
            c * TILE_W:(c + 1) * TILE_W,
            :,
        ] = frames
    return mosaic


def render_one(args) -> dict:
    """Worker: render a single episode parquet to MP4 + thumb JPG."""
    dataset_dir, out_dir, ep = args
    parquet_path = Path(dataset_dir) / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    video_path = Path(out_dir) / "videos" / f"ep_{ep:03d}.mp4"
    thumb_path = Path(out_dir) / "thumbs" / f"ep_{ep:03d}.jpg"
    try:
        table = pq.read_table(
            parquet_path,
            columns=list(SLOT_ORDER),
        )
        mosaic = _build_mosaic(table)
        encode_video(mosaic, video_path, fps=FPS)
        save_thumb(mosaic[0], thumb_path)
        return {"ep": ep, "ok": True, "frames": int(mosaic.shape[0])}
    except Exception as e:
        return {"ep": ep, "ok": False, "error": f"{type(e).__name__}: {e}"}


def render_dataset(
    dataset_dir: Path,
    out_dir: Path,
    workers: int = 4,
    episodes: list[int] | None = None,
) -> list[dict]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    n_eps = int(info["total_episodes"])
    eps = episodes if episodes is not None else list(range(n_eps))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "thumbs").mkdir(exist_ok=True)

    tasks = [(str(dataset_dir), str(out_dir), ep) for ep in eps]
    print(
        f"[render_lerobot_pi] {dataset_dir.name} -> {out_dir.name}: "
        f"{len(tasks)} episodes, {workers} workers"
    )

    rows: list[dict] = []
    t0 = time.time()
    if workers == 1:
        for t in tasks:
            r = render_one(t)
            rows.append(r)
            tag = "OK" if r.get("ok") else f"ERR ({r.get('error')})"
            print(f"  ep {r['ep']:3d}: {tag}")
    else:
        ctx = mp.get_context("forkserver")
        with ctx.Pool(workers) as pool:
            for r in pool.imap_unordered(render_one, tasks):
                rows.append(r)
                if not r.get("ok"):
                    print(f"  ep {r['ep']:3d}: ERR ({r.get('error')})")
        rows.sort(key=lambda d: d["ep"])
    dt = time.time() - t0
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"[render_lerobot_pi] done: {n_ok}/{len(rows)} ok in {dt:.1f}s")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", choices=("2SC", "3SC"), help="Task tag.")
    g.add_argument("--dataset-path", type=Path,
                   help="Direct path to a LeRobot dataset dir (skips --task/--cam).")
    ap.add_argument("--cam", choices=("ws", "wc"),
                    help="Required with --task.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override output dir (default derived from task/cam).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", default="all",
                    help="comma-separated ints, or 'all'.")
    args = ap.parse_args()

    if args.dataset_path is not None:
        dataset_dir = args.dataset_path
        if args.out_dir is None:
            raise SystemExit("--out-dir is required when --dataset-path is given")
        out_dir = args.out_dir
    else:
        if args.cam is None:
            raise SystemExit("--cam is required with --task")
        ds_name = TASK_DATASETS[(args.task, args.cam)]
        dataset_dir = LEROBOT_ROOT / ds_name
        out_name = TASK_OUTPUT_NAME[(args.task, args.cam)]
        out_dir = args.out_dir or (SITE_ROOT / "media" / out_name)

    eps = None if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    rows = render_dataset(dataset_dir, out_dir, workers=args.workers, episodes=eps)

    # Write a tiny manifest for downstream wiring
    manifest = {
        "dataset": str(dataset_dir),
        "out_dir": str(out_dir),
        "fps": FPS,
        "mosaic_size": [MOSAIC_W, MOSAIC_H],
        "slot_order": list(SLOT_ORDER),
        "episodes": rows,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
    n_err = sum(1 for r in rows if not r.get("ok"))
    if n_err:
        raise SystemExit(f"{n_err} episodes failed; see {out_dir}/render_manifest.json")


if __name__ == "__main__":
    main()
