"""HDF5 -> tiled MP4 + first-frame JPG, per episode.

Extracts RGB frames already stored in the RoboFactory HDF5 demos (no re-simulation),
tiles multi-agent cameras into a single frame, encodes with libx264/yuv420p/faststart
for HTML5 playback. Designed to be called from a multiprocessing.Pool worker.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/hf_download")
SITE_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/site")
FPS = 30
THUMB_MAX = (160, 120)  # (w, h) bbox; PIL thumbnail preserves aspect

GRID_LAYOUTS = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (2, 3),  # one blank cell
    6: (2, 3),
    7: (3, 3),  # two blank cells
    8: (3, 3),  # one blank cell
    9: (3, 3),
}

CAM_SHORT_LABELS = {
    "head_camera": "head",
    "head_camera_global": "global",
    "head_camera_agent0": "agent 0",
    "head_camera_agent1": "agent 1",
    "head_camera_agent2": "agent 2",
    "head_camera_agent3": "agent 3",
    "head_camera_agent4": "agent 4",
    "hand_camera_0": "wrist 0",
    "hand_camera_1": "wrist 1",
    "hand_camera_2": "wrist 2",
    "hand_camera_3": "wrist 3",
}

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _short_label(name: str, dup_groups: list[list[str]]) -> str:
    base = CAM_SHORT_LABELS.get(name, name)
    for group in dup_groups:
        if group[0] == name and len(group) > 1:
            members = [CAM_SHORT_LABELS.get(m, m) for m in group[1:]]
            return f"{base} (=" + ", ".join(members) + ")"
    return base


def _annotate_labels(
    tiled: np.ndarray,
    cam_names: list[str],
    dup_groups: list[list[str]],
) -> np.ndarray:
    """Burn a short text label into the top-left of each tile cell, copied
    across all timesteps. Useful for telling cameras apart at a glance.

    Rather than alpha-blend per frame (slow), we draw opaque label pixels once
    and overwrite the same pixel region on every frame via numpy masking.
    """
    n = len(cam_names)
    if n == 0:
        return tiled
    rows, cols = GRID_LAYOUTS[n]
    T, H_total, W_total, _ = tiled.shape
    cell_h = H_total // rows
    cell_w = W_total // cols

    overlay = Image.new("RGBA", (W_total, H_total), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(14)
    for i, name in enumerate(cam_names):
        r, c = divmod(i, cols)
        x0 = c * cell_w + 5
        y0 = r * cell_h + 4
        text = _short_label(name, dup_groups)
        bbox = draw.textbbox((x0, y0), text, font=font)
        # opaque black background box + white text
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=(0, 0, 0, 255))
        draw.text((x0, y0), text, fill=(255, 255, 255, 255), font=font)

    ov = np.asarray(overlay)  # (H, W, 4)
    mask = ov[..., 3] > 0     # (H, W) bool
    rgb = ov[..., :3]
    out = tiled.copy()
    out[:, mask] = rgb[mask]
    return out


def discover_tasks() -> list[str]:
    return sorted(
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and (d / f"{d.name}.h5").exists()
    )


def _resize_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (T, H, W, 3) to (T, target_h, target_w, 3) using PIL LANCZOS."""
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    out = np.empty((frames.shape[0], target_h, target_w, 3), dtype=np.uint8)
    for t, frame in enumerate(frames):
        out[t] = np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS))
    return out


def tile_cameras(cam_frames: dict[str, np.ndarray]) -> np.ndarray:
    """cam_frames: {name: (T, H, W, 3) uint8}, sorted alphabetically on output.

    Returns (T, H*rows, W*cols, 3) uint8. Blank cells are left as zeros.
    Cameras with different resolutions are resized to match the first (alphabetical) camera.
    """
    names = sorted(cam_frames.keys())
    n = len(names)
    if n not in GRID_LAYOUTS:
        raise ValueError(f"no grid layout for {n} cameras ({names})")
    rows, cols = GRID_LAYOUTS[n]
    T, H, W, _ = cam_frames[names[0]].shape
    out = np.zeros((T, H * rows, W * cols, 3), dtype=np.uint8)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        out[:, r * H:(r + 1) * H, c * W:(c + 1) * W, :] = _resize_frames(cam_frames[name], H, W)
    return out


def encode_video(frames: np.ndarray, out_path: Path, fps: int = FPS) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}: {proc.stderr.decode()[:500]}")


def save_thumb(first_frame: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(first_frame)
    img.thumbnail(THUMB_MAX, Image.LANCZOS)
    img.save(out_path, format="JPEG", quality=70, optimize=True)


def _dedup_cameras(cam_frames: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[list[str]]]:
    """Collapse cameras with byte-identical first frames into a single entry.

    Some RoboFactory HDF5 files (notably ThreeRobotsStackCube) duplicate the same
    camera under multiple names. Rendering them as N tiles wastes space and
    misleads the viewer. We keep the alphabetically-first key of each group.

    Returns the deduped dict plus the list of groups (for reporting).
    """
    names = sorted(cam_frames.keys())
    # hash first frame of each camera
    groups: dict[bytes, list[str]] = {}
    for n in names:
        key = cam_frames[n][0].tobytes()
        groups.setdefault(key, []).append(n)
    if all(len(g) == 1 for g in groups.values()):
        return cam_frames, []
    keep: dict[str, np.ndarray] = {}
    dup_groups: list[list[str]] = []
    for key, members in groups.items():
        canon = members[0]
        keep[canon] = cam_frames[canon]
        if len(members) > 1:
            dup_groups.append(members)
    return keep, dup_groups


def _load_actions(grp: h5py.Group) -> np.ndarray | None:
    """Return (T, D) actions array. Multi-agent tasks store per-arm actions under
    a Group (panda-0, panda-1, ...); we concatenate along the feature axis so a
    single jerk metric makes sense across the whole system."""
    if "actions" not in grp:
        return None
    node = grp["actions"]
    if isinstance(node, h5py.Dataset):
        return node[:]
    # Group: concatenate per-arm arrays in sorted key order
    arrs = [node[k][:] for k in sorted(node.keys())]
    return np.concatenate(arrs, axis=-1)


def _load_sidecar(task: str) -> dict[int, dict]:
    p = DATA_ROOT / task / f"{task}.json"
    if not p.exists():
        return {}
    meta = json.loads(p.read_text())
    return {ep["episode_id"]: ep for ep in meta.get("episodes", [])}


def render_one(task: str, ep: int) -> dict:
    """Open HDF5, render one episode to MP4+JPG, return manifest row."""
    h5_path = DATA_ROOT / task / f"{task}.h5"
    media_dir = SITE_ROOT / "media" / task
    video_path = media_dir / "videos" / f"ep_{ep:03d}.mp4"
    thumb_path = media_dir / "thumbs" / f"ep_{ep:03d}.jpg"
    try:
        with h5py.File(h5_path, "r") as f:
            ep_key = f"traj_{ep}"
            if ep_key not in f:
                raise KeyError(ep_key)
            grp = f[ep_key]
            sensor = grp["obs/sensor_data"]
            cam_frames = {name: sensor[name]["rgb"][:] for name in sensor.keys()}
            cam_frames, dup_cameras = _dedup_cameras(cam_frames)
            tiled = tile_cameras(cam_frames)
            tiled = _annotate_labels(tiled, sorted(cam_frames.keys()), dup_cameras)

            success = bool(grp["success"][:].any()) if "success" in grp else False
            actions = _load_actions(grp)  # (T, D) for single-agent, (T, D*n_agents) concatenated for multi
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
        if dup_cameras:
            row["duplicate_cameras"] = dup_cameras
        return row
    except Exception as e:
        return {"task": task, "ep": ep, "error": f"{type(e).__name__}: {e}"}


def render_task(task: str, episodes: list[int] | None = None) -> list[dict]:
    """Serial per-task entry point (for CLI + smoke tests)."""
    h5_path = DATA_ROOT / task / f"{task}.h5"
    with h5py.File(h5_path, "r") as f:
        all_keys = [k for k in f.keys() if k.startswith("traj_")]
    all_eps = sorted(int(k.split("_")[1]) for k in all_keys)
    eps = episodes if episodes is not None else all_eps
    out = []
    for ep in eps:
        row = render_one(task, ep)
        out.append(row)
        tag = "OK" if "error" not in row else "ERR"
        print(f"[{task}] ep {ep:3d}: {tag}")
    return out


def write_episodes_json(task: str, rows: list[dict]) -> Path:
    """Merge new rows into site/media/<task>/episodes.json, keyed by episode id.

    Preserves existing entries so partial re-runs don't clobber unrelated episodes.
    """
    p = SITE_ROOT / "media" / task / "episodes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if p.exists():
        for r in json.loads(p.read_text()):
            existing[r["ep"]] = r
    for r in rows:
        existing[r["ep"]] = r
    merged = [existing[k] for k in sorted(existing)]
    p.write_text(json.dumps(merged, indent=2))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", default="all", help="comma-separated ints, or 'all'")
    args = ap.parse_args()
    eps = None if args.episodes == "all" else [int(x) for x in args.episodes.split(",")]
    rows = render_task(args.task, eps)
    write_episodes_json(args.task, rows)
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"[{args.task}] {n_ok}/{len(rows)} episodes rendered")


if __name__ == "__main__":
    main()
