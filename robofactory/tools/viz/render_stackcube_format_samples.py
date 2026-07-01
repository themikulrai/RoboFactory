"""Render sample-episode MP4s from the four StackCube converted-data sources.

Outputs four MP4s (one per quadrant of the field-notes grid cell):
  zarr_ws.mp4   - zarr workspace_cent (head_camera only, 224x224)
  zarr_wc.mp4   - zarr wristcam_cent  (head_camera = hand, + head_camera_global = head, tiled)
  libero_ws.mp4 - lerobot workspace   (base_0_rgb_raw, 240x320)
  libero_wc.mp4 - lerobot wristcam    (base_0_rgb_raw + left_wrist_0_rgb_raw, tiled)

Reads episode 0 from each source. Run after the conversion pipeline completes.
"""
from __future__ import annotations

import argparse
import io
import os
import os.path as osp
import subprocess
from pathlib import Path

import numpy as np
import zarr
import pyarrow.parquet as pq
from PIL import Image

OUT_ROOT = Path("/iris/u/mikulrai/datasets/multi_robot/RoboFactory/site/media/StackCube_formats")
FPS = 20  # true sim control rate (matches LeRobot info.json fps); all four panels share this


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((4, 4), text, font=font)
    pad = 3
    draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def _to_tile(frames: list[np.ndarray]) -> np.ndarray:
    """Horizontal tile of frames; resize to common height."""
    if len(frames) == 1:
        return frames[0]
    target_h = max(f.shape[0] for f in frames)
    out = []
    for f in frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = int(f.shape[1] * scale)
            pil = Image.fromarray(f).resize((new_w, target_h), Image.BILINEAR)
            f = np.asarray(pil)
        out.append(f)
    return np.concatenate(out, axis=1)


def _ensure_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    nh = h - (h % 2)
    nw = w - (w % 2)
    if nh == h and nw == w:
        return img
    return img[:nh, :nw]


def _write_mp4(frames_thwc: np.ndarray, out_path: Path, fps: int = FPS) -> None:
    """Write (T,H,W,3) uint8 -> H.264 yuv420p mp4 via ffmpeg pipe."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames_thwc.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}", "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-preset", "fast", "-crf", "23",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames_thwc.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def _zarr_episode_bounds(root, ep_idx: int = 0):
    ends = np.asarray(root["meta/episode_ends"][:])
    start = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
    end = int(ends[ep_idx])
    return start, end


def render_zarr_ws(zarr_path: Path, out: Path) -> None:
    root = zarr.open(str(zarr_path), mode="r")
    s, e = _zarr_episode_bounds(root, 0)
    cam = np.asarray(root["data/head_camera"][s:e])  # (T,3,224,224)
    cam = np.moveaxis(cam, 1, -1)
    frames = [_label(cam[t], "zarr · workspace · head_camera (224)") for t in range(cam.shape[0])]
    arr = _ensure_even(np.stack(frames))
    _write_mp4(arr, out)


def render_zarr_wc(zarr_path: Path, out: Path) -> None:
    root = zarr.open(str(zarr_path), mode="r")
    s, e = _zarr_episode_bounds(root, 0)
    hand = np.asarray(root["data/head_camera"][s:e])  # actually hand_camera since cam-family=wristcam
    hand = np.moveaxis(hand, 1, -1)
    if "data/head_camera_global" in root:
        head = np.asarray(root["data/head_camera_global"][s:e])
        head = np.moveaxis(head, 1, -1)
    else:
        head = None
    frames = []
    for t in range(hand.shape[0]):
        parts = []
        if head is not None:
            parts.append(_label(head[t], "head (global)"))
        parts.append(_label(hand[t], "hand_0"))
        frames.append(_to_tile(parts))
    arr = _ensure_even(np.stack(frames))
    _write_mp4(arr, out)


def _decode_image_cell(cell) -> np.ndarray:
    """LeRobot v2.1 image columns store struct{bytes: bytes, path: str}; pyarrow returns dicts."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
    else:
        data = cell
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def render_libero_ws(repo_root: Path, out: Path) -> None:
    parquet = repo_root / "data/chunk-000/episode_000000.parquet"
    tbl = pq.read_table(parquet, columns=["base_0_rgb_raw"])
    col = tbl["base_0_rgb_raw"].to_pylist()
    frames = [_label(_decode_image_cell(c), "libero · workspace · base_0_rgb_raw") for c in col]
    arr = _ensure_even(np.stack(frames))
    _write_mp4(arr, out, fps=20)


def render_libero_wc(repo_root: Path, out: Path) -> None:
    parquet = repo_root / "data/chunk-000/episode_000000.parquet"
    tbl = pq.read_table(parquet, columns=["base_0_rgb_raw", "left_wrist_0_rgb_raw"])
    base = tbl["base_0_rgb_raw"].to_pylist()
    wrist = tbl["left_wrist_0_rgb_raw"].to_pylist()
    frames = []
    for b, w in zip(base, wrist):
        bi = _label(_decode_image_cell(b), "base (head)")
        wi = _label(_decode_image_cell(w), "left_wrist (hand)")
        frames.append(_to_tile([bi, wi]))
    arr = _ensure_even(np.stack(frames))
    _write_mp4(arr, out, fps=20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr-ws", default="/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data/StackCube-rf_workspace_cent_150.zarr")
    ap.add_argument("--zarr-wc", default="/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data/StackCube-rf_wristcam_cent_150.zarr")
    ap.add_argument("--libero-ws", default="/iris/u/mikulrai/datasets/multi_robot/RoboFactory/lerobot/robofactory_stack_cube_workspace_seedfix_v1")
    ap.add_argument("--libero-wc", default="/iris/u/mikulrai/datasets/multi_robot/RoboFactory/lerobot/robofactory_stack_cube_wristcam_seedfix_v1")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}")

    render_zarr_ws(Path(args.zarr_ws), out_dir / "zarr_ws.mp4")
    print("  ok zarr_ws.mp4")
    render_zarr_wc(Path(args.zarr_wc), out_dir / "zarr_wc.mp4")
    print("  ok zarr_wc.mp4")
    render_libero_ws(Path(args.libero_ws), out_dir / "libero_ws.mp4")
    print("  ok libero_ws.mp4")
    render_libero_wc(Path(args.libero_wc), out_dir / "libero_wc.mp4")
    print("  ok libero_wc.mp4")


if __name__ == "__main__":
    main()
