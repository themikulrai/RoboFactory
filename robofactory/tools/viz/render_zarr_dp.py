"""Zarr -> tiled MP4 + first-frame JPG, per episode, for DP training inputs.

Renders the *exact* visual inputs that a Diffusion-Policy training run consumes,
straight from the converted zarr stores under
/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data/.

Cameras are tiled HORIZONTALLY in alphabetical key order to match the existing
LiftBarrier_dp_* output convention (672x224 for 3 cams, 448x224 for 2 cams,
896x224 for 4 cams). Encoding: libx264 + yuv420p + faststart, 30 fps.

Output:
  /iris/u/mikulrai/datasets/multi_robot/RoboFactory/site/media/<task>_dp_<cam>_<scheme>/
    videos/ep_NNN.mp4
    thumbs/ep_NNN.jpg

Schemes per task:
  2SC: cent, dec_a0, dec_a1                  -> 6 variants (ws + wc)
  3SC: cent, dec_a0, dec_a1, dec_a2          -> 8 variants (ws + wc)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import zarr

from .render_videos import encode_video, save_thumb

ZARR_ROOT = Path("/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data")
SITE_ROOT = Path("/iris/u/mikulrai/datasets/multi_robot/RoboFactory/site")
FPS = 30
NON_IMAGE_KEYS = {"action", "state", "tcp_action"}


def zarr_path_for(task: str, cam: str, scheme: str) -> Path:
    """Map (task, cam, scheme) -> zarr filename.

    task: TwoRobotsStackCube | ThreeRobotsStackCube
    cam:  ws | wc
    scheme: cent | dec_aN
    """
    cam_full = {"ws": "workspace", "wc": "wristcam"}[cam]
    if scheme == "cent":
        suffix = "cent_150"
    elif scheme.startswith("dec_a"):
        i = int(scheme[len("dec_a"):])
        suffix = f"decent_agent{i}_150"
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")
    return ZARR_ROOT / f"{task}-rf_{cam_full}_{suffix}.zarr"


def out_dir_for(task: str, cam: str, scheme: str) -> Path:
    return SITE_ROOT / "media" / f"{task}_dp_{cam}_{scheme}"


def discover_image_keys(root: zarr.Group) -> list[str]:
    """Return alphabetically-sorted list of camera keys under data/."""
    keys = [k for k in root["data"].keys() if k not in NON_IMAGE_KEYS]
    return sorted(keys)


def episode_slices(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, stop) per episode. ep 0 is [0:episode_ends[0]]."""
    out = []
    prev = 0
    for end in episode_ends:
        out.append((int(prev), int(end)))
        prev = end
    return out


def _render_one(args: tuple) -> dict:
    """Worker entry: one episode."""
    zarr_path, image_keys, ep_idx, start, stop, out_video, out_thumb = args
    try:
        z = zarr.open(str(zarr_path), mode="r")
        # (T_ep, H, W*n_cams, 3)
        tiles = []
        for k in image_keys:
            # zarr arr is (N, 3, 224, 224) channel-first
            chw = z["data"][k][start:stop]  # (T_ep, 3, H, W) uint8
            hwc = np.transpose(chw, (0, 2, 3, 1))  # (T_ep, H, W, 3)
            tiles.append(hwc)
        tiled = np.concatenate(tiles, axis=2)  # cat along width
        encode_video(tiled, out_video, fps=FPS)
        save_thumb(tiled[0], out_thumb)
        return {"ep": ep_idx, "ok": True, "shape": list(tiled.shape)}
    except Exception as e:
        return {"ep": ep_idx, "ok": False, "error": f"{type(e).__name__}: {e}"}


def render_variant(task: str, cam: str, scheme: str, workers: int = 8) -> dict:
    zarr_path = zarr_path_for(task, cam, scheme)
    if not zarr_path.exists():
        return {"variant": f"{task}_dp_{cam}_{scheme}", "error": f"missing zarr: {zarr_path}"}

    z = zarr.open(str(zarr_path), mode="r")
    image_keys = discover_image_keys(z)
    episode_ends = z["meta"]["episode_ends"][:]
    slices = episode_slices(episode_ends)

    out_dir = out_dir_for(task, cam, scheme)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)

    print(f"[{task} {cam} {scheme}] zarr={zarr_path.name} cams={image_keys} N_eps={len(slices)}")

    jobs = []
    for ep_idx, (start, stop) in enumerate(slices):
        out_video = out_dir / "videos" / f"ep_{ep_idx:03d}.mp4"
        out_thumb = out_dir / "thumbs" / f"ep_{ep_idx:03d}.jpg"
        jobs.append((zarr_path, image_keys, ep_idx, start, stop, out_video, out_thumb))

    t0 = time.time()
    results = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(_render_one, jobs):
            tag = "OK" if r["ok"] else f"ERR {r.get('error', '')}"
            print(f"[{task} {cam} {scheme}] ep {r['ep']:3d}: {tag}")
            results.append(r)
    elapsed = time.time() - t0

    n_ok = sum(1 for r in results if r["ok"])
    n_err = len(results) - n_ok
    print(f"[{task} {cam} {scheme}] done {n_ok}/{len(results)} in {elapsed:.1f}s")
    return {
        "variant": f"{task}_dp_{cam}_{scheme}",
        "n_ok": n_ok,
        "n_err": n_err,
        "elapsed_s": elapsed,
        "errors": [r for r in results if not r["ok"]],
    }


# All 14 variants
ALL_VARIANTS: list[tuple[str, str, str]] = []
for task, n_arms in [("TwoRobotsStackCube", 2), ("ThreeRobotsStackCube", 3)]:
    for cam in ("ws", "wc"):
        ALL_VARIANTS.append((task, cam, "cent"))
        for i in range(n_arms):
            ALL_VARIANTS.append((task, cam, f"dec_a{i}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["TwoRobotsStackCube", "ThreeRobotsStackCube"])
    ap.add_argument("--cam", choices=["ws", "wc"])
    ap.add_argument("--scheme", help="cent or dec_a<i>")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="render all 14 variants in sequence")
    args = ap.parse_args()

    t0 = time.time()
    summaries = []
    if args.all:
        for task, cam, scheme in ALL_VARIANTS:
            summaries.append(render_variant(task, cam, scheme, workers=args.workers))
    else:
        if not (args.task and args.cam and args.scheme):
            ap.error("must pass --task, --cam, --scheme  (or --all)")
        summaries.append(render_variant(args.task, args.cam, args.scheme, workers=args.workers))
    total = time.time() - t0
    print("\n===== SUMMARY =====")
    for s in summaries:
        if "error" in s:
            print(f"  {s['variant']}: SKIPPED ({s['error']})")
            continue
        print(f"  {s['variant']}: {s['n_ok']} ok, {s['n_err']} err, {s['elapsed_s']:.1f}s")
    print(f"TOTAL wall: {total:.1f}s")


if __name__ == "__main__":
    main()
