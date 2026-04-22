"""Create a side-by-side composite video from a wristcam H5.

Layout (per trajectory):
  +-----------------------+----------+----------+
  |                       | wrist 0  | wrist 1  |
  |   global camera       +----------+----------|
  |   (head_camera_global)|  wrist 2 | wrist 3  |
  +-----------------------+----------+----------+

Usage:
    python scripts/feasibility_wrist_replay/make_grid_video.py \
        --h5 /path/to/rollout.h5 --out-dir /tmp/grid_vids
"""
import argparse
import os
import os.path as osp

import cv2
import h5py
import imageio
import numpy as np

WRIST_CELL = 224  # each wrist camera cell (px); 2x = 448 (divisible by 16)
SEP = 0           # no separator so total width stays divisible by 16


def _label(frame, text, font_scale=0.55, thickness=1):
    frame = frame.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # dark background pill behind text
    cv2.rectangle(frame, (4, 4), (8 + tw, 10 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 6 + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _resize(frame_bgr, h, w):
    return cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def make_frame(global_rgb, wrist_rgbs, wrist_labels):
    """Compose one video frame. All inputs are HWC uint8 RGB numpy arrays."""
    grid_h = WRIST_CELL * 2
    grid_w = WRIST_CELL * 2

    # Scale global to match grid height, preserving aspect ratio; round to even for H.264
    gh, gw = global_rgb.shape[:2]
    global_h = grid_h
    global_w = int(global_h * gw / gh)
    global_w = global_w + (global_w % 2)  # ensure even
    global_bgr = cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR)
    global_resized = _resize(global_bgr, global_h, global_w)
    global_labeled = _label(global_resized, "global")

    # 2x2 wrist grid
    cells = []
    for i in range(4):
        if i < len(wrist_rgbs):
            c = cv2.cvtColor(wrist_rgbs[i], cv2.COLOR_RGB2BGR)
            c = _resize(c, WRIST_CELL, WRIST_CELL)
            c = _label(c, wrist_labels[i], font_scale=0.45)
        else:
            c = np.zeros((WRIST_CELL, WRIST_CELL, 3), dtype=np.uint8)
        cells.append(c)

    top_row = np.concatenate([cells[0], cells[1]], axis=1)
    bot_row = np.concatenate([cells[2], cells[3]], axis=1)
    wrist_grid = np.concatenate([top_row, bot_row], axis=0)

    # Separator
    sep_col = np.zeros((grid_h, SEP, 3), dtype=np.uint8)

    # Composite
    composite_bgr = np.concatenate([global_labeled, sep_col, wrist_grid], axis=1)
    return cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)


def main(h5_path, out_dir, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        for tk in sorted(f.keys()):
            sd = f[tk]["obs"]["sensor_data"]
            keys = sorted(sd.keys())

            # Pick global camera: prefer head_camera_global, else first head_camera_*
            global_keys = [k for k in keys if k == "head_camera_global"]
            if not global_keys:
                global_keys = [k for k in keys if "head_camera" in k]
            if not global_keys:
                print(f"  {tk}: no head_camera found, skipping")
                continue

            wrist_keys = sorted(k for k in keys if k.startswith("hand_camera_"))
            if not wrist_keys:
                print(f"  {tk}: no hand_camera found, skipping")
                continue

            global_rgb = np.asarray(sd[global_keys[0]]["rgb"])  # T,H,W,3
            T = global_rgb.shape[0]

            wrist_rgbs = [np.asarray(sd[k]["rgb"]) for k in wrist_keys[:4]]
            wrist_labels = [f"arm {k.split('_')[-1]}" for k in wrist_keys[:4]]

            h5_stem = osp.splitext(osp.basename(h5_path))[0]
            out_path = osp.join(out_dir, f"{h5_stem}_{tk}_grid.mp4")
            with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
                for i in range(T):
                    frame = make_frame(
                        global_rgb[i],
                        [w[i] for w in wrist_rgbs],
                        wrist_labels,
                    )
                    writer.append_data(frame)
            dims = make_frame(global_rgb[0], [w[0] for w in wrist_rgbs], wrist_labels)
            print(f"  wrote {out_path}  ({T} frames, {dims.shape[1]}x{dims.shape[0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    out_dir = args.out_dir or osp.join(osp.dirname(args.h5), "grid_videos")
    main(args.h5, out_dir, fps=args.fps)
