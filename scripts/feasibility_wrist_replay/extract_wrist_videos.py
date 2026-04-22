"""Extract per-wrist-camera MP4s from the feasibility H5.

Writes one MP4 per (trajectory, wrist-camera) so you can inspect each arm's
egocentric view separately from the tiled `sensors`-mode render.
"""
import argparse
import glob
import os
import os.path as osp

import h5py
import numpy as np
import imageio


def main(h5_path, out_dir, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        for tk in sorted(f.keys()):
            t = f[tk]
            sd = t["obs"]["sensor_data"]
            wrist_cams = sorted([k for k in sd.keys() if k.startswith("hand_camera")])
            for cam in wrist_cams:
                rgb = np.asarray(sd[cam]["rgb"])  # (T, H, W, 3)
                out = osp.join(out_dir, f"{tk}_{cam}.mp4")
                with imageio.get_writer(out, fps=fps) as w:
                    for frame in rgb:
                        w.append_data(frame)
                print(f"  wrote {out}  ({rgb.shape[0]} frames, {rgb.shape[1]}x{rgb.shape[2]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    out_dir = args.out_dir or osp.join(osp.dirname(args.h5), "per_wrist_cam")
    main(args.h5, out_dir, fps=args.fps)
