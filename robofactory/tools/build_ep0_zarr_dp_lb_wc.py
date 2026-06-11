"""Build a 1-episode zarr (ep0 of LB-wc-decent) from the existing per-agent zarrs.

DP overfit-on-ep0 counterpart to the Pi0.5 overfit. Keeps everything else constant:
same source data (frames 0..98 of agent0/agent1 zarrs), same DP recipe.

Output paths:
  data/zarr_data/LiftBarrier-rf_wristcam_decent_agent{0,1}_ep0.zarr
"""

from __future__ import annotations

import pathlib

import numpy as np
import zarr

SRC_DIR = pathlib.Path("/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data")
DST_DIR = SRC_DIR
EP0_LEN = 98


def slice_one(arm: int):
    src = zarr.open(str(SRC_DIR / f"LiftBarrier-rf_wristcam_decent_agent{arm}_150.zarr"), mode="r")
    ee = np.asarray(src["meta/episode_ends"])
    n = int(ee[0])
    print(f"[arm{arm}] source frames: {sum(src['data/state'].shape[:1])} | episode_ends[0]={n}")
    assert n == EP0_LEN, f"unexpected ep0 length {n} for arm {arm}"
    dst_path = DST_DIR / f"LiftBarrier-rf_wristcam_decent_agent{arm}_ep0.zarr"
    if dst_path.exists():
        print(f"[arm{arm}] overwriting existing {dst_path}")
        import shutil
        shutil.rmtree(dst_path)
    dst = zarr.open(str(dst_path), mode="w")
    dst_data = dst.create_group("data")
    dst_meta = dst.create_group("meta")
    for k in src["data"].keys():
        arr = np.asarray(src["data"][k][:n])
        dst_data.create_dataset(k, data=arr, chunks=arr.shape, dtype=arr.dtype)
        print(f"  {k}: {arr.shape} {arr.dtype}")
    dst_meta.create_dataset("episode_ends", data=np.array([n], dtype=np.int64))
    print(f"[arm{arm}] wrote {dst_path}")


def main():
    for arm in (0, 1):
        slice_one(arm)


if __name__ == "__main__":
    main()
