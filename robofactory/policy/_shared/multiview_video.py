"""Tile multiple camera views into a single video frame (pure numpy + cv2).

No sim / torch dependency — everything operates on HWC uint8 RGB numpy arrays
so it can be unit-tested anywhere. `cv2` is used only for `resize` (already a
project dependency via the eval drivers).

Tiling contract (`tile_views`):

- Every view is resized to a common cell size `(cell_h, cell_w)` =
  `(max height among frames, max width among frames)`. Minor aspect distortion
  is accepted on purpose — simplicity over letterboxing.
- `layout="auto"`:
    * n == 1 -> that frame (contiguous copy).
    * 2 <= n <= 3 -> one row of n columns (side by side).
    * n > 3 -> grid with `ncols = ceil(sqrt(n))`, `nrows = ceil(n / ncols)`,
      filled row-major. Trailing empty cells are black (zeros), so any padding
      lands in the bottom-right.
- `layout="horizontal"` always lays out a single row of `n` cells.
- `layout="grid"` forces the grid path even for `n <= 2`.

Output is always uint8 and C-contiguous.

dtype handling: only uint8 is officially supported. Non-uint8 frames are
`np.clip(frame, 0, 255).astype(uint8)` — NO [0,1]->[0,255] rescaling. A float
value of 100.0 maps to 100; 0.5 maps to 0; 300.0 clips to 255.
"""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np


def ordered_unique(names: List[Optional[str]]) -> List[str]:
    """Drop None entries and duplicates, preserving first-seen order.

    Example: ordered_unique(["g", "wl", None, "g", "wr"]) -> ["g", "wl", "wr"].
    """
    seen = set()
    out: List[str] = []
    for name in names:
        if name is None or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    """Validate a single frame is HWC and coerce to uint8 (clip, no rescale)."""
    if frame.ndim != 3:
        raise ValueError(
            f"each frame must be 3D (H, W, C); got ndim={frame.ndim} "
            f"with shape {frame.shape}"
        )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def tile_views(frames: List[np.ndarray], layout: str = "auto") -> np.ndarray:
    """Combine HWC uint8 RGB frames into one contiguous tiled uint8 frame.

    See the module docstring for the full layout / sizing contract.
    """
    if not frames:
        raise ValueError("tile_views requires at least one frame; got empty list")

    frames = [_as_uint8(f) for f in frames]
    n = len(frames)

    cell_h = max(f.shape[0] for f in frames)
    cell_w = max(f.shape[1] for f in frames)

    # cv2.resize takes (width, height)
    cells = [
        cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        for f in frames
    ]

    # Decide grid geometry.
    if layout == "horizontal":
        nrows, ncols = 1, n
    elif layout == "grid":
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
    elif layout == "auto":
        if n == 1:
            return np.ascontiguousarray(cells[0])
        if n <= 3:
            nrows, ncols = 1, n
        else:
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)
    else:
        raise ValueError(
            f"unknown layout {layout!r}; expected 'auto', 'horizontal', or 'grid'"
        )

    black = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    row_imgs = []
    for r in range(nrows):
        row_cells = []
        for c in range(ncols):
            idx = r * ncols + c
            row_cells.append(cells[idx] if idx < n else black)
        row_imgs.append(np.concatenate(row_cells, axis=1))
    out = np.concatenate(row_imgs, axis=0)
    return np.ascontiguousarray(out)


def subsample(frames: list, stride: int) -> list:
    """Return `frames[::stride]`, always including the last frame.

    `stride <= 1` returns frames unchanged (full sequence). For `stride >= 2`
    the final frame is appended if striding skipped it, so the terminal state
    is always visible in the rendered video.
    """
    if stride <= 1:
        return frames
    out = list(frames[::stride])
    if frames and (not out or out[-1] is not frames[-1]):
        out.append(frames[-1])
    return out
