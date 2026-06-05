"""Tests for the pure-numpy multi-view video tiling helper.

These run anywhere with just numpy + cv2 (no sim / torch). Each test builds
small synthetic uint8 RGB frames so assertions on exact pixel regions are cheap.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from robofactory.policy._shared.multiview_video import (
    ordered_unique,
    subsample,
    tile_views,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _solid(h, w, color):
    """HWC uint8 frame filled with a single RGB color tuple."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, :] = color
    return f


# ---------------------------------------------------------------------------
# ordered_unique
# ---------------------------------------------------------------------------
def test_ordered_unique_dedupe_and_order():
    assert ordered_unique(["g", "wl", None, "g", "wr"]) == ["g", "wl", "wr"]


def test_ordered_unique_all_none():
    assert ordered_unique([None, None, None]) == []


def test_ordered_unique_empty():
    assert ordered_unique([]) == []


def test_ordered_unique_no_dupes_preserves_order():
    assert ordered_unique(["a", "b", "c"]) == ["a", "b", "c"]


def test_ordered_unique_dupes_only():
    assert ordered_unique(["x", "x", "x"]) == ["x"]


# ---------------------------------------------------------------------------
# tile_views — single frame
# ---------------------------------------------------------------------------
def test_tile_single_returns_same_pixels_and_contiguous():
    f = _solid(50, 60, (10, 20, 30))
    out = tile_views([f])
    assert out.shape == (50, 60, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, f)
    # must be a copy, not an alias
    assert out is not f
    out[0, 0, 0] = 255
    assert f[0, 0, 0] == 10


# ---------------------------------------------------------------------------
# tile_views — n == 2 horizontal
# ---------------------------------------------------------------------------
def test_tile_two_horizontal_layout():
    f0 = _solid(40, 50, (255, 0, 0))
    f1 = _solid(40, 50, (0, 255, 0))
    out = tile_views([f0, f1])
    assert out.shape == (40, 100, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]
    # left half == f0, right half == f1
    np.testing.assert_array_equal(out[:, :50], f0)
    np.testing.assert_array_equal(out[:, 50:], f1)


def test_tile_two_different_sizes_uses_max_cell():
    f0 = _solid(40, 50, (1, 2, 3))
    f1 = _solid(60, 30, (4, 5, 6))
    out = tile_views([f0, f1])
    # cell = (max h, max w) = (60, 50)
    assert out.shape == (60, 100, 3)
    # left half is f0 resized to (60,50), still solid color 1,2,3
    left = out[:, :50]
    right = out[:, 50:]
    assert left.shape == (60, 50, 3)
    assert right.shape == (60, 50, 3)
    # solid frames stay solid through resize
    assert np.all(left == np.array([1, 2, 3], dtype=np.uint8))
    assert np.all(right == np.array([4, 5, 6], dtype=np.uint8))


# ---------------------------------------------------------------------------
# tile_views — grids
# ---------------------------------------------------------------------------
def test_tile_three_horizontal_single_row():
    # New contract: n <= 3 -> single horizontal row (no grid, no black padding).
    frames = [_solid(20, 20, (i + 1, i + 1, i + 1)) for i in range(3)]
    out = tile_views(frames)
    # 1 row x 3 cols of 20x20 cells
    assert out.shape == (20, 60, 3)
    # all three cells are non-black, left-to-right in order
    assert np.all(out[:, 0:20] == 1)
    assert np.all(out[:, 20:40] == 2)
    assert np.all(out[:, 40:60] == 3)


def test_tile_four_grid_no_padding():
    frames = [_solid(20, 20, (i + 1, i + 1, i + 1)) for i in range(4)]
    out = tile_views(frames)
    # ncols = ceil(sqrt(4)) = 2, nrows = 2
    assert out.shape == (40, 40, 3)
    # no black cell — every cell non-zero
    for r in (0, 20):
        for c in (0, 20):
            cell = out[r:r + 20, c:c + 20]
            assert np.any(cell != 0)


def test_tile_five_grid_one_black_cell():
    frames = [_solid(10, 10, (i + 1, i + 1, i + 1)) for i in range(5)]
    out = tile_views(frames)
    # ncols = ceil(sqrt(5)) = 3, nrows = ceil(5/3) = 2 -> 6 cells, 1 black
    assert out.shape == (20, 30, 3)
    # count black cells
    black = 0
    for r_idx in range(2):
        for c_idx in range(3):
            cell = out[r_idx * 10:(r_idx + 1) * 10, c_idx * 10:(c_idx + 1) * 10]
            if np.all(cell == 0):
                black += 1
    assert black == 1
    # the single black cell must be bottom-right (last cell, row-major)
    assert np.all(out[10:20, 20:30] == 0)


def test_tile_six_grid_no_padding():
    frames = [_solid(10, 10, (i + 1, i + 1, i + 1)) for i in range(6)]
    out = tile_views(frames)
    # ncols = ceil(sqrt(6)) = 3, nrows = ceil(6/3) = 2
    assert out.shape == (20, 30, 3)
    black = 0
    for r_idx in range(2):
        for c_idx in range(3):
            cell = out[r_idx * 10:(r_idx + 1) * 10, c_idx * 10:(c_idx + 1) * 10]
            if np.all(cell == 0):
                black += 1
    assert black == 0


def test_grid_dimensions_formula_many_n():
    """Cross-check the documented ncols/nrows formula for n = 4..16 (grid path)."""
    for n in range(4, 17):
        frames = [_solid(8, 8, (1, 1, 1)) for _ in range(n)]
        out = tile_views(frames)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        assert out.shape == (nrows * 8, ncols * 8, 3), f"n={n}"


# ---------------------------------------------------------------------------
# mixed resolutions
# ---------------------------------------------------------------------------
def test_mixed_resolutions_cell_is_max():
    f0 = _solid(224, 224, (10, 10, 10))
    f1 = _solid(240, 320, (20, 20, 20))
    out = tile_views([f0, f1])
    # cell = (240, 320), n=2 horizontal
    assert out.shape == (240, 640, 3)
    assert out[:, :320].shape == (240, 320, 3)
    assert out[:, 320:].shape == (240, 320, 3)


# ---------------------------------------------------------------------------
# explicit layouts
# ---------------------------------------------------------------------------
def test_layout_horizontal_forces_single_row_for_four():
    frames = [_solid(15, 15, (i + 1, i + 1, i + 1)) for i in range(4)]
    out = tile_views(frames, layout="horizontal")
    # 1 row, 4 cols
    assert out.shape == (15, 60, 3)


def test_layout_grid_forces_grid_for_two():
    f0 = _solid(15, 15, (1, 1, 1))
    f1 = _solid(15, 15, (2, 2, 2))
    out = tile_views([f0, f1], layout="grid")
    # ncols = ceil(sqrt(2)) = 2, nrows = ceil(2/2) = 1
    assert out.shape == (15, 30, 3)


def test_layout_grid_single_frame():
    f = _solid(15, 15, (3, 3, 3))
    out = tile_views([f], layout="grid")
    # ncols = ceil(sqrt(1)) = 1, nrows = 1
    assert out.shape == (15, 15, 3)


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------
def test_float_frame_clipped_not_rescaled():
    # values like 100.0 should map to 100 (no [0,1] rescaling)
    f = np.full((10, 10, 3), 100.0, dtype=np.float32)
    out = tile_views([f])
    assert out.dtype == np.uint8
    assert np.all(out == 100)
    assert np.any(out != 0)


def test_float_frame_in_unit_range_clips_low():
    # documented behavior: floats in [0,1] are NOT rescaled, so 0.5 -> 0
    f = np.full((10, 10, 3), 0.5, dtype=np.float32)
    out = tile_views([f])
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_float_frame_above_255_clipped():
    f = np.full((10, 10, 3), 300.0, dtype=np.float32)
    out = tile_views([f])
    assert np.all(out == 255)


# ---------------------------------------------------------------------------
# subsample
# ---------------------------------------------------------------------------
def test_subsample_stride_two_keeps_every_other_plus_last():
    frames = list(range(7))  # 0..6
    out = subsample(frames, 2)
    # frames[::2] = [0,2,4,6]; last (6) already present
    assert out == [0, 2, 4, 6]


def test_subsample_stride_two_appends_missing_last():
    frames = list(range(6))  # 0..5
    out = subsample(frames, 2)
    # frames[::2] = [0,2,4]; last is 5, not present -> appended
    assert out == [0, 2, 4, 5]


def test_subsample_stride_one_unchanged():
    frames = list(range(5))
    assert subsample(frames, 1) == frames
    assert subsample(frames, 0) == frames
    assert subsample(frames, -3) == frames


def test_subsample_stride_three_on_ten():
    frames = list(range(10))  # 0..9
    out = subsample(frames, 3)
    # frames[::3] = [0,3,6,9]; last (9) already present
    assert out == [0, 3, 6, 9]


def test_subsample_does_not_mutate_input():
    frames = list(range(6))
    original = list(frames)
    subsample(frames, 2)
    assert frames == original


def test_subsample_single_frame():
    assert subsample([42], 5) == [42]


def test_subsample_empty():
    assert subsample([], 2) == []


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_empty_list_raises_valueerror():
    with pytest.raises(ValueError):
        tile_views([])


def test_ndim_not_three_raises_valueerror():
    bad = np.zeros((10, 10), dtype=np.uint8)  # 2D
    with pytest.raises(ValueError):
        tile_views([bad])


def test_ndim_four_raises_valueerror():
    bad = np.zeros((2, 10, 10, 3), dtype=np.uint8)  # 4D
    with pytest.raises(ValueError):
        tile_views([bad])
