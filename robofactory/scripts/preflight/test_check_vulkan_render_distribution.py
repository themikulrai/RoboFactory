"""Tests for scripts/preflight/check_vulkan_render_distribution.py.

These cover the *compare* path only — rendering is tested by the SLURM
driver on real compute nodes.  The login node renders SAPIEN scenes ~32%
darker than compute nodes (per CLAUDE.md / the very bug this preflight
detects), so unit-testing the renderer here would be both wrong (login
output != compute output) and slow (SAPIEN startup).

Run from repo root::

    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \\
        -m pytest robofactory/scripts/preflight/test_check_vulkan_render_distribution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the script importable regardless of pytest cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))  # repo root

from robofactory.scripts.preflight import check_vulkan_render_distribution as cv  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: synthetic image batches with controllable per-channel statistics
# ---------------------------------------------------------------------------


def _const_image(mean_rgb: tuple[float, float, float], shape=(8, 8)) -> np.ndarray:
    """Return a single (H, W, 3) uint8 image with each channel set to its mean.

    A constant image has std == 0 on each channel, so it's the cleanest
    fixture for tests that want to control mean exactly.
    """
    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in enumerate(mean_rgb):
        img[..., k] = int(round(v))
    return img


def _gauss_batch(
    mean_rgb: tuple[float, float, float],
    std_rgb: tuple[float, float, float],
    n: int = 4,
    shape: tuple[int, int] = (16, 16),
    seed: int = 0,
) -> np.ndarray:
    """Return (N, H, W, 3) uint8 with controllable per-channel mean+std.

    We use a fixed RNG seed and large pixel counts so empirical channel
    stats land within ~0.5% of the requested values — well below the 2%
    default tolerance, so tests don't flake.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    out = np.empty((n, h, w, 3), dtype=np.float64)
    for k in range(3):
        out[..., k] = rng.normal(mean_rgb[k], std_rgb[k], size=(n, h, w))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Identical images -> all diffs == 0, passed=True
# ---------------------------------------------------------------------------


class TestIdentical:
    def test_identical_arrays_pass_with_zero_diff(self):
        imgs = _gauss_batch((128, 96, 64), (10, 10, 10), n=4, seed=1)
        report = cv.compare_renders(imgs, imgs.copy(), tolerance_pct=2.0)
        assert report.passed is True
        assert report.failed_channels == []
        for d in report.mean_diff_pct:
            assert d == pytest.approx(0.0, abs=1e-12)
        for d in report.std_diff_pct:
            assert d == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. 1% mean shift on red -> under tolerance
# ---------------------------------------------------------------------------


class TestSmallMeanShift:
    def test_one_percent_red_shift_passes(self):
        # Constant images: per-channel std == 0 on both sides, so std diffs
        # use the b==0/diff==0 branch -> 0.0 (true agreement).  Only the
        # red mean shifts.  Percent diff is |a-b|/|b|*100, so a=101 vs
        # b=100 yields exactly 1.0%.
        a = _const_image((101.0, 100.0, 100.0))[None, ...]
        b = _const_image((100.0, 100.0, 100.0))[None, ...]
        report = cv.compare_renders(a, b, tolerance_pct=2.0)
        assert report.passed is True
        assert report.failed_channels == []
        # Sanity: the diff is exactly 1.0% on mean_R, 0% elsewhere.
        assert report.mean_diff_pct[0] == pytest.approx(1.0, rel=1e-6)
        assert report.mean_diff_pct[1] == pytest.approx(0.0, abs=1e-12)
        assert report.mean_diff_pct[2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. 3% mean shift on red -> fail mean_R only
# ---------------------------------------------------------------------------


class TestMeanShiftFails:
    def test_three_percent_red_shift_fails_only_mean_r(self):
        # Percent diff is |a-b|/|b|*100, so for a clean 3% diff we put
        # the elevated value on side a and the reference on side b.
        a = _const_image((103.0, 100.0, 100.0))[None, ...]
        b = _const_image((100.0, 100.0, 100.0))[None, ...]
        report = cv.compare_renders(a, b, tolerance_pct=2.0)
        assert report.passed is False
        assert report.failed_channels == ["mean_R"]
        assert report.mean_diff_pct[0] == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. 3% std shift on green -> fail std_G only
# ---------------------------------------------------------------------------


class TestStdShiftFails:
    def test_three_percent_green_std_shift_fails_only_std_g(self):
        # Use Gaussian batches so std is well-defined.  Mean is identical,
        # std differs by 3% on G.
        a = _gauss_batch((128, 128, 128), (20.0, 20.0, 20.0), n=8, seed=2)
        b = _gauss_batch((128, 128, 128), (20.0, 20.6, 20.0), n=8, seed=2)
        report = cv.compare_renders(a, b, tolerance_pct=2.0)
        # Only std_G should fail.  Mean diffs are at sample-noise level
        # (well under 2%), and other channels' stds match.
        assert "std_G" in report.failed_channels
        # Filter to just the channels we care about being clean.
        clean_kinds = [c for c in report.failed_channels if c != "std_G"]
        assert clean_kinds == [], f"unexpected failures: {clean_kinds}"
        assert report.passed is False


# ---------------------------------------------------------------------------
# 5. Multiple channels failing -> all listed
# ---------------------------------------------------------------------------


class TestMultiChannelFailure:
    def test_multiple_channels_listed_in_failed_channels(self):
        # Mean shifts on all three channels, all > 2%.
        a = _const_image((100.0, 100.0, 100.0))[None, ...]
        b = _const_image((105.0, 110.0, 120.0))[None, ...]
        report = cv.compare_renders(a, b, tolerance_pct=2.0)
        assert report.passed is False
        # Exactly the three mean channels should fail (std is 0/0 -> 0%).
        assert set(report.failed_channels) == {"mean_R", "mean_G", "mean_B"}


# ---------------------------------------------------------------------------
# 6. Mismatched array shapes -> ValueError
# ---------------------------------------------------------------------------


class TestShapeMismatch:
    def test_mismatched_per_image_shape_raises(self):
        a = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        b = np.zeros((4, 16, 16, 3), dtype=np.uint8)  # different H,W
        with pytest.raises(ValueError, match="image shape mismatch"):
            cv.compare_renders(a, b, tolerance_pct=2.0)

    def test_wrong_channel_dim_raises(self):
        a = np.zeros((4, 8, 8, 4), dtype=np.uint8)  # RGBA, not RGB
        b = np.zeros((4, 8, 8, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="last dim must be 3"):
            cv.compare_renders(a, b, tolerance_pct=2.0)

    def test_wrong_ndim_raises(self):
        a = np.zeros((8, 3), dtype=np.uint8)  # 2-D garbage
        b = np.zeros((8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="unsupported shape"):
            cv.compare_renders(a, b, tolerance_pct=2.0)


# ---------------------------------------------------------------------------
# 7. uint8 vs float32 -> handled gracefully (cast both to float)
# ---------------------------------------------------------------------------


class TestDtypeMixing:
    def test_uint8_vs_float32_compared_as_float(self):
        a_u8 = _const_image((100.0, 100.0, 100.0))[None, ...]
        # Same content, but as float32 in [0,255].
        b_f32 = a_u8.astype(np.float32)
        report = cv.compare_renders(a_u8, b_f32, tolerance_pct=2.0)
        assert report.passed is True
        for d in report.mean_diff_pct:
            assert d == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 8. Single image (H,W,3) -> reshaped to (1,H,W,3)
# ---------------------------------------------------------------------------


class TestSingleImageShape:
    def test_three_dim_input_treated_as_n_equals_one(self):
        # Percent diff is |a-b|/|b|*100, so a=101, b=100 -> exactly 1.0%.
        a = _const_image((101.0, 100.0, 100.0))  # shape (H, W, 3)
        b = _const_image((100.0, 100.0, 100.0))
        assert a.ndim == 3 and b.ndim == 3
        report = cv.compare_renders(a, b, tolerance_pct=2.0)
        assert report.passed is True
        assert report.mean_diff_pct[0] == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 9. Empty arrays -> ValueError
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_array_raises(self):
        a = np.zeros((0, 8, 8, 3), dtype=np.uint8)
        b = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            cv.compare_renders(a, b, tolerance_pct=2.0)


# ---------------------------------------------------------------------------
# 10. JSON round-trip of RenderReport
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_report_serialises_and_deserialises(self, tmp_path):
        a = _const_image((100.0, 100.0, 100.0))[None, ...]
        b = _const_image((101.0, 100.0, 100.0))[None, ...]
        report = cv.compare_renders(
            a, b, tolerance_pct=2.0,
            node_a_path="/tmp/a.npy",
            node_b_path="/tmp/b.npy",
        )
        out = tmp_path / "report.json"
        out.write_text(json.dumps(report.to_dict(), indent=2))

        loaded = json.loads(out.read_text())
        for k in ("node_a_path", "node_b_path",
                  "per_channel_mean_a", "per_channel_mean_b",
                  "per_channel_std_a", "per_channel_std_b",
                  "mean_diff_pct", "std_diff_pct",
                  "tolerance_pct", "passed", "failed_channels"):
            assert k in loaded, f"missing key: {k}"
        assert loaded["passed"] is True
        assert loaded["tolerance_pct"] == 2.0
        assert loaded["failed_channels"] == []


# ---------------------------------------------------------------------------
# 11. Tolerance of 0.0 -> any non-zero diff fails
# ---------------------------------------------------------------------------


class TestZeroTolerance:
    def test_zero_tolerance_fails_on_any_diff(self):
        a = _const_image((100.0, 100.0, 100.0))[None, ...]
        b = _const_image((100.0, 100.0, 101.0))[None, ...]  # 1% on B
        report = cv.compare_renders(a, b, tolerance_pct=0.0)
        assert report.passed is False
        assert "mean_B" in report.failed_channels

    def test_zero_tolerance_zero_diff_still_passes(self):
        a = _const_image((100.0, 100.0, 100.0))[None, ...]
        report = cv.compare_renders(a, a.copy(), tolerance_pct=0.0)
        assert report.passed is True
        assert report.failed_channels == []


# ---------------------------------------------------------------------------
# 12. NaN in input -> ValueError ("rendered images contain NaN")
# ---------------------------------------------------------------------------


class TestNaNRejection:
    def test_nan_in_node_a_raises(self):
        a = np.ones((1, 4, 4, 3), dtype=np.float32)
        a[0, 0, 0, 0] = np.nan
        b = np.ones((1, 4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            cv.compare_renders(a, b, tolerance_pct=2.0)

    def test_nan_in_node_b_raises(self):
        a = np.ones((1, 4, 4, 3), dtype=np.float32)
        b = np.ones((1, 4, 4, 3), dtype=np.float32)
        b[0, 0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            cv.compare_renders(a, b, tolerance_pct=2.0)


# ---------------------------------------------------------------------------
# Bonus: dependency-injection seam for render_scene_image
# ---------------------------------------------------------------------------


class TestRenderInjection:
    """Verify the env_factory + rgb_extractor seams work (no SAPIEN).

    Catches accidental top-level SAPIEN imports — if the module ever
    imports SAPIEN at import time, this whole test file would fail to
    import on the login node.
    """

    def test_render_scene_image_uses_injected_factory(self):
        captured = {}

        class _FakeEnv:
            def reset(self, seed=None):
                captured["seed"] = seed
                rgb = np.full((8, 8, 3), 42, dtype=np.uint8)
                return {"rgb": rgb}, {}

            def close(self):
                captured["closed"] = True

        def factory(scene_cfg, seed):
            captured["scene_cfg"] = scene_cfg
            captured["factory_seed"] = seed
            return _FakeEnv()

        out = cv.render_scene_image(
            Path("/dev/null/scene.yaml"), seed=99,
            env_factory=factory,
        )
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8
        assert (out == 42).all()
        assert captured["seed"] == 99
        assert captured["factory_seed"] == 99
        assert captured["closed"] is True

    def test_render_scene_image_coerces_float_to_uint8(self):
        class _FakeEnv:
            def reset(self, seed=None):
                rgb = np.full((4, 4, 3), 200.7, dtype=np.float32)
                return {"rgb": rgb}, {}

            def close(self):
                pass

        out = cv.render_scene_image(
            Path("/dev/null/scene.yaml"), seed=0,
            env_factory=lambda *_: _FakeEnv(),
        )
        assert out.dtype == np.uint8
        # Values clipped to uint8 range.
        assert int(out[0, 0, 0]) == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
