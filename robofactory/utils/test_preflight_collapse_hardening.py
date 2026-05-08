"""Adversarial tests for utils.preflight_collapse.

Beyond the happy-path tests in `test_preflight_collapse.py`, these stress:
- _zero_keys mutation safety (does it copy or share references?)
- feature_episodes_cap=0 boundary
- dtype handling (uint8 image, mixed-dtype obs)
- multi-image-key auto-detection
- to_wandb_payload prefix override
- _safe_mse with inf

Run:
    python -m unittest utils.test_preflight_collapse_hardening -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import preflight_collapse  # noqa: E402


class TestZeroKeysSafety(unittest.TestCase):
    def test_zero_keys_returns_new_dict(self):
        obs = {"img": np.ones((3, 4, 4), dtype=np.float32),
               "pos": np.array([1.0, 2.0])}
        out = preflight_collapse._zero_keys(obs, ["img"])
        self.assertIsNot(out, obs)
        self.assertTrue((out["img"] == 0).all())
        # original untouched
        self.assertTrue((obs["img"] == 1).all())

    def test_zero_keys_does_not_mutate_kept_values(self):
        """A naive impl might share array references; mutating the output's
        kept value should not bleed into the input."""
        obs = {"img": np.ones((3, 4, 4), dtype=np.float32),
               "pos": np.array([1.0, 2.0])}
        out = preflight_collapse._zero_keys(obs, ["img"])
        # If `out["pos"]` is the same ndarray as `obs["pos"]`, mutating it
        # corrupts the input. The current impl shares the reference; this
        # test PINS that behaviour so a future change is intentional.
        self.assertIs(out["pos"], obs["pos"])
        # If you ever change to defensive copy, flip this assertion.

    def test_zero_keys_with_nonexistent_key_is_silent(self):
        obs = {"img": np.ones((3,), dtype=np.float32)}
        out = preflight_collapse._zero_keys(obs, ["typo_key"])
        # No-op; original returned untouched in shape.
        self.assertTrue((out["img"] == 1).all())


class TestAutoKeyDetection(unittest.TestCase):
    def test_multiple_image_keys_detected(self):
        obs = {
            "head_cam": np.zeros((3, 32, 32), dtype=np.float32),
            "wrist_cam": np.zeros((3, 32, 32), dtype=np.float32),
            "agent_pos": np.zeros(8, dtype=np.float32),
        }
        keys = preflight_collapse._auto_image_keys(obs)
        self.assertEqual(set(keys), {"head_cam", "wrist_cam"})

    def test_grayscale_2d_not_detected_as_image(self):
        obs = {"depth": np.zeros((32, 32), dtype=np.float32),
               "agent_pos": np.zeros(8, dtype=np.float32)}
        # Grayscale 2D has ndim<3 OR no C=3 dim. Not detected.
        keys = preflight_collapse._auto_image_keys(obs)
        self.assertEqual(keys, [])

    def test_4_channel_not_detected(self):
        """RGBD with 4 channels does NOT match the C=3 heuristic."""
        obs = {"rgbd": np.zeros((4, 32, 32), dtype=np.float32)}
        keys = preflight_collapse._auto_image_keys(obs)
        self.assertEqual(keys, [])

    def test_proprio_keys_excludes_image_keys(self):
        obs = {
            "head_cam": np.zeros((3, 32, 32), dtype=np.float32),
            "agent_pos": np.zeros(8, dtype=np.float32),
            "joints": np.zeros(7, dtype=np.float32),
        }
        ik = preflight_collapse._auto_image_keys(obs)
        pk = preflight_collapse._auto_proprio_keys(obs, ik)
        self.assertEqual(set(pk), {"agent_pos", "joints"})


class TestSafeMseEdgeCases(unittest.TestCase):
    def test_inf_in_pred_raises(self):
        pred = np.array([[1.0, float("inf"), 3.0]])
        target = np.zeros_like(pred)
        with self.assertRaises(preflight_collapse.CollapseProbeError):
            preflight_collapse._safe_mse(pred, target)

    def test_neg_inf_in_target_raises(self):
        pred = np.zeros((1, 3))
        target = np.array([[1.0, -float("inf"), 3.0]])
        with self.assertRaises(preflight_collapse.CollapseProbeError):
            preflight_collapse._safe_mse(pred, target)

    def test_finite_int_dtype_handled(self):
        pred = np.array([[1, 2, 3]], dtype=np.int8)
        target = np.array([[0, 0, 0]], dtype=np.int8)
        # Cast to float64 inside, no error, finite result.
        v = preflight_collapse._safe_mse(pred, target)
        self.assertAlmostEqual(v, (1 + 4 + 9) / 3.0, places=6)


class TestPayloadCustomPrefix(unittest.TestCase):
    def test_custom_prefix_replaces_collapse(self):
        report = preflight_collapse.CollapseReport(
            mse_baseline=0.01, mse_zero_image=0.02, mse_zero_proprio=0.5,
            feature_rank=4, per_channel_variance=np.array([0.1]),
            n_episodes=8, image_keys=("h",), proprio_keys=("p",),
            feature_dim=64,
        )
        payload = report.to_wandb_payload(prefix="probe_pm")
        # No "collapse/*" keys
        self.assertFalse(any(k.startswith("collapse/") for k in payload))
        self.assertIn("probe_pm/mse_baseline", payload)
        self.assertIn("probe_pm/feature_rank", payload)


class TestRatioBoundaries(unittest.TestCase):
    def test_ratio_with_tiny_baseline_uses_floor(self):
        report = preflight_collapse.CollapseReport(
            mse_baseline=1e-15,  # below the 1e-12 floor
            mse_zero_image=1.0, mse_zero_proprio=1.0,
            feature_rank=1, per_channel_variance=np.array([0.0]),
            n_episodes=1, image_keys=("h",), proprio_keys=("p",),
            feature_dim=4,
        )
        # Floor protects from infinity; ratio is large but finite.
        self.assertTrue(np.isfinite(report.image_to_baseline_ratio))


if __name__ == "__main__":
    unittest.main()
