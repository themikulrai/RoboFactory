"""Tests for scripts/preflight/train_eval_consistency.py.

Run from repo root:
    python -m unittest scripts.preflight.test_train_eval_consistency -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preflight import train_eval_consistency as tec  # noqa: E402


def _good_cfg() -> dict:
    return {
        "image": {
            "shape": [3, 240, 320],
            "channel_order": "CHW",
            "normalize_range": "[0,1]",
        },
        "action": {
            "min": [-1.0] * 8,
            "max": [1.0] * 8,
            "mean": [0.0] * 8,
            "std": [0.5] * 8,
            "mode": "delta",
            "gripper_range": "[0,1]",
            "control_target": "joint",
        },
        "control": {"fps": 15},
        "cameras": {
            "head_camera": {"fx": 280.0, "fy": 280.0, "cx": 160.0, "cy": 120.0},
        },
    }


class TestImagePreprocess(unittest.TestCase):
    def test_match(self):
        a = _good_cfg()
        b = _good_cfg()
        result = tec.Check("img", tec.check_image_preprocess).run(a, b)
        self.assertTrue(result.passed)

    def test_shape_mismatch_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["image"]["shape"] = [3, 224, 224]
        result = tec.Check("img", tec.check_image_preprocess).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("shape mismatch", result.message)

    def test_channel_order_mismatch_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["image"]["channel_order"] = "HWC"
        result = tec.Check("img", tec.check_image_preprocess).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("channel_order", result.message)

    def test_normalize_range_mismatch_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["image"]["normalize_range"] = "[-1,1]"
        result = tec.Check("img", tec.check_image_preprocess).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("normalize_range", result.message)


class TestActionSpaceSemantics(unittest.TestCase):
    def test_match(self):
        a = _good_cfg()
        b = _good_cfg()
        result = tec.Check("act", tec.check_action_space_semantics).run(a, b)
        self.assertTrue(result.passed)

    def test_mode_swap_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["action"]["mode"] = "absolute"
        result = tec.Check("act", tec.check_action_space_semantics).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("action.mode", result.message)

    def test_gripper_convention_swap_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["action"]["gripper_range"] = "[-1,1]"
        result = tec.Check("act", tec.check_action_space_semantics).run(a, b)
        self.assertFalse(result.passed)


class TestActionNormStats(unittest.TestCase):
    def test_byte_equal_passes(self):
        a = _good_cfg()
        b = _good_cfg()
        result = tec.Check("an", tec.check_action_norm_stats).run(a, b)
        self.assertTrue(result.passed)

    def test_min_off_by_epsilon_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["action"]["min"] = [-1.0 + 1e-7] * 8
        result = tec.Check("an", tec.check_action_norm_stats).run(a, b)
        self.assertFalse(result.passed)


class TestControlRate(unittest.TestCase):
    def test_match(self):
        a = _good_cfg()
        b = _good_cfg()
        result = tec.Check("cr", tec.check_control_rate).run(a, b)
        self.assertTrue(result.passed)

    def test_double_fps_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["control"]["fps"] = 30
        result = tec.Check("cr", tec.check_control_rate).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("fps mismatch", result.message)

    def test_missing_fps_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        del b["control"]["fps"]
        result = tec.Check("cr", tec.check_control_rate).run(a, b)
        self.assertFalse(result.passed)


class TestCameraIntrinsics(unittest.TestCase):
    def test_match(self):
        a = _good_cfg()
        b = _good_cfg()
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertTrue(result.passed)

    def test_camera_name_mismatch_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["cameras"]["head_camera_global"] = b["cameras"].pop("head_camera")
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("camera names mismatch", result.message)

    def test_intrinsic_drift_within_tolerance_passes(self):
        a = _good_cfg()
        b = _good_cfg()
        b["cameras"]["head_camera"]["fx"] = 280.4  # within 0.5 tol
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertTrue(result.passed)

    def test_intrinsic_drift_above_tolerance_fails(self):
        a = _good_cfg()
        b = _good_cfg()
        b["cameras"]["head_camera"]["fx"] = 281.0  # > 0.5 tol
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertFalse(result.passed)


class TestRunAll(unittest.TestCase):
    def test_all_pass_on_good_config(self):
        a = _good_cfg()
        b = _good_cfg()
        results = tec.run_all(a, b)
        n_fail = sum(1 for r in results if not r.passed)
        self.assertEqual(n_fail, 0)
        self.assertEqual(len(results), len(tec.CHEAP_CHECKS))

    def test_continues_after_failure(self):
        """The driver collects every check's result rather than short-circuiting."""
        a = _good_cfg()
        b = _good_cfg()
        b["image"]["shape"] = [3, 224, 224]   # fails check 1
        b["control"]["fps"] = 30              # fails check 4
        results = tec.run_all(a, b)
        self.assertEqual(sum(1 for r in results if not r.passed), 2)


if __name__ == "__main__":
    unittest.main()
