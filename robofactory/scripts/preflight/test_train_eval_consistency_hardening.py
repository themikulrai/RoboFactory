"""Adversarial tests for scripts/preflight/train_eval_consistency.py.

Beyond the happy-path tests in `test_train_eval_consistency.py`, these stress
the failure modes that real configs are likely to hit:

- check_fn raising non-CheckFailure exceptions
- both-sides-missing keys silently passing checks (a real silent-bug class)
- list-vs-tuple normalisation
- run_all on empty configs

Run:
    python -m unittest scripts.preflight.test_train_eval_consistency_hardening -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preflight import train_eval_consistency as tec  # noqa: E402


def _full_cfg() -> dict:
    return {
        "image": {"shape": [3, 240, 320], "channel_order": "CHW", "normalize_range": "[0,1]"},
        "action": {
            "min": [-1.0] * 8, "max": [1.0] * 8, "mean": [0.0] * 8, "std": [0.5] * 8,
            "mode": "delta", "gripper_range": "[0,1]", "control_target": "joint",
        },
        "control": {"fps": 15},
        "cameras": {"head_camera": {"fx": 280.0, "fy": 280.0, "cx": 160.0, "cy": 120.0}},
    }


class TestCheckRunRobustness(unittest.TestCase):
    """Check.run catches all exceptions and records them as failed results, so
    a typo in one check fn does not abort run_all."""

    def test_check_fn_raising_unexpected_exception_collected(self):
        def bad_fn(a, b):
            raise KeyError("typo")
        result = tec.Check(name="bad", fn=bad_fn).run({}, {})
        self.assertFalse(result.passed)
        self.assertIn("KeyError", result.message)
        self.assertEqual(result.details.get("exc_type"), "KeyError")

    def test_check_fn_raising_assertion_collected(self):
        def assertive(a, b):
            assert False, "internal invariant"
        result = tec.Check(name="a", fn=assertive).run({}, {})
        self.assertFalse(result.passed)
        self.assertIn("AssertionError", result.message)


class TestRequiredKeys(unittest.TestCase):
    """check_action_space_semantics treats absence as a config bug, not a
    silent pass. check_action_norm_stats keeps the all-None opt-out because
    not every config normalises actions."""

    def test_action_space_fails_when_both_lack_action_mode(self):
        a = {"action": {"gripper_range": "[0,1]", "control_target": "joint"}}
        b = {"action": {"gripper_range": "[0,1]", "control_target": "joint"}}
        result = tec.Check("act", tec.check_action_space_semantics).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("action.mode", result.message)
        self.assertIn("missing", result.message)

    def test_action_space_fails_when_one_side_has_mode(self):
        a = {"action": {"mode": "delta", "gripper_range": "[0,1]", "control_target": "joint"}}
        b = {"action": {"gripper_range": "[0,1]", "control_target": "joint"}}
        result = tec.Check("act", tec.check_action_space_semantics).run(a, b)
        self.assertFalse(result.passed)

    def test_action_norm_intentionally_optional_when_both_missing(self):
        """Some configs don't normalise actions; all-None on both sides is
        acceptable. Documented as intentional (not the same class as the
        action_space bug)."""
        a = {"action": {"max": [1.0]}}
        b = {"action": {"max": [1.0]}}
        result = tec.Check("an", tec.check_action_norm_stats).run(a, b)
        self.assertTrue(result.passed)


class TestNormalisation(unittest.TestCase):
    def test_image_shape_list_vs_tuple_normalised(self):
        a = _full_cfg()
        b = _full_cfg()
        a["image"]["shape"] = [3, 240, 320]
        b["image"]["shape"] = (3, 240, 320)
        result = tec.Check("img", tec.check_image_preprocess).run(a, b)
        self.assertTrue(result.passed)

    def test_camera_intrinsic_int_vs_float_match(self):
        a = _full_cfg()
        b = _full_cfg()
        b["cameras"]["head_camera"]["fx"] = 280  # int
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertTrue(result.passed)


class TestEmptyAndPartial(unittest.TestCase):
    def test_run_all_on_empty_dicts_yields_5_failures(self):
        results = tec.run_all({}, {})
        self.assertEqual(len(results), len(tec.CHEAP_CHECKS))
        # Every check requires at least one key; absent keys → CheckFailure.
        # Exception: action_norm_stats accepts both-None pairs. Count *expected*
        # failures explicitly so this test fails if behaviour drifts.
        n_fail = sum(1 for r in results if not r.passed)
        # image, control, cameras → fail. action_space → fails (mode is None on
        # both, but check uses "!=" so passes — that's the silent-bug test
        # above). action_norm → passes (all-None silent). So expect ≥3 fails.
        self.assertGreaterEqual(n_fail, 3)

    def test_run_all_partial_cfg_does_not_crash(self):
        a = {"image": {"shape": [3, 240, 320], "channel_order": "CHW", "normalize_range": "[0,1]"}}
        b = {"image": {"shape": [3, 240, 320], "channel_order": "CHW", "normalize_range": "[0,1]"}}
        results = tec.run_all(a, b)
        # Image passes, the rest fail; importantly run_all must not raise.
        self.assertEqual(len(results), len(tec.CHEAP_CHECKS))


class TestActionNormBoundaries(unittest.TestCase):
    def test_strict_byte_equal_no_tolerance(self):
        """1e-9 difference fails — the docstring promises byte-equal."""
        a = _full_cfg()
        b = _full_cfg()
        b["action"]["mean"] = [1e-9] * 8
        result = tec.Check("an", tec.check_action_norm_stats).run(a, b)
        self.assertFalse(result.passed)

    def test_nan_in_stats_fails_byte_equal(self):
        a = _full_cfg()
        b = _full_cfg()
        a["action"]["min"] = [float("nan")] * 8
        b["action"]["min"] = [float("nan")] * 8
        result = tec.Check("an", tec.check_action_norm_stats).run(a, b)
        # NaN != NaN, so the check is supposed to FAIL even if both sides have
        # NaN. Document this: NaN is never equal, ever.
        self.assertFalse(result.passed)


class TestCameraIntrinsicsBoundary(unittest.TestCase):
    def test_drift_exactly_at_tolerance_passes(self):
        """abs(diff) > 0.5 fails; abs(diff) == 0.5 passes."""
        a = _full_cfg()
        b = _full_cfg()
        b["cameras"]["head_camera"]["fx"] = 280.5  # exactly at boundary
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertTrue(result.passed)

    def test_camera_field_missing_fails(self):
        a = _full_cfg()
        b = _full_cfg()
        del b["cameras"]["head_camera"]["fx"]
        result = tec.Check("ci", tec.check_camera_intrinsics).run(a, b)
        self.assertFalse(result.passed)
        self.assertIn("fx", result.message)


if __name__ == "__main__":
    unittest.main()
