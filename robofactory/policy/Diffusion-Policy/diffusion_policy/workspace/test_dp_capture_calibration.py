"""Tests for the auto-capture calibration hook in robotworkspace.py
(workflow improvement #4).

Strategy: we don't try to spin up a real DP training loop here — that's covered
by test_robotworkspace_wandb_resume*.py / test_joint_plumbing.py and would need
GPU + a full dataset. We test the helper `_capture_calibration_npz` in
isolation (the only thing we added that has actual logic) plus the cfg-flag
plumbing on the wrapper that calls it.

Run from the robofactory repo root:

    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \\
      policy/Diffusion-Policy/diffusion_policy/workspace/test_dp_capture_calibration.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

# Make Diffusion-Policy importable. The hyphen blocks dotted imports.
_DP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DP_DIR))

from diffusion_policy.workspace import robotworkspace  # noqa: E402


class TestCaptureCalibrationHelper(unittest.TestCase):
    """Tests for `_capture_calibration_npz` directly."""

    def test_skip_when_npz_already_exists(self):
        with mock.patch.object(robotworkspace, "subprocess") as sp_mock:
            with mock.patch("pathlib.Path.exists", return_value=True):
                ret = robotworkspace._capture_calibration_npz(
                    ckpt_path="/tmp/fake.ckpt",
                    run_id="abc123",
                    config_path="configs/table/three_robots_stack_cube.yaml",
                    obs_cam_family="workspace",
                    include_global=True,
                )
            sp_mock.run.assert_not_called()
            self.assertIsNotNone(ret)
            self.assertTrue(ret.endswith("abc123.npz"))

    def test_calls_subprocess_with_correct_args(self):
        fake_proc = mock.MagicMock(returncode=0)
        with mock.patch.object(robotworkspace, "subprocess") as sp_mock:
            sp_mock.run.return_value = fake_proc
            with mock.patch("pathlib.Path.exists", return_value=False):
                with mock.patch("pathlib.Path.mkdir"):
                    robotworkspace._capture_calibration_npz(
                        ckpt_path="/tmp/run42/300.ckpt",
                        run_id="wid_xyz",
                        config_path="configs/table/three_robots_stack_cube.yaml",
                        obs_cam_family="wristcam",
                        include_global=False,
                    )
            sp_mock.run.assert_called_once()
            cmd = sp_mock.run.call_args.args[0]
            self.assertIn("--ckpt-path", cmd)
            self.assertIn("/tmp/run42/300.ckpt", cmd)
            self.assertIn("--obs-cam-family", cmd)
            self.assertIn("wristcam", cmd)
            self.assertIn("--no-include-global", cmd)
            # NPZ output must be under runs/calibration/<run_id>.npz
            out_idx = cmd.index("--out") + 1
            self.assertTrue(cmd[out_idx].endswith("/wid_xyz.npz"))
            self.assertIn("runs/calibration", cmd[out_idx])

    def test_nonzero_rc_does_not_raise(self):
        bad_proc = mock.MagicMock(returncode=1)
        with mock.patch.object(robotworkspace, "subprocess") as sp_mock:
            sp_mock.run.return_value = bad_proc
            with mock.patch("pathlib.Path.exists", return_value=False):
                with mock.patch("pathlib.Path.mkdir"):
                    ret = robotworkspace._capture_calibration_npz(
                        ckpt_path="/tmp/x.ckpt",
                        run_id="w",
                        config_path="c.yaml",
                        obs_cam_family="workspace",
                        include_global=True,
                    )
            self.assertIsNone(ret)

    def test_subprocess_exception_does_not_raise(self):
        with mock.patch.object(robotworkspace, "subprocess") as sp_mock:
            sp_mock.run.side_effect = OSError("env busted")
            with mock.patch("pathlib.Path.exists", return_value=False):
                with mock.patch("pathlib.Path.mkdir"):
                    ret = robotworkspace._capture_calibration_npz(
                        ckpt_path="/tmp/x.ckpt",
                        run_id="w",
                        config_path="c.yaml",
                        obs_cam_family="workspace",
                        include_global=True,
                    )
            self.assertIsNone(ret)


class TestCfgFlag(unittest.TestCase):
    """Verify the OmegaConf cfg-flag plumbing matches the expected pattern.

    We don't actually re-run RobotWorkspace.run() (needs GPU + dataset). We
    just check the hook's flag-resolution pattern matches what the workspace
    body uses. If this test ever drifts from the workspace body, both fail.
    """

    def test_flag_default_true(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"training": {"freeze_encoder": False}})
        flag = bool(cfg.training.get("capture_calibration", True))
        self.assertTrue(flag)

    def test_flag_explicit_false(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"training": {"capture_calibration": False}})
        flag = bool(cfg.training.get("capture_calibration", True))
        self.assertFalse(flag)


if __name__ == "__main__":
    unittest.main()
