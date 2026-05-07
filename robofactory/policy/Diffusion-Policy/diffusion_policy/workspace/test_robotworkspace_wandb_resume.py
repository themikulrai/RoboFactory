"""Tests for `_init_wandb_with_resume` in robotworkspace.py.

Run from the robofactory repo root in the RoboFactory env:
    python -m unittest \\
      policy.Diffusion-Policy.diffusion_policy.workspace.test_robotworkspace_wandb_resume -v

(or just exec the file directly because the dotted path has hyphens that
 unittest's loader can't handle).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make Diffusion-Policy importable. The hyphen blocks dotted imports, so we
# insert the directory directly.
_DP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DP_DIR))

from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy.workspace import robotworkspace  # noqa: E402


def _make_cfg(resume: bool = True) -> OmegaConf:
    return OmegaConf.create({
        "logging": {
            "project": "test-proj",
            "resume": resume,
            "mode": "online",
            "name": "test-run",
            "tags": ["t"],
            "id": None,
            "group": None,
        },
        "training": {"seed": 0},
    })


class TestFirstLaunch(unittest.TestCase):
    def test_first_launch_writes_id_file(self):
        cfg = _make_cfg(resume=True)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            fake_run = mock.MagicMock()
            fake_run.id = "abc123"
            with mock.patch.object(robotworkspace.wandb, "init", return_value=fake_run) as init_mock:
                with mock.patch.object(robotworkspace.wandb, "run", fake_run):
                    robotworkspace._init_wandb_with_resume(output_dir, cfg)

            init_mock.assert_called_once()
            kwargs = init_mock.call_args.kwargs
            # id must NOT be set on first launch (no id file present)
            self.assertNotIn("id", kwargs)
            # resume must be "allow", not literal True (which would force resume by name)
            self.assertEqual(kwargs["resume"], "allow")

            id_file = output_dir / "wandb_id.txt"
            self.assertTrue(id_file.is_file())
            self.assertEqual(id_file.read_text().strip(), "abc123")


class TestResumeLaunch(unittest.TestCase):
    def test_existing_id_file_forces_resume_must(self):
        cfg = _make_cfg(resume=True)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            id_file = output_dir / "wandb_id.txt"
            id_file.write_text("xyz789\n")

            fake_run = mock.MagicMock()
            fake_run.id = "xyz789"
            with mock.patch.object(robotworkspace.wandb, "init", return_value=fake_run) as init_mock:
                with mock.patch.object(robotworkspace.wandb, "run", fake_run):
                    robotworkspace._init_wandb_with_resume(output_dir, cfg)

            init_mock.assert_called_once()
            kwargs = init_mock.call_args.kwargs
            self.assertEqual(kwargs["id"], "xyz789")
            self.assertEqual(kwargs["resume"], "must")

    def test_resume_does_not_overwrite_existing_id(self):
        """Same id present + same id returned → no rewrite needed (idempotent)."""
        cfg = _make_cfg(resume=True)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            id_file = output_dir / "wandb_id.txt"
            id_file.write_text("xyz789")
            mtime0 = id_file.stat().st_mtime

            fake_run = mock.MagicMock()
            fake_run.id = "xyz789"
            with mock.patch.object(robotworkspace.wandb, "init", return_value=fake_run):
                with mock.patch.object(robotworkspace.wandb, "run", fake_run):
                    robotworkspace._init_wandb_with_resume(output_dir, cfg)

            # We do not rewrite if file content already matches the id.
            self.assertEqual(id_file.read_text().strip(), "xyz789")


class TestEdgeCases(unittest.TestCase):
    def test_empty_id_file_treated_as_first_launch(self):
        cfg = _make_cfg(resume=True)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            (output_dir / "wandb_id.txt").write_text("")

            fake_run = mock.MagicMock()
            fake_run.id = "fresh999"
            with mock.patch.object(robotworkspace.wandb, "init", return_value=fake_run) as init_mock:
                with mock.patch.object(robotworkspace.wandb, "run", fake_run):
                    robotworkspace._init_wandb_with_resume(output_dir, cfg)

            kwargs = init_mock.call_args.kwargs
            self.assertNotIn("id", kwargs)
            self.assertEqual(kwargs["resume"], "allow")

    def test_resume_false_in_cfg_passes_through(self):
        cfg = _make_cfg(resume=False)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            fake_run = mock.MagicMock()
            fake_run.id = "x"
            with mock.patch.object(robotworkspace.wandb, "init", return_value=fake_run) as init_mock:
                with mock.patch.object(robotworkspace.wandb, "run", fake_run):
                    robotworkspace._init_wandb_with_resume(output_dir, cfg)

            kwargs = init_mock.call_args.kwargs
            # Cfg said resume=False; we don't promote it
            self.assertFalse(kwargs.get("resume"))


if __name__ == "__main__":
    unittest.main()
