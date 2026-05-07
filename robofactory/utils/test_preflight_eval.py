"""Tests for robofactory.utils.preflight_eval.

Run from the robofactory env:
    python -m unittest robofactory.utils.test_preflight_eval -v
or directly:
    python robofactory/utils/test_preflight_eval.py
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from robofactory.utils import preflight_eval as pe


class TestWandbOnline(unittest.TestCase):
    def test_default_online_with_api_key(self):
        env = {"WANDB_API_KEY": "abc"}
        pe.check_wandb_online(env)  # no raise

    def test_offline_mode_fails(self):
        env = {"WANDB_API_KEY": "abc", "WANDB_MODE": "offline"}
        with self.assertRaises(pe.PreflightFailure):
            pe.check_wandb_online(env)

    def test_disabled_fails(self):
        env = {"WANDB_API_KEY": "abc", "WANDB_DISABLED": "true"}
        with self.assertRaises(pe.PreflightFailure):
            pe.check_wandb_online(env)

    def test_disabled_value_one_fails(self):
        env = {"WANDB_API_KEY": "abc", "WANDB_DISABLED": "1"}
        with self.assertRaises(pe.PreflightFailure):
            pe.check_wandb_online(env)

    def test_disabled_value_no_does_not_fail(self):
        env = {"WANDB_API_KEY": "abc", "WANDB_DISABLED": "no"}
        pe.check_wandb_online(env)  # not in {1,true,yes}

    def test_missing_api_key_fails(self):
        env = {"WANDB_MODE": "online"}  # no WANDB_API_KEY
        with self.assertRaises(pe.PreflightFailure):
            pe.check_wandb_online(env)


class TestPathExists(unittest.TestCase):
    def test_existing_path_passes(self):
        with tempfile.NamedTemporaryFile() as f:
            pe.check_path_exists("test", Path(f.name))

    def test_missing_path_fails(self):
        with self.assertRaises(pe.PreflightFailure):
            pe.check_path_exists("test", Path("/no/such/file/xxx"))


class TestSceneNameMatch(unittest.TestCase):
    def test_pm_match(self):
        cfg = Path("configs/table/pick_meat.yaml")
        ckpt = Path("/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt")
        pe.check_scene_name_match(cfg, ckpt)  # no raise

    def test_tsc_match(self):
        cfg = Path("configs/table/three_robots_stack_cube.yaml")
        ckpt = Path("/iris/u/mikulrai/checkpoints/RoboFactory/"
                    "ThreeRobotsStackCube-rf_agent0_d2_wristcam_150/300.ckpt")
        pe.check_scene_name_match(cfg, ckpt)  # no raise

    def test_pm_ckpt_with_tsc_config_fails(self):
        cfg = Path("configs/table/three_robots_stack_cube.yaml")
        ckpt = Path("/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt")
        with self.assertRaises(pe.PreflightFailure) as ctx:
            pe.check_scene_name_match(cfg, ckpt)
        self.assertIn("scene-name mismatch", str(ctx.exception))

    def test_tsc_ckpt_with_pm_config_fails(self):
        cfg = Path("configs/table/pick_meat.yaml")
        ckpt = Path("/iris/u/mikulrai/checkpoints/RoboFactory/"
                    "ThreeRobotsStackCube-rf_agent0_d2_wristcam_150/300.ckpt")
        with self.assertRaises(pe.PreflightFailure):
            pe.check_scene_name_match(cfg, ckpt)

    def test_unknown_ckpt_prefix_warns_but_passes(self):
        # Permissive when ckpt dir doesn't match any known task prefix
        # (e.g. an opaque hash dir). Caller is responsible for enforcement
        # downstream; the guard's job is to catch the COMMON mistake.
        cfg = Path("configs/table/pick_meat.yaml")
        ckpt = Path("/some/random/path/abc123def/300.ckpt")
        pe.check_scene_name_match(cfg, ckpt)  # no raise


class TestSeedFileSha256(unittest.TestCase):
    def test_matching_sha_passes(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "seeds.txt"
            content = b"10000\n10001\n10002\n"
            f.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            pe.check_seed_file_sha256(f, expected)

    def test_mismatched_sha_fails(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "seeds.txt"
            f.write_bytes(b"10000\n")
            with self.assertRaises(pe.PreflightFailure) as ctx:
                pe.check_seed_file_sha256(f, "0" * 64)
            self.assertIn("sha256", str(ctx.exception))

    def test_missing_file_fails(self):
        with self.assertRaises(pe.PreflightFailure):
            pe.check_seed_file_sha256(Path("/no/such/seeds.txt"), "0" * 64)

    def test_sha_check_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "seeds.txt"
            content = b"abc"
            f.write_bytes(content)
            expected_upper = hashlib.sha256(content).hexdigest().upper()
            pe.check_seed_file_sha256(f, expected_upper)


class TestCliMain(unittest.TestCase):
    """End-to-end: CLI exit code reflects success/failure correctly."""

    def setUp(self):
        # Each test gets a fresh tempdir + matching ckpt + scene config.
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        self.ckpt_dir = td / "PickMeat-rf_150"
        self.ckpt_dir.mkdir()
        self.ckpt = self.ckpt_dir / "300.ckpt"
        self.ckpt.write_text("")
        self.scene_cfg = td / "pick_meat.yaml"
        self.scene_cfg.write_text("dummy: 1\n")
        self.seeds = td / "seeds.txt"
        self.seeds.write_text("10000\n10001\n")
        self.seeds_sha = hashlib.sha256(self.seeds.read_bytes()).hexdigest()

        # Stash + restore wandb env vars per test.
        import os
        self._old_env = {k: os.environ.get(k) for k in (
            "WANDB_MODE", "WANDB_DISABLED", "WANDB_API_KEY",
        )}
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_API_KEY"] = "fake-key-for-test"

    def tearDown(self):
        import os
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._td.cleanup()

    def test_all_checks_pass(self):
        rc = pe.main([
            "--ckpt-path", str(self.ckpt),
            "--scene-config", str(self.scene_cfg),
            "--seeds-file", str(self.seeds),
            "--seeds-sha256", self.seeds_sha,
        ])
        self.assertEqual(rc, 0)

    def test_missing_ckpt_returns_1(self):
        rc = pe.main([
            "--ckpt-path", "/no/such/ckpt.ckpt",
            "--scene-config", str(self.scene_cfg),
        ])
        self.assertEqual(rc, 1)

    def test_scene_mismatch_returns_1(self):
        # ckpt PickMeat with TSC config -> mismatch
        bad_cfg = self.scene_cfg.parent / "three_robots_stack_cube.yaml"
        bad_cfg.write_text("dummy: 1\n")
        rc = pe.main([
            "--ckpt-path", str(self.ckpt),
            "--scene-config", str(bad_cfg),
        ])
        self.assertEqual(rc, 1)

    def test_scene_mismatch_can_be_disabled(self):
        bad_cfg = self.scene_cfg.parent / "three_robots_stack_cube.yaml"
        bad_cfg.write_text("dummy: 1\n")
        rc = pe.main([
            "--ckpt-path", str(self.ckpt),
            "--scene-config", str(bad_cfg),
            "--no-scene-name-match",
        ])
        self.assertEqual(rc, 0)

    def test_wandb_offline_returns_1(self):
        import os
        os.environ["WANDB_MODE"] = "offline"
        rc = pe.main([
            "--ckpt-path", str(self.ckpt),
            "--scene-config", str(self.scene_cfg),
        ])
        self.assertEqual(rc, 1)

    def test_wandb_check_can_be_disabled(self):
        import os
        os.environ["WANDB_MODE"] = "offline"
        rc = pe.main([
            "--ckpt-path", str(self.ckpt),
            "--scene-config", str(self.scene_cfg),
            "--no-wandb-online",
        ])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
