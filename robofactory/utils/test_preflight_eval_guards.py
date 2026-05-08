"""Tests for robofactory.utils.preflight_eval_guards (Stage-3 guards)."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import yaml

from robofactory.utils import preflight_eval_guards as g


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _make_train_cfg(tmp: Path, env_id: str = "PickMeat-rf",
                    cam_family: str = "workspace") -> Path:
    """Hydra-style trainer dump with task.env_runner.env_id."""
    cfg = {
        "task": {
            "env_runner": {"env_id": env_id},
            "obs_cam_family": cam_family,
        },
    }
    return _write_yaml(tmp / "train.yaml", cfg)


def _make_eval_cfg(tmp: Path, env_id: str = "PickMeat-rf",
                   cam_family: str = "workspace") -> Path:
    """Eval-style YAML — uses task_name (RoboFactory convention) plus a
    dataset_path-like field that signals the camera family."""
    cfg = {
        "task_name": env_id,
        "task": {
            "dataset_path": f"/data/{env_id}_{cam_family}_150.zarr",
        },
    }
    return _write_yaml(tmp / "eval.yaml", cfg)


# ---------------------------------------------------------------------------
# Scene match
# ---------------------------------------------------------------------------
class TestSceneMatch(unittest.TestCase):
    def test_match_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p, env_id="PickMeat-rf")
            ev = _make_eval_cfg(p, env_id="PickMeat-rf")
            g.assert_scene_match(train, ev)  # no raise

    def test_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p, env_id="PickMeat-rf")
            ev = _make_eval_cfg(p, env_id="ThreeRobotsStackCube-rf")
            with self.assertRaises(g.SceneMismatchError):
                g.assert_scene_match(train, ev)

    def test_missing_env_id_skips_with_warning(self):
        # Old configs with no env_id should not crash — just warn.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _write_yaml(p / "t.yaml", {"unrelated": 1})
            ev = _write_yaml(p / "e.yaml", {"unrelated": 2})
            with self.assertWarns(UserWarning):
                g.assert_scene_match(train, ev)  # no raise


# ---------------------------------------------------------------------------
# Camera family match
# ---------------------------------------------------------------------------
class TestCameraFamilyMatch(unittest.TestCase):
    def test_match_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p, cam_family="workspace")
            ev = _make_eval_cfg(p, cam_family="workspace")
            g.assert_camera_family_match(train, ev)  # no raise

    def test_workspace_vs_wristcam_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p, cam_family="workspace")
            ev = _make_eval_cfg(p, cam_family="wristcam")
            with self.assertRaises(g.CameraFamilyMismatchError):
                g.assert_camera_family_match(train, ev)

    def test_indeterminate_skips_with_warning(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _write_yaml(p / "t.yaml", {"task": {"env_runner": {"env_id": "X"}}})
            ev = _write_yaml(p / "e.yaml", {"task_name": "X"})
            with self.assertWarns(UserWarning):
                g.assert_camera_family_match(train, ev)  # no raise


# ---------------------------------------------------------------------------
# Eval seed file sha
# ---------------------------------------------------------------------------
class TestEvalSeedFile(unittest.TestCase):
    def test_match_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "seeds.txt"
            content = b"1\n2\n3\n"
            p.write_bytes(content)
            sha = hashlib.sha256(content).hexdigest()
            g.assert_eval_seed_file(p, sha)  # no raise

    def test_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "seeds.txt"
            p.write_bytes(b"1\n2\n3\n")
            with self.assertRaises(g.EvalSeedFileDriftError):
                g.assert_eval_seed_file(p, "0" * 64)

    def test_missing_seed_file_raises(self):
        with self.assertRaises(g.EvalSeedFileDriftError):
            g.assert_eval_seed_file(Path("/no/such/file/xxx"), "0" * 64)

    def test_empty_expected_falls_back_to_pinned(self):
        # The repo ships /iris/u/mikulrai/runs/eval_seeds_60.txt whose hash is
        # pinned in EVAL_SEEDS_60_SHA256. Verify that path verifies cleanly
        # via the empty-string fallback.
        canonical = Path("/iris/u/mikulrai/runs/eval_seeds_60.txt")
        if not canonical.exists():
            self.skipTest(f"canonical seed file not present: {canonical}")
        g.assert_eval_seed_file(canonical, "")  # no raise

    def test_load_pinned_sha256_strips_sha256sum_format(self):
        # `sha256sum file` writes "<hex>  <path>"; loader should accept that.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pin.sha256"
            p.write_text("abc123  /some/file\n")
            self.assertEqual(g._load_pinned_sha256(p), "abc123")

    def test_load_pinned_sha256_strips_bare_hex(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pin.sha256"
            p.write_text("abc123\n")
            self.assertEqual(g._load_pinned_sha256(p), "abc123")

    def test_load_pinned_sha256_missing_returns_empty(self):
        self.assertEqual(g._load_pinned_sha256(Path("/no/such/x")), "")
        self.assertEqual(g._load_pinned_sha256(None), "")


# ---------------------------------------------------------------------------
# Wandb live
# ---------------------------------------------------------------------------
class TestWandbLive(unittest.TestCase):
    def test_online_with_api_key_passes(self):
        env = {"WANDB_API_KEY": "abc"}
        g.assert_wandb_live(env)

    def test_offline_raises(self):
        env = {"WANDB_API_KEY": "abc", "WANDB_MODE": "offline"}
        with self.assertRaises(g.WandbOfflineError):
            g.assert_wandb_live(env)

    def test_disabled_raises(self):
        env = {"WANDB_API_KEY": "abc", "WANDB_DISABLED": "true"}
        with self.assertRaises(g.WandbOfflineError):
            g.assert_wandb_live(env)

    def test_missing_api_key_raises(self):
        env = {"WANDB_MODE": "online"}
        with self.assertRaises(g.WandbOfflineError):
            g.assert_wandb_live(env)

    def test_default_uses_os_environ(self):
        # When env arg is None, falls back to os.environ.
        with mock.patch.dict(
            os.environ,
            {"WANDB_API_KEY": "abc", "WANDB_MODE": "online", "WANDB_DISABLED": ""},
            clear=True,
        ):
            g.assert_wandb_live()  # no raise

    def test_project_probe_passes(self):
        # When `project` is given, wandb.Api().runs(project) is invoked. We
        # patch sys.modules so the import-inside-function returns a stub.
        env = {"WANDB_API_KEY": "abc"}
        fake_wandb = mock.MagicMock()
        fake_api = mock.MagicMock()
        fake_api.runs.return_value = iter([])
        fake_wandb.Api.return_value = fake_api
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            g.assert_wandb_live(env=env, project="entity/proj")
        fake_wandb.Api.assert_called_once()
        fake_api.runs.assert_called_once_with("entity/proj", per_page=1)

    def test_project_probe_unreachable_raises(self):
        env = {"WANDB_API_KEY": "abc"}
        fake_wandb = mock.MagicMock()
        fake_api = mock.MagicMock()
        fake_api.runs.side_effect = RuntimeError("404 not found")
        fake_wandb.Api.return_value = fake_api
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            with self.assertRaises(g.WandbOfflineError) as cm:
                g.assert_wandb_live(env=env, project="entity/missing")
            self.assertIn("unreachable", str(cm.exception))

    def test_api_construction_failure_raises(self):
        env = {"WANDB_API_KEY": "abc"}
        fake_wandb = mock.MagicMock()
        fake_wandb.Api.side_effect = RuntimeError("auth failure")
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            with self.assertRaises(g.WandbOfflineError) as cm:
                g.assert_wandb_live(env=env, project="entity/proj")
            self.assertIn("Api()", str(cm.exception))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCli(unittest.TestCase):
    @staticmethod
    def _run(argv: list[str], env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.setdefault("WANDB_API_KEY", "x")
        env["WANDB_MODE"] = "online"
        env.pop("WANDB_DISABLED", None)
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "robofactory.utils.preflight_eval_guards", *argv],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),  # RoboFactory repo root
            timeout=60,
        )

    def test_cli_help(self):
        r = self._run(["--help"])
        self.assertEqual(r.returncode, 0)
        self.assertIn("--train-cfg", r.stdout)
        self.assertIn("--eval-cfg", r.stdout)

    def test_cli_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p)
            ev = _make_eval_cfg(p)
            seeds = p / "seeds.txt"
            seeds.write_bytes(b"1\n2\n")
            sha = hashlib.sha256(seeds.read_bytes()).hexdigest()
            r = self._run([
                "--train-cfg", str(train),
                "--eval-cfg", str(ev),
                "--seed-file", str(seeds),
                "--expected-sha256", sha,
            ])
            self.assertEqual(r.returncode, 0, msg=f"stderr={r.stderr}\nstdout={r.stdout}")
            self.assertIn("all guards passed", r.stdout)

    def test_cli_pass_with_sha256_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p)
            ev = _make_eval_cfg(p)
            seeds = p / "seeds.txt"
            seeds.write_bytes(b"1\n2\n")
            sha = hashlib.sha256(seeds.read_bytes()).hexdigest()
            sha_file = p / "seeds.sha256"
            sha_file.write_text(f"{sha}  {seeds}\n")
            r = self._run([
                "--train-cfg", str(train),
                "--eval-cfg", str(ev),
                "--seed-file", str(seeds),
                "--expected-sha256-file", str(sha_file),
            ])
            self.assertEqual(r.returncode, 0, msg=f"stderr={r.stderr}\nstdout={r.stdout}")

    def test_cli_fail_offline_mode(self):
        # WANDB_MODE=offline should cause a non-zero exit with a clear message.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p)
            ev = _make_eval_cfg(p)
            seeds = p / "seeds.txt"
            seeds.write_bytes(b"1\n2\n")
            sha = hashlib.sha256(seeds.read_bytes()).hexdigest()
            r = self._run(
                [
                    "--train-cfg", str(train),
                    "--eval-cfg", str(ev),
                    "--seed-file", str(seeds),
                    "--expected-sha256", sha,
                ],
                env_extra={"WANDB_MODE": "offline"},
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("WandbOfflineError", r.stderr)

    def test_cli_fail_scene_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = _make_train_cfg(p, env_id="PickMeat-rf")
            ev = _make_eval_cfg(p, env_id="ThreeRobotsStackCube-rf")
            seeds = p / "seeds.txt"
            seeds.write_bytes(b"1\n2\n")
            sha = hashlib.sha256(seeds.read_bytes()).hexdigest()
            r = self._run([
                "--train-cfg", str(train),
                "--eval-cfg", str(ev),
                "--seed-file", str(seeds),
                "--expected-sha256", sha,
            ])
            self.assertEqual(r.returncode, 1)
            self.assertIn("SceneMismatchError", r.stderr)


if __name__ == "__main__":
    unittest.main()
