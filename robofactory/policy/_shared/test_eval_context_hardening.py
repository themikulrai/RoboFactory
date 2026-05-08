"""Adversarial tests for policy._shared.eval_context.

Beyond the happy-path tests in `test_eval_context.py`, these stress:
- assert_wandb_live positive case (valid setup → no exception)
- log_summary / log_raw routing correctness
- post-exit logging is silent (run is None)
- defensive copies of tags/config
- assert_wandb_live failure prevents wandb.init from being called
- VideoRecorder edge cases (max_recorded=0, large seed_idx)

Run:
    python -m unittest policy._shared.test_eval_context_hardening -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy._shared import eval_context  # noqa: E402


class TestAssertWandbLivePositive(unittest.TestCase):
    def test_valid_env_with_api_key_passes(self):
        env = {"WANDB_API_KEY": "fake-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            # No exception — positive case the existing tests miss.
            eval_context.assert_wandb_live()

    def test_valid_env_with_netrc_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".netrc") as netrc_file:
            netrc_file.write(b"machine api.wandb.ai login user password p\n")
            netrc_file.flush()
            env = {"HOME": str(Path(netrc_file.name).parent)}
            # Move the netrc to the fake HOME and rename to .netrc
            fake_home = Path(env["HOME"])
            real_netrc = fake_home / ".netrc"
            try:
                real_netrc.write_text("machine api.wandb.ai\n")
                with mock.patch.dict(os.environ, env, clear=True):
                    eval_context.assert_wandb_live()
            finally:
                if real_netrc.is_file():
                    real_netrc.unlink()

    def test_explicit_online_mode_passes(self):
        env = {"WANDB_MODE": "online", "WANDB_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            eval_context.assert_wandb_live()

    def test_disabled_mode_string_raises(self):
        """WANDB_MODE='disabled' is the third documented value besides 'offline'."""
        with mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
            with self.assertRaises(RuntimeError):
                eval_context.assert_wandb_live()


class TestWandbRunGuards(unittest.TestCase):
    def _live_env(self):
        return mock.patch.dict(os.environ, {"WANDB_API_KEY": "k"}, clear=False)

    def test_assert_failure_prevents_init_call(self):
        """If assert_wandb_live raises, wandb.init must NOT be called."""
        env = {}  # no key, no netrc → assert raises
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("os.path.exists", return_value=False), \
             mock.patch("wandb.init") as init_mock:
            with self.assertRaises(RuntimeError):
                with eval_context.WandbRun(
                    enabled=True, project="p", job_type="eval", name="n"
                ):
                    self.fail("body should not execute")
        init_mock.assert_not_called()

    def test_tags_none_does_not_crash(self):
        with self._live_env(), mock.patch("wandb.init", return_value=mock.MagicMock()) as init_mock, \
             mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n", tags=None
            ):
                pass
        self.assertEqual(init_mock.call_args.kwargs["tags"], [])

    def test_config_none_does_not_crash(self):
        with self._live_env(), mock.patch("wandb.init", return_value=mock.MagicMock()) as init_mock, \
             mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n", config=None
            ):
                pass
        self.assertEqual(init_mock.call_args.kwargs["config"], {})

    def test_tags_defensive_copy(self):
        """Mutating caller-provided tags after construction must not affect run."""
        tags = ["a"]
        run = eval_context.WandbRun(
            enabled=False, project="p", job_type="eval", name="n", tags=tags
        )
        tags.append("b")
        self.assertEqual(run.tags, ["a"])

    def test_config_defensive_copy(self):
        cfg = {"x": 1}
        run = eval_context.WandbRun(
            enabled=False, project="p", job_type="eval", name="n", config=cfg
        )
        cfg["x"] = 999
        self.assertEqual(run.config, {"x": 1})


class TestLoggingMethods(unittest.TestCase):
    def _live_env(self):
        return mock.patch.dict(os.environ, {"WANDB_API_KEY": "k"}, clear=False)

    def test_log_summary_routes_correctly(self):
        with self._live_env(), \
             mock.patch("wandb.init", return_value=mock.MagicMock()), \
             mock.patch("wandb.log") as log_mock, \
             mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n"
            ) as r:
                r.log_summary(success_rate=0.5, n_episodes=60)
        log_mock.assert_called_once_with(
            {"summary/success_rate": 0.5, "summary/n_episodes": 60}
        )

    def test_log_raw_bypasses_prefix(self):
        with self._live_env(), \
             mock.patch("wandb.init", return_value=mock.MagicMock()), \
             mock.patch("wandb.log") as log_mock, \
             mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n"
            ) as r:
                r.log_raw({"custom/key": 42, "another": 3.14})
        log_mock.assert_called_once_with({"custom/key": 42, "another": 3.14})

    def test_log_after_exit_silent(self):
        """Logging after `with` block exited must be a silent no-op."""
        with self._live_env(), \
             mock.patch("wandb.init", return_value=mock.MagicMock()), \
             mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n"
            ) as r:
                pass
            with mock.patch("wandb.log") as log_mock:
                r.log_episode(0, success=1)
                r.log_summary(success_rate=0.5)
                r.log_raw({"k": 1})
            log_mock.assert_not_called()


class TestVideoRecorderEdges(unittest.TestCase):
    def test_max_recorded_zero_returns_none_for_all(self):
        with tempfile.TemporaryDirectory() as td:
            with eval_context.VideoRecorder(td, max_recorded=0) as v:
                self.assertIsNone(v.video_path_for(0, 100))
                self.assertIsNone(v.video_path_for(99, 9999))

    def test_max_recorded_one(self):
        with tempfile.TemporaryDirectory() as td:
            with eval_context.VideoRecorder(td, max_recorded=1) as v:
                self.assertIsNotNone(v.video_path_for(0, 100))
                self.assertIsNone(v.video_path_for(1, 101))

    def test_path_format_with_default_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            with eval_context.VideoRecorder(td, max_recorded=3) as v:
                p = v.video_path_for(0, 42)
                self.assertTrue(p.endswith("seed0000042.mp4"))
                self.assertNotIn("__", Path(p).name)  # no double-underscore


class TestExitCodeOnException(unittest.TestCase):
    def _live_env(self):
        return mock.patch.dict(os.environ, {"WANDB_API_KEY": "k"}, clear=False)

    def test_exit_does_not_suppress_exception(self):
        """Even though we call wandb.finish(exit_code=1), the original
        exception must propagate up (return False from __exit__)."""
        with self._live_env(), \
             mock.patch("wandb.init", return_value=mock.MagicMock()), \
             mock.patch("wandb.finish"):
            with self.assertRaises(RuntimeError) as ctx:
                with eval_context.WandbRun(
                    enabled=True, project="p", job_type="eval", name="n"
                ):
                    raise RuntimeError("rollout died")
            self.assertEqual(str(ctx.exception), "rollout died")


if __name__ == "__main__":
    unittest.main()
