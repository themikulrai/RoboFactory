"""Tests for policy._shared.eval_context.

Run from the robofactory env:
    python -m unittest policy._shared.test_eval_context -v
or directly:
    python policy/_shared/test_eval_context.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# allow running directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy._shared import eval_context  # noqa: E402


class TestAssertWandbLive(unittest.TestCase):
    def test_offline_mode_raises(self):
        with mock.patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                eval_context.assert_wandb_live()
            self.assertIn("WANDB_MODE", str(ctx.exception))

    def test_disabled_raises(self):
        env = dict(os.environ)
        env.pop("WANDB_MODE", None)
        env["WANDB_DISABLED"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                eval_context.assert_wandb_live()
            self.assertIn("WANDB_DISABLED", str(ctx.exception))

    def test_no_credentials_raises(self):
        env = {"HOME": "/nonexistent_home_12345"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                eval_context.assert_wandb_live()
            self.assertIn("WANDB_API_KEY", str(ctx.exception))


class TestWandbRunDisabled(unittest.TestCase):
    """`enabled=False` short-circuits to a no-op so existing CLI flow keeps working."""

    def test_disabled_does_not_init(self):
        with mock.patch("wandb.init") as init_mock:
            with eval_context.WandbRun(
                enabled=False, project="x", job_type="eval", name="n"
            ) as run:
                run.log_episode(0, success=1)
                run.log_summary(success_rate=0.5)
        init_mock.assert_not_called()

    def test_disabled_yields_self(self):
        with eval_context.WandbRun(
            enabled=False, project="x", job_type="eval", name="n"
        ) as run:
            self.assertIsInstance(run, eval_context.WandbRun)
            self.assertIsNone(run.run)


class TestWandbRunEnabled(unittest.TestCase):
    """When enabled=True, init must be called and a None return must crash."""

    def _live_env(self):
        return mock.patch.dict(
            os.environ, {"WANDB_API_KEY": "fake-key-for-test"}, clear=False
        )

    def test_init_called_with_correct_kwargs(self):
        fake_run = mock.MagicMock()
        with self._live_env(), mock.patch("wandb.init", return_value=fake_run) as init_mock, mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True,
                project="proj",
                job_type="eval",
                name="run-name",
                tags=["a", "b"],
                config={"x": 1},
            ):
                pass
        init_mock.assert_called_once()
        kwargs = init_mock.call_args.kwargs
        self.assertEqual(kwargs["project"], "proj")
        self.assertEqual(kwargs["name"], "run-name")
        self.assertEqual(kwargs["tags"], ["a", "b"])
        self.assertEqual(kwargs["config"], {"x": 1})

    def test_init_returning_none_raises(self):
        with self._live_env(), mock.patch("wandb.init", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                with eval_context.WandbRun(
                    enabled=True, project="p", job_type="eval", name="n"
                ):
                    pass
            self.assertIn("None", str(ctx.exception))

    def test_finish_called_on_exit_clean(self):
        fake_run = mock.MagicMock()
        with self._live_env(), mock.patch("wandb.init", return_value=fake_run), mock.patch("wandb.finish") as finish_mock:
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n"
            ):
                pass
        finish_mock.assert_called_once_with(exit_code=0)

    def test_finish_called_on_exception(self):
        fake_run = mock.MagicMock()
        with self._live_env(), mock.patch("wandb.init", return_value=fake_run), mock.patch("wandb.finish") as finish_mock:
            with self.assertRaises(ValueError):
                with eval_context.WandbRun(
                    enabled=True, project="p", job_type="eval", name="n"
                ):
                    raise ValueError("rollout crashed")
        finish_mock.assert_called_once_with(exit_code=1)

    def test_log_episode_routes_to_wandb_log(self):
        fake_run = mock.MagicMock()
        with self._live_env(), mock.patch("wandb.init", return_value=fake_run), mock.patch("wandb.log") as log_mock, mock.patch("wandb.finish"):
            with eval_context.WandbRun(
                enabled=True, project="p", job_type="eval", name="n"
            ) as r:
                r.log_episode(7, success=1, steps=42)
        log_mock.assert_called_once_with(
            {"episode/idx": 7, "episode/success": 1, "episode/steps": 42}
        )


class TestVideoRecorder(unittest.TestCase):
    def test_creates_dir_on_enter(self):
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "videos")
            self.assertFalse(os.path.exists(target))
            with eval_context.VideoRecorder(target):
                pass
            self.assertTrue(os.path.exists(target))

    def test_caps_at_max_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            with eval_context.VideoRecorder(td, max_recorded=3) as v:
                self.assertIsNotNone(v.video_path_for(0, 1000))
                self.assertIsNotNone(v.video_path_for(1, 1001))
                self.assertIsNotNone(v.video_path_for(2, 1002))
                self.assertIsNone(v.video_path_for(3, 1003))
                self.assertIsNone(v.video_path_for(59, 9999))

    def test_all_seeds_overrides_cap(self):
        with tempfile.TemporaryDirectory() as td:
            with eval_context.VideoRecorder(td, max_recorded=3, all_seeds=True) as v:
                self.assertIsNotNone(v.video_path_for(59, 12345))

    def test_empty_record_dir_returns_none(self):
        with eval_context.VideoRecorder("", max_recorded=3) as v:
            self.assertIsNone(v.video_path_for(0, 1))

    def test_path_format_includes_seed(self):
        with tempfile.TemporaryDirectory() as td:
            with eval_context.VideoRecorder(td, max_recorded=3) as v:
                p = v.video_path_for(0, 1234567, suffix="_global")
                self.assertTrue(p.endswith("seed1234567_global.mp4"))


class TestDeriveTagsFromName(unittest.TestCase):
    """Workflow improvement #3 — tags inferred from run name at init time."""

    def test_dp_pm_workspace_train(self):
        tags = eval_context._derive_tags_from_name("dp_ws_pm_in1k_cent_train_s100_a3c736d")
        self.assertIn("dp", tags)
        self.assertIn("ws", tags)
        self.assertIn("pm", tags)
        self.assertIn("in1k", tags)
        self.assertIn("cent", tags)
        self.assertIn("train", tags)

    def test_pi05_decent_tsc_eval(self):
        tags = eval_context._derive_tags_from_name(
            "pi05_wc_tsc_decent_eval_c19999_s100_a3c736d"
        )
        self.assertIn("pi05", tags)
        self.assertIn("wc", tags)
        self.assertIn("tsc", tags)
        self.assertIn("decent", tags)
        self.assertIn("eval", tags)

    def test_substring_does_not_falsely_match(self):
        # "dp" should not match arbitrary "depth"-like words; word-boundary check.
        tags = eval_context._derive_tags_from_name("depthsensor_predefined_run")
        self.assertNotIn("dp", tags)
        self.assertNotIn("pm", tags)

    def test_legacy_dp_dirname_style(self):
        # Existing on-disk DP run dirs follow 'PickMeat-rf_150' / 'ThreeRobotsStackCube-rf_agent0_d2_wristcam_150'
        # — derived tags should still surface useful tokens.
        tags = eval_context._derive_tags_from_name(
            "ThreeRobotsStackCube-rf_agent0_d2_wristcam_150"
        )
        self.assertIn("agent0", tags)
        self.assertIn("d2", tags)
        self.assertIn("wristcam", tags)

    def test_freezeenc_token(self):
        tags = eval_context._derive_tags_from_name(
            "ThreeRobotsStackCube-rf_agent1_freezeenc_150"
        )
        self.assertIn("freezeenc", tags)
        self.assertIn("agent1", tags)

    def test_empty_name_returns_empty(self):
        self.assertEqual(eval_context._derive_tags_from_name(""), [])
        self.assertEqual(eval_context._derive_tags_from_name(None or ""), [])

    def test_tags_sorted_dedup(self):
        # Same token appearing twice (eval_eval_pm) shouldn't show twice.
        tags = eval_context._derive_tags_from_name("eval_eval_pm_pm")
        self.assertEqual(tags, sorted(tags))
        self.assertEqual(tags.count("eval"), 1)
        self.assertEqual(tags.count("pm"), 1)

    def test_explicit_tags_merge_with_derived_in_wandbrun(self):
        # Run is disabled so we can introspect self.tags without hitting wandb.
        run = eval_context.WandbRun(
            enabled=False,
            project="diffusion-robofactory",
            job_type="eval",
            name="dp_ws_pm_in1k_cent_eval_c300_s100",
            tags=["custom-tag", "in1k"],  # in1k overlaps; 'custom-tag' is novel
        )
        self.assertIn("custom-tag", run.tags)
        self.assertIn("dp", run.tags)
        self.assertIn("eval", run.tags)
        self.assertIn("in1k", run.tags)
        # No duplicates.
        self.assertEqual(len(run.tags), len(set(run.tags)))


if __name__ == "__main__":
    unittest.main()
