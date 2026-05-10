"""C2-S2 default-on video smoke test.

Plan section "Phase C2 — Detailed Plan" S2 specifies:
    Default-on video → first-3-seeds-per-task only, full video on `--video-all`.
    Disk math: 60 seeds × ~50 MB × N evals/wk = unsustainable on /iris.

Existing `test_eval_context.py::TestVideoRecorder` already covers the cap and
override behavior, but it always passes `max_recorded=3` explicitly. This file
asserts the *constructor defaults* themselves so a future refactor that changes
`max_recorded`'s default value can't silently regress disk usage.

Run:
    cd /iris/u/mikulrai/projects/RoboFactory/robofactory
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
        -m pytest policy/_shared/test_video_recorder_defaults.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Allow running directly from repo root (mirrors test_eval_context.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy._shared.eval_context import VideoRecorder  # noqa: E402


class TestVideoRecorderDefaults(unittest.TestCase):
    def test_default_max_recorded_is_3(self):
        """Default cap must be 3 — see C2-S2 disk-math rationale."""
        v = VideoRecorder(record_dir="/tmp/does_not_matter")
        self.assertEqual(v.max_recorded, 3)
        self.assertIs(v.all_seeds, False)

    def test_max_recorded_kwarg_overrides(self):
        v = VideoRecorder(record_dir="/tmp/does_not_matter", max_recorded=10)
        self.assertEqual(v.max_recorded, 10)
        self.assertIs(v.all_seeds, False)

    def test_all_seeds_kwarg_overrides(self):
        v = VideoRecorder(
            record_dir="/tmp/does_not_matter", max_recorded=3, all_seeds=True
        )
        self.assertIs(v.all_seeds, True)

    def test_only_three_writes_over_60_seeds(self):
        """Simulate a 60-seed eval; mp4 writer should be invoked exactly 3 times.

        We emulate the driver pattern: for each seed, call `video_path_for(idx, seed)`
        and only write when the returned path is not None. This is the same gate
        every eval driver uses, so this test guards the integration contract.
        """
        with tempfile.TemporaryDirectory() as td:
            with VideoRecorder(record_dir=td) as v:
                writer = mock.MagicMock()
                for seed_idx in range(60):
                    seed = 100 + seed_idx
                    path = v.video_path_for(seed_idx, seed)
                    if path is not None:
                        # Stand-in for `imageio.mimsave(path, frames)` etc.
                        writer(path)
            self.assertEqual(writer.call_count, 3)
            # First three seed indices wrote; rest didn't.
            written_paths = [c.args[0] for c in writer.call_args_list]
            self.assertTrue(all(p.endswith(".mp4") for p in written_paths))
            self.assertTrue(any("seed0000100" in p for p in written_paths))
            self.assertTrue(any("seed0000102" in p for p in written_paths))

    def test_all_seeds_writes_all_60(self):
        """Sanity check the override path: all_seeds=True bypasses the cap."""
        with tempfile.TemporaryDirectory() as td:
            with VideoRecorder(record_dir=td, all_seeds=True) as v:
                writer = mock.MagicMock()
                for seed_idx in range(60):
                    path = v.video_path_for(seed_idx, 100 + seed_idx)
                    if path is not None:
                        writer(path)
            self.assertEqual(writer.call_count, 60)


if __name__ == "__main__":
    unittest.main()
