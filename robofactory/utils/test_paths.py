"""Tests for robofactory.utils.paths.

Run from the robofactory env:
    python -m unittest robofactory.utils.test_paths -v
or directly:
    python robofactory/utils/test_paths.py
"""
from __future__ import annotations

import datetime
import unittest
from pathlib import Path

from robofactory.utils import paths


class TestPathConstants(unittest.TestCase):
    def test_root_constants_are_paths(self):
        self.assertIsInstance(paths.RUN_ROOT, Path)
        self.assertIsInstance(paths.DEBUG_OUTPUT_ROOT, Path)
        self.assertIsInstance(paths.MANIFEST_PATH, Path)

    def test_root_constants_are_under_user_home(self):
        self.assertEqual(paths.RUN_ROOT, Path("/iris/u/mikulrai/runs"))
        self.assertEqual(paths.DEBUG_OUTPUT_ROOT, Path("/iris/u/mikulrai/debug_output"))
        self.assertEqual(paths.MANIFEST_PATH, paths.RUN_ROOT / "manifest.csv")


class TestDebugOutputDir(unittest.TestCase):
    def test_default_uses_today(self):
        today_str = datetime.date.today().strftime("%Y%m%d")
        out = paths.debug_output_dir("foo.py")
        self.assertEqual(out, paths.DEBUG_OUTPUT_ROOT / f"foo_{today_str}")

    def test_explicit_date(self):
        out = paths.debug_output_dir("foo.py", date="20990131")
        self.assertEqual(out, paths.DEBUG_OUTPUT_ROOT / "foo_20990131")

    def test_strips_directories(self):
        out = paths.debug_output_dir(
            "script/debug/dump_seed_poses.py", date="20260507"
        )
        self.assertEqual(
            out, paths.DEBUG_OUTPUT_ROOT / "dump_seed_poses_20260507"
        )

    def test_accepts_path_object(self):
        p = Path("script/debug/inspect_arm2_demos.py")
        out = paths.debug_output_dir(p, date="20260507")
        self.assertEqual(
            out, paths.DEBUG_OUTPUT_ROOT / "inspect_arm2_demos_20260507"
        )

    def test_strips_py_and_sh_extensions_consistently(self):
        out_py = paths.debug_output_dir("foo.py", date="20260507")
        out_sh = paths.debug_output_dir("foo.sh", date="20260507")
        self.assertEqual(out_py, out_sh)
        self.assertEqual(out_py, paths.DEBUG_OUTPUT_ROOT / "foo_20260507")

    def test_only_strips_last_extension(self):
        out = paths.debug_output_dir("a.b.c.py", date="20260507")
        self.assertEqual(out.name, "a.b.c_20260507")

    def test_simple_stem(self):
        out = paths.debug_output_dir("a", date="20260507")
        self.assertEqual(out.name, "a_20260507")


if __name__ == "__main__":
    unittest.main(verbosity=2)
