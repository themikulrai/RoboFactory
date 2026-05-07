"""Tests for script/lint/check_eval_drivers.py.

Run:
    python -m unittest script.lint.test_check_eval_drivers -v
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

# allow running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestLintCheck(unittest.TestCase):
    def setUp(self):
        # import fresh so we can monkeypatch REPO_ROOT per test
        if "script.lint.check_eval_drivers" in sys.modules:
            del sys.modules["script.lint.check_eval_drivers"]
        from script.lint import check_eval_drivers as mod
        self.mod = mod

    def _make_driver(self, dirpath: Path, name: str, body: str):
        path = dirpath / "policy" / "fake" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
        return path

    def test_passes_when_all_drivers_use_wandbrun(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._make_driver(tdp, "eval_a.py", "with WandbRun(enabled=True): pass\n")
            self._make_driver(tdp, "eval_b.py", "with WandbRun(\n    enabled=True): pass\n")
            with mock.patch.object(self.mod, "REPO_ROOT", tdp):
                rc = self.mod.main()
            self.assertEqual(rc, 0)

    def test_fails_when_a_driver_misses_wandbrun(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._make_driver(tdp, "eval_good.py", "with WandbRun(enabled=True): pass\n")
            self._make_driver(tdp, "eval_bad.py", "import wandb\nwandb.init()\n")
            with mock.patch.object(self.mod, "REPO_ROOT", tdp):
                rc = self.mod.main()
            self.assertEqual(rc, 1)

    def test_eval_context_is_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # eval_context.py itself does NOT contain `with WandbRun(` — it defines it
            self._make_driver(tdp, "eval_context.py", "class WandbRun: pass\n")
            self._make_driver(tdp, "eval_real.py", "with WandbRun(enabled=True): pass\n")
            with mock.patch.object(self.mod, "REPO_ROOT", tdp):
                rc = self.mod.main()
            self.assertEqual(rc, 0)

    def test_no_drivers_found_is_failure(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            with mock.patch.object(self.mod, "REPO_ROOT", tdp):
                rc = self.mod.main()
            self.assertEqual(rc, 1)


class TestLintCheckOnRealRepo(unittest.TestCase):
    """Smoke test: actual repo passes."""

    def test_repo_passes(self):
        if "script.lint.check_eval_drivers" in sys.modules:
            del sys.modules["script.lint.check_eval_drivers"]
        from script.lint import check_eval_drivers as mod
        self.assertEqual(mod.main(), 0)


if __name__ == "__main__":
    unittest.main()
