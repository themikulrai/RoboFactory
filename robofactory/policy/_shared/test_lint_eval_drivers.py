"""Tests for scripts/lint/check_eval_drivers.py.

The lint script ASTs each `eval_*.py` driver under `robofactory/policy/**/`
and asserts the call graph contains `with WandbRun(` (or `with EvalRunContext(`).
This guards against future drivers regressing back to bare `wandb.init` calls
without a context manager — the canonical cause of the C1 incident where
~30 GPU-hours of decent-pi05 eval ran with no wandb run at all.

Run from repo root:
    python -m pytest robofactory/policy/_shared/test_lint_eval_drivers.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LINT_SCRIPT = REPO_ROOT / "robofactory" / "scripts" / "lint" / "check_eval_drivers.py"
PYTHON = sys.executable


class TestLintCatchesMissingPattern(unittest.TestCase):
    """When an eval_*.py lacks `with WandbRun(`, the lint must fail."""

    def test_missing_pattern_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "robofactory" / "policy" / "fake"
            tree.mkdir(parents=True)
            bad = tree / "eval_bad.py"
            bad.write_text(textwrap.dedent("""\
                import wandb
                def main():
                    wandb.init(project='x')
                    wandb.finish()
                if __name__ == '__main__':
                    main()
                """))
            result = subprocess.run(
                [PYTHON, str(LINT_SCRIPT), "--root", td],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0,
                                f"expected non-zero exit; stdout={result.stdout!r} stderr={result.stderr!r}")
            combined = result.stdout + result.stderr
            self.assertIn("eval_bad.py", combined)


class TestLintPassesWithPattern(unittest.TestCase):
    """When every eval_*.py uses `with WandbRun(` or `with EvalRunContext(`,
    the lint must exit 0."""

    def test_with_wandbrun_pattern_exits_zero(self):
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "robofactory" / "policy" / "fake"
            tree.mkdir(parents=True)
            good = tree / "eval_good.py"
            good.write_text(textwrap.dedent("""\
                from policy._shared.eval_context import WandbRun
                def main():
                    with WandbRun(enabled=True, project='p',
                                  job_type='eval', name='n') as run:
                        run.log_summary(success_rate=1.0)
                if __name__ == '__main__':
                    main()
                """))
            result = subprocess.run(
                [PYTHON, str(LINT_SCRIPT), "--root", td],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"expected zero exit; stdout={result.stdout!r} stderr={result.stderr!r}")

    def test_with_eval_run_context_pattern_exits_zero(self):
        """`EvalRunContext` is the higher-level wrapper allowed by the lint."""
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "robofactory" / "policy" / "fake"
            tree.mkdir(parents=True)
            good = tree / "eval_alt.py"
            good.write_text(textwrap.dedent("""\
                from policy._shared.eval_context import EvalRunContext
                def main():
                    with EvalRunContext(run_id='x', run_dir='/tmp',
                                        project='p', tags=[],
                                        config={}) as (wr, vr):
                        pass
                if __name__ == '__main__':
                    main()
                """))
            result = subprocess.run(
                [PYTHON, str(LINT_SCRIPT), "--root", td],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"stdout={result.stdout!r} stderr={result.stderr!r}")


class TestLintRealRepo(unittest.TestCase):
    """Sanity: the actual repo's eval_*.py drivers must pass the lint right now."""

    def test_real_drivers_pass(self):
        # Restrict to robofactory/policy/ via --root pointing at repo root; the
        # script's own glob narrows from there.
        result = subprocess.run(
            [PYTHON, str(LINT_SCRIPT), "--root", str(REPO_ROOT)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"real-repo lint failed; stdout={result.stdout!r} stderr={result.stderr!r}")


class TestLintIgnoresEvalContextItself(unittest.TestCase):
    """The lint must not flag eval_context.py — it's the impl, not a driver."""

    def test_eval_context_module_not_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "robofactory" / "policy" / "_shared"
            tree.mkdir(parents=True)
            ec = tree / "eval_context.py"
            ec.write_text("class WandbRun: pass\n")
            result = subprocess.run(
                [PYTHON, str(LINT_SCRIPT), "--root", td],
                capture_output=True, text=True,
            )
            # No drivers found -> exit 0 (vacuously true).
            self.assertEqual(result.returncode, 0,
                             f"stdout={result.stdout!r} stderr={result.stderr!r}")


if __name__ == "__main__":
    unittest.main()
