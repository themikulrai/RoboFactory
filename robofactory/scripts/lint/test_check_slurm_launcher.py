"""Tests for scripts/lint/check_slurm_launcher.py.

Run from repo root via:
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
        -m pytest robofactory/scripts/lint/test_check_slurm_launcher.py -q
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

# Load the module by path because scripts/ isn't a package.
_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "check_slurm_launcher", _HERE / "check_slurm_launcher.py"
)
check_slurm_launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_slurm_launcher)


class TestRuleR1RelativeSource(unittest.TestCase):
    """The bug that ate today's S6 PM 7-second failure."""

    def test_relative_source_fails(self):
        text = 'source "$(dirname "$0")/_resolve_train_cfg.sh"'
        errs = check_slurm_launcher.rule_R1_no_relative_source(text)
        self.assertEqual(len(errs), 1)
        self.assertIn("relative `source", errs[0])

    def test_absolute_source_passes(self):
        text = "source /iris/u/mikulrai/projects/foo/bar.sh"
        errs = check_slurm_launcher.rule_R1_no_relative_source(text)
        self.assertEqual(errs, [])

    def test_unquoted_dirname_form_also_caught(self):
        text = "source $(dirname $0)/helper.sh"
        errs = check_slurm_launcher.rule_R1_no_relative_source(text)
        self.assertEqual(len(errs), 1)


class TestRuleR2WandbApiKey(unittest.TestCase):
    """The eval_decent_pi05 lapse class."""

    def test_export_present_passes(self):
        text = "export WANDB_API_KEY=abc123\nfoo"
        errs = check_slurm_launcher.rule_R2_wandb_api_key_exported(text, "pm_eval_in1k.sh")
        self.assertEqual(errs, [])

    def test_missing_export_fails(self):
        text = "echo hello\n"
        errs = check_slurm_launcher.rule_R2_wandb_api_key_exported(text, "pm_eval_in1k.sh")
        self.assertEqual(len(errs), 1)

    def test_calibration_jobs_skip_check(self):
        # Calibration jobs don't log to wandb.
        text = "echo hello\n"
        errs = check_slurm_launcher.rule_R2_wandb_api_key_exported(
            text, "pm_in1k_calibration_capture.sh"
        )
        self.assertEqual(errs, [])


class TestRuleR3AbsoluteOutputPaths(unittest.TestCase):
    def test_absolute_passes(self):
        text = "#SBATCH --output=/iris/u/mikulrai/logs/foo_%j.out\n"
        errs = check_slurm_launcher.rule_R3_absolute_output_paths(text)
        self.assertEqual(errs, [])

    def test_relative_fails(self):
        text = "#SBATCH --output=logs/foo_%j.out\n"
        errs = check_slurm_launcher.rule_R3_absolute_output_paths(text)
        self.assertEqual(len(errs), 1)

    def test_short_form_e_caught(self):
        text = "#SBATCH -e logs/foo.err\n"
        errs = check_slurm_launcher.rule_R3_absolute_output_paths(text)
        self.assertEqual(len(errs), 1)


class TestRuleR4PartitionGres(unittest.TestCase):
    def test_iris_hi_with_3_gpus_passes(self):
        text = "#SBATCH --partition=iris-hi\n#SBATCH --gres=gpu:a40:3\n"
        errs = check_slurm_launcher.rule_R4_partition_gres_match(text)
        self.assertEqual(errs, [])

    def test_iris_hi_with_8_gpus_fails(self):
        text = "#SBATCH --partition=iris-hi\n#SBATCH --gres=gpu:8\n"
        errs = check_slurm_launcher.rule_R4_partition_gres_match(text)
        self.assertEqual(len(errs), 1)
        self.assertIn("cap is 6", errs[0])

    def test_orion_with_a5000_passes(self):
        text = "#SBATCH --partition=orion\n#SBATCH --gres=gpu:a5000:1\n"
        errs = check_slurm_launcher.rule_R4_partition_gres_match(text)
        self.assertEqual(errs, [])

    def test_orion_with_h100_fails(self):
        text = "#SBATCH --partition=orion\n#SBATCH --gres=gpu:h100:1\n"
        errs = check_slurm_launcher.rule_R4_partition_gres_match(text)
        self.assertEqual(len(errs), 1)
        self.assertIn("forbidden", errs[0])

    def test_no_partition_directive_skipped(self):
        text = "#SBATCH --gres=gpu:1\n"
        errs = check_slurm_launcher.rule_R4_partition_gres_match(text)
        self.assertEqual(errs, [])


class TestLintOne(unittest.TestCase):
    """Integration: a complete launcher with multiple rule violations."""

    def _write(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, dir=tempfile.gettempdir()
        )
        f.write(content)
        f.close()
        self.addCleanup(lambda p=f.name: Path(p).unlink(missing_ok=True))
        return Path(f.name)

    def test_clean_launcher_no_errors(self):
        path = self._write(
            "#!/bin/bash\n"
            "#SBATCH --partition=iris\n"
            "#SBATCH --output=/iris/u/mikulrai/logs/foo_%j.out\n"
            "#SBATCH --gres=gpu:1\n"
            "export WANDB_API_KEY=abc\n"
            "source /abs/path.sh\n"
        )
        path = path.rename(path.parent / "pm_eval_in1k_60seeds.sh")
        self.addCleanup(lambda p=path: Path(p).unlink(missing_ok=True))
        errs = check_slurm_launcher.lint_one(path)
        self.assertEqual(errs, [])

    def test_buggy_launcher_lists_all_failures(self):
        # Today's bug + missing wandb + relative output + iris-hi over cap.
        path = self._write(
            "#!/bin/bash\n"
            "#SBATCH --partition=iris-hi\n"
            "#SBATCH --gres=gpu:8\n"
            "#SBATCH --output=logs/foo_%j.out\n"
            'source "$(dirname "$0")/helper.sh"\n'
        )
        path = path.rename(path.parent / "tsc_d2_eval_60seeds.sh")
        self.addCleanup(lambda p=path: Path(p).unlink(missing_ok=True))
        errs = check_slurm_launcher.lint_one(path)
        # 4 rule failures expected
        self.assertGreaterEqual(len(errs), 4)
        joined = "\n".join(errs)
        self.assertIn("R1_no_relative_source", joined)
        self.assertIn("R2_wandb_api_key", joined)
        self.assertIn("R3_absolute_output_paths", joined)
        self.assertIn("R4_partition_gres_match", joined)


class TestEndToEndAgainstRealLaunchers(unittest.TestCase):
    """The 15 canonical launchers must pass lint after the recent fixes."""

    def test_canonical_launchers_clean(self):
        root = Path(__file__).resolve().parents[3]
        launchers = check_slurm_launcher._collect_launchers(root)
        self.assertGreaterEqual(len(launchers), 10, "expected ≥10 canonical launchers")
        bad = []
        for p in launchers:
            errs = check_slurm_launcher.lint_one(p)
            if errs:
                bad.append((p.name, errs))
        if bad:
            msg = "\n".join(f"{n}: {e}" for n, e in bad)
            self.fail(f"canonical launchers failed lint:\n{msg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
