"""Tests for scripts/canonical/ and scripts/ablations/ launcher layout.

Verifies that the C1#7 launcher split keeps the directories well-formed:
- both dirs exist and contain the expected files
- every .sh file has a shebang
- every .sh file declares an SBATCH --job-name
- every .sh file passes `bash -n` syntax check
- both dirs ship a README.md indexing every launcher

Run from the robofactory env:
    python -m unittest robofactory.scripts.test_layout -v
or directly:
    python robofactory/scripts/test_layout.py
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / "canonical"
ABLATIONS = HERE / "ablations"

EXPECTED_CANONICAL = {
    "pm_eval_in1k_60seeds.sh",
    "resume_dp_tsc_d2_in1k_a0_from285.sh",
    "resume_dp_tsc_d2_in1k_a1_from285.sh",
    "resume_dp_tsc_d2_in1k_a2_from285.sh",
    "retrain_dp_pm_d1_ep300_in1k.sh",
    "retrain_dp_tsc_d2_ep300_in1k_a0.sh",
    "retrain_dp_tsc_d2_ep300_in1k_a1.sh",
    "retrain_dp_tsc_d2_ep300_in1k_a2.sh",
    "tsc_d1_eval_decent_in1k_60seeds.sh",
    "tsc_d2_wristcam_table_60seeds_reeval.sh",
}

EXPECTED_ABLATIONS = {
    "pm_eval_dino_blora_60seeds.sh",
    "pm_eval_dino_spatch_60seeds.sh",
    "pm_eval_in1k_crop_60seeds.sh",
    "pm_eval_r3m_60seeds.sh",
    "retrain_dp_pm_d1_ep300_crop.sh",
    "retrain_dp_pm_d1_ep300_dinov2_blora.sh",
    "retrain_dp_pm_d1_ep300_dinov2_spatch.sh",
    "retrain_dp_pm_d1_ep300_in1k_crop.sh",
    "retrain_dp_pm_d1_ep300_r3m.sh",
}


def _shells(dirpath: Path) -> set[str]:
    return {p.name for p in dirpath.glob("*.sh")}


class TestDirectoryLayout(unittest.TestCase):
    def test_canonical_dir_exists(self):
        self.assertTrue(CANONICAL.is_dir(), f"missing dir: {CANONICAL}")

    def test_ablations_dir_exists(self):
        self.assertTrue(ABLATIONS.is_dir(), f"missing dir: {ABLATIONS}")

    def test_canonical_has_expected_launchers(self):
        self.assertEqual(_shells(CANONICAL), EXPECTED_CANONICAL)

    def test_ablations_has_expected_launchers(self):
        self.assertEqual(_shells(ABLATIONS), EXPECTED_ABLATIONS)

    def test_canonical_readme_exists(self):
        self.assertTrue((CANONICAL / "README.md").is_file())

    def test_ablations_readme_exists(self):
        self.assertTrue((ABLATIONS / "README.md").is_file())


class TestLauncherWellFormedness(unittest.TestCase):
    """Every launcher in either dir must be a sane bash + sbatch script."""

    @classmethod
    def setUpClass(cls):
        cls.all_launchers = sorted(
            list(CANONICAL.glob("*.sh")) + list(ABLATIONS.glob("*.sh"))
        )

    def test_has_shebang(self):
        for p in self.all_launchers:
            with self.subTest(script=p.name):
                first_line = p.read_text().splitlines()[0]
                self.assertTrue(
                    first_line.startswith("#!"),
                    f"{p.name} missing shebang (first line: {first_line!r})",
                )

    def test_declares_job_name(self):
        pat = re.compile(r"^#SBATCH\s+--job-name=", re.MULTILINE)
        for p in self.all_launchers:
            with self.subTest(script=p.name):
                self.assertRegex(
                    p.read_text(),
                    pat,
                    f"{p.name} has no #SBATCH --job-name= line",
                )

    def test_bash_syntax_check(self):
        for p in self.all_launchers:
            with self.subTest(script=p.name):
                result = subprocess.run(
                    ["bash", "-n", str(p)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{p.name} failed bash -n: {result.stderr}",
                )


class TestEvalLaunchersHavePreflight(unittest.TestCase):
    """Stage-3 per-eval guards (plan v2 C1#10): every eval launcher in the
    canonical/ablations dirs must invoke robofactory.utils.preflight_eval
    before its eval entrypoint, so a scene-mismatch / wandb-offline / missing
    ckpt aborts the SLURM job before any GPU work."""

    def _eval_launchers(self) -> list[Path]:
        return list(CANONICAL.glob("*eval*.sh")) + list(ABLATIONS.glob("*eval*.sh"))

    def test_each_eval_launcher_invokes_preflight(self):
        for p in self._eval_launchers():
            with self.subTest(launcher=p.name):
                content = p.read_text()
                self.assertIn(
                    "robofactory.utils.preflight_eval",
                    content,
                    f"{p.name} does not invoke preflight_eval",
                )

    def test_preflight_runs_before_eval_entrypoint(self):
        # The preflight call must appear BEFORE any python invocation of
        # eval_dp.py / eval_multi_dp.py / eval_decent_pi05.py — otherwise it
        # can't gate compute.
        eval_entrypoint_re = re.compile(
            r"python\s.*\beval_(?:dp|multi_dp|decent_pi05)\.py", re.MULTILINE
        )
        preflight_re = re.compile(
            r"python\s.*\brobofactory\.utils\.preflight_eval", re.MULTILINE
        )
        for p in self._eval_launchers():
            with self.subTest(launcher=p.name):
                content = p.read_text()
                pf = preflight_re.search(content)
                ee = eval_entrypoint_re.search(content)
                self.assertIsNotNone(pf, f"{p.name} no preflight call")
                self.assertIsNotNone(ee, f"{p.name} no eval entrypoint")
                self.assertLess(
                    pf.start(), ee.start(),
                    f"{p.name}: preflight comes after eval entrypoint",
                )


class TestReadmeIndexesEveryLauncher(unittest.TestCase):
    """The README in each dir must mention every launcher in that dir."""

    def test_canonical_readme_indexes_all_launchers(self):
        readme = (CANONICAL / "README.md").read_text()
        for name in EXPECTED_CANONICAL:
            with self.subTest(launcher=name):
                self.assertIn(
                    name,
                    readme,
                    f"canonical/README.md does not mention {name}",
                )

    def test_ablations_readme_indexes_all_launchers(self):
        readme = (ABLATIONS / "README.md").read_text()
        for name in EXPECTED_ABLATIONS:
            with self.subTest(launcher=name):
                self.assertIn(
                    name,
                    readme,
                    f"ablations/README.md does not mention {name}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
