"""Tests for robofactory.utils.manifest.

Run from the robofactory env:
    python -m unittest robofactory.utils.test_manifest -v
or directly:
    python robofactory/utils/test_manifest.py
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from robofactory.utils import manifest


class TestCanonicalRunIdParser(unittest.TestCase):
    def test_dp_ws_pm_train(self):
        d = manifest.parse_canonical_run_id("dp_ws_pm_in1k_cent_train_s100_a3c736d")
        assert d is not None
        self.assertEqual(d["model"], "dp")
        self.assertEqual(d["dataset"], "ws")
        self.assertEqual(d["task"], "pm")
        self.assertEqual(d["encoder"], "in1k")
        self.assertEqual(d["scheme"], "cent")
        self.assertEqual(d["arm"], "")
        self.assertEqual(d["phase"], "train")
        self.assertEqual(d["seed"], "100")
        self.assertEqual(d["git_sha"], "a3c736d")

    def test_dp_ws_pm_eval_with_step(self):
        d = manifest.parse_canonical_run_id("dp_ws_pm_in1k_cent_eval_c30000_s100_a3c736d")
        assert d is not None
        self.assertEqual(d["phase"], "eval")
        self.assertEqual(d["step"], "30000")

    def test_pi05_wc_tsc_decent_arm0_train(self):
        d = manifest.parse_canonical_run_id("pi05_wc_tsc_none_decent_arm0_train_s100_a3c736d")
        assert d is not None
        self.assertEqual(d["model"], "pi05")
        self.assertEqual(d["dataset"], "wc")
        self.assertEqual(d["task"], "tsc")
        self.assertEqual(d["encoder"], "none")
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "0")
        self.assertEqual(d["phase"], "train")

    def test_pi05_wc_tsc_decent_eval_no_arm(self):
        # Decent eval has NO arm tag (eval fans out across all 3 arms).
        d = manifest.parse_canonical_run_id("pi05_wc_tsc_none_decent_eval_c19999_s100_a3c736d")
        assert d is not None
        self.assertEqual(d["phase"], "eval")
        self.assertEqual(d["arm"], "")
        self.assertEqual(d["step"], "19999")

    def test_encoder_with_dash(self):
        d = manifest.parse_canonical_run_id(
            "dp_ws_pm_dino-s-lora_cent_train_s100_a3c736d"
        )
        assert d is not None
        self.assertEqual(d["encoder"], "dino-s-lora")

    def test_in1k_crop_encoder(self):
        d = manifest.parse_canonical_run_id(
            "dp_ws_pm_in1k-crop_cent_train_s100_a3c736d"
        )
        assert d is not None
        self.assertEqual(d["encoder"], "in1k-crop")

    def test_rejects_unknown_model(self):
        self.assertIsNone(manifest.parse_canonical_run_id(
            "rt2_ws_pm_in1k_cent_train_s100_a3c736d"
        ))

    def test_rejects_short_sha(self):
        self.assertIsNone(manifest.parse_canonical_run_id(
            "dp_ws_pm_in1k_cent_train_s100_a3c73"
        ))

    def test_rejects_legacy_dirname(self):
        self.assertIsNone(manifest.parse_canonical_run_id(
            "PickMeat-rf_150"
        ))


class TestLegacyDpDirParser(unittest.TestCase):
    def test_pickmeat_150(self):
        d = manifest.parse_legacy_dp_dirname("PickMeat-rf_150")
        self.assertEqual(d["model"], "dp")
        self.assertEqual(d["task"], "pm")
        # No scheme/dataset/encoder inferable from this minimal name
        self.assertNotIn("scheme", d)

    def test_pickmeat_in1k_crop(self):
        d = manifest.parse_legacy_dp_dirname("PickMeat-rf_150_in1k_crop")
        self.assertEqual(d["task"], "pm")
        self.assertEqual(d["encoder"], "in1k-crop")

    def test_tsc_decent_d2_wristcam(self):
        d = manifest.parse_legacy_dp_dirname(
            "ThreeRobotsStackCube-rf_agent0_d2_wristcam_150"
        )
        self.assertEqual(d["task"], "tsc")
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "0")
        self.assertEqual(d["dataset"], "wc")

    def test_tsc_cent_d1_workspace_in1k(self):
        d = manifest.parse_legacy_dp_dirname(
            "ThreeRobotsStackCube-rf_joint_d1_workspace_150_in1k"
        )
        self.assertEqual(d["task"], "tsc")
        self.assertEqual(d["scheme"], "cent")
        self.assertEqual(d["dataset"], "ws")
        self.assertEqual(d["encoder"], "in1k")

    def test_tsc_decent_freezeenc(self):
        d = manifest.parse_legacy_dp_dirname(
            "ThreeRobotsStackCube-rf_agent2_freezeenc_150"
        )
        self.assertEqual(d["task"], "tsc")
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "2")
        self.assertEqual(d["encoder"], "in1k")  # freezeenc = frozen in1k

    def test_unknown_prefix(self):
        d = manifest.parse_legacy_dp_dirname("RandomDirName")
        self.assertEqual(d["model"], "dp")
        self.assertNotIn("task", d)


class TestPi05ConfigNameParser(unittest.TestCase):
    def test_pm_lora_finetune(self):
        d = manifest._parse_pi05_config_name("pi05_robofactory_pm_lora_finetune")
        self.assertEqual(d["task"], "pm")
        self.assertEqual(d["dataset"], "ws")  # default
        self.assertEqual(d["scheme"], "cent")

    def test_decent_arm1(self):
        d = manifest._parse_pi05_config_name(
            "pi05_robofactory_decent_lora_finetune_arm1"
        )
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "1")
        self.assertEqual(d["dataset"], "ws")

    def test_wristcam_decent_arm0(self):
        d = manifest._parse_pi05_config_name(
            "pi05_robofactory_decent_wristcam_lora_finetune_arm0"
        )
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "0")
        self.assertEqual(d["dataset"], "wc")

    def test_libero_marked_ablation(self):
        d = manifest._parse_pi05_config_name("pi05_libero_spatial_lora")
        self.assertEqual(d["task"], "libero")
        self.assertEqual(d["category"], "ablation")


class TestWalkDpCheckpoints(unittest.TestCase):
    def test_walks_legacy_dp_dir_with_two_ckpts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "PickMeat-rf_150_in1k"
            run_dir.mkdir()
            (run_dir / "100.ckpt").write_text("")
            (run_dir / "300.ckpt").write_text("")
            (run_dir / "junk.txt").write_text("")  # ignored

            recs = list(manifest.walk_dp_checkpoints(root))
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual(r.model, "dp")
            self.assertEqual(r.task, "pm")
            self.assertEqual(r.encoder, "in1k")
            self.assertEqual(r.step, "300")
            self.assertEqual(r.status, "done")
            # ckpt_paths stores the LATEST ckpt only; the count goes in notes.
            self.assertEqual(r.ckpt_paths, str(run_dir / "300.ckpt"))
            self.assertIn("2 step ckpts", r.notes)

    def test_skips_dirs_with_no_ckpts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "EmptyDir").mkdir()
            (root / "EmptyDir" / "config.yaml").write_text("")
            self.assertEqual(list(manifest.walk_dp_checkpoints(root)), [])

    def test_missing_root_returns_nothing(self):
        self.assertEqual(
            list(manifest.walk_dp_checkpoints(Path("/no/such/dir/xxx"))),
            [],
        )


class TestWalkPi05Checkpoints(unittest.TestCase):
    def test_walks_config_exp_step_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "pi05_robofactory_pm_lora_finetune"
            exp = cfg / "pi05_pm_d1_v1"
            exp.mkdir(parents=True)
            (exp / "10000").mkdir()
            (exp / "19999").mkdir()
            (exp / "wandb_id.txt").write_text("abc1234x\n")

            recs = list(manifest.walk_pi05_checkpoints(root))
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual(r.model, "pi05")
            self.assertEqual(r.task, "pm")
            self.assertEqual(r.step, "19999")
            # Latest ckpt path only; not the intermediate 10000.
            self.assertEqual(r.ckpt_paths, str(exp / "19999"))
            self.assertNotIn("10000", r.ckpt_paths)
            self.assertIn("2 step ckpts", r.notes)
            self.assertIn("abc1234x", r.wandb_url)

    def test_skips_exp_with_no_step_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "cfg" / "exp").mkdir(parents=True)
            (root / "cfg" / "exp" / "wandb_id.txt").write_text("x")
            self.assertEqual(list(manifest.walk_pi05_checkpoints(root)), [])

    def test_libero_dir_does_not_collide_on_category(self):
        # Regression: _parse_pi05_config_name returns category='ablation' for
        # libero dirs; walker must not also pass category= literally or
        # RunRecord(**fields, category=...) raises duplicate-kwarg TypeError.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "pi05_libero_spatial_lora"
            exp = cfg / "phase1"
            exp.mkdir(parents=True)
            (exp / "5000").mkdir()
            recs = list(manifest.walk_pi05_checkpoints(root))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].category, "ablation")
            self.assertEqual(recs[0].task, "libero")


class TestWriteManifestCsv(unittest.TestCase):
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "manifest.csv"
            recs = [
                manifest.RunRecord(
                    run_id="row1", model="dp", task="pm", status="done",
                ),
                manifest.RunRecord(
                    run_id="row2", model="pi05", task="tsc", status="running",
                ),
            ]
            n = manifest.write_manifest_csv(recs, out)
            self.assertEqual(n, 2)
            with out.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "row1")
            self.assertEqual(rows[1]["model"], "pi05")
            # Schema column count check
            self.assertEqual(set(rows[0].keys()), set(manifest.MANIFEST_COLUMNS))

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "deep" / "nested" / "manifest.csv"
            manifest.write_manifest_csv([], out)
            self.assertTrue(out.is_file())


class TestSchemaConsistency(unittest.TestCase):
    def test_runrecord_fields_match_columns(self):
        # Every column must be a RunRecord field — no silent drops.
        recfields = {f.name for f in __import__("dataclasses").fields(manifest.RunRecord)}
        self.assertEqual(recfields, set(manifest.MANIFEST_COLUMNS))

    def test_status_default_is_valid(self):
        r = manifest.RunRecord()
        self.assertIn(r.status, manifest.VALID_STATUS)
        self.assertIn(r.category, manifest.VALID_CATEGORY)


class TestCliDryRun(unittest.TestCase):
    def test_dry_run_against_empty_roots(self):
        # Pass nonexistent roots so walkers yield nothing — exit 0.
        rc = manifest.main([
            "--dry-run",
            "--source", "filesystem",
            "--dp-ckpt-root", "/no/such/dp",
            "--pi05-ckpt-root", "/no/such/pi05",
        ])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
