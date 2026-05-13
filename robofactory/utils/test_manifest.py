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
from typing import Optional

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

    def test_decent_arm1_defaults_to_tsc(self):
        d = manifest._parse_pi05_config_name(
            "pi05_robofactory_decent_lora_finetune_arm1"
        )
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "1")
        self.assertEqual(d["dataset"], "ws")
        self.assertEqual(d["task"], "tsc")  # codebase convention
        self.assertIn("inferred", d.get("notes", ""))

    def test_wristcam_decent_arm0_defaults_to_tsc(self):
        d = manifest._parse_pi05_config_name(
            "pi05_robofactory_decent_wristcam_lora_finetune_arm0"
        )
        self.assertEqual(d["scheme"], "decent")
        self.assertEqual(d["arm"], "0")
        self.assertEqual(d["dataset"], "wc")
        self.assertEqual(d["task"], "tsc")
        self.assertIn("inferred", d.get("notes", ""))

    def test_libero_marked_ablation(self):
        d = manifest._parse_pi05_config_name("pi05_libero_spatial_lora")
        self.assertEqual(d["task"], "libero")
        self.assertEqual(d["category"], "ablation")
        self.assertNotIn("notes", d)  # libero is decisive, no inference note

    def test_exp_name_disambiguates_task(self):
        # Config dir doesn't have a task token but exp_name does
        d = manifest._parse_pi05_config_name(
            "pi05_robofactory_decent_lora_finetune_arm0",
            exp_name="pi05_lp_d1_decent_arm0_v0",  # hypothetical LP exp
        )
        self.assertEqual(d["task"], "lp")
        # task came from exp_name, not the decent-default fallback
        self.assertNotIn("notes", d)

    def test_explicit_pm_no_inference_note(self):
        d = manifest._parse_pi05_config_name("pi05_robofactory_pm_lora_finetune")
        self.assertEqual(d["task"], "pm")
        self.assertNotIn("notes", d)  # task explicit; no inference


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


class TestEvalSrFromJsonl(unittest.TestCase):
    """Workflow improvement #6: SLURM-killed wandb run still gets SR from JSONL."""

    def _write_jsonl(self, lines):
        import json
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, dir=tempfile.gettempdir()
        )
        for line in lines:
            tmp.write(json.dumps(line) + "\n")
        tmp.close()
        self.addCleanup(lambda p=tmp.name: Path(p).unlink(missing_ok=True))
        return tmp.name

    def test_no_path_in_config_returns_none(self):
        sr, n = manifest._eval_sr_from_jsonl({})
        self.assertIsNone(sr)
        self.assertIsNone(n)

    def test_missing_jsonl_file_returns_none(self):
        sr, n = manifest._eval_sr_from_jsonl({"jsonl_path": "/no/such/file.jsonl"})
        self.assertIsNone(sr)
        self.assertIsNone(n)

    def test_clean_summary_line_is_used(self):
        path = self._write_jsonl([
            {"kind": "manifest", "task": "PickMeat-rf"},
            {"kind": "episode", "success": 1},
            {"kind": "episode", "success": 0},
            {"kind": "summary", "success_rate": 0.5, "n_total": 60},
        ])
        sr, n = manifest._eval_sr_from_jsonl({"jsonl_path": path})
        self.assertEqual(sr, 0.5)
        self.assertEqual(n, 60)

    def test_killed_mid_eval_computes_from_episodes(self):
        # Slurm killed before summary line was written; recover from episodes.
        rows = [{"kind": "manifest"}]
        for i in range(27):
            rows.append({"kind": "episode", "success": 0})
        path = self._write_jsonl(rows)
        sr, n = manifest._eval_sr_from_jsonl({"jsonl_path": path})
        self.assertEqual(sr, 0.0)
        self.assertEqual(n, 27)

    def test_killed_mid_eval_partial_success_rate(self):
        rows = [{"kind": "manifest"}]
        for i in range(10):
            rows.append({"kind": "episode", "success": 1 if i < 3 else 0})
        path = self._write_jsonl(rows)
        sr, n = manifest._eval_sr_from_jsonl({"jsonl_path": path})
        self.assertAlmostEqual(sr, 0.3)
        self.assertEqual(n, 10)

    def test_results_json_path_alias_works(self):
        # eval_decent_pi05.py uses `results_json_path` instead of `jsonl_path`.
        path = self._write_jsonl([
            {"kind": "manifest"},
            {"kind": "episode", "success": 1},
            {"kind": "summary", "success_rate": 1.0, "n_total": 1},
        ])
        sr, n = manifest._eval_sr_from_jsonl({"results_json_path": path})
        self.assertEqual(sr, 1.0)
        self.assertEqual(n, 1)

    def test_malformed_lines_skipped(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, dir=tempfile.gettempdir()
        )
        path.write('{"kind": "manifest"}\n')
        path.write("not valid json at all\n")
        path.write('{"kind": "episode", "success": 1}\n')
        path.write('{"kind": "summary", "success_rate": 1.0, "n_total": 1}\n')
        path.close()
        self.addCleanup(lambda: Path(path.name).unlink(missing_ok=True))
        sr, n = manifest._eval_sr_from_jsonl({"jsonl_path": path.name})
        self.assertEqual(sr, 1.0)
        self.assertEqual(n, 1)

    def test_zero_episodes_returns_none(self):
        path = self._write_jsonl([{"kind": "manifest"}])
        sr, n = manifest._eval_sr_from_jsonl({"jsonl_path": path})
        self.assertIsNone(sr)
        self.assertIsNone(n)


class TestCkptFormatMarker(unittest.TestCase):
    """Phase D6 — manifest reads ``ckpt_format.json`` marker for Pi0.5 step dirs."""

    def test_no_marker_means_full(self):
        with tempfile.TemporaryDirectory() as td:
            step_dir = Path(td) / "10000"
            step_dir.mkdir()
            self.assertEqual(manifest._read_ckpt_format_marker(step_dir), "full")

    def test_marker_lora_only_is_read(self):
        with tempfile.TemporaryDirectory() as td:
            step_dir = Path(td) / "10000"
            step_dir.mkdir()
            (step_dir / manifest._MARKER_FILENAME).write_text(
                '{"format": "lora_only", "base_weight_loader": "gs://x"}'
            )
            self.assertEqual(manifest._read_ckpt_format_marker(step_dir), "lora_only")

    def test_marker_full_is_read(self):
        with tempfile.TemporaryDirectory() as td:
            step_dir = Path(td) / "10000"
            step_dir.mkdir()
            (step_dir / manifest._MARKER_FILENAME).write_text(
                '{"format": "full"}'
            )
            self.assertEqual(manifest._read_ckpt_format_marker(step_dir), "full")

    def test_corrupted_json_falls_back_to_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            step_dir = Path(td) / "10000"
            step_dir.mkdir()
            (step_dir / manifest._MARKER_FILENAME).write_text("not valid json {")
            # Manifest generation must never abort on one bad marker.
            self.assertEqual(manifest._read_ckpt_format_marker(step_dir), "unknown")

    def test_unrecognized_format_value_is_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            step_dir = Path(td) / "10000"
            step_dir.mkdir()
            (step_dir / manifest._MARKER_FILENAME).write_text(
                '{"format": "bogus_value"}'
            )
            self.assertEqual(manifest._read_ckpt_format_marker(step_dir), "unknown")


class TestPi05WalkerPopulatesCkptFormat(unittest.TestCase):
    """Phase D6 — ``walk_pi05_checkpoints`` populates ``ckpt_format`` on each row."""

    def _make_exp(self, td: str, *, marker_format: Optional[str] = None) -> Path:
        root = Path(td)
        cfg = root / "pi05_robofactory_pm_lora_finetune"
        exp = cfg / "pi05_pm_d1_v1"
        exp.mkdir(parents=True)
        step_dir = exp / "19999"
        step_dir.mkdir()
        if marker_format is not None:
            (step_dir / manifest._MARKER_FILENAME).write_text(
                f'{{"format": "{marker_format}", "base_weight_loader": "gs://x"}}'
            )
        return root

    def test_full_default_when_no_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_exp(td)
            recs = list(manifest.walk_pi05_checkpoints(root))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].ckpt_format, "full")

    def test_lora_only_when_marker_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_exp(td, marker_format="lora_only")
            recs = list(manifest.walk_pi05_checkpoints(root))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].ckpt_format, "lora_only")

    def test_explicit_full_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_exp(td, marker_format="full")
            recs = list(manifest.walk_pi05_checkpoints(root))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].ckpt_format, "full")

    def test_marker_only_read_from_latest_step(self):
        """Only the latest step's marker matters; intermediate-step markers
        don't determine the row's reported format (one row per exp dir)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = root / "pi05_robofactory_pm_lora_finetune" / "pi05_pm_d1_v1"
            exp.mkdir(parents=True)
            # Intermediate step says "lora_only" but latest has no marker.
            (exp / "10000").mkdir()
            (exp / "10000" / manifest._MARKER_FILENAME).write_text(
                '{"format": "lora_only", "base_weight_loader": "gs://x"}'
            )
            (exp / "19999").mkdir()  # no marker → default "full"
            recs = list(manifest.walk_pi05_checkpoints(root))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].step, "19999")
            self.assertEqual(recs[0].ckpt_format, "full")

    def test_csv_round_trip_carries_format_through(self):
        """Write a manifest row with ckpt_format set; read it back; confirm
        the column is in the CSV header and the value is preserved."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "manifest.csv"
            recs = [
                manifest.RunRecord(
                    run_id="row1", model="pi05", ckpt_format="lora_only",
                    status="done",
                ),
            ]
            manifest.write_manifest_csv(recs, out)
            with out.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertIn("ckpt_format", rows[0])
            self.assertEqual(rows[0]["ckpt_format"], "lora_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
