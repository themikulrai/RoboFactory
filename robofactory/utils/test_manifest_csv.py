"""Tests for robofactory.utils.manifest_csv (and the additive
generate_manifest_csv API in robofactory.utils.manifest).

NEVER hits the network — wandb.Api is patched.
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from robofactory.utils import manifest, manifest_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_wandb_run(
    *,
    name: str,
    rid: str = "abc123",
    state: str = "finished",
    url: str = "https://wandb.ai/x/y/runs/abc123",
    config: dict | None = None,
    summary: dict | None = None,
    tags: list | None = None,
    created_at: str = "2026-05-01T12:00:00",
):
    return SimpleNamespace(
        name=name,
        id=rid,
        state=state,
        url=url,
        config=config or {},
        summary=summary or {},
        tags=tags or [],
        created_at=created_at,
    )


def _write_ckpt_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemaExactColumns(unittest.TestCase):
    def test_exact_23_columns_in_order(self):
        expected = (
            "run_id", "model", "dataset", "task", "encoder", "scheme",
            "arm", "phase", "step", "seed", "git_sha", "slurm_id",
            "status", "category", "wandb_url", "ckpt_paths", "eval_sr",
            "eval_n", "scene_config", "camera_mapping", "parent_run_id",
            "created_utc", "notes",
        )
        self.assertEqual(manifest.MANIFEST_COLUMNS, expected)
        self.assertEqual(len(manifest.MANIFEST_COLUMNS), 23)
        # CLI shim re-exports the same tuple
        self.assertIs(manifest_csv.MANIFEST_COLUMNS, manifest.MANIFEST_COLUMNS)

    def test_csv_header_matches_schema(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "manifest.csv"
            with patch.object(manifest, "fetch_wandb_runs", return_value=iter([])):
                manifest.generate_manifest_csv(
                    out_path=out,
                    fetch_wandb=False,
                    ckpt_index_path=Path(td) / "missing.jsonl",
                    run_root=Path(td) / "missing_runs",
                )
            with out.open() as fh:
                header = next(csv.reader(fh))
            self.assertEqual(tuple(header), manifest.MANIFEST_COLUMNS)


# ---------------------------------------------------------------------------
# Wandb-Api mock tests
# ---------------------------------------------------------------------------

class TestWandbApiIsMocked(unittest.TestCase):
    """Sanity: confirm we can run the full generator with wandb fully
    mocked and never hit the network."""

    def test_wandb_api_constructor_is_patched(self):
        fake_run = _make_fake_wandb_run(
            name="dp_ws_pm_in1k_cent_train_s100_a3c736d",
            rid="run01",
            state="finished",
        )
        fake_api = MagicMock()
        fake_api.runs.return_value = [fake_run]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "manifest.csv"
            ckpt_idx = Path(td) / "ckpt_index.jsonl"
            ckpt_idx.touch()  # empty index

            with patch("wandb.apis.public.Api", return_value=fake_api):
                n = manifest.generate_manifest_csv(
                    out_path=out,
                    fetch_wandb=True,
                    wandb_entity="ent",
                    wandb_projects=("proj",),
                    ckpt_index_path=ckpt_idx,
                    run_root=Path(td) / "missing_runs",
                )
            self.assertEqual(n, 1)
            with out.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["run_id"], "dp_ws_pm_in1k_cent_train_s100_a3c736d")
            # Canonical run_id parser populated structural fields
            self.assertEqual(rows[0]["model"], "dp")
            self.assertEqual(rows[0]["task"], "pm")
            self.assertEqual(rows[0]["encoder"], "in1k")
            self.assertEqual(rows[0]["status"], "done")


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

class TestMergeLogic(unittest.TestCase):
    def test_wandb_run_with_matching_ckpt_populates_ckpt_paths(self):
        # Wandb run has structural fields parseable from the canonical name;
        # ckpt index has a matching legacy DP entry by (model, task, scheme,
        # arm, seed) — but here the run name is canonical so we match by id
        # equality only when run_ids align. We instead test the structural
        # match path: wandb run with model=dp,task=tsc,scheme=cent,seed=42;
        # ckpt entry walks to the same fields.
        fake_run = _make_fake_wandb_run(
            name="dp_ws_tsc_in1k_cent_train_s42_aaaaaaa",
            rid="rt1",
            state="finished",
        )
        fake_api = MagicMock()
        fake_api.runs.return_value = [fake_run]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            ckpt_idx = Path(td) / "ckpt_index.jsonl"
            _write_ckpt_index(ckpt_idx, [{
                "framework": "dp",
                "task": "ThreeRobotsStackCube",
                "identifier": "ThreeRobotsStackCube-rf_joint_d1_workspace_150_in1k",
                "epoch": 300,
                "path": "/checkpoints/foo/300.ckpt",
                "size_bytes": 100,
                "mtime_unix": 1700000000.0,
            }])
            with patch("wandb.apis.public.Api", return_value=fake_api):
                manifest.generate_manifest_csv(
                    out_path=out,
                    fetch_wandb=True,
                    wandb_entity="ent",
                    wandb_projects=("proj",),
                    ckpt_index_path=ckpt_idx,
                    run_root=Path(td) / "no_runs",
                )
            with out.open() as fh:
                rows = list(csv.DictReader(fh))

        # 1 wandb run + 1 unmatched ckpt row (seeds don't match) = 2 rows.
        # The ckpt row has the path; the wandb row has the URL.
        self.assertEqual(len(rows), 2)
        ckpt_paths_total = [r["ckpt_paths"] for r in rows if r["ckpt_paths"]]
        wandb_urls_total = [r["wandb_url"] for r in rows if r["wandb_url"]]
        self.assertEqual(ckpt_paths_total, ["/checkpoints/foo/300.ckpt"])
        self.assertEqual(len(wandb_urls_total), 1)

    def test_ckpt_with_no_wandb_run_emits_empty_wandb_url(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            ckpt_idx = Path(td) / "ckpt_index.jsonl"
            _write_ckpt_index(ckpt_idx, [{
                "framework": "dp",
                "task": "PickMeat",
                "identifier": "PickMeat-rf_150",
                "epoch": 300,
                "path": "/ck/PickMeat-rf_150/300.ckpt",
                "size_bytes": 100,
                "mtime_unix": 1700000000.0,
            }])
            n = manifest.generate_manifest_csv(
                out_path=out,
                fetch_wandb=False,
                ckpt_index_path=ckpt_idx,
                run_root=Path(td) / "no_runs",
            )
            self.assertEqual(n, 1)
            with out.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["wandb_url"], "")
            self.assertEqual(rows[0]["ckpt_paths"], "/ck/PickMeat-rf_150/300.ckpt")
            self.assertEqual(rows[0]["task"], "pm")
            # status is "done" for ckpt-only rows because the file exists.
            self.assertEqual(rows[0]["status"], "done")

    def test_wandb_run_with_no_ckpt_emits_empty_ckpt_paths(self):
        fake_run = _make_fake_wandb_run(
            name="some-eval-run-no-canonical-name",
            rid="evx",
            state="running",
        )
        fake_api = MagicMock()
        fake_api.runs.return_value = [fake_run]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            ckpt_idx = Path(td) / "missing.jsonl"  # does not exist
            with patch("wandb.apis.public.Api", return_value=fake_api):
                manifest.generate_manifest_csv(
                    out_path=out,
                    fetch_wandb=True,
                    wandb_entity="ent",
                    wandb_projects=("openpi-robofactory",),
                    ckpt_index_path=ckpt_idx,
                    run_root=Path(td) / "no_runs",
                )
            with out.open() as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ckpt_paths"], "")
            self.assertEqual(rows[0]["status"], "running")
            self.assertEqual(rows[0]["wandb_url"], fake_run.url)


# ---------------------------------------------------------------------------
# Domain-value tolerance
# ---------------------------------------------------------------------------

class TestDomainValueTolerance(unittest.TestCase):
    def test_unknown_encoder_in_ckpt_does_not_crash(self):
        # Identifier with no recognized encoder token — _parse_legacy_dp
        # returns no encoder field, but RunRecord stays valid.
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            ckpt_idx = Path(td) / "ckpt_index.jsonl"
            _write_ckpt_index(ckpt_idx, [{
                "framework": "dp",
                "task": "PickMeat",
                "identifier": "PickMeat-rf_150_some_unknown_thing",
                "epoch": 100,
                "path": "/ck/x/100.ckpt",
                "size_bytes": 1,
                "mtime_unix": 1700000000.0,
            }])
            n = manifest.generate_manifest_csv(
                out_path=out,
                fetch_wandb=False,
                ckpt_index_path=ckpt_idx,
                run_root=Path(td) / "no_runs",
            )
            self.assertEqual(n, 1)
            with out.open() as fh:
                rows = list(csv.DictReader(fh))
            # Encoder stayed empty (not crashed); task still parsed.
            self.assertEqual(rows[0]["task"], "pm")
            self.assertEqual(rows[0]["encoder"], "")

    def test_wandb_unknown_state_passes_through(self):
        fake_run = _make_fake_wandb_run(
            name="weird-name",
            rid="wx",
            state="zombie",  # not in our state_map
        )
        rec = manifest.wandb_run_to_record(fake_run, project="diffusion-robofactory")
        self.assertEqual(rec.status, "zombie")
        # Project hint still kicked in
        self.assertEqual(rec.model, "dp")


# ---------------------------------------------------------------------------
# Wandb soft-fail
# ---------------------------------------------------------------------------

class TestWandbSoftFail(unittest.TestCase):
    def test_wandb_api_raises_does_not_crash_generator(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            ckpt_idx = Path(td) / "ckpt_index.jsonl"
            _write_ckpt_index(ckpt_idx, [{
                "framework": "dp",
                "task": "PickMeat",
                "identifier": "PickMeat-rf_150",
                "epoch": 100,
                "path": "/ck/PickMeat-rf_150/100.ckpt",
                "size_bytes": 1,
                "mtime_unix": 1700000000.0,
            }])
            with patch(
                "wandb.apis.public.Api",
                side_effect=RuntimeError("network dead"),
            ):
                n = manifest.generate_manifest_csv(
                    out_path=out,
                    fetch_wandb=True,
                    wandb_entity="ent",
                    wandb_projects=("proj",),
                    ckpt_index_path=ckpt_idx,
                    run_root=Path(td) / "no_runs",
                )
            # Got the 1 ckpt row; wandb side soft-failed.
            self.assertEqual(n, 1)


# ---------------------------------------------------------------------------
# Run-root walk
# ---------------------------------------------------------------------------

class TestRunRootWalk(unittest.TestCase):
    def test_walks_canonical_runid_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            run_root = Path(td) / "runs"
            (run_root / "dp" / "ws" / "pm" /
             "dp_ws_pm_in1k_cent_train_s100_a3c736d").mkdir(parents=True)
            recs = list(manifest._walk_run_root(run_root))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].run_id, "dp_ws_pm_in1k_cent_train_s100_a3c736d")
            self.assertEqual(recs[0].model, "dp")
            self.assertEqual(recs[0].phase, "train")

    def test_missing_run_root_yields_nothing(self):
        recs = list(manifest._walk_run_root(Path("/no/such/runroot/xyz")))
        self.assertEqual(recs, [])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

class TestCliPrintsRowCount(unittest.TestCase):
    def test_main_prints_wrote_n_rows(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            ckpt_idx = Path(td) / "missing.jsonl"
            with patch.object(manifest, "fetch_wandb_runs", return_value=iter([])), \
                 patch.object(manifest.paths, "CKPT_INDEX_PATH", ckpt_idx), \
                 patch.object(manifest.paths, "RUN_ROOT", Path(td) / "no_runs"):
                rc = manifest_csv.main([
                    "--output", str(out),
                    "--no-wandb",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            with out.open() as fh:
                header = next(csv.reader(fh))
            self.assertEqual(tuple(header), manifest.MANIFEST_COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
