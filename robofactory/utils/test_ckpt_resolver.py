"""Tests for utils.ckpt_resolver.

Run from the robofactory env (cwd = robofactory repo root):
    python -m unittest utils.test_ckpt_resolver -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ckpt_resolver  # noqa: E402


def _make_dp_layout(root: Path):
    """Create a minimal DP-like ckpt tree under root."""
    pm = root / "PickMeat-rf_150"
    pm.mkdir(parents=True)
    (pm / "300.ckpt").write_bytes(b"x" * 100)
    (pm / "295.ckpt").write_bytes(b"x" * 100)
    backup = pm / "backup"
    backup.mkdir()
    (backup / "300_in1k.ckpt").write_bytes(b"x" * 100)
    (backup / "300_dino_blora.ckpt").write_bytes(b"x" * 100)

    tsc0 = root / "ThreeRobotsStackCube-rf_Agent0_d2_wristcam_150"
    tsc0.mkdir(parents=True)
    (tsc0 / "300.ckpt").write_bytes(b"x" * 100)
    (tsc0 / "295.ckpt").write_bytes(b"x" * 100)


def _make_pi05_layout(root: Path):
    """Create a minimal Pi0.5-like ckpt tree under root."""
    cfg = root / "pi05_robofactory_pm_lora_finetune"
    exp = cfg / "pi05_pm_d1_v1"
    for step in (1000, 5000, 18000):
        d = exp / str(step)
        for sub in ("params", "assets", "train_state"):
            (d / sub).mkdir(parents=True)
            (d / sub / "data.bin").write_bytes(b"x" * 1024)
    # one trash dir that should be ignored
    (exp / "1000.trash_12345_99").mkdir()


class TestScanLegacyTree(unittest.TestCase):
    def test_scan_dp_finds_numeric_ckpts(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "RoboFactory").mkdir()
            (tdp / "openpi").mkdir()
            _make_dp_layout(tdp / "RoboFactory")

            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(roots=[tdp / "RoboFactory", tdp / "openpi"], out_path=out)

            # PickMeat: 300, 295, 300_in1k, 300_dino_blora — 4 entries
            # TSC: 300, 295 — 2 entries
            self.assertEqual(n, 6)
            pm_entries = r.find(framework="dp", task="PickMeat")
            self.assertEqual(len(pm_entries), 4)

    def test_scan_pi05_finds_step_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "RoboFactory").mkdir()
            (tdp / "openpi").mkdir()
            _make_pi05_layout(tdp / "openpi")

            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(roots=[tdp / "RoboFactory", tdp / "openpi"], out_path=out)

            self.assertEqual(n, 3)  # 1000, 5000, 18000 (trash dir ignored)
            entries = r.find(framework="pi05")
            self.assertEqual(sorted(e.epoch for e in entries), [1000, 5000, 18000])

    def test_load_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "RoboFactory").mkdir()
            (tdp / "openpi").mkdir()
            _make_dp_layout(tdp / "RoboFactory")

            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            r.scan_legacy_tree(roots=[tdp / "RoboFactory", tdp / "openpi"], out_path=out)

            loaded = ckpt_resolver.CkptResolver.load(out)
            self.assertEqual(len(loaded), 6)


class TestFindAndLatest(unittest.TestCase):
    def test_find_filters(self):
        entries = [
            ckpt_resolver.CkptEntry(framework="dp", task="A", identifier="A_150", epoch=100, path="/p1", size_bytes=1, mtime_unix=1.0),
            ckpt_resolver.CkptEntry(framework="dp", task="A", identifier="A_150", epoch=300, path="/p2", size_bytes=1, mtime_unix=2.0),
            ckpt_resolver.CkptEntry(framework="dp", task="B", identifier="B_150", epoch=300, path="/p3", size_bytes=1, mtime_unix=3.0),
            ckpt_resolver.CkptEntry(framework="pi05", task="A", identifier="cfg/exp", epoch=500, path="/p4", size_bytes=1, mtime_unix=4.0),
        ]
        r = ckpt_resolver.CkptResolver(entries)
        self.assertEqual(len(r.find(framework="dp")), 3)
        self.assertEqual(len(r.find(framework="dp", task="A")), 2)
        self.assertEqual(len(r.find(epoch=300)), 2)
        self.assertEqual(len(r.find(identifier_substr="cfg/")), 1)

    def test_latest_picks_highest_epoch(self):
        entries = [
            ckpt_resolver.CkptEntry(framework="dp", task="A", identifier="A_150", epoch=100, path="/p1", size_bytes=1, mtime_unix=10.0),
            ckpt_resolver.CkptEntry(framework="dp", task="A", identifier="A_150", epoch=300, path="/p2", size_bytes=1, mtime_unix=2.0),
        ]
        r = ckpt_resolver.CkptResolver(entries)
        latest = r.latest(framework="dp", task="A")
        self.assertEqual(latest.epoch, 300)

    def test_latest_breaks_ties_by_mtime(self):
        entries = [
            ckpt_resolver.CkptEntry(framework="dp", task="A", identifier="A_150", epoch=300, path="/p1", size_bytes=1, mtime_unix=10.0),
            ckpt_resolver.CkptEntry(framework="dp", task="A", identifier="A_150", epoch=300, path="/p2", size_bytes=1, mtime_unix=20.0),
        ]
        r = ckpt_resolver.CkptResolver(entries)
        latest = r.latest(framework="dp", task="A")
        self.assertEqual(latest.path, "/p2")

    def test_latest_returns_none_when_empty(self):
        r = ckpt_resolver.CkptResolver([])
        self.assertIsNone(r.latest(framework="dp"))


class TestEdgeCases(unittest.TestCase):
    def test_load_missing_file_returns_empty(self):
        r = ckpt_resolver.CkptResolver.load("/nonexistent/index.jsonl")
        self.assertEqual(len(r), 0)

    def test_invalid_ckpt_filename_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "RoboFactory").mkdir()
            run = tdp / "RoboFactory" / "Foo_100"
            run.mkdir()
            # neither matches _DP_NUMERIC_CKPT
            (run / "best.ckpt").write_bytes(b"x")
            (run / "snapshot.pt").write_bytes(b"x")
            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
