"""Adversarial tests for utils.ckpt_resolver.

These deliberately stress edge cases that the happy-path test file misses.
The most important one — `test_identifier_collision_*` — pins the known bug
where `300_in1k.ckpt` / `300_dino_blora.ckpt` / `300_r3m.ckpt` /
`300_dino_spatch.ckpt` all parse to the same `(identifier, epoch)`, so
`find()` returns four indistinguishable entries and `latest()` is
non-deterministic across the four backup-encoder variants.

Run from robofactory repo root:
    python -m unittest utils.test_ckpt_resolver_hardening -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import ckpt_resolver  # noqa: E402


class TestIdentifierEncodesSuffix(unittest.TestCase):
    """Backup-encoder variants live as `300_<encoder>.ckpt`. The suffix is
    encoded in `identifier` so `find()`/`latest()` can disambiguate."""

    def _make_pm_with_backups(self, root: Path):
        pm = root / "PickMeat-rf_150"
        backup = pm / "backup"
        backup.mkdir(parents=True)
        for variant in ("in1k", "dino_blora", "dino_spatch", "r3m"):
            (backup / f"300_{variant}.ckpt").write_bytes(b"x" * 100)

    def test_four_backup_variants_become_four_index_rows(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._make_pm_with_backups(tdp / "RoboFactory")
            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            self.assertEqual(n, 4)

    def test_find_at_epoch_300_returns_all_four(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._make_pm_with_backups(tdp / "RoboFactory")
            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            entries = r.find(framework="dp", task="PickMeat", epoch=300)
            self.assertEqual(len(entries), 4)
            paths = sorted(e.path for e in entries)
            self.assertEqual(len(set(paths)), 4)

    def test_identifier_encodes_encoder_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._make_pm_with_backups(tdp / "RoboFactory")
            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            idents = {e.identifier for e in r.find(epoch=300)}
            self.assertEqual(idents, {
                "PickMeat-rf_150:in1k",
                "PickMeat-rf_150:dino_blora",
                "PickMeat-rf_150:dino_spatch",
                "PickMeat-rf_150:r3m",
            })

    def test_latest_with_substr_disambiguates(self):
        """latest(identifier_substr=':in1k') returns exactly the in1k variant."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self._make_pm_with_backups(tdp / "RoboFactory")
            out = tdp / "ckpt_index.jsonl"
            r = ckpt_resolver.CkptResolver()
            r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            in1k_latest = r.latest(framework="dp", identifier_substr=":in1k")
            self.assertIsNotNone(in1k_latest)
            self.assertIn("300_in1k.ckpt", in1k_latest.path)

    def test_unsuffixed_ckpt_keeps_bare_identifier(self):
        """A non-suffixed ckpt (e.g. `300.ckpt`) should keep the bare
        run-dir name as identifier — no spurious colon."""
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            run = tdp / "RoboFactory" / "Foo_150"
            run.mkdir(parents=True)
            (run / "300.ckpt").write_bytes(b"x")
            out = tdp / "ix.jsonl"
            r = ckpt_resolver.CkptResolver()
            r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            self.assertEqual(r.find()[0].identifier, "Foo_150")


class TestMalformedIndex(unittest.TestCase):
    def test_load_malformed_line_skipped_with_warning(self):
        """One bad line must not break the whole index — log warning + skip."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.jsonl"
            p.write_text(
                '{"framework": "dp", "task": "X", "identifier": "X_1", "epoch": 1, "path": "/p1", "size_bytes": 1, "mtime_unix": 1.0}\n'
                '{not valid json\n'
                '{"framework": "dp", "task": "Y", "identifier": "Y_1", "epoch": 2, "path": "/p2", "size_bytes": 1, "mtime_unix": 2.0}\n'
            )
            with self.assertLogs("utils.ckpt_resolver", level="WARNING") as cm:
                r = ckpt_resolver.CkptResolver.load(p)
            self.assertEqual(len(r), 2)
            self.assertTrue(any("malformed" in m.lower() for m in cm.output))

    def test_load_blank_lines_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ix.jsonl"
            p.write_text(
                "\n"
                '{"framework": "dp", "task": "X", "identifier": "X_1", "epoch": 1, "path": "/p", "size_bytes": 1, "mtime_unix": 1.0}\n'
                "   \n"
                "\n"
            )
            r = ckpt_resolver.CkptResolver.load(p)
            self.assertEqual(len(r), 1)

    def test_load_extra_fields_ignored(self):
        """Forward-compat: a writer adding new fields must not break older readers."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ix.jsonl"
            p.write_text(
                '{"framework": "dp", "task": "X", "identifier": "X_1", "epoch": 1, '
                '"path": "/p", "size_bytes": 1, "mtime_unix": 1.0, "future_field": "x", "another": 42}\n'
            )
            r = ckpt_resolver.CkptResolver.load(p)
            self.assertEqual(len(r), 1)
            e = r.find()[0]
            self.assertEqual(e.identifier, "X_1")
            self.assertEqual(e.epoch, 1)
            # CkptEntry has no `future_field`; it's silently dropped.
            self.assertFalse(hasattr(e, "future_field"))


class TestScanRobustness(unittest.TestCase):
    def test_scan_missing_roots_yields_zero(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "ix.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(
                roots=[Path(td) / "no_such_root_RoboFactory",
                       Path(td) / "no_such_openpi"],
                out_path=out,
            )
            self.assertEqual(n, 0)

    def test_scan_skips_broken_symlink_in_run_dir(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            run = tdp / "RoboFactory" / "Foo_150"
            run.mkdir(parents=True)
            (run / "100.ckpt").write_bytes(b"x")
            # broken symlink whose name happens to look like a ckpt
            os.symlink("/nonexistent/target.ckpt", run / "200.ckpt")
            out = tdp / "ix.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(roots=[tdp / "RoboFactory"], out_path=out)
            # Real file is indexed; broken symlink's stat fails → skipped.
            self.assertEqual(n, 1)
            self.assertEqual(r.find()[0].epoch, 100)

    def test_scan_pi05_skips_hidden_exp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cfg = tdp / "openpi" / "some_cfg"
            for exp_name in (".tmp", "real_exp"):
                exp = cfg / exp_name / "1000"
                for sub in ("params", "assets", "train_state"):
                    (exp / sub).mkdir(parents=True)
                    (exp / sub / "data.bin").write_bytes(b"x" * 16)
            out = tdp / "ix.jsonl"
            r = ckpt_resolver.CkptResolver()
            n = r.scan_legacy_tree(roots=[tdp / "openpi"], out_path=out)
            self.assertEqual(n, 1)
            self.assertNotIn(".tmp", r.find()[0].identifier)


class TestFindEdgeCases(unittest.TestCase):
    def test_find_no_filters_returns_all(self):
        es = [
            ckpt_resolver.CkptEntry("dp", "A", "A_1", 1, "/p1", 1, 1.0),
            ckpt_resolver.CkptEntry("pi05", "B", "B/exp", 2, "/p2", 1, 2.0),
        ]
        r = ckpt_resolver.CkptResolver(es)
        self.assertEqual(len(r.find()), 2)

    def test_find_no_match_returns_empty_not_none(self):
        r = ckpt_resolver.CkptResolver([
            ckpt_resolver.CkptEntry("dp", "A", "A_1", 1, "/p", 1, 1.0)
        ])
        self.assertEqual(r.find(framework="pi05"), [])

    def test_ckpt_entry_is_frozen(self):
        e = ckpt_resolver.CkptEntry("dp", "A", "A_1", 1, "/p", 1, 1.0)
        with self.assertRaises(dataclasses_FrozenError := __import__("dataclasses").FrozenInstanceError):
            e.epoch = 999  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
