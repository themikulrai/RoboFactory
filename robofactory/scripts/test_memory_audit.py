"""Tests for scripts/memory_audit.py.

Run from repo root:
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
        -m pytest robofactory/scripts/test_memory_audit.py -q
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "memory_audit", _HERE / "memory_audit.py"
)
memory_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(memory_audit)


class TestSplitFrontmatter(unittest.TestCase):
    def test_well_formed(self):
        text = "---\nname: foo\ntype: project\n---\nbody\n"
        fields, body = memory_audit._split_frontmatter(text)
        self.assertEqual(fields, {"name": "foo", "type": "project"})
        self.assertEqual(body, "body\n")

    def test_no_frontmatter(self):
        text = "just body content"
        fields, body = memory_audit._split_frontmatter(text)
        self.assertEqual(fields, {})
        self.assertEqual(body, text)

    def test_truncated_frontmatter(self):
        # Opening --- but no closing — treat as no frontmatter.
        text = "---\nname: foo\nno closing fence"
        fields, body = memory_audit._split_frontmatter(text)
        self.assertEqual(fields, {})

    def test_value_with_colons(self):
        # Description fields routinely contain colons; only first one splits.
        text = "---\ndescription: 2026-05-08 thing: with colons\n---\n"
        fields, _ = memory_audit._split_frontmatter(text)
        self.assertEqual(fields["description"], "2026-05-08 thing: with colons")


class TestInferLastTouched(unittest.TestCase):
    def _write(self, body: str) -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, dir=tempfile.gettempdir()
        )
        f.write(body)
        f.close()
        self.addCleanup(lambda p=f.name: Path(p).unlink(missing_ok=True))
        return Path(f.name)

    def test_updated_field_wins(self):
        path = self._write(
            "---\n"
            "name: x\n"
            "description: 2025-01-01 something old\n"
            "updated: 2026-05-08\n"
            "---\nbody"
        )
        fields, _ = memory_audit._split_frontmatter(path.read_text())
        date, source = memory_audit.infer_last_touched(path, fields)
        self.assertEqual(date, _dt.date(2026, 5, 8))
        self.assertEqual(source, "frontmatter")

    def test_description_iso_prefix_used_if_no_updated(self):
        path = self._write(
            "---\n"
            "name: x\n"
            "description: 2026-05-08 the thing happened\n"
            "---\nbody"
        )
        fields, _ = memory_audit._split_frontmatter(path.read_text())
        date, source = memory_audit.infer_last_touched(path, fields)
        self.assertEqual(date, _dt.date(2026, 5, 8))
        self.assertEqual(source, "description-prefix")

    def test_mtime_fallback(self):
        path = self._write(
            "---\nname: x\ndescription: no date here\n---\nbody"
        )
        fields, _ = memory_audit._split_frontmatter(path.read_text())
        date, source = memory_audit.infer_last_touched(path, fields)
        self.assertEqual(source, "mtime")
        # Just a sanity check; not equality.
        self.assertLessEqual(date, _dt.date.today())

    def test_invalid_iso_in_updated_falls_back_to_description(self):
        path = self._write(
            "---\n"
            "name: x\n"
            "description: 2026-05-08 dated\n"
            "updated: not-a-date\n"
            "---\nbody"
        )
        fields, _ = memory_audit._split_frontmatter(path.read_text())
        date, source = memory_audit.infer_last_touched(path, fields)
        self.assertEqual(date, _dt.date(2026, 5, 8))
        self.assertEqual(source, "description-prefix")


class TestTouch(unittest.TestCase):
    def _make_dir_with(self, fname: str, content: str) -> tuple[Path, Path]:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda dd=d: __import__("shutil").rmtree(dd, ignore_errors=True))
        f = d / fname
        f.write_text(content)
        return d, f

    def test_touch_one_adds_updated_field(self):
        d, f = self._make_dir_with(
            "project_x.md",
            "---\nname: x\ntype: project\n---\nbody\n",
        )
        today = _dt.date(2026, 5, 8)
        changed = memory_audit.touch_one(f, today)
        self.assertTrue(changed)
        fields, body = memory_audit._split_frontmatter(f.read_text())
        self.assertEqual(fields["updated"], "2026-05-08")
        self.assertIn("body", body)

    def test_touch_idempotent_same_date(self):
        d, f = self._make_dir_with(
            "project_x.md",
            "---\nname: x\nupdated: 2026-05-08\n---\nbody\n",
        )
        today = _dt.date(2026, 5, 8)
        changed = memory_audit.touch_one(f, today)
        self.assertFalse(changed)

    def test_touch_skips_no_frontmatter(self):
        d, f = self._make_dir_with("flat.md", "no frontmatter here\n")
        today = _dt.date(2026, 5, 8)
        changed = memory_audit.touch_one(f, today)
        self.assertFalse(changed)
        # File contents unchanged.
        self.assertEqual(f.read_text(), "no frontmatter here\n")

    def test_touch_all_stale_only_touches_old(self):
        d, _ = self._make_dir_with(
            "project_old.md",
            "---\nname: old\ndescription: 2025-01-01 dated\n---\nbody\n",
        )
        (d / "project_fresh.md").write_text(
            "---\nname: fresh\nupdated: 2026-05-07\n---\nbody\n"
        )
        # Use real audit() pathway via touch_command
        rc = memory_audit.touch_command(
            d, target=None, all_stale=True, max_age_days=30
        )
        self.assertEqual(rc, 0)
        old_fields, _ = memory_audit._split_frontmatter(
            (d / "project_old.md").read_text()
        )
        fresh_fields, _ = memory_audit._split_frontmatter(
            (d / "project_fresh.md").read_text()
        )
        self.assertIn("updated", old_fields)
        self.assertEqual(fresh_fields["updated"], "2026-05-07")  # unchanged


class TestAuditExitCodes(unittest.TestCase):
    def _populate(self) -> Path:
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda dd=d: __import__("shutil").rmtree(dd, ignore_errors=True))
        return d

    def test_no_stale_returns_zero(self):
        d = self._populate()
        today = _dt.date.today().isoformat()
        (d / "project_fresh.md").write_text(
            f"---\nname: x\nupdated: {today}\n---\nbody\n"
        )
        rc = memory_audit.audit(d, max_age_days=90)
        self.assertEqual(rc, 0)

    def test_stale_returns_one(self):
        d = self._populate()
        (d / "project_old.md").write_text(
            "---\nname: x\nupdated: 2024-01-01\n---\nbody\n"
        )
        rc = memory_audit.audit(d, max_age_days=90)
        self.assertEqual(rc, 1)

    def test_empty_dir_returns_zero(self):
        d = self._populate()
        rc = memory_audit.audit(d, max_age_days=90)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
