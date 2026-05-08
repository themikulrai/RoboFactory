"""Audit Claude memory files for staleness.

Workflow improvement #9. The memory file accumulates `project_*` entries
that decay quickly (e.g. `project_d1_freezeenc_decent_zero` is fresh today
but goes stale once the encoder-collapse genus is solved). Without a
triage signal these entries silently misinform future sessions.

This tool:

  audit    : print a table of memory files with last-touched dates and
             a STALE flag if older than --max-age-days (default 90).
  touch    : add or update an `updated: YYYY-MM-DD` frontmatter field
             on a specific memory file (or all of them) — the agent /
             user runs this whenever a memory is read and confirmed
             still accurate.

Usage:
    python -m robofactory.scripts.memory_audit audit
    python -m robofactory.scripts.memory_audit audit --max-age-days 30
    python -m robofactory.scripts.memory_audit touch <name.md>
    python -m robofactory.scripts.memory_audit touch --all-stale --max-age-days 90

Source of truth for staleness, in order of preference:
    1. `updated:` frontmatter field (ISO date, set by `touch`).
    2. ISO date prefix in the `description:` field (a convention several
       project memories already follow: "2026-05-08 ... ").
    3. File mtime (fallback; least informative because reading a file
       doesn't update mtime).

Default memory dir: /iris/u/mikulrai/.claude/projects/-iris-u-mikulrai/memory/
Override with --memory-dir.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Optional

DEFAULT_MEMORY_DIR = Path(
    "/iris/u/mikulrai/.claude/projects/-iris-u-mikulrai/memory/"
)
ISO_DATE_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\b")


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (fields, body). Tolerant of missing/malformed frontmatter.

    Frontmatter is the block between the first two `---` lines at the start
    of the file. Each line inside is `key: value`. We don't try to be a
    full YAML parser — every memory file in this project uses the simple
    flat key:value form.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body_start = end + len("\n---")
    body = text[body_start:].lstrip("\n")
    fields: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        fields[k.strip()] = v.strip()
    return fields, body


def _serialize_frontmatter(fields: dict) -> str:
    lines = ["---"]
    for k, v in fields.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Date inference
# ---------------------------------------------------------------------------


def infer_last_touched(path: Path, fields: dict) -> tuple[_dt.date, str]:
    """Return (date, source) for the most-recent date we can attribute to this file.

    `source` is one of: 'frontmatter', 'description-prefix', 'mtime'.
    """
    upd = fields.get("updated")
    if upd:
        try:
            return _dt.date.fromisoformat(upd), "frontmatter"
        except ValueError:
            pass

    desc = fields.get("description", "")
    m = ISO_DATE_RE.match(desc)
    if m:
        try:
            return _dt.date.fromisoformat(m.group(0).strip()), "description-prefix"
        except ValueError:
            pass

    return _dt.date.fromtimestamp(path.stat().st_mtime), "mtime"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def collect_memory_files(memory_dir: Path) -> list[Path]:
    if not memory_dir.is_dir():
        return []
    return sorted(p for p in memory_dir.glob("*.md") if p.name != "MEMORY.md")


def audit(memory_dir: Path, max_age_days: int) -> int:
    """Print a table; exit 0 if no stale entries, 1 otherwise."""
    files = collect_memory_files(memory_dir)
    if not files:
        print(f"WARN: no memory files under {memory_dir}", file=sys.stderr)
        return 0

    today = _dt.date.today()
    threshold = today - _dt.timedelta(days=max_age_days)
    rows = []
    n_stale = 0
    for path in files:
        text = path.read_text()
        fields, _ = _split_frontmatter(text)
        date, source = infer_last_touched(path, fields)
        is_stale = date < threshold
        if is_stale:
            n_stale += 1
        rows.append((date, source, path.name, fields.get("type", "?"), is_stale))

    rows.sort(key=lambda r: r[0])  # oldest first
    name_w = max(len(r[2]) for r in rows)
    type_w = max(len(r[3]) for r in rows)
    print(f"{'last':10}  {'src':18}  {'type':<{type_w}}  {'name':<{name_w}}  flag")
    print("-" * (10 + 18 + type_w + name_w + 5 + 6))
    for date, source, name, typ, is_stale in rows:
        flag = "STALE" if is_stale else ""
        print(f"{date.isoformat()}  {source:18}  {typ:<{type_w}}  {name:<{name_w}}  {flag}")

    print()
    print(f"summary: {len(rows)} entries; {n_stale} STALE (>{max_age_days}d)")
    return 1 if n_stale else 0


# ---------------------------------------------------------------------------
# Touch
# ---------------------------------------------------------------------------


def touch_one(path: Path, today: _dt.date) -> bool:
    """Set/update the `updated:` frontmatter field on path. Return True if changed."""
    text = path.read_text()
    fields, body = _split_frontmatter(text)
    if not fields:
        # No frontmatter at all — refuse to write, that's a separate problem.
        print(f"SKIP {path.name}: no frontmatter detected", file=sys.stderr)
        return False
    new_val = today.isoformat()
    if fields.get("updated") == new_val:
        return False
    fields["updated"] = new_val
    new_text = _serialize_frontmatter(fields) + body
    path.write_text(new_text)
    return True


def touch_command(
    memory_dir: Path,
    target: Optional[str],
    all_stale: bool,
    max_age_days: int,
) -> int:
    today = _dt.date.today()
    threshold = today - _dt.timedelta(days=max_age_days)
    if target:
        path = memory_dir / target
        if not path.is_file():
            print(f"FAIL: {path} not found", file=sys.stderr)
            return 1
        changed = touch_one(path, today)
        print(f"{'touched' if changed else 'no change'}: {path.name}")
        return 0
    if not all_stale:
        print("FAIL: pass either <name.md> or --all-stale", file=sys.stderr)
        return 2
    files = collect_memory_files(memory_dir)
    n = 0
    for path in files:
        text = path.read_text()
        fields, _ = _split_frontmatter(text)
        date, _ = infer_last_touched(path, fields)
        if date >= threshold:
            continue
        if touch_one(path, today):
            n += 1
            print(f"touched: {path.name}")
    print(f"summary: touched {n} stale files")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--memory-dir",
        default=str(DEFAULT_MEMORY_DIR),
        help=f"Memory directory (default: {DEFAULT_MEMORY_DIR})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="report memory file ages + STALE flag")
    a.add_argument("--max-age-days", type=int, default=90)

    t = sub.add_parser("touch", help="set updated: today on a memory file")
    t.add_argument("target", nargs="?", help="memory filename (e.g. project_X.md)")
    t.add_argument("--all-stale", action="store_true",
                   help="touch every entry older than --max-age-days")
    t.add_argument("--max-age-days", type=int, default=90)

    args = p.parse_args(argv)
    memory_dir = Path(args.memory_dir).resolve()

    if args.cmd == "audit":
        return audit(memory_dir, args.max_age_days)
    if args.cmd == "touch":
        return touch_command(memory_dir, args.target, args.all_stale, args.max_age_days)
    return 2


if __name__ == "__main__":
    sys.exit(main())
