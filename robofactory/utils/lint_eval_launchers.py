"""Offline lint over RoboFactory eval launcher shell scripts.

Catches the launcher-drift patterns that Stage-3 preflight's argv↔seed-file
cross-check enforces at runtime — but at commit time, so a bad launcher never
even makes it to slurm.

Scope: ``scripts/{canonical,ablations}/*_60seeds.sh`` under the RoboFactory
project root. Smoke / single-seed / overfit scripts are out of scope.

Rules
-----
A) The preflight call must NOT use the empty-fallback ``--expected-sha256
   "${EVAL_SEEDS_SHA256:-}"`` form. That used to silently soft-warn through;
   the new preflight hard-fails it, but the lint catches it one layer earlier.
   Use ``--expected-sha256-file <pin_path>`` instead.

B) The launcher must define a ``SEEDS=$(paste -sd<delim> <seed_file>)`` line
   so the same seed list feeds both preflight (via ``--argv-seeds "$SEEDS"``)
   and the eval driver (via ``-s $SEEDS`` or ``--seeds "$SEEDS"``).

C) The path in ``--seed-file <path>`` must equal the seed file used by
   ``SEEDS=$(paste -sd... <path>)``.

D) The preflight call must pass ``--argv-seeds "$SEEDS"``. Without it, the
   runtime cross-check is bypassed.

E) The eval driver's seed argv must be ``$SEEDS`` (unquoted for DP nargs='+'
   consumers, quoted for Pi0.5 comma-list consumers). A literal multi-line
   numeric list is the drift pattern this lint is designed to kill.

Exit code: 0 if clean, 1 if any violation found.

Usage
-----

    python -m robofactory.utils.lint_eval_launchers
    python -m robofactory.utils.lint_eval_launchers --root /path/to/repo
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]  # robofactory/
LAUNCHER_GLOBS = (
    "scripts/canonical/*_60seeds.sh",
    "scripts/canonical/*_60seeds_*.sh",  # variants e.g. *_60seeds_reeval.sh
    "scripts/ablations/*_60seeds.sh",
    "scripts/ablations/*_60seeds_*.sh",
)


@dataclass
class LauncherFinding:
    path: Path
    line_no: int
    rule: str  # A | B | C | D | E
    message: str


@dataclass
class LauncherReport:
    path: Path
    findings: list[LauncherFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings


# Regex patterns
RE_EMPTY_SHA_FALLBACK = re.compile(r'--expected-sha256\s+["\']?\$\{[^}]*:-[^}]*\}["\']?')
RE_SEEDS_FROM_PASTE = re.compile(
    # delim can be '<x>' (quoted, often `' '`) or a single non-space token (e.g. `,`).
    r"^\s*SEEDS\s*=\s*\$\(\s*paste\s+-sd(?:'[^']*'|\"[^\"]*\"|\S+)\s+(\S+)\s*\)",
    re.MULTILINE,
)
RE_SEED_FILE_FLAG = re.compile(r'--seed-file\s+(\S+)')
RE_ARGV_SEEDS_FLAG = re.compile(r'--argv-seeds\s+["\']?\$\{?SEEDS\}?["\']?')
RE_PREFLIGHT_BLOCK = re.compile(
    # Capture `preflight_eval_guards` and the bash command continuation lines
    # that follow (ending at the first newline NOT preceded by `\`).
    r'preflight_eval_guards\b(?:[^\n]*\\\n)*[^\n]*',
)
# A literal multi-line seed list looks like `-s 10000 10001 ... 1029 \\`
# spanning several backslash-continued lines. We detect "the driver got
# 5+ integer tokens in a row right after -s or --seeds or --seed".
RE_LITERAL_SEED_BLOCK = re.compile(
    r'(?:^|\s)(-s|--seed|--seeds)\s+(?:\d+\s+){5,}',
    re.MULTILINE,
)


def _line_no_of(text: str, span_start: int) -> int:
    return text.count("\n", 0, span_start) + 1


def lint_one(path: Path) -> LauncherReport:
    rpt = LauncherReport(path=path)
    text = path.read_text()

    # Rule A — empty-fallback sha pattern is banned.
    for m in RE_EMPTY_SHA_FALLBACK.finditer(text):
        rpt.findings.append(LauncherFinding(
            path=path,
            line_no=_line_no_of(text, m.start()),
            rule="A",
            message=(
                "uses --expected-sha256 with an empty-fallback env var "
                f'({m.group(0)!r}). Use --expected-sha256-file <pin> instead.'
            ),
        ))

    # Rule B — must define SEEDS via `paste -sd<delim> <seed_file>`.
    paste_match = RE_SEEDS_FROM_PASTE.search(text)
    if paste_match is None:
        rpt.findings.append(LauncherFinding(
            path=path,
            line_no=0,
            rule="B",
            message=(
                "no `SEEDS=$(paste -sd<delim> <seed_file>)` line found. "
                "Drive both preflight and the driver from one file via $SEEDS."
            ),
        ))
        paste_file: str | None = None
    else:
        paste_file = paste_match.group(1)

    # Rule C — --seed-file path must match the paste source.
    seed_file_match = RE_SEED_FILE_FLAG.search(text)
    if seed_file_match is None:
        rpt.findings.append(LauncherFinding(
            path=path,
            line_no=0,
            rule="C",
            message="no `--seed-file <path>` flag found in preflight call.",
        ))
    elif paste_file is not None:
        seed_file_arg = seed_file_match.group(1)
        if seed_file_arg != paste_file:
            rpt.findings.append(LauncherFinding(
                path=path,
                line_no=_line_no_of(text, seed_file_match.start()),
                rule="C",
                message=(
                    f"--seed-file is {seed_file_arg!r} but SEEDS is pasted from "
                    f"{paste_file!r}. They must be the same path."
                ),
            ))

    # Rule D — preflight call must include --argv-seeds "$SEEDS".
    preflight_match = RE_PREFLIGHT_BLOCK.search(text)
    if preflight_match is None:
        rpt.findings.append(LauncherFinding(
            path=path,
            line_no=0,
            rule="D",
            message="no `preflight_eval_guards` invocation found.",
        ))
    else:
        block = preflight_match.group(0)
        if not RE_ARGV_SEEDS_FLAG.search(block):
            rpt.findings.append(LauncherFinding(
                path=path,
                line_no=_line_no_of(text, preflight_match.start()),
                rule="D",
                message=(
                    "preflight call is missing `--argv-seeds \"$SEEDS\"`. "
                    "Without it, the runtime argv↔file cross-check is skipped."
                ),
            ))

    # Rule E — eval driver must not receive a literal multi-line seed list.
    for m in RE_LITERAL_SEED_BLOCK.finditer(text):
        rpt.findings.append(LauncherFinding(
            path=path,
            line_no=_line_no_of(text, m.start()),
            rule="E",
            message=(
                "literal multi-line seed list passed to the eval driver. "
                "Replace with `-s $SEEDS` (DP) or `--seeds \"$SEEDS\"` (Pi0.5)."
            ),
        ))

    return rpt


def collect_launchers(root: Path) -> list[Path]:
    seen: set[Path] = set()
    for glob in LAUNCHER_GLOBS:
        for p in root.glob(glob):
            seen.add(p.resolve())
    return sorted(seen)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="RoboFactory project root (default: this file's repo root).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Also print OK files (default: only print files with findings).",
    )
    args = p.parse_args(argv)

    launchers = collect_launchers(args.root)
    if not launchers:
        print(
            f"[lint_eval_launchers] no launchers matched globs "
            f"{list(LAUNCHER_GLOBS)} under {args.root}",
            file=sys.stderr,
        )
        return 1

    reports = [lint_one(p) for p in launchers]
    total_findings = 0
    bad_files = 0
    for rpt in reports:
        if rpt.ok:
            if args.verbose:
                print(f"OK  {rpt.path.relative_to(args.root)}")
            continue
        bad_files += 1
        total_findings += len(rpt.findings)
        for f in rpt.findings:
            rel = f.path.relative_to(args.root)
            print(f"{rel}:{f.line_no}: rule {f.rule}: {f.message}")

    if total_findings:
        print(
            f"\n[lint_eval_launchers] {total_findings} finding(s) across "
            f"{bad_files} file(s).",
            file=sys.stderr,
        )
        return 1

    print(f"[lint_eval_launchers] clean: {len(launchers)} launcher(s) checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
