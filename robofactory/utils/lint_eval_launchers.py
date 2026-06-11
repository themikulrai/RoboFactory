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

F) No hardcoded ``80[0-9][0-9]`` PORT literal may appear in a targeted launcher
   (``scripts/{canonical,ablations}/*_60seeds.sh`` or any
   ``sbatch_hierarchical_*.sh`` at the repo root). Co-scheduled evals that share
   a hardcoded server port silently route a client to the wrong policy server
   (solution_E6.md §PR1). Ports must be allocated job-uniquely via
   ``_lib/free_ports.sh`` (bash) or ``free_ports()`` (python). Checkpoint-step
   path components (e.g. ``/18000``, ``_v1/18000``) are NOT flagged — only
   standalone ``80xx`` numeric tokens that are not part of a longer number and
   are not preceded by a path/word character.

G) A 60-seed launcher must source its seeds from the single seed-convention
   module (solution_E6.md §PR4): either a ``--seed-pool <name>`` flag (a name in
   ``robofactory.utils.eval_seeds.POOL_NAMES``) OR one of the PINNED env-seed
   files under ``/iris/u/mikulrai/runs/`` (``eval_seeds_env_60.txt`` =
   canonical_env_60, the frozen ``eval_seeds_60_dp.txt`` = dp_legacy, or
   ``eval_seeds_60.txt``). A launcher with neither is feeding raw/ad-hoc seeds —
   the exact pattern that produced never-seed-paired cross-method comparisons and
   the silently-unpaired "paired" manifest entry. A ``--seed-pool`` referencing an
   unknown pool name is also flagged.

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
# Rule F also covers the repo-root hierarchical sbatch launchers, which live in
# `<repo>/scripts/` (one level above `robofactory/`), NOT under `robofactory/`.
SBATCH_HIER_GLOBS = (
    "scripts/sbatch_hierarchical_*.sh",
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
# Rule F — a hardcoded 80xx PORT literal. Match a standalone `80[0-9][0-9]`
# token: not preceded by a digit (so `18000` -> the inner `8000` is rejected),
# not preceded by a path/word char `/`, `_`, `.`, `-` (so ckpt-step paths like
# `/18000`, `_v1/18000`, `step-8000` are not flagged), and not followed by a
# digit (so `80001` is rejected). Bare port usages — `8000`, `8000,8001`,
# `8000/8001`, `,8000)`, `--port 8000`, `start_server 0 8000` — DO match.
RE_HARDCODED_PORT = re.compile(r'(?<![\w./-])80[0-9][0-9](?![0-9])')

# Rule G — seed-convention sourcing. A 60-seed launcher must source its seeds from
# the single seed-convention module (PR4): either a `--seed-pool <name>` flag or a
# PINNED env-seed file. The pinned files are the frozen seed pins under
# /iris/u/mikulrai/runs/ (matched by basename so the lint is path-prefix robust).
RE_SEED_POOL_FLAG = re.compile(r'--seed-pool[=\s]+([A-Za-z0-9_]+)')
# Pinned env-seed files (basenames). canonical_env_60, the frozen dp_legacy pin, and
# the historical bases pin (kept for launchers that still reference it).
PINNED_SEED_FILES = (
    "eval_seeds_env_60.txt",
    "eval_seeds_60_dp.txt",
    "eval_seeds_60.txt",
)
RE_PINNED_SEED_FILE = re.compile(
    r'(?:' + "|".join(re.escape(name) for name in PINNED_SEED_FILES) + r')'
)
# Does the launcher set seeds at all? (If it never passes seeds, rule G is N/A — the
# driver default / manifest handles it; rule G only governs how an explicitly-seeded
# 60-seed launcher names its seeds.)
RE_SETS_SEEDS = re.compile(
    r'(?:^|\s)(?:-s|--seeds?|--env-seeds|--seed-pool)\b'
    r'|^\s*SEEDS\s*='
    r'|\bpaste\s+-sd',
    re.MULTILINE,
)
# Valid pool names live in eval_seeds; import lazily so the lint has no hard dep at
# import time (e.g. when invoked outside the RoboFactory env). Falls back to a static
# copy if the import fails.
def _valid_pool_names() -> frozenset:
    try:
        from robofactory.utils.eval_seeds import POOL_NAMES
        return POOL_NAMES
    except Exception:
        return frozenset({
            "train_datagen", "canonical_env_60", "fresh_ood_60",
            "upstream_100", "dp_legacy_60",
        })


def _line_no_of(text: str, span_start: int) -> int:
    return text.count("\n", 0, span_start) + 1


def _rel_display(path: Path, root: Path) -> str:
    """Path for display, robust to launchers outside `root` (the repo-root sbatch
    hier scripts live at root.parent, which relative_to(root) cannot express)."""
    for base in (root, root.parent):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


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

    # Rule F — no hardcoded 80xx PORT literal. Scanned per-line so we can skip
    # pure-comment lines (a comment explaining the free-port fix is allowed).
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # comment line: free-port explanation is fine
        if RE_HARDCODED_PORT.search(line):
            for pm in RE_HARDCODED_PORT.finditer(line):
                rpt.findings.append(LauncherFinding(
                    path=path,
                    line_no=i,
                    rule="F",
                    message=(
                        f"hardcoded port literal {pm.group(0)!r}. Allocate "
                        "job-unique ports via _lib/free_ports.sh (bash) or "
                        "free_ports() (python) — a shared hardcoded port routes "
                        "co-scheduled clients to the wrong policy server (PR1)."
                    ),
                ))

    # Rule G — seed-convention sourcing (PR4). Only 60-seed launchers are in scope
    # (sbatch_hierarchical_*.sh are covered for rule F only). A launcher that sets
    # seeds at all must do so via --seed-pool <valid-name> OR a pinned env-seed file.
    if "60seeds" in path.name and RE_SETS_SEEDS.search(text):
        pool_m = RE_SEED_POOL_FLAG.search(text)
        pinned_m = RE_PINNED_SEED_FILE.search(text)
        if pool_m is not None:
            pool_name = pool_m.group(1)
            if pool_name not in _valid_pool_names():
                rpt.findings.append(LauncherFinding(
                    path=path,
                    line_no=_line_no_of(text, pool_m.start()),
                    rule="G",
                    message=(
                        f"--seed-pool {pool_name!r} is not a known pool. Valid pools: "
                        f"{sorted(_valid_pool_names())} (robofactory.utils.eval_seeds)."
                    ),
                ))
        elif pinned_m is None:
            rpt.findings.append(LauncherFinding(
                path=path,
                line_no=0,
                rule="G",
                message=(
                    "60-seed launcher sets seeds but uses neither `--seed-pool "
                    "<name>` nor a PINNED env-seed file "
                    f"({list(PINNED_SEED_FILES)}). Raw/ad-hoc seeds are the "
                    "never-seed-paired pattern PR4 retires — drive seeds from "
                    "robofactory.utils.eval_seeds (solution_E6.md §PR4)."
                ),
            ))

    return rpt


def collect_launchers(root: Path) -> list[Path]:
    seen: set[Path] = set()
    for glob in LAUNCHER_GLOBS:
        for p in root.glob(glob):
            seen.add(p.resolve())
    # Rule-F also covers repo-root hierarchical sbatch launchers. `root` is the
    # `robofactory/` package dir; the sbatch launchers live in `<repo>/scripts/`
    # (root.parent/scripts). Glob both root and root.parent to be robust to
    # whether `root` is passed as the package dir or the repo root.
    for base in (root, root.parent):
        for glob in SBATCH_HIER_GLOBS:
            for p in base.glob(glob):
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
                print(f"OK  {_rel_display(rpt.path, args.root)}")
            continue
        bad_files += 1
        total_findings += len(rpt.findings)
        for f in rpt.findings:
            rel = _rel_display(f.path, args.root)
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
