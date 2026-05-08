"""Lint check: every `eval_*.py` driver must use `with WandbRun(` or
`with EvalRunContext(` somewhere in its call graph.

Motivation: in C1 we found ~30 GPU-hours of decentralised pi0.5 eval that
had run with no wandb run at all because `wandb.init` was wrapped in
`if args.wandb:` and the launch flag had been dropped. Forcing a context
manager (which does `assert_wandb_live()` in `__enter__`) makes the
failure mode "loud crash on launch" instead of "silent no-op for hours".

Usage:
    python robofactory/scripts/lint/check_eval_drivers.py [--root REPO_ROOT]

Exit codes:
    0 — every driver under <root>/robofactory/policy/**/eval_*.py contains
        the required `with` statement.
    1 — at least one driver lacks the pattern (path printed to stderr).

The lint deliberately ignores:
    - `eval_context.py` itself (it's the implementation, not a driver).
    - Any file whose name starts with `test_` (tests can mock around it).
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ALLOWED_CONTEXT_NAMES = frozenset({"WandbRun", "EvalRunContext"})

# File globs to lint (relative to --root). The first hit wins; we walk every
# path matching any pattern and dedupe.
GLOB_PATTERNS = (
    "robofactory/policy/**/eval_*.py",
    # The openpi tree (if present alongside RoboFactory in the same monorepo)
    # has its own eval scripts; opt-in via convention.
    "projects/openpi/scripts/eval*.py",
)

EXCLUDED_BASENAMES = frozenset({"eval_context.py"})


def _is_allowed_with(node: ast.With) -> bool:
    """True iff any item in this `with` calls one of the allowed context managers."""
    for item in node.items:
        ctx = item.context_expr
        if isinstance(ctx, ast.Call):
            func = ctx.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in ALLOWED_CONTEXT_NAMES:
                return True
    return False


def _file_uses_required_pattern(path: Path) -> bool:
    """AST-walk `path` and return True iff it contains a `with WandbRun(` /
    `with EvalRunContext(` somewhere (regardless of nesting)."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        # Treat unparseable as "fails lint" — the file is almost certainly
        # broken and the dev needs to know.
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)) and _is_allowed_with(node):
            return True
    return False


def _collect_drivers(root: Path) -> list[Path]:
    seen: set[Path] = set()
    for pattern in GLOB_PATTERNS:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            if p.name in EXCLUDED_BASENAMES:
                continue
            if p.name.startswith("test_"):
                continue
            seen.add(p.resolve())
    return sorted(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repo root to scan from (default: this script's repo root).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    drivers = _collect_drivers(root)

    if not drivers:
        print(f"[check_eval_drivers] no eval_*.py drivers found under {root}", file=sys.stderr)
        return 0

    bad: list[Path] = []
    for d in drivers:
        if not _file_uses_required_pattern(d):
            bad.append(d)

    if bad:
        print(
            "[check_eval_drivers] FAIL: the following drivers lack a "
            "`with WandbRun(` / `with EvalRunContext(` block:",
            file=sys.stderr,
        )
        for p in bad:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nFix: import `WandbRun` from `policy._shared.eval_context` and "
            "wrap the eval loop in a `with WandbRun(...) as wandb_run:` block.",
            file=sys.stderr,
        )
        return 1

    print(f"[check_eval_drivers] OK: {len(drivers)} drivers checked.")
    for d in drivers:
        print(f"  - {d.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
