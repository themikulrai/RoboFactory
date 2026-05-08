"""Lint check: every canonical SLURM launcher passes a fixed safety battery.

Motivation: workflow improvement #7. Today's S6 PM go/no-go failed in 7 s
because the launcher used `source "$(dirname "$0")/_resolve_train_cfg.sh"`,
and SLURM copies the script to `/var/lib/slurm/slurmd/job<id>/` before
exec — `$(dirname "$0")` resolved to that copy dir, the helper wasn't
there, the source silently failed, and the next command read an empty
TRAIN_CFG_PATH as the cwd `'.'`. This lint catches that class of bug
before submission, plus a small number of related ones we already ate.

Each lint rule below corresponds to a real failure mode this project hit:

    R1 — `source "$(dirname "$0")/...` (today's S6 7-second failure)
    R2 — `WANDB_API_KEY` export missing (the eval_decent_pi05 lapse class)
    R3 — `#SBATCH --output=` / `--error=` must be absolute paths
    R4 — partition / GRES mismatch:
            iris-hi requires gpu count ≤ 6
            orion requires gpu_type ∈ {a5000, a6000} (a100/h100 forbidden)

Usage:
    python -m robofactory.scripts.lint.check_slurm_launcher [--root REPO_ROOT]

Exit codes:
    0 — every canonical launcher passes every rule.
    1 — at least one launcher fails at least one rule (printed to stderr).

The lint deliberately ignores:
    - `_resolve_train_cfg.sh` (sourced helper, no #SBATCH header)
    - `install_*.sh` (one-shot user installers)
    - `slurm_*.sh` under `scripts/preflight/` (those are dispatch
      orchestrators with their own validation)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

GLOB_PATTERNS = (
    "robofactory/scripts/canonical/*.sh",
)

EXCLUDED_BASENAMES = frozenset({"_resolve_train_cfg.sh", "install_kinit_renew_cron.sh"})

ORION_FORBIDDEN_GPU_TYPES = frozenset({"a100", "h100", "h200"})


# ---------------------------------------------------------------------------
# Individual rules — each returns a list of error strings (empty = pass)
# ---------------------------------------------------------------------------


def rule_R1_no_relative_source(text: str) -> list[str]:
    """source "$(dirname "$0")/..." breaks under SLURM script copy."""
    errs = []
    pattern = re.compile(r'source\s+"?\$\(\s*dirname\s+"?\$0"?\s*\)')
    for ln_no, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            errs.append(
                f"L{ln_no}: relative `source $(dirname $0)/...` will break under SLURM "
                f"(script gets copied to /var/lib/slurm/slurmd/job<id>/ before exec). "
                f"Use an absolute path. Line: {line.strip()}"
            )
    return errs


def rule_R2_wandb_api_key_exported(text: str, basename: str) -> list[str]:
    """Long-running launchers must export WANDB_API_KEY; a stale key surfaces
    as a wandb-init failure (with assert_wandb_live() now) instead of a
    silent no-op (the eval_decent_pi05 lapse class)."""
    if "calibration" in basename:
        # Calibration capture jobs don't log to wandb; skip.
        return []
    if re.search(r"^\s*export\s+WANDB_API_KEY\s*=", text, re.MULTILINE):
        return []
    return ["missing `export WANDB_API_KEY=...` (long-running jobs must log to wandb online)"]


def rule_R3_absolute_output_paths(text: str) -> list[str]:
    """#SBATCH --output and --error must be absolute paths, or SLURM resolves
    relative to the user's HOME on the *compute* node (which on iris-* is
    /iris/u/mikulrai but on orion may differ — file lands in a surprise dir)."""
    errs = []
    pattern = re.compile(r'^#SBATCH\s+(?:--output|-o|--error|-e)[\s=]+(\S+)', re.MULTILINE)
    for m in pattern.finditer(text):
        path = m.group(1)
        if not path.startswith("/"):
            errs.append(f"#SBATCH path '{path}' is not absolute — output may land in a surprise dir on a different compute node.")
    return errs


def rule_R4_partition_gres_match(text: str) -> list[str]:
    """iris-hi has a 6-GPU/user cap; orion forbids a100/h100/h200."""
    errs = []
    partition_match = re.search(r'^#SBATCH\s+--partition[\s=]+(\S+)', text, re.MULTILINE)
    gres_match = re.search(r'^#SBATCH\s+--gres[\s=]+(\S+)', text, re.MULTILINE)
    if not partition_match or not gres_match:
        return errs
    partition = partition_match.group(1)
    gres = gres_match.group(1)

    # GRES forms we recognise: gpu:N, gpu:<type>:N
    gpu_count = 0
    gpu_type = ""
    if gres.startswith("gpu:"):
        parts = gres.split(":")
        if len(parts) == 2:
            try:
                gpu_count = int(parts[1])
            except ValueError:
                pass
        elif len(parts) >= 3:
            gpu_type = parts[1]
            try:
                gpu_count = int(parts[2])
            except ValueError:
                pass

    if partition == "iris-hi" and gpu_count > 6:
        errs.append(f"partition=iris-hi but --gres requests {gpu_count} GPUs (cap is 6/user).")
    if partition == "orion" and gpu_type and gpu_type.lower() in ORION_FORBIDDEN_GPU_TYPES:
        errs.append(f"partition=orion but --gres requests gpu_type='{gpu_type}' (forbidden; only a5000/a6000 allowed).")
    return errs


RULES = [
    ("R1_no_relative_source", rule_R1_no_relative_source),
    ("R2_wandb_api_key", rule_R2_wandb_api_key_exported),
    ("R3_absolute_output_paths", rule_R3_absolute_output_paths),
    ("R4_partition_gres_match", rule_R4_partition_gres_match),
]


def lint_one(path: Path) -> list[str]:
    """Return list of '<rule>: <message>' for path; empty if clean.

    Files with no `#SBATCH` directives at all are wrappers (e.g.
    submit_with_preflights.sh) and skip the SLURM-specific rules.
    """
    text = path.read_text()
    if not re.search(r"^#SBATCH\s", text, re.MULTILINE):
        # Still apply R1 (relative source paths) — wrappers can have that bug too.
        errs = rule_R1_no_relative_source(text)
        return [f"R1_no_relative_source: {e}" for e in errs]
    out = []
    for name, fn in RULES:
        if fn is rule_R2_wandb_api_key_exported:
            errs = fn(text, path.name)
        else:
            errs = fn(text)
        for e in errs:
            out.append(f"{name}: {e}")
    return out


def _collect_launchers(root: Path) -> list[Path]:
    seen: set[Path] = set()
    for pat in GLOB_PATTERNS:
        for p in root.glob(pat):
            if p.name in EXCLUDED_BASENAMES:
                continue
            if not p.is_file():
                continue
            seen.add(p.resolve())
    return sorted(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repo root (default: 3 levels up from this file).",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    launchers = _collect_launchers(root)
    if not launchers:
        print(f"WARN: no launchers matched under {root} (looked at {GLOB_PATTERNS})", file=sys.stderr)
        return 0

    bad = 0
    for path in launchers:
        errs = lint_one(path)
        if errs:
            bad += 1
            rel = path.relative_to(root)
            print(f"FAIL: {rel}", file=sys.stderr)
            for e in errs:
                print(f"  {e}", file=sys.stderr)
    if bad:
        print(f"\n{bad}/{len(launchers)} launcher(s) failed lint.", file=sys.stderr)
        return 1
    print(f"OK: all {len(launchers)} launcher(s) passed lint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
