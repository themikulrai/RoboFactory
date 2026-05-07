"""Lint check: every production eval driver must use the shared WandbRun context manager.

Why:
    Plan v2 root cause for the lost ~30 GPU-hours of `eval_decent_pi05.py` runs was
    that the driver was a fork of the centralised version that never got the wandb
    plumbing copied over. By construction, this lint forces every `eval_*.py` under
    `policy/` to use `with WandbRun(...)` — a future LLM that writes a fresh driver
    cannot accidentally skip wandb.

Scope:
    Only files matching `policy/**/eval_*.py` are checked.  Debug-only one-off
    scripts under `script/debug/eval_*.py` are excluded.

Run:
    python script/lint/check_eval_drivers.py

Exit code 0 if every driver uses `with WandbRun(`, else 1 (with a per-file report).
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_GLOB = "policy/**/eval_*.py"
EXCLUDE_NAMES = {"eval_context.py"}
REQUIRED_TOKEN = "with WandbRun("


def find_drivers() -> list[Path]:
    return sorted(
        p
        for p in REPO_ROOT.glob(DRIVER_GLOB)
        if p.name not in EXCLUDE_NAMES and "__pycache__" not in p.parts
    )


def check(path: Path) -> bool:
    src = path.read_text()
    return REQUIRED_TOKEN in src


def main() -> int:
    drivers = find_drivers()
    if not drivers:
        print(f"ERROR: no eval drivers found under {REPO_ROOT}/{DRIVER_GLOB}", file=sys.stderr)
        return 1

    failed: list[Path] = []
    for d in drivers:
        if check(d):
            print(f"OK   {d.relative_to(REPO_ROOT)}")
        else:
            print(f"FAIL {d.relative_to(REPO_ROOT)}  (missing `{REQUIRED_TOKEN}`)")
            failed.append(d)

    if failed:
        print()
        print(
            f"{len(failed)}/{len(drivers)} driver(s) miss the shared WandbRun context manager.\n"
            f"Each must use `with WandbRun(...) as wandb_run:` from "
            f"`policy._shared.eval_context`. See `policy/Diffusion-Policy/eval_dp.py` "
            f"for the canonical example.",
            file=sys.stderr,
        )
        return 1

    print(f"\n{len(drivers)} driver(s) checked, all use WandbRun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
