"""Thin CLI shim for the 23-column manifest CSV generator.

The actual schema, parsers, and merge logic live in
``robofactory.utils.manifest``. This module exists so that
``python -m robofactory.utils.manifest_csv`` is a single, network-tolerant
entry point that:

  1. fetches wandb runs (best-effort; soft-fail on network errors),
  2. loads the ckpt index produced by CkptResolver,
  3. walks the canonical RUN_ROOT directory tree,
  4. merges them into one CSV at MANIFEST_PATH.

CLI:
    python -m robofactory.utils.manifest_csv [--output PATH] [--no-wandb]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robofactory.utils import paths
from robofactory.utils.manifest import (
    MANIFEST_COLUMNS,  # re-exported for tests / downstream tooling
    RunRecord,         # re-exported
    generate_manifest_csv,
)

__all__ = ["MANIFEST_COLUMNS", "RunRecord", "generate_manifest_csv", "main"]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=paths.MANIFEST_PATH,
        help=f"Output CSV path (default: {paths.MANIFEST_PATH})",
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="Skip the wandb API call (filesystem + ckpt-index only)",
    )
    p.add_argument(
        "--wandb-entity", default="mikulrai-stanford-university",
    )
    p.add_argument(
        "--wandb-projects",
        default="diffusion-robofactory,openpi-robofactory",
        help="Comma-separated wandb project names",
    )
    args = p.parse_args(argv)

    projects = tuple(s.strip() for s in args.wandb_projects.split(",") if s.strip())
    n = generate_manifest_csv(
        out_path=args.output,
        fetch_wandb=not args.no_wandb,
        wandb_entity=args.wandb_entity,
        wandb_projects=projects,
    )
    print(f"wrote {n} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
