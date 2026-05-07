"""Stage-3 per-eval guards (plan v2 C1#10).

Run inside an eval SLURM job before any GPU work. Crashes the job loudly
(non-zero exit) when a guard fails so we don't burn compute on a broken
launch.

Checks (each can be disabled via flag):
  1. wandb-online       — WANDB_MODE/_DISABLED clean, WANDB_API_KEY set
  2. ckpt-path-exists   — --ckpt-path is a real file
  3. scene-config-exists — --scene-config is a real file
  4. scene-name match   — heuristic: eval-config scene token agrees with
                           the ckpt-dir scene token (catches scene mismatch
                           like running a TSC ckpt against a robocasa cfg).
  5. seed-file SHA      — sha256(--seeds-file) == --seeds-sha256

CLI:
  python -m robofactory.utils.preflight_eval \
      --ckpt-path PATH \
      --scene-config configs/table/pick_meat.yaml \
      [--seeds-file PATH --seeds-sha256 HEX] \
      [--no-wandb-online] [--no-scene-name-match]

Designed to be sourced from eval launcher .sh files immediately after
`conda activate` and before the eval entrypoint invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Individual checks (each returns None on success, raises on failure)
# ---------------------------------------------------------------------------

class PreflightFailure(RuntimeError):
    """Raised when a Stage-3 guard fails. Caught only by main() -> exit 1."""


def check_wandb_online(env: Optional[dict[str, str]] = None) -> None:
    """Refuse to run if wandb is offline / disabled / unauthenticated."""
    e = env if env is not None else os.environ
    mode = e.get("WANDB_MODE", "online")
    if mode != "online":
        raise PreflightFailure(
            f"WANDB_MODE={mode!r}; only 'online' is permitted (no offline runs)."
        )
    if e.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        raise PreflightFailure(
            "WANDB_DISABLED is set — refusing to run a wandb-blind eval."
        )
    if not e.get("WANDB_API_KEY"):
        raise PreflightFailure(
            "WANDB_API_KEY not set in environment; "
            "wandb.init() would fail silently or block."
        )


def check_path_exists(label: str, path: Path) -> None:
    if not path.exists():
        raise PreflightFailure(f"{label} does not exist: {path}")


# Scene-name tokens we know about. Each entry is (yaml_token, ckpt_token).
# Eval YAML at configs/table/<yaml_token>.yaml goes with ckpt dirs whose
# basename starts with <ckpt_token>.
_SCENE_PAIRS = (
    ("pick_meat", "PickMeat-rf"),
    ("three_robots_stack_cube", "ThreeRobotsStackCube-rf"),
    ("two_robots_stack_cube", "TwoRobotsStackCube-rf"),
    ("long_pipeline", "LongPipeline-rf"),
)


def check_scene_name_match(scene_config: Path, ckpt_path: Path) -> None:
    """Heuristic scene-mismatch guard.

    Looks at the eval YAML stem (e.g. 'pick_meat') and the ckpt dir name
    (e.g. 'PickMeat-rf_150') and asserts they refer to the same task.

    A future BaseEvalDriver (Phase C2) will replace this with a real read
    of the trainer-dumped .hydra_config.yaml. Until then this catches the
    most common mistake — running a ckpt against a sibling task's config.
    """
    yaml_stem = scene_config.stem
    # ckpt_path is /.../<task_dir>/<step>.ckpt or /.../<task_dir>/<step>/...
    # walk up until we find a known task prefix.
    ckpt_dir_name = ""
    for parent in [ckpt_path, *ckpt_path.parents]:
        for _, ckpt_token in _SCENE_PAIRS:
            if parent.name.startswith(ckpt_token):
                ckpt_dir_name = parent.name
                break
        if ckpt_dir_name:
            break

    if not ckpt_dir_name:
        # Unknown ckpt-dir convention — can't verify. Be permissive but warn.
        print(
            f"[preflight_eval] WARNING: scene-name match skipped — "
            f"ckpt path has no recognized task prefix: {ckpt_path}",
            file=sys.stderr,
        )
        return

    expected_yaml: Optional[str] = None
    for yaml_tok, ckpt_tok in _SCENE_PAIRS:
        if ckpt_dir_name.startswith(ckpt_tok):
            expected_yaml = yaml_tok
            break

    if expected_yaml is None or yaml_stem != expected_yaml:
        raise PreflightFailure(
            f"scene-name mismatch: eval config {scene_config} (stem={yaml_stem!r}) "
            f"does not match ckpt dir {ckpt_dir_name!r} "
            f"(expected stem={expected_yaml!r})"
        )


def check_seed_file_sha256(seeds_file: Path, expected_hex: str) -> None:
    if not seeds_file.exists():
        raise PreflightFailure(f"seeds file not found: {seeds_file}")
    h = hashlib.sha256(seeds_file.read_bytes()).hexdigest()
    if h.lower() != expected_hex.lower():
        raise PreflightFailure(
            f"seeds file {seeds_file} sha256={h} but expected {expected_hex}"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all_checks(args: argparse.Namespace) -> list[str]:
    """Run every requested check. Returns a list of human-readable PASS lines.
    Raises PreflightFailure on the first failure."""
    passed: list[str] = []

    if args.wandb_online:
        check_wandb_online()
        passed.append("wandb-online: OK")

    if args.ckpt_path is not None:
        check_path_exists("ckpt-path", args.ckpt_path)
        passed.append(f"ckpt-path-exists: {args.ckpt_path}")

    if args.scene_config is not None:
        check_path_exists("scene-config", args.scene_config)
        passed.append(f"scene-config-exists: {args.scene_config}")

    if args.scene_name_match and args.scene_config and args.ckpt_path:
        check_scene_name_match(args.scene_config, args.ckpt_path)
        passed.append("scene-name-match: OK")

    if args.seeds_file is not None and args.seeds_sha256:
        check_seed_file_sha256(args.seeds_file, args.seeds_sha256)
        passed.append(f"seeds-sha256: matches {args.seeds_file}")

    return passed


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-path", type=Path, default=None)
    p.add_argument("--scene-config", type=Path, default=None)
    p.add_argument("--seeds-file", type=Path, default=None)
    p.add_argument("--seeds-sha256", default=None,
                   help="64-char sha256 hex of --seeds-file. If unset, the "
                        "seed-file check is skipped.")
    # Toggles — default ON.
    p.add_argument("--no-wandb-online", dest="wandb_online",
                   action="store_false", default=True)
    p.add_argument("--no-scene-name-match", dest="scene_name_match",
                   action="store_false", default=True)
    args = p.parse_args(argv)

    try:
        passed = run_all_checks(args)
    except PreflightFailure as exc:
        print(f"[preflight_eval] FAIL: {exc}", file=sys.stderr)
        return 1

    print("[preflight_eval] all checks passed:")
    for line in passed:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
