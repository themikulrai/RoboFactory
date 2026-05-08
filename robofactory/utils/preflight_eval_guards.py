"""Stage-3 per-eval guards (Phase C2 / Tier 1.2).

Refuses to run an eval if any of the four invariants below are violated.
Designed to be invoked at the very top of a `*_eval*.sh` SLURM script,
BEFORE the GPU is allocated (well, before any expensive work) so that we
don't burn compute on a misconfigured launch.

Invariants:
  1. Scene match — train cfg's task / env_id agrees with eval cfg's.
  2. Camera-family match — train and eval agree on `*_workspace_*` vs
     `*_wristcam_*`.
  3. Eval seed file frozen — `sha256(seed_file) == EVAL_SEEDS_60_SHA256`.
  4. Wandb live — `WANDB_MODE=online`, `WANDB_DISABLED` unset, API key set.

Each guard has its own exception type so callers (and tests) can target
specific failures. `run_all_guards` raises on the FIRST failure — there is
no point in checking camera match if scene match already failed.

CLI:
    python -m robofactory.utils.preflight_eval_guards \\
        --train-cfg /path/to/.hydra_config.yaml \\
        --eval-cfg  /path/to/eval/config.yaml \\
        --seed-file /iris/u/mikulrai/runs/eval_seeds_60.txt \\
        --expected-sha256 <hex>          # optional; defaults to pinned constant

Exits 0 on success, 1 on first guard failure.

Note: a sister module `preflight_eval` (singular) ships an older
ckpt-name-based heuristic. This new module is the proper cfg-vs-cfg path
the plan v2 calls for, so we keep both during the transition; new sbatch
templates should call THIS one.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Pinned hash for the canonical eval seed file. Update this in lock-step with
# manifest.csv whenever the seed list legitimately changes.
# ---------------------------------------------------------------------------
EVAL_SEEDS_60_SHA256 = (
    "a97e2f8be50aca97e56db992f5c41420b1556b0732fcd907e1c31c61d14f3816"
)


# ---------------------------------------------------------------------------
# Exceptions — one per invariant so tests / callers can target them.
# ---------------------------------------------------------------------------
class PreflightGuardError(Exception):
    """Base class for all Stage-3 guard failures."""


class SceneMismatchError(PreflightGuardError):
    """Train and eval configs disagree on the task / env id."""


class CameraFamilyMismatchError(PreflightGuardError):
    """Train and eval configs disagree on workspace vs wristcam cameras."""


class EvalSeedFileDriftError(PreflightGuardError):
    """sha256 of eval-seed file does not match the pinned/expected hash."""


class WandbOfflineError(PreflightGuardError):
    """Wandb is offline / disabled / unauthenticated."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file. Returns {} for empty docs. Raises on missing files."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PreflightGuardError(
            f"expected mapping at top of {path}, got {type(data).__name__}"
        )
    return data


def _walk_keys(cfg: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Optional[Any]:
    """Try each dotted key path in order, return the first non-None hit."""
    for keys in paths:
        cur: Any = cfg
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def _extract_env_id(cfg: dict[str, Any]) -> Optional[str]:
    """Pull the task / env identifier out of a heterogeneous YAML cfg.

    Handles:
      - hydra-style trainer dump: cfg.task.env_runner.env_id
      - shorter trainer dump:    cfg.env_runner.env_id
      - top-level:               cfg.env_id
      - eval-style yaml:         cfg.task_name (RoboFactory eval cfgs)
    """
    val = _walk_keys(
        cfg,
        (
            ("task", "env_runner", "env_id"),
            ("env_runner", "env_id"),
            ("env_id",),
            ("task", "task_name"),
            ("task_name",),
        ),
    )
    if val is None:
        return None
    return str(val)


def _extract_camera_family(cfg: dict[str, Any]) -> Optional[str]:
    """Return 'workspace' or 'wristcam' or None if undetermined."""
    candidate = _walk_keys(
        cfg,
        (
            ("task", "obs_cam_family"),
            ("obs_cam_family",),
            ("task", "dataset_path"),
            ("task", "data", "zarr_path"),
            ("data", "zarr_path"),
            ("zarr_path",),
            ("dataset_path",),
            ("task", "dataset"),
        ),
    )
    if candidate is None:
        return None
    s = str(candidate)
    if "_workspace_" in s or s == "workspace":
        return "workspace"
    if "_wristcam_" in s or s == "wristcam":
        return "wristcam"
    return None


# ---------------------------------------------------------------------------
# Individual guards
# ---------------------------------------------------------------------------
def assert_scene_match(train_cfg_path: Path, eval_cfg_path: Path) -> None:
    """Crash if train and eval configs disagree on the task identifier."""
    train_cfg = _load_yaml(train_cfg_path)
    eval_cfg = _load_yaml(eval_cfg_path)
    train_id = _extract_env_id(train_cfg)
    eval_id = _extract_env_id(eval_cfg)
    if train_id is None or eval_id is None:
        warnings.warn(
            f"scene-match check skipped — missing env_id "
            f"(train={train_id!r}, eval={eval_id!r}) in "
            f"{train_cfg_path} / {eval_cfg_path}",
            stacklevel=2,
        )
        return
    if train_id != eval_id:
        raise SceneMismatchError(
            f"scene mismatch: train cfg ({train_cfg_path}) has env_id={train_id!r} "
            f"but eval cfg ({eval_cfg_path}) has env_id={eval_id!r}"
        )


def assert_camera_family_match(train_cfg_path: Path, eval_cfg_path: Path) -> None:
    """Crash if train and eval configs disagree on workspace vs wristcam."""
    train_cfg = _load_yaml(train_cfg_path)
    eval_cfg = _load_yaml(eval_cfg_path)
    train_fam = _extract_camera_family(train_cfg)
    eval_fam = _extract_camera_family(eval_cfg)
    if train_fam is None or eval_fam is None:
        warnings.warn(
            f"camera-family check skipped — could not infer "
            f"(train={train_fam!r}, eval={eval_fam!r}) from "
            f"{train_cfg_path} / {eval_cfg_path}",
            stacklevel=2,
        )
        return
    if train_fam != eval_fam:
        raise CameraFamilyMismatchError(
            f"camera-family mismatch: train cfg ({train_cfg_path}) is {train_fam!r} "
            f"but eval cfg ({eval_cfg_path}) is {eval_fam!r}"
        )


def assert_eval_seed_file(seed_file_path: Path, expected_sha256: str) -> None:
    """Crash if the seed file's sha256 doesn't match `expected_sha256`.

    If `expected_sha256` is empty/falsy, fall back to the module-level
    `EVAL_SEEDS_60_SHA256`. If the pinned constant is itself empty (rare —
    only when bootstrapping), warn and skip.
    """
    if not seed_file_path.exists():
        raise EvalSeedFileDriftError(f"seed file not found: {seed_file_path}")

    expected = expected_sha256 or EVAL_SEEDS_60_SHA256
    if not expected:
        warnings.warn(
            f"eval-seed-file check skipped — no expected sha256 supplied "
            f"and EVAL_SEEDS_60_SHA256 is empty",
            stacklevel=2,
        )
        return

    actual = hashlib.sha256(seed_file_path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise EvalSeedFileDriftError(
            f"seed file {seed_file_path} sha256={actual} but expected {expected}"
        )


def assert_wandb_live(
    env: Optional[dict[str, str]] = None,
    project: Optional[str] = None,
) -> None:
    """Crash if wandb is offline / disabled / missing API key.

    Env-var checks always run (fast). If `project` is supplied (non-empty),
    we additionally construct `wandb.Api()` and try to resolve the project,
    so that an unreachable wandb backend or a typo'd project name is caught
    BEFORE the eval allocates a GPU.

    The deeper probe is opt-in (and skipped silently if `wandb` isn't
    importable) so that unit tests stay hermetic.
    """
    e = env if env is not None else os.environ
    mode = e.get("WANDB_MODE", "online")
    if mode != "online":
        raise WandbOfflineError(
            f"WANDB_MODE={mode!r}; only 'online' is permitted (no offline runs)."
        )
    disabled = e.get("WANDB_DISABLED", "")
    if disabled and disabled.lower() in ("1", "true", "yes"):
        raise WandbOfflineError(
            "WANDB_DISABLED is set — refusing to run a wandb-blind eval."
        )
    if not e.get("WANDB_API_KEY"):
        raise WandbOfflineError(
            "WANDB_API_KEY not set in environment; "
            "wandb.init() would fail silently or block."
        )
    if not project:
        return
    try:
        import wandb  # local import — keeps tests hermetic
    except ImportError:
        warnings.warn(
            "wandb not importable; skipping deeper Api() probe.",
            stacklevel=2,
        )
        return
    try:
        api = wandb.Api()
    except Exception as exc:  # noqa: BLE001 — wandb raises a wide range
        raise WandbOfflineError(
            f"wandb.Api() construction failed: {exc!r}"
        ) from exc
    # Resolve the project so a typo / missing-permission failure surfaces
    # BEFORE the eval starts. We don't care about the run list — just that
    # a 0-arg .runs() call doesn't 404.
    try:
        list(api.runs(project, per_page=1))
    except Exception as exc:  # noqa: BLE001
        raise WandbOfflineError(
            f"wandb project {project!r} unreachable: {exc!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _load_pinned_sha256(sha_path: Optional[Path]) -> str:
    """Load a single-line sha256 hex digest from a file. Empty/missing -> ''."""
    if sha_path is None:
        return ""
    if not sha_path.exists():
        return ""
    text = sha_path.read_text().strip()
    # Tolerate `sha256sum` output: "<hex>  <path>"
    return text.split()[0] if text else ""


def run_all_guards(
    train_cfg_path: Path,
    eval_cfg_path: Path,
    seed_file_path: Path,
    expected_sha256: str,
    project: Optional[str] = None,
) -> None:
    """Run every guard. Raises on the first failure."""
    assert_scene_match(train_cfg_path, eval_cfg_path)
    assert_camera_family_match(train_cfg_path, eval_cfg_path)
    assert_eval_seed_file(seed_file_path, expected_sha256)
    assert_wandb_live(project=project)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage-3 preflight guards for eval SLURM jobs.",
    )
    p.add_argument(
        "--train-cfg",
        type=Path,
        required=True,
        help="Path to trainer-dumped .hydra_config.yaml",
    )
    p.add_argument(
        "--eval-cfg",
        type=Path,
        required=True,
        help="Path to eval-time YAML config",
    )
    p.add_argument(
        "--seed-file",
        type=Path,
        required=True,
        help="Path to the canonical eval seed file (eval_seeds_60.txt)",
    )
    p.add_argument(
        "--expected-sha256",
        default="",
        help="Override pinned EVAL_SEEDS_60_SHA256. Empty -> use --expected-sha256-file or pinned constant.",
    )
    p.add_argument(
        "--expected-sha256-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a file containing the expected sha256 (e.g. "
            "/iris/u/mikulrai/runs/eval_seeds.sha256). Used when --expected-sha256 is empty."
        ),
    )
    p.add_argument(
        "--project",
        default="",
        help=(
            "Optional wandb project (entity/project or just project) to probe with "
            "wandb.Api(). Empty -> skip the network round-trip."
        ),
    )
    args = p.parse_args(argv)

    expected = args.expected_sha256
    if not expected:
        expected = _load_pinned_sha256(args.expected_sha256_file)

    try:
        run_all_guards(
            args.train_cfg,
            args.eval_cfg,
            args.seed_file,
            expected,
            project=(args.project or None),
        )
    except PreflightGuardError as exc:
        print(
            f"[preflight_eval_guards] FAIL ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"[preflight_eval_guards] FAIL (FileNotFoundError): {exc}", file=sys.stderr)
        return 1

    print("[preflight_eval_guards] all guards passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
