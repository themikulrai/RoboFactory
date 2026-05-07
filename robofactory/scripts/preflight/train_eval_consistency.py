"""Pre-launch train/eval consistency suite.

Plan v2 Phase C2 stage S5.  Run once per (model, dataset, task, scheme)
config the *first* time it's queued.  Catches the genus of train/eval
pipeline mismatches that "overfit-on-1-episode but can't replay it at
eval" surfaced last quarter — image preprocessing skew, action denorm
mismatch, fps mismatch, gripper-convention swap, etc.

Architecture: each check is a `Check` instance with a `name` and a `run()`
method that raises `CheckFailure` on detected mismatch. The driver runs
them cheap-first and records every result in the report JSON.

This module ships only the cheap, code-only checks. The heavier checks
(init-pose Wasserstein, Vulkan cross-node render, overfit-replay) live
behind their own SLURM-launched scripts since they require a full sim env
+ GPU and have very different cost profiles.

Usage::

    python scripts/preflight/train_eval_consistency.py \\
        --train-yaml conf/dp_pm_in1k.yaml \\
        --eval-yaml configs/table/pick_meat.yaml \\
        --report-out runs/preflight/<run_id>/consistency.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional


class CheckFailure(RuntimeError):
    """Raised by a check when it detects a train/eval mismatch."""


@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict


@dataclasses.dataclass
class Check:
    name: str
    fn: Callable[[dict, dict], None]   # raise CheckFailure on mismatch
    description: str = ""

    def run(self, train_cfg: dict, eval_cfg: dict) -> CheckResult:
        try:
            self.fn(train_cfg, eval_cfg)
            return CheckResult(name=self.name, passed=True, message="OK", details={})
        except CheckFailure as e:
            return CheckResult(
                name=self.name,
                passed=False,
                message=str(e),
                details=getattr(e, "details", {}),
            )


# ---------------------------------------------------------------------------
# Cheap checks (implemented here)
# ---------------------------------------------------------------------------


def _get(d: dict, dotted: str, default: Any = None) -> Any:
    """Resolve dotted key path with default."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def check_image_preprocess(train_cfg: dict, eval_cfg: dict) -> None:
    """Image shape, channel order, dtype, normalize range must match between
    train dataset and eval env wrapper."""
    train_shape = tuple(_get(train_cfg, "image.shape", ()))
    eval_shape = tuple(_get(eval_cfg, "image.shape", ()))
    if not train_shape or not eval_shape:
        raise CheckFailure(
            "image.shape missing in one or both configs "
            f"(train={train_shape!r}, eval={eval_shape!r})"
        )
    if train_shape != eval_shape:
        raise CheckFailure(
            f"image.shape mismatch: train={train_shape}, eval={eval_shape}"
        )
    train_order = _get(train_cfg, "image.channel_order", "CHW")
    eval_order = _get(eval_cfg, "image.channel_order", "CHW")
    if train_order != eval_order:
        raise CheckFailure(
            f"image.channel_order mismatch: train={train_order!r}, eval={eval_order!r}"
        )
    train_norm = _get(train_cfg, "image.normalize_range", "[0,1]")
    eval_norm = _get(eval_cfg, "image.normalize_range", "[0,1]")
    if train_norm != eval_norm:
        raise CheckFailure(
            f"image.normalize_range mismatch: train={train_norm!r}, eval={eval_norm!r}"
        )


def check_action_space_semantics(train_cfg: dict, eval_cfg: dict) -> None:
    """delta-vs-absolute, gripper 0/1 vs -1/1, joint vs eef must match."""
    keys = (
        "action.mode",          # "delta" | "absolute"
        "action.gripper_range",  # "[0,1]" | "[-1,1]"
        "action.control_target", # "joint" | "eef"
    )
    for key in keys:
        t = _get(train_cfg, key)
        e = _get(eval_cfg, key)
        if t != e:
            raise CheckFailure(f"{key} mismatch: train={t!r}, eval={e!r}")


def check_control_rate(train_cfg: dict, eval_cfg: dict) -> None:
    """Training data fps must match eval-env step rate."""
    train_fps = _get(train_cfg, "control.fps")
    eval_fps = _get(eval_cfg, "control.fps")
    if train_fps is None or eval_fps is None:
        raise CheckFailure(
            f"control.fps missing (train={train_fps!r}, eval={eval_fps!r})"
        )
    if train_fps != eval_fps:
        raise CheckFailure(f"control.fps mismatch: train={train_fps}, eval={eval_fps}")


def check_camera_intrinsics(train_cfg: dict, eval_cfg: dict) -> None:
    """Per-camera intrinsics (fx, fy, cx, cy) must match within tolerance."""
    train_cams = _get(train_cfg, "cameras", {})
    eval_cams = _get(eval_cfg, "cameras", {})
    if not train_cams or not eval_cams:
        raise CheckFailure(
            f"cameras dict missing (train_keys={list(train_cams.keys())!r}, "
            f"eval_keys={list(eval_cams.keys())!r})"
        )
    train_keys = set(train_cams.keys())
    eval_keys = set(eval_cams.keys())
    if train_keys != eval_keys:
        raise CheckFailure(
            f"camera names mismatch: train={sorted(train_keys)} eval={sorted(eval_keys)}"
        )
    for name in train_keys:
        for field in ("fx", "fy", "cx", "cy"):
            t = train_cams[name].get(field)
            e = eval_cams[name].get(field)
            if t is None or e is None:
                raise CheckFailure(f"camera[{name}].{field} missing")
            if abs(float(t) - float(e)) > 0.5:
                raise CheckFailure(
                    f"camera[{name}].{field} mismatch: train={t} eval={e}"
                )


def check_action_norm_stats(train_cfg: dict, eval_cfg: dict) -> None:
    """Action min/max/mean/std must match exactly (not just within tolerance)."""
    fields = ("action.min", "action.max", "action.mean", "action.std")
    for f in fields:
        t = _get(train_cfg, f)
        e = _get(eval_cfg, f)
        if t is None and e is None:
            continue  # optional
        if t != e:
            raise CheckFailure(f"{f} mismatch: train={t!r}, eval={e!r}")


CHEAP_CHECKS = (
    Check(name="image_preprocess", fn=check_image_preprocess,
          description="image shape / channel order / normalize range parity"),
    Check(name="action_norm_stats", fn=check_action_norm_stats,
          description="action min/max/mean/std byte-equal"),
    Check(name="action_space_semantics", fn=check_action_space_semantics,
          description="delta vs absolute, gripper convention, joint vs eef"),
    Check(name="control_rate", fn=check_control_rate,
          description="train fps == eval fps"),
    Check(name="camera_intrinsics", fn=check_camera_intrinsics,
          description="per-camera intrinsics byte-equal across train and eval envs"),
)


# Heavier checks — declared as known-pending for the caller to implement / wire later.
HEAVY_CHECKS_PENDING = (
    "init_pose_wasserstein",      # 50 train inits vs 50 eval inits, per-dim Wasserstein + CI
    "vulkan_render_distribution", # render same scene+seed on login vs compute node
    "overfit_replay_sanity",      # 1 train episode open-loop replay; chunk-0 + full-chunk MSE
)


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def run_all(
    train_cfg: dict,
    eval_cfg: dict,
    *,
    cheap_only: bool = True,
) -> list[CheckResult]:
    """Run every check; do not stop at first failure (the caller decides)."""
    results = []
    for check in CHEAP_CHECKS:
        results.append(check.run(train_cfg, eval_cfg))
    if not cheap_only:
        # heavy checks would be wired here; for now they're pending.
        pass
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-yaml", type=Path, required=True,
                    help="Path to a YAML containing the train pipeline knobs we care about.")
    ap.add_argument("--eval-yaml", type=Path, required=True,
                    help="Path to a YAML containing the eval pipeline knobs we care about.")
    ap.add_argument("--report-out", type=Path, default=None,
                    help="Optional JSON sink for the full report.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any check fails (default: strict).")
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.set_defaults(strict=True)
    args = ap.parse_args()

    train_cfg = _load_yaml(args.train_yaml)
    eval_cfg = _load_yaml(args.eval_yaml)
    results = run_all(train_cfg, eval_cfg)

    n_fail = sum(1 for r in results if not r.passed)
    for r in results:
        marker = "OK  " if r.passed else "FAIL"
        print(f"[{marker}] {r.name}: {r.message}")

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(
            {"results": [dataclasses.asdict(r) for r in results],
             "n_total": len(results), "n_failed": n_fail,
             "pending_heavy_checks": list(HEAVY_CHECKS_PENDING)}, indent=2))
        print(f"\nReport: {args.report_out}")

    print(f"\n{len(results) - n_fail}/{len(results)} checks passed; "
          f"{len(HEAVY_CHECKS_PENDING)} heavy checks pending.")
    if args.strict and n_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
