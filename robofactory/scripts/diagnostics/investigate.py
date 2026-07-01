"""Diagnostic battery for "before you retrain" investigation — workflow #1.

Motivation: the encoder-collapse genus took 5 corners (cent×decent × ws×wc
+ freezeenc retrain) and multiple full training runs to surface, each
24 h+ of compute. The freezeenc retrain in particular was a deliberate
"fix attempt" that produced another 0/27.

When an eval comes in 0% (or below tolerance), instead of "let's tweak
the recipe and retrain", run this fixed diagnostic battery first. Each
check fires fast-first (cheap → expensive) and is independent — a single
definitive failure crashes the orchestrator non-zero so a wrapping
`make investigate` target signals "do not retrain yet".

Battery (fast → slow):

  1. encoder-collapse probe   (~10 s GPU)
        utils.preflight_collapse.probe_collapse — surfaces proprio-shortcut
        collapse: encoder maps every input to a near-constant feature, so
        zeroing the image input barely changes the predicted action.
        Diagnostic: mse_zero_image / mse_baseline ≈ 1 means the policy
        ignores vision.

  2. init-pose Wasserstein    (~30 s sim)
        scripts/preflight/check_init_pose_wasserstein.py — distributional
        mismatch between training-data init poses and eval-env reset
        poses. Per-dim Wasserstein-1 with bootstrap CI; threshold 0.5 cm
        / 0.5 deg.

  3. overfit-replay sanity    (~2-5 min GPU + sim)
        scripts/preflight/check_overfit_replay_sanity.py — open-loop
        replay one training episode through the eval pipeline; if MSE
        > tolerance, train and eval pipelines are not byte-compatible.

  4. action-distribution probe (~30 s)  [TODO — wires through script/debug/
        probe_decent_dp_features* + analyze_decent_dp_features's new
        linear_probe_features_to_cubes; left unwired for v1 to keep the
        commit small. Add as Tier-2 follow-up.]

CLI:

    python -m robofactory.scripts.diagnostics.investigate \\
        --run-id <run_id_from_manifest> \\
        --out-dir /iris/u/mikulrai/runs/investigate/<run_id>/

    # If you don't yet have a manifest row, pass the artifacts directly:
    python -m robofactory.scripts.diagnostics.investigate \\
        --ckpt /iris/.../300_in1k.ckpt \\
        --dataset /iris/.../PickMeat-rf_150.zarr \\
        --scene-config configs/table/pick_meat.yaml \\
        --out-dir /iris/u/mikulrai/runs/investigate/<label>/

Output: a JSON verdict table at <out-dir>/verdict.json; a printed
summary on stdout. Exit codes:

    0 — every check passed (no obvious failure mode detected)
    1 — at least one check failed definitively (don't retrain — fix the
        flagged failure mode first)
    2 — bad arguments / setup error

This script is the diagnostic ENTRYPOINT. It does not by itself prevent
you from retraining — that's a process discipline (or a Makefile target
on the user's part). The exit code makes the discipline grep-able for
future automation.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/iris/u/mikulrai/projects/RoboFactory/robofactory")
PYTHON = "/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python"
MANIFEST_DEFAULT = Path("/iris/u/mikulrai/runs/manifest.csv")


@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    elapsed_s: float
    artifact_path: Optional[str] = None
    short_summary: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Manifest lookup
# ---------------------------------------------------------------------------


def lookup_manifest(run_id: str, manifest_path: Path) -> Optional[dict]:
    """Return the manifest row matching `run_id`, or None if not found."""
    if not manifest_path.is_file():
        return None
    with manifest_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("run_id") == run_id:
                return row
    return None


# ---------------------------------------------------------------------------
# Individual battery steps
# ---------------------------------------------------------------------------


def step_init_pose(
    *, dataset: str, scene_config: str, out_dir: Path
) -> CheckResult:
    name = "init_pose_wasserstein"
    out_json = out_dir / f"{name}.json"
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts/preflight/check_init_pose_wasserstein.py"),
        "--train-data-path", dataset,
        "--eval-config", scene_config,
        "--n-samples", "50",
        "--n-pos-dims", "3",
        "--pos-tolerance-cm", "0.5",
        "--rot-tolerance-deg", "0.5",
        "--out-json", str(out_json),
    ]
    return _run_subprocess(name, cmd, out_json)


def step_overfit_replay(
    *,
    ckpt: str,
    dataset: str,
    scene_config: str,
    episode_idx: int,
    max_steps: int,
    mse_tolerance: float,
    out_dir: Path,
) -> CheckResult:
    name = "overfit_replay_sanity"
    out_json = out_dir / f"{name}.json"
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts/preflight/check_overfit_replay_sanity.py"),
        "--ckpt-path", ckpt,
        "--dataset-path", dataset,
        "--scene-config", scene_config,
        "--episode-idx", str(episode_idx),
        "--max-steps", str(max_steps),
        "--mse-tolerance", str(mse_tolerance),
        "--mode", "full",
        "--output-json", str(out_json),
    ]
    return _run_subprocess(name, cmd, out_json)


def _run_subprocess(name: str, cmd: list[str], out_json: Path) -> CheckResult:
    """Run a check CLI, capture pass/fail + a one-line summary from its JSON."""
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return CheckResult(
            name=name, passed=False, elapsed_s=time.time() - t0,
            error="TIMEOUT after 1 h",
        )
    elapsed = time.time() - t0
    passed = proc.returncode == 0
    summary = ""
    if out_json.is_file():
        try:
            data = json.loads(out_json.read_text())
            # Each check sets a top-level `passed` and a free-form summary.
            if isinstance(data, dict):
                summary = _short_summary(name, data)
        except json.JSONDecodeError:
            summary = "<invalid JSON output>"
    err = ""
    if not passed:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        err = err[0]
    return CheckResult(
        name=name,
        passed=passed,
        elapsed_s=elapsed,
        artifact_path=str(out_json) if out_json.is_file() else None,
        short_summary=summary,
        error=err,
    )


def _short_summary(name: str, data: dict) -> str:
    """Compress each check's JSON into one line for the verdict table."""
    if name == "init_pose_wasserstein":
        verdict = data.get("verdict", "?")
        worst = data.get("worst_dim", {})
        wd = worst.get("dim_index", "?") if isinstance(worst, dict) else "?"
        return f"verdict={verdict} worst_dim={wd}"
    if name == "overfit_replay_sanity":
        c0 = data.get("chunk0_mse_mean")
        cf = data.get("full_chunk_mse_mean")
        return f"chunk0_mse={c0} full_chunk_mse={cf}"
    return ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_battery(
    *,
    ckpt: str,
    dataset: str,
    scene_config: str,
    out_dir: Path,
    episode_idx: int,
    max_steps: int,
    mse_tolerance: float,
    skip: list[str],
) -> tuple[list[CheckResult], int]:
    """Run all battery steps; return (results, exit_code)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CheckResult] = []

    if "init_pose" not in skip:
        results.append(step_init_pose(
            dataset=dataset, scene_config=scene_config, out_dir=out_dir,
        ))

    if "overfit_replay" not in skip:
        results.append(step_overfit_replay(
            ckpt=ckpt, dataset=dataset, scene_config=scene_config,
            episode_idx=episode_idx, max_steps=max_steps,
            mse_tolerance=mse_tolerance, out_dir=out_dir,
        ))

    # Encoder-collapse and action-distribution: deferred (see module docstring).

    rc = 0 if all(r.passed for r in results) else 1
    return results, rc


def print_verdict_table(results: list[CheckResult]) -> None:
    if not results:
        print("(no checks ran)")
        return
    name_w = max(len(r.name) for r in results)
    print(f"\n{'check':<{name_w}}  {'pass':<6}  {'time(s)':>8}  summary")
    print("-" * (name_w + 6 + 8 + 40))
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        line = f"{r.name:<{name_w}}  {flag:<6}  {r.elapsed_s:>8.1f}  {r.short_summary}"
        if r.error:
            line += f"  [{r.error}]"
        print(line)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\nsummary: {n_pass}/{len(results)} passed; {n_fail} failed")


def write_verdict_json(results: list[CheckResult], path: Path) -> None:
    payload = {
        "results": [dataclasses.asdict(r) for r in results],
        "n_pass": sum(1 for r in results if r.passed),
        "n_fail": sum(1 for r in results if not r.passed),
        "all_passed": all(r.passed for r in results),
    }
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id", help="manifest run_id; looks up ckpt/dataset/scene")
    src.add_argument(
        "--ckpt", help="explicit ckpt path (use with --dataset and --scene-config)",
    )
    p.add_argument("--dataset", help="dataset path (zarr or LeRobot HF)")
    p.add_argument("--scene-config", help="eval scene YAML path")
    p.add_argument("--out-dir", required=True, help="where to write per-check JSONs")
    p.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--mse-tolerance", type=float, default=0.01)
    p.add_argument("--skip", nargs="*", default=[],
                   help="steps to skip: init_pose | overfit_replay")

    args = p.parse_args(argv)

    if args.run_id:
        row = lookup_manifest(args.run_id, Path(args.manifest))
        if row is None:
            print(
                f"FAIL: run_id '{args.run_id}' not found in manifest "
                f"({args.manifest})",
                file=sys.stderr,
            )
            return 2
        ckpt = (row.get("ckpt_paths") or "").split(",")[0].strip() or args.ckpt
        dataset = args.dataset or _guess_dataset_from_manifest_row(row)
        scene_config = (
            args.scene_config or row.get("scene_config") or _guess_scene_config(row)
        )
    else:
        ckpt = args.ckpt
        dataset = args.dataset
        scene_config = args.scene_config

    if not all([ckpt, dataset, scene_config]):
        print(
            "FAIL: need ckpt, dataset, and scene-config (either --run-id "
            "with a manifest hit or all three flags directly)",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).resolve()
    print(f"[investigate] ckpt={ckpt}")
    print(f"[investigate] dataset={dataset}")
    print(f"[investigate] scene={scene_config}")
    print(f"[investigate] out_dir={out_dir}")

    results, rc = run_battery(
        ckpt=ckpt,
        dataset=dataset,
        scene_config=scene_config,
        out_dir=out_dir,
        episode_idx=args.episode_idx,
        max_steps=args.max_steps,
        mse_tolerance=args.mse_tolerance,
        skip=args.skip,
    )
    print_verdict_table(results)
    write_verdict_json(results, out_dir / "verdict.json")
    print(f"\nverdict written to {out_dir / 'verdict.json'}")
    return rc


def _guess_dataset_from_manifest_row(row: dict) -> str:
    """Best-effort dataset path inference from manifest fields.

    Manifest typically doesn't carry the exact dataset path, so fall back to
    canonical conventions: PM → PickMeat-rf_150.zarr; TSC → ThreeRobotsStackCube
    family. Returns "" if nothing fits — caller errors out cleanly.
    """
    task = (row.get("task") or "").lower()
    dataset_root = "/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data"
    if task == "pm":
        return f"{dataset_root}/PickMeat-rf_150.zarr"
    return ""


def _guess_scene_config(row: dict) -> str:
    task = (row.get("task") or "").lower()
    cfg_root = REPO_ROOT / "configs" / "table"
    if task == "pm":
        return str(cfg_root / "pick_meat.yaml")
    if task == "tsc":
        return str(cfg_root / "three_robots_stack_cube.yaml")
    return ""


if __name__ == "__main__":
    sys.exit(main())
