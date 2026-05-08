"""Heavy preflight check: Vulkan cross-node render distribution parity.

Plan v2 Phase C2 stage S5 step 7 ("Vulkan render-distribution check").  The
cheap consistency suite (`scripts/preflight/train_eval_consistency.py`) covers
byte-equal config parity; this script covers the *renderer* mismatch that
those checks can't see: the SAPIEN/Vulkan stack on the IRIS login node renders
scenes ~32% darker than on compute nodes (a real, observed bug — see CLAUDE.md
"SAPIEN / Vulkan rendering"), and even between compute nodes a driver/GPU SKU
delta can shift per-channel statistics enough that a model trained on one
node's pixels rolls out on another node's pixels with measurable degradation.

Approach: render the *same* scene + seed on two nodes, save the RGB
observation as ``.npy`` per node, then assert per-channel mean and std agree
within ``--tolerance-pct`` (default 2%).  This is the simplest invariant we
can check without a perceptual metric, and 2% on per-channel mean is enough
to catch the 32%-darker login-node failure mode by orders of magnitude.

Two modes (split so SLURM can run the heavy half on each node, then the
cheap compare wherever):

- ``--mode render`` — instantiate the env from ``--scene-config``, reset
  with ``--seed``, capture the RGB observation as ``np.ndarray[uint8,
  (H, W, 3)]``, save to ``--output-npy``.  Lazily imports SAPIEN / gym so
  this module stays importable on the login node and from tests.
- ``--mode compare`` — load both ``--node-a-images`` and ``--node-b-images``
  npys, compute per-channel mean/std on each side, compare percent-deltas
  against ``--tolerance-pct``, write a JSON report, exit non-zero if any
  channel exceeds tolerance.

Standalone-script note: like ``check_init_pose_wasserstein.py``, this file
does NOT plug into the ``Check`` / ``CheckResult`` machinery in
``train_eval_consistency.py``.  It needs a sim env reset on a real GPU node
(rendering on the login node is the bug we're trying to detect, not a
sensible fallback), so it lives behind its own SLURM launcher
(``slurm_vulkan_render_check.sh``) and emits its own JSON report shape.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RenderReport:
    """JSON-serialisable verdict from comparing two renderings."""

    node_a_path: str
    node_b_path: str
    per_channel_mean_a: list[float]
    per_channel_mean_b: list[float]
    per_channel_std_a: list[float]
    per_channel_std_b: list[float]
    mean_diff_pct: list[float]
    std_diff_pct: list[float]
    tolerance_pct: float
    passed: bool
    failed_channels: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# Channel labels used in ``failed_channels``.  Three RGB channels, two
# stat kinds -> six possible failure tags.
_CHANNEL_NAMES = ("R", "G", "B")


# ---------------------------------------------------------------------------
# Compare mode (the part tests cover)
# ---------------------------------------------------------------------------


def _normalise_image_array(arr: np.ndarray, name: str) -> np.ndarray:
    """Return a (N, H, W, 3) float64 array, raising on malformed input.

    - Accepts (H, W, 3) and reshapes to (1, H, W, 3).
    - Rejects empty arrays and arrays without a trailing 3-channel dim.
    - Casts uint8 / float32 / etc. to float64 so mean/std math is uniform.
    - Rejects NaN payloads (silent NaN -> false-pass would be the worst
      possible failure mode for a preflight check).
    """
    if arr.size == 0:
        raise ValueError(f"{name} is empty (size=0)")
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(
            f"{name} has unsupported shape {arr.shape}; expected (H,W,3) or (N,H,W,3)"
        )
    if arr.shape[-1] != 3:
        raise ValueError(
            f"{name} last dim must be 3 (RGB); got shape {arr.shape}"
        )
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"rendered images contain NaN or inf in {name}")
    return arr


def _per_channel_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of shape (3,) over all pixels of all images."""
    # Collapse N, H, W; keep channel.
    flat = arr.reshape(-1, arr.shape[-1])
    return flat.mean(axis=0), flat.std(axis=0)


def _percent_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``abs(a - b) / |b| * 100`` with safe handling of b == 0.

    If both a and b are exactly zero on a channel, the diff is 0% (true
    agreement).  If b is zero and a is not, the diff is +inf (which will
    fail any finite tolerance — correct behaviour).
    """
    diff = np.abs(a - b)
    denom = np.abs(b)
    out = np.zeros_like(diff, dtype=np.float64)
    nz = denom > 0
    out[nz] = diff[nz] / denom[nz] * 100.0
    # Where denom == 0: 0/0 -> 0 (both zero, agreement); else nonzero/0 -> inf.
    zero_mask = ~nz
    if np.any(zero_mask):
        out[zero_mask] = np.where(diff[zero_mask] == 0.0, 0.0, np.inf)
    return out


def compare_renders(
    images_a: np.ndarray,
    images_b: np.ndarray,
    tolerance_pct: float,
    *,
    node_a_path: str = "<a>",
    node_b_path: str = "<b>",
) -> RenderReport:
    """Compare per-channel mean/std of two image arrays; build a RenderReport."""
    a = _normalise_image_array(images_a, "node_a images")
    b = _normalise_image_array(images_b, "node_b images")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(
            f"image shape mismatch: a={a.shape}, b={b.shape} "
            f"(per-image dims must match; N may differ)"
        )

    mean_a, std_a = _per_channel_stats(a)
    mean_b, std_b = _per_channel_stats(b)

    mean_diff = _percent_diff(mean_a, mean_b)
    std_diff = _percent_diff(std_a, std_b)

    failed: list[str] = []
    for k, ch in enumerate(_CHANNEL_NAMES):
        if mean_diff[k] > tolerance_pct:
            failed.append(f"mean_{ch}")
        if std_diff[k] > tolerance_pct:
            failed.append(f"std_{ch}")

    return RenderReport(
        node_a_path=node_a_path,
        node_b_path=node_b_path,
        per_channel_mean_a=mean_a.tolist(),
        per_channel_mean_b=mean_b.tolist(),
        per_channel_std_a=std_a.tolist(),
        per_channel_std_b=std_b.tolist(),
        mean_diff_pct=mean_diff.tolist(),
        std_diff_pct=std_diff.tolist(),
        tolerance_pct=float(tolerance_pct),
        passed=len(failed) == 0,
        failed_channels=failed,
    )


# ---------------------------------------------------------------------------
# Render mode (NOT exercised by tests; only the SLURM driver hits this on
# real compute nodes)
# ---------------------------------------------------------------------------


def render_scene_image(
    scene_config: Path,
    seed: int,
    *,
    env_factory: Optional[Callable[[Path, int], Any]] = None,
    rgb_extractor: Optional[Callable[[Any], np.ndarray]] = None,
) -> np.ndarray:
    """Reset the env from ``scene_config`` with ``seed`` and return one RGB frame.

    ``env_factory(scene_config, seed)`` returns an env-like object with
    ``.reset(seed=...)``; ``rgb_extractor(obs)`` pulls the RGB frame out.
    Both default to the production SAPIEN/gym path (lazily imported so this
    module stays importable on the login node and from unit tests).
    """
    if env_factory is None:
        env_factory = _default_env_factory
    if rgb_extractor is None:
        rgb_extractor = _default_rgb_extractor

    env = env_factory(scene_config, seed)
    try:
        obs, _info = env.reset(seed=seed)
        rgb = np.asarray(rgb_extractor(obs))
    finally:
        if hasattr(env, "close"):
            env.close()
    if rgb.dtype != np.uint8:
        # Coerce to the canonical on-disk dtype so the compare side doesn't
        # have to special-case float renderers.
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _default_env_factory(scene_config: Path, seed: int):
    """Production env factory.  Imported lazily so login-node tests don't
    pull in SAPIEN."""
    import gymnasium as gym  # noqa: F401  (local import — heavy dep)
    import yaml  # noqa: F401

    cfg = yaml.safe_load(Path(scene_config).read_text())
    env_id = cfg["task_name"] + "-rf"
    return gym.make(
        env_id,
        config=str(scene_config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
    )


def _default_rgb_extractor(obs: Any) -> np.ndarray:
    """Best-effort RGB extractor for the SAPIEN reset obs.

    Tries the common keys used by ManiSkill / RoboFactory observations.
    On multi-camera obs we pick the first camera in sorted-key order so
    the choice is deterministic across nodes.
    """
    if isinstance(obs, np.ndarray):
        return obs
    if not isinstance(obs, dict):
        raise TypeError(f"unexpected obs type {type(obs)}; expected dict or ndarray")

    # Common nested layouts: obs["image"][cam]["rgb"], obs["sensor_data"][cam]["rgb"].
    for top in ("image", "sensor_data", "images"):
        if top in obs and isinstance(obs[top], dict) and obs[top]:
            cams = sorted(obs[top].keys())
            cam0 = obs[top][cams[0]]
            if isinstance(cam0, dict) and "rgb" in cam0:
                return np.asarray(cam0["rgb"])
            if isinstance(cam0, np.ndarray):
                return np.asarray(cam0)

    # Flat: obs["rgb"].
    if "rgb" in obs:
        return np.asarray(obs["rgb"])

    raise KeyError(
        f"could not locate RGB frame in obs; top-level keys={list(obs.keys())}"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_render(args: argparse.Namespace) -> int:
    if args.scene_config is None or args.output_npy is None:
        raise SystemExit(
            "--mode render requires --scene-config and --output-npy"
        )
    rgb = render_scene_image(args.scene_config, args.seed)
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, rgb)
    print(
        f"vulkan_render: rendered {args.scene_config.name} "
        f"seed={args.seed} -> {args.output_npy} (shape={rgb.shape}, dtype={rgb.dtype})"
    )
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    if args.node_a_images is None or args.node_b_images is None:
        raise SystemExit(
            "--mode compare requires --node-a-images and --node-b-images"
        )
    if args.output_json is None:
        raise SystemExit("--mode compare requires --output-json")

    a = np.load(args.node_a_images)
    b = np.load(args.node_b_images)
    report = compare_renders(
        a, b,
        tolerance_pct=args.tolerance_pct,
        node_a_path=str(args.node_a_images),
        node_b_path=str(args.node_b_images),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report.to_dict(), indent=2))
    print(
        f"vulkan_render_compare: passed={report.passed} "
        f"failed_channels={report.failed_channels} -> {args.output_json}"
    )
    return 0 if report.passed else 1


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", choices=("render", "compare"), required=True,
        help="render: one node, save npy; compare: two npys -> JSON verdict.",
    )
    ap.add_argument("--scene-config", type=Path, default=None,
                    help="Path to scene yaml, e.g. configs/table/three_robots_stack_cube.yaml")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Reset seed used for the rendered frame.")
    ap.add_argument("--output-npy", type=Path, default=None,
                    help="(render mode) Where to save the rendered RGB frame.")
    ap.add_argument("--node-a-images", type=Path, default=None,
                    help="(compare mode) npy from node A.")
    ap.add_argument("--node-b-images", type=Path, default=None,
                    help="(compare mode) npy from node B.")
    ap.add_argument("--tolerance-pct", type=float, default=2.0,
                    help="Per-channel mean/std percent-diff tolerance.")
    ap.add_argument("--output-json", type=Path, default=None,
                    help="(compare mode) Where to write the JSON report.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.mode == "render":
        return _run_render(args)
    return _run_compare(args)


if __name__ == "__main__":
    sys.exit(main())
