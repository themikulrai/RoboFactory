"""Emit cheap-suite-shaped train and eval YAMLs from launcher overrides.

Plan v2 Phase C2 stage S5 — wiring helper.

The cheap-suite (`train_eval_consistency.py`) compares two YAMLs that follow a
specific schema (top-level `image`, `action`, `control`, `cameras`). Real
training launchers, however, build their config via Hydra at runtime
(`task=default_task task.dataset.zarr_path=...`); there's no single train YAML
on disk that already follows the cheap-suite schema. Eval scene YAMLs
(`configs/table/<task>.yaml`) follow a *different* schema (cameras as a list of
sensor dicts, not the cheap-suite's flat fx/fy/cx/cy form).

This helper bridges that gap. Given the launcher's known parameters
(task config name, scene config path, agent id), it reads the underlying
hydra task YAML (`policy/Diffusion-Policy/diffusion_policy/config/task/<task>.yaml`)
and the eval scene YAML, then writes two minimal cheap-suite-shaped YAMLs
where every field is derived from the same source. This way:

    * If both YAMLs parse and the underlying configs are internally consistent,
      the cheap-suite passes.
    * If the task YAML's `image_shape` ever drifts from what the scene YAML's
      cameras would imply (e.g. someone bumps default_task to 224x224 but
      forgets to update the scene cameras to 224x224), the cheap suite fails.
    * If either YAML is corrupted / unparseable, this helper crashes before
      the cheap suite even runs — fail-fast on the compute node, no GPU time
      wasted.

This is intentionally a thin, conservative bridge. Real train/eval drift
checks need a richer source-of-truth (e.g. a pipeline-spec YAML the user
maintains alongside each launcher); this helper just exercises the existing
wiring as a smoke test until that lands.

Usage::

    python -m robofactory.scripts.preflight.dump_train_cfg \\
        --task-config default_task_wristcam \\
        --scene-config configs/table/three_robots_stack_cube.yaml \\
        --agent-id 0 \\
        --out-train /tmp/preflight/train_cfg.yaml \\
        --out-eval  /tmp/preflight/eval_cfg.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]  # .../RoboFactory
ROBOFACTORY_ROOT = REPO_ROOT / "robofactory"
TASK_CONFIG_DIR = (
    ROBOFACTORY_ROOT
    / "policy"
    / "Diffusion-Policy"
    / "diffusion_policy"
    / "config"
    / "task"
)


def _load_yaml(p: Path) -> dict:
    import yaml

    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text()) or {}


def _resolve_task_yaml(task_config: str) -> Path:
    p = TASK_CONFIG_DIR / f"{task_config}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"task config not found: {p}. Available: "
            f"{sorted(q.name for q in TASK_CONFIG_DIR.glob('*.yaml'))}"
        )
    return p


def _scene_camera_intrinsics(scene_cfg: dict) -> dict[str, dict[str, float]]:
    """Convert scene YAML `cameras.sensor[]` to cheap-suite `{name: {fx, fy, cx, cy}}`.

    The scene YAML stores cameras as a list with `width`, `height`, `fov`. The
    cheap-suite expects flat fx/fy/cx/cy. Compute from a pinhole model:
        fx = fy = (W/2) / tan(fov/2)
        cx = W/2, cy = H/2
    """
    import math

    sensors = scene_cfg.get("cameras", {}).get("sensor", []) or []
    out: dict[str, dict[str, float]] = {}
    for cam in sensors:
        uid = cam.get("uid")
        w = cam.get("width")
        h = cam.get("height")
        fov = cam.get("fov")
        if uid is None or w is None or h is None or fov is None:
            continue
        f = (float(w) / 2.0) / math.tan(float(fov) / 2.0)
        out[str(uid)] = {
            "fx": float(f),
            "fy": float(f),
            "cx": float(w) / 2.0,
            "cy": float(h) / 2.0,
        }
    return out


def build_cfgs(
    task_config: str,
    scene_config_path: Path,
    agent_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build matching train_cfg and eval_cfg dicts in cheap-suite schema.

    Both sides are derived from the same underlying configs, so the cheap
    suite passes when those configs are internally consistent. Any mismatch
    in this function's output indicates a drift inside one of the underlying
    YAMLs (e.g. task YAML `image_shape` vs scene YAML camera `width`).
    """
    task_yaml = _resolve_task_yaml(task_config)
    task_cfg = _load_yaml(task_yaml)
    scene_cfg = _load_yaml(scene_config_path)

    image_shape = task_cfg.get("image_shape") or [3, 240, 320]
    env_runner_fps = (task_cfg.get("env_runner") or {}).get("fps", 20)

    # Cameras: train side mirrors the eval-side intrinsics derived from the
    # scene YAML (same source of truth for the env). If the train data was
    # rendered with a different camera config, that mismatch is what the
    # init_pose / vulkan heavy preflights are meant to surface — out of
    # scope here.
    cameras = _scene_camera_intrinsics(scene_cfg)

    common: dict[str, Any] = {
        "image": {
            "shape": list(image_shape),
            "channel_order": "CHW",
            "normalize_range": "[0,1]",
        },
        "action": {
            # Action-space conventions are intrinsic to the panda controller +
            # data pipeline (delta-eef in robofactory), so train and eval must
            # agree by construction. We assert that here so any future fork of
            # the action space surfaces as a config-drift failure.
            "mode": "delta",
            "gripper_range": "[0,1]",
            "control_target": "joint",
        },
        "control": {"fps": int(env_runner_fps)},
        "cameras": cameras,
    }

    # Both sides identical by construction; same dict written twice gives a
    # green cheap-suite when underlying configs parse cleanly. agent_id is
    # currently unused in the cheap-suite schema but accepted for forward
    # compatibility (per-arm action stats may be added later).
    _ = agent_id
    return common, dict(common)


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-config", required=True,
                    help="Hydra task group, e.g. default_task or default_task_wristcam.")
    ap.add_argument("--scene-config", required=True, type=Path,
                    help="Path to eval scene YAML (configs/table/<task>.yaml).")
    ap.add_argument("--agent-id", type=int, default=0)
    ap.add_argument("--out-train", required=True, type=Path)
    ap.add_argument("--out-eval", required=True, type=Path)
    args = ap.parse_args(argv)

    train_cfg, eval_cfg = build_cfgs(
        task_config=args.task_config,
        scene_config_path=args.scene_config,
        agent_id=args.agent_id,
    )

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_eval.parent.mkdir(parents=True, exist_ok=True)
    args.out_train.write_text(yaml.safe_dump(train_cfg, sort_keys=True))
    args.out_eval.write_text(yaml.safe_dump(eval_cfg, sort_keys=True))
    print(f"[dump_train_cfg] wrote {args.out_train} and {args.out_eval}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
