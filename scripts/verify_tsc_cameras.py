#!/usr/bin/env python
"""Render-verify gate for the ThreeRobotsStackCube (TSC) per-agent head cameras.

WHY THIS EXISTS
---------------
The TSC scene yamls used to give head_camera_agent0/1/2 an IDENTICAL look_at
pose -> byte-identical RGB across agents (mean-abs-diff 0.0), so per-arm
policies got no per-arm visual signal. FIX 3 gave each agent camera a distinct
viewpoint aimed at its own arm's workspace. This script is the gate the
researcher runs ON A COMPUTE NODE (NOT the login node, NOT a bad-Vulkan node)
before mass datagen to confirm the three per-agent cameras now differ.

WHAT IT DOES
------------
  1. resets the TSC env with shader_pack="default" (matches data-gen; "minimal"
     blacks out the upper global cam -- see SAPIEN shader_pack memory note),
  2. dumps the 4 head-camera RGBs to PNG,
  3. asserts pairwise mean-abs-diff > THRESH (default 10/255) between the THREE
     per-agent cameras (head_camera_global is intentionally NOT part of the
     gate -- it is the shared overhead view and may legitimately overlap),
  4. prints the full 4x4 diff matrix + PASS/FAIL and exits non-zero on FAIL.

RUN IT ON A COMPUTE NODE (example, IRIS):
  srun --partition=iris-hi-interactive --account=iris --gres=gpu:1 \
       --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash -lc '
    source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
    conda activate RoboFactory
    export HOME=/iris/u/mikulrai SAPIEN_HEADLESS=1
    WT=/iris/u/mikulrai/projects/RoboFactory/.claude/worktrees/datafix-rf
    PYTHONPATH="$WT/robofactory:$WT/robofactory/policy/Diffusion-Policy" \
      /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
      "$WT/scripts/verify_tsc_cameras.py" --scene table
  '

Add `--scene robocasa` to also gate the robocasa-scene yaml (same fix mirrored).
PNGs land in <worktree>/images/tsc_camera_verify/<scene>/ by default.
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np

# --- make the WORKTREE's robofactory importable (so we test the FIXED yaml) ---
# Derive the worktree root from this file's location instead of hardcoding the
# main checkout -- otherwise we'd import the unfixed task/config.
WT_ROOT = Path(__file__).resolve().parents[1]
RF_PKG = WT_ROOT / "robofactory"
DP_PKG = RF_PKG / "policy" / "Diffusion-Policy"
for p in (str(RF_PKG), str(DP_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym  # noqa: E402
from PIL import Image  # noqa: E402

from robofactory.tasks import *  # noqa: E402,F401,F403  (side-effect: register_env)

ENV_ID = "ThreeRobotsStackCube-rf"
AGENT_CAMS = ["head_camera_agent0", "head_camera_agent1", "head_camera_agent2"]
GLOBAL_CAM = "head_camera_global"
ALL_CAMS = AGENT_CAMS + [GLOBAL_CAM]


def _to_hwc_uint8(rgb):
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    if arr.ndim == 4:  # (num_envs, H, W, 3) with num_envs==1
        arr = arr[0]
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="table", choices=["table", "robocasa"],
                    help="which TSC yaml to verify (default: table)")
    ap.add_argument("--config", default=None,
                    help="explicit config yaml path (overrides --scene)")
    ap.add_argument("--seed", type=int, default=10000, help="reset seed")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="min pairwise mean-abs-diff (0-255 scale) to PASS")
    ap.add_argument("--out-dir", default=None,
                    help="PNG output dir (default <worktree>/images/tsc_camera_verify/<scene>)")
    args = ap.parse_args()

    config = args.config or str(RF_PKG / "configs" / args.scene / "three_robots_stack_cube.yaml")
    if not os.path.isfile(config):
        print(f"[verify] ERROR  config yaml not found: {config}", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(args.out_dir or (WT_ROOT / "images" / "tsc_camera_verify" / args.scene))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[verify] env={ENV_ID}  scene={args.scene}")
    print(f"[verify] config={config}")
    print(f"[verify] shader_pack=default  seed={args.seed}  threshold={args.threshold}/255")
    print(f"[verify] out_dir={out_dir}")

    env = gym.make(
        ENV_ID,
        config=config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        num_envs=1,
        sim_backend="cpu",
        enable_shadow=False,
        sensor_configs=dict(shader_pack="default"),  # CRITICAL: match data-gen
    )
    obs, _ = env.reset(seed=args.seed)
    sd = obs["sensor_data"]

    missing = [c for c in ALL_CAMS if c not in sd or sd[c].get("rgb") is None]
    if missing:
        print(f"[verify] ERROR  missing camera RGB in obs: {missing}", file=sys.stderr)
        print(f"[verify]        available sensor_data keys: {sorted(sd.keys())}", file=sys.stderr)
        env.close()
        sys.exit(2)

    imgs = {}
    for cam in ALL_CAMS:
        img = _to_hwc_uint8(sd[cam]["rgb"])
        imgs[cam] = img
        png = out_dir / f"{cam}.png"
        Image.fromarray(img).save(png)
        print(f"[verify] saved {png}  shape={img.shape}")
    env.close()

    # --- pairwise mean-abs-diff matrix over ALL 4 cams (for context) ----------
    def mad(a, b):
        return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))

    print("\n[verify] pairwise mean-abs-diff (0-255 scale):")
    hdr = "                    " + "".join(f"{c.replace('head_camera_',''):>10}" for c in ALL_CAMS)
    print(hdr)
    for ci in ALL_CAMS:
        row = f"{ci.replace('head_camera_',''):>20}"
        for cj in ALL_CAMS:
            row += f"{mad(imgs[ci], imgs[cj]):>10.3f}"
        print(row)

    # --- the GATE: only the 3 per-agent cams must be mutually distinct --------
    print(f"\n[verify] GATE: pairwise MAD among {AGENT_CAMS} must each be > {args.threshold}")
    ok = True
    for a, b in itertools.combinations(AGENT_CAMS, 2):
        d = mad(imgs[a], imgs[b])
        verdict = "PASS" if d > args.threshold else "FAIL"
        if d <= args.threshold:
            ok = False
        print(f"  {a:>18} vs {b:<18} MAD={d:8.3f}  [{verdict}]")

    print("\n[verify] RESULT:", "PASS  (per-agent cameras are distinct)" if ok
          else "FAIL  (per-agent cameras still too similar -- DO NOT run datagen)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
