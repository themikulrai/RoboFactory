#!/bin/bash
# SLURM driver for the Vulkan cross-node render-distribution preflight.
#
# Plan v2 Phase C2 stage S5 step 7.  Renders the same scene+seed on two
# different compute nodes, then compares per-channel mean/std within a 2%
# tolerance.  Catches the SAPIEN/Vulkan ~32%-darker login-node bug and
# any cross-node renderer drift (driver / GPU SKU / Vulkan loader).
#
# DO NOT render on the login node — the login node's render output is the
# bug we're trying to detect.  Both render jobs MUST run on real GPU
# compute nodes.  The compare job is cheap and can run anywhere with a
# CPU + numpy.
#
# ---------------------------------------------------------------------------
# Required env vars (set before sourcing or invoking this script's blocks):
#   SCENE_CONFIG  absolute path to a configs/table/*.yaml
#   SEED          integer reset seed (default 12345)
#   NODE_A        target nodename for render A (e.g. iris-1)
#   NODE_B        target nodename for render B (different physical node)
#   OUT_DIR       absolute path; ${OUT_DIR}/{a.npy, b.npy, report.json}
#
# ---------------------------------------------------------------------------
# Manual two-step the user runs (DO NOT auto-submit):
#
#   export SCENE_CONFIG=/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml
#   export SEED=12345
#   export NODE_A=iris-1
#   export NODE_B=iris-2
#   export OUT_DIR=/iris/u/mikulrai/runs/preflight/vulkan_render_$(date +%Y%m%d_%H%M%S)
#   mkdir -p "$OUT_DIR"
#
#   # Step 1: submit both render jobs (parallel, different nodes), capture jobids.
#   JOB_A=$(sbatch --parsable \
#       --job-name=vk_render_a \
#       --partition=iris --account=iris --gres=gpu:1 --cpus-per-task=4 \
#       --mem=32G --time=00:15:00 \
#       --exclude=iris-hp-z8,iris-hgx-1 \
#       -w "$NODE_A" \
#       --output="$OUT_DIR/render_a_%j.out" --error="$OUT_DIR/render_a_%j.err" \
#       --wrap="bash $0 render $OUT_DIR/a.npy")
#
#   JOB_B=$(sbatch --parsable \
#       --job-name=vk_render_b \
#       --partition=iris --account=iris --gres=gpu:1 --cpus-per-task=4 \
#       --mem=32G --time=00:15:00 \
#       --exclude=iris-hp-z8,iris-hgx-1 \
#       -w "$NODE_B" \
#       --output="$OUT_DIR/render_b_%j.out" --error="$OUT_DIR/render_b_%j.err" \
#       --wrap="bash $0 render $OUT_DIR/b.npy")
#
#   # Step 2: submit the compare job, gated on both renders finishing OK.
#   sbatch --job-name=vk_compare \
#       --partition=iris --account=iris --cpus-per-task=2 \
#       --mem=8G --time=00:05:00 \
#       --dependency=afterok:${JOB_A}:${JOB_B} \
#       --output="$OUT_DIR/compare_%j.out" --error="$OUT_DIR/compare_%j.err" \
#       --wrap="bash $0 compare $OUT_DIR"
#
# ---------------------------------------------------------------------------
# When sbatch invokes this script via --wrap="bash $0 <subcommand> ...",
# the dispatch below picks the right code path.

set -euo pipefail

SUBCMD="${1:-}"
shift || true

REPO_ROOT=/iris/u/mikulrai/projects/RoboFactory/robofactory
PYBIN=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
SCRIPT="$REPO_ROOT/robofactory/scripts/preflight/check_vulkan_render_distribution.py"

# Defaults (overridable via env).
SCENE_CONFIG="${SCENE_CONFIG:-}"
SEED="${SEED:-12345}"
TOLERANCE_PCT="${TOLERANCE_PCT:-2.0}"

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HYDRA_FULL_ERROR=1

case "$SUBCMD" in
  render)
    OUT_NPY="${1:?missing output npy path}"
    if [[ -z "$SCENE_CONFIG" ]]; then
      echo "ERROR: SCENE_CONFIG env var must be set for render mode" >&2
      exit 2
    fi
    echo "[$(hostname)] rendering $SCENE_CONFIG seed=$SEED -> $OUT_NPY"
    "$PYBIN" -u "$SCRIPT" \
      --mode render \
      --scene-config "$SCENE_CONFIG" \
      --seed "$SEED" \
      --output-npy "$OUT_NPY"
    ;;

  compare)
    OUT_DIR="${1:?missing OUT_DIR}"
    echo "[$(hostname)] comparing $OUT_DIR/a.npy vs $OUT_DIR/b.npy (tol=${TOLERANCE_PCT}%)"
    "$PYBIN" -u "$SCRIPT" \
      --mode compare \
      --node-a-images "$OUT_DIR/a.npy" \
      --node-b-images "$OUT_DIR/b.npy" \
      --tolerance-pct "$TOLERANCE_PCT" \
      --output-json "$OUT_DIR/report.json"
    ;;

  *)
    echo "usage: $0 {render <out_npy> | compare <out_dir>}" >&2
    echo "see header comment for the full sbatch two-step." >&2
    exit 2
    ;;
esac
