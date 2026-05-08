#!/bin/bash
#SBATCH --job-name=auto_calib
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/iris/u/mikulrai/runs/calibration/slurm-%j.out
#SBATCH --error=/iris/u/mikulrai/runs/calibration/slurm-%j.err
#
# Workflow improvement #4 — auto-capture calibration NPZ from a freshly
# trained ckpt. Designed to be submitted with `--dependency=afterok:<train>`
# so calibration runs iff training succeeds; failed training auto-cancels.
#
# Goal: every clean training pass produces a calibration NPZ at
# /iris/u/mikulrai/runs/calibration/<run_id>.npz so the S1 encoder-collapse
# probe graduates from metric-only to hard-gate naturally as new working
# ckpts emerge (the probe needs ≥4 calibration ckpts before threshold-
# based gating becomes safe; today there are 2).
#
# Required env vars (passed via `sbatch --export=ALL,...`):
#   CKPT       — path to the just-saved ckpt
#   CONFIG     — eval/scene config YAML
#   OUT_NPZ    — output NPZ path (e.g. runs/calibration/<run_id>.npz)
#
# Optional env vars (with defaults):
#   PROBE      — calibration script; default chosen by task name in CONFIG:
#                  pick_meat.yaml         -> script/debug/probe_pm_in1k_calibration.py
#                  three_robots_*.yaml    -> script/debug/probe_decent_dp_features.py
#                anything else: must be set explicitly
#   IMG_HEIGHT — default 240
#   IMG_WIDTH  — default 320
#   SEED_START — default 10000
#   N_SEEDS    — default 30
#
# Usage:
#   sbatch --dependency=afterok:<train_jid> \
#          --export=ALL,CKPT=...,CONFIG=...,OUT_NPZ=... \
#          scripts/canonical/auto_capture_calibration.sh

set -euo pipefail

: "${CKPT:?CKPT must be set}"
: "${CONFIG:?CONFIG must be set}"
: "${OUT_NPZ:?OUT_NPZ must be set}"

IMG_HEIGHT="${IMG_HEIGHT:-240}"
IMG_WIDTH="${IMG_WIDTH:-320}"
SEED_START="${SEED_START:-10000}"
N_SEEDS="${N_SEEDS:-30}"

if [ -z "${PROBE:-}" ]; then
    base=$(basename "$CONFIG")
    case "$base" in
        pick_meat.yaml)         PROBE="script/debug/probe_pm_in1k_calibration.py" ;;
        three_robots_stack_cube.yaml) PROBE="script/debug/probe_decent_dp_features.py" ;;
        *)
            echo "ERROR: don't know which probe to use for config '$base'." >&2
            echo "       Set PROBE=<script/debug/probe_*.py> explicitly." >&2
            exit 2
            ;;
    esac
fi

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch

cd /iris/u/mikulrai/projects/RoboFactory/robofactory
mkdir -p "$(dirname "$OUT_NPZ")"

SEEDS=$(seq "$SEED_START" $((SEED_START + N_SEEDS - 1)) | tr '\n' ' ')

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(ts)] auto_capture_calibration starting"
echo "[$(ts)]   CKPT=$CKPT"
echo "[$(ts)]   CONFIG=$CONFIG"
echo "[$(ts)]   PROBE=$PROBE"
echo "[$(ts)]   OUT_NPZ=$OUT_NPZ"
echo "[$(ts)]   seeds=$SEED_START..$((SEED_START + N_SEEDS - 1))"

python "$PROBE" \
    --config="$CONFIG" \
    --ckpt-path="$CKPT" \
    --img-height="$IMG_HEIGHT" \
    --img-width="$IMG_WIDTH" \
    --seeds $SEEDS \
    --out "$OUT_NPZ"

echo "[$(ts)] done. NPZ at $OUT_NPZ"
