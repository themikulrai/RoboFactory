#!/bin/bash
# DART pilot: run the SIM tests then a small sigma sweep on a GPU compute node.
#
# DO NOT submit by hand and DO NOT run on the login node — SAPIEN/Vulkan renders
# scenes ~32% darker on the login node (see CLAUDE.md), which corrupts the data.
#
# Submit via iris-mcp, which sets partition / gpu_type / mem / time as ARGS
# (#SBATCH partition/gpu/time are intentionally NOT hardcoded here — iris-mcp
# overrides them; pass e.g. gpu_type=a40, mem=64G). iris-mcp also forces stdout
# to /iris/u/mikulrai/slurm/<jobid>.out regardless of --output below.
#
# Example (conceptual — actual submit goes through iris-mcp.submit_job):
#   submit_job(script=run_pilot.sbatch, partition=iris-hi, gpu_type=a40,
#              mem=64G, time=08:00:00)
#
#SBATCH --job-name=dart-pilot
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# NOTE: deliberately NOT using `set -e`. A high-sigma run may keep few/zero
# demos (that low keep-rate IS the sweep signal) and a SAPIEN rollout can
# segfault; neither must abort the rest of the sweep. Each sigma runs in its
# own process + record-dir and is guarded with `|| true`.
set -uo pipefail

PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
REPO=/iris/u/mikulrai/projects/RoboFactory
cd "$REPO"

export SAPIEN_HEADLESS=1

echo "=== [1/2] DART tests (incl. SIM) ==="
DART_RUN_SIM=1 "$PY" -m pytest scripts/dart/tests -v || true

echo "=== [2/2] DART sigma sweep (LiftBarrier, --num 5 each) ==="
TASK=LiftBarrier
ROOT=/iris/u/mikulrai/data/RoboFactory/hf_download_dart
# sigma=0.0 is the clean baseline; 0.15/0.30/0.45 are the disturbance arms.
# Each gets its own subdir + dart_meta.json so the sweep figure has one
# keep-rate point per sigma even if a sibling run fails/segfaults.
for SIGMA in 0.0 0.15 0.30 0.45; do
    echo "--- sigma=${SIGMA} ---"
    "$PY" scripts/dart/run_dart_rollouts.py \
        --task "$TASK" \
        --num 5 \
        --sigma "$SIGMA" \
        --record-dir "${ROOT}/sigma${SIGMA}" \
        --save-video || true
done

echo "=== DART pilot done ==="
