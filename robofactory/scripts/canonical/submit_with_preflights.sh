#!/bin/bash
# Submit a training launcher gated on the heavy-preflight battery.
#
# Workflow improvement #2: every new training run is gated on
# slurm_heavy_preflights.sh (init-pose Wasserstein + overfit-replay
# sanity, single-node) via --dependency=afterok. Without this, a
# pipeline regression burns the full training time before anyone
# notices. With this, the regression crashes in 5 min before the
# real GPU is allocated.
#
# Usage:
#   bash scripts/canonical/submit_with_preflights.sh \
#       --train-launcher scripts/canonical/retrain_dp_pm_d1_ep300_in1k.sh \
#       --ckpt   /iris/u/mikulrai/checkpoints/.../300_in1k.ckpt \
#       --dataset /iris/u/mikulrai/data/RoboFactory/zarr_data/PickMeat-rf_150.zarr \
#       --scene-config configs/table/pick_meat.yaml \
#       [--out-dir /iris/u/mikulrai/runs/preflight/<run_id>]
#       [--max-steps 50] [--mse-tolerance 0.01]
#       [--skip-vulkan]   # cross-node vulkan check stays manual for now
#
# For RETRAIN launchers (retrain_dp_*, resume_dp_*) you typically already
# have a baseline ckpt to preflight against — pass its path as --ckpt.
#
# For FRESH training, run a 100-step overfit pass first, then preflight
# against THAT tiny ckpt. The same heavy-preflight artifact is reusable
# across full training attempts on the same (model, dataset, task) combo.

set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# //;s/^#//' >&2
    exit 2
}

TRAIN_LAUNCHER=""
CKPT_PATH=""
DATASET_PATH=""
SCENE_CONFIG=""
OUT_DIR=""
EXTRA_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-launcher) TRAIN_LAUNCHER="$2"; shift 2 ;;
        --ckpt)           CKPT_PATH="$2"; shift 2 ;;
        --dataset)        DATASET_PATH="$2"; shift 2 ;;
        --scene-config)   SCENE_CONFIG="$2"; shift 2 ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        --max-steps)      EXTRA_ENV="${EXTRA_ENV},MAX_STEPS=$2"; shift 2 ;;
        --mse-tolerance)  EXTRA_ENV="${EXTRA_ENV},MSE_TOLERANCE=$2"; shift 2 ;;
        --episode-idx)    EXTRA_ENV="${EXTRA_ENV},EPISODE_IDX=$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

[ -n "$TRAIN_LAUNCHER" ] || { echo "ERROR: --train-launcher required" >&2; usage; }
[ -f "$TRAIN_LAUNCHER" ] || { echo "ERROR: $TRAIN_LAUNCHER not found" >&2; exit 1; }
[ -n "$CKPT_PATH" ]      || { echo "ERROR: --ckpt required" >&2; usage; }
[ -n "$DATASET_PATH" ]   || { echo "ERROR: --dataset required" >&2; usage; }
[ -n "$SCENE_CONFIG" ]   || { echo "ERROR: --scene-config required" >&2; usage; }

# Default OUT_DIR includes the train launcher basename + timestamp so
# parallel preflights for different launchers don't collide.
if [ -z "$OUT_DIR" ]; then
    base=$(basename "$TRAIN_LAUNCHER" .sh)
    ts=$(date +%Y%m%d_%H%M%S)
    OUT_DIR="/iris/u/mikulrai/runs/preflight/${base}_${ts}"
fi
mkdir -p "$OUT_DIR"

PREFLIGHT_SCRIPT="/iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/preflight/slurm_heavy_preflights.sh"
[ -f "$PREFLIGHT_SCRIPT" ] || { echo "ERROR: $PREFLIGHT_SCRIPT not found" >&2; exit 1; }

# Submit heavy preflights with the inputs as env vars.
PREFLIGHT_EXPORT="ALL,CKPT_PATH=${CKPT_PATH},DATASET_PATH=${DATASET_PATH},SCENE_CONFIG=${SCENE_CONFIG},OUT_DIR=${OUT_DIR}${EXTRA_ENV}"

echo "[submit_with_preflights] Submitting heavy preflights..."
JID_PRE=$(sbatch --parsable --export="${PREFLIGHT_EXPORT}" "$PREFLIGHT_SCRIPT")
echo "  preflight JID: ${JID_PRE}"
echo "  out dir:       ${OUT_DIR}"

echo "[submit_with_preflights] Submitting training (gated afterok:${JID_PRE})..."
JID_TRAIN=$(sbatch --parsable --dependency="afterok:${JID_PRE}" "$TRAIN_LAUNCHER")
echo "  training JID:  ${JID_TRAIN}"

cat <<EOF

Done. Two jobs queued:
  ${JID_PRE}  heavy preflights (single-node)
  ${JID_TRAIN}  training (waits afterok)

Cancel both:
  scancel ${JID_PRE} ${JID_TRAIN}

Reports:
  ${OUT_DIR}/init_pose_wasserstein.json
  ${OUT_DIR}/overfit_replay.json

If preflights fail, training is auto-cancelled by the dependency.
EOF
