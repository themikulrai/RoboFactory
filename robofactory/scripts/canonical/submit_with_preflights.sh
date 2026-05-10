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
#
# Env vars:
#   DRYRUN=1   echo the sbatch commands instead of submitting.
#              Use this to verify wiring before committing real GPU time.

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
CAPTURE_CALIBRATION=0
CALIB_RUN_ID=""
TRAIN_CKPT_OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-launcher)    TRAIN_LAUNCHER="$2"; shift 2 ;;
        --ckpt)              CKPT_PATH="$2"; shift 2 ;;
        --dataset)           DATASET_PATH="$2"; shift 2 ;;
        --scene-config)      SCENE_CONFIG="$2"; shift 2 ;;
        --out-dir)           OUT_DIR="$2"; shift 2 ;;
        --max-steps)         EXTRA_ENV="${EXTRA_ENV},MAX_STEPS=$2"; shift 2 ;;
        --mse-tolerance)     EXTRA_ENV="${EXTRA_ENV},MSE_TOLERANCE=$2"; shift 2 ;;
        --episode-idx)       EXTRA_ENV="${EXTRA_ENV},EPISODE_IDX=$2"; shift 2 ;;
        --capture-calibration) CAPTURE_CALIBRATION=1; shift ;;
        --calib-run-id)      CALIB_RUN_ID="$2"; shift 2 ;;
        --train-ckpt-out)    TRAIN_CKPT_OUT="$2"; shift 2 ;;
        -h|--help)           usage ;;
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

# DRYRUN=1 echoes the sbatch commands instead of submitting. Useful for CI
# parse-checks and for showing the user what will happen before they commit.
DRYRUN="${DRYRUN:-0}"

echo "[submit_with_preflights] Submitting heavy preflights..."
if [ "$DRYRUN" = "1" ]; then
    echo "  DRYRUN: sbatch --parsable --export=\"${PREFLIGHT_EXPORT}\" \"$PREFLIGHT_SCRIPT\""
    JID_PRE="<DRY_PRE>"
else
    JID_PRE=$(sbatch --parsable --export="${PREFLIGHT_EXPORT}" "$PREFLIGHT_SCRIPT")
fi
echo "  preflight JID: ${JID_PRE}"
echo "  out dir:       ${OUT_DIR}"

# --kill-on-invalid-dep=yes ensures the training job is auto-cancelled if the
# preflight job exits non-zero (instead of sitting forever in DependencyNeverSatisfied).
echo "[submit_with_preflights] Submitting training (gated afterok:${JID_PRE})..."
if [ "$DRYRUN" = "1" ]; then
    echo "  DRYRUN: sbatch --parsable --dependency=\"afterok:${JID_PRE}\" --kill-on-invalid-dep=yes \"$TRAIN_LAUNCHER\""
    JID_TRAIN="<DRY_TRAIN>"
else
    JID_TRAIN=$(sbatch --parsable --dependency="afterok:${JID_PRE}" --kill-on-invalid-dep=yes "$TRAIN_LAUNCHER")
fi
echo "  training JID:  ${JID_TRAIN}"

# Optionally chain calibration capture after training success (workflow#4).
JID_CALIB=""
if [ "$CAPTURE_CALIBRATION" -eq 1 ]; then
    if [ -z "$TRAIN_CKPT_OUT" ]; then
        echo "ERROR: --capture-calibration set but --train-ckpt-out missing" >&2
        echo "       Pass --train-ckpt-out <path-the-trainer-will-save-to>" >&2
        echo "       (e.g. /iris/.../PickMeat-rf_150/300.ckpt for ep300)" >&2
        exit 1
    fi
    if [ -z "$CALIB_RUN_ID" ]; then
        CALIB_RUN_ID="$(basename "$TRAIN_LAUNCHER" .sh)_$(date +%Y%m%d_%H%M%S)"
    fi
    CALIB_NPZ="/iris/u/mikulrai/runs/calibration/${CALIB_RUN_ID}.npz"
    CALIB_SCRIPT="/iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/canonical/auto_capture_calibration.sh"
    CALIB_EXPORT="ALL,CKPT=${TRAIN_CKPT_OUT},CONFIG=${SCENE_CONFIG},OUT_NPZ=${CALIB_NPZ}"
    echo "[submit_with_preflights] Submitting calibration capture (gated afterok:${JID_TRAIN})..."
    if [ "$DRYRUN" = "1" ]; then
        echo "  DRYRUN: sbatch --parsable --dependency=\"afterok:${JID_TRAIN}\" --kill-on-invalid-dep=yes --export=\"${CALIB_EXPORT}\" \"$CALIB_SCRIPT\""
        JID_CALIB="<DRY_CALIB>"
    else
        JID_CALIB=$(sbatch --parsable --dependency="afterok:${JID_TRAIN}" --kill-on-invalid-dep=yes \
                           --export="${CALIB_EXPORT}" "$CALIB_SCRIPT")
    fi
    echo "  calibration JID: ${JID_CALIB}"
    echo "  calib NPZ:       ${CALIB_NPZ}"
fi

cat <<EOF

Done. Job chain queued:
  ${JID_PRE}  heavy preflights (single-node)
  ${JID_TRAIN}  training (waits afterok)
EOF
if [ -n "$JID_CALIB" ]; then
    echo "  ${JID_CALIB}  calibration capture (waits afterok train)"
fi
cat <<EOF

Cancel all:
  scancel ${JID_PRE} ${JID_TRAIN}${JID_CALIB:+ }${JID_CALIB}

Reports:
  ${OUT_DIR}/init_pose_wasserstein.json
  ${OUT_DIR}/overfit_replay.json
EOF
if [ -n "$JID_CALIB" ]; then
    echo "  /iris/u/mikulrai/runs/calibration/${CALIB_RUN_ID}.npz"
fi
echo
echo "If preflights fail, training is auto-cancelled by the dependency."
if [ -n "$JID_CALIB" ]; then
    echo "If training fails, calibration is auto-cancelled too."
fi
