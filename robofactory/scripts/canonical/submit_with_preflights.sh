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
# For RETRAIN launchers (retrain_dp_*, resume_dp_*) on a (model, dataset,
# task, scheme) combo where a baseline ckpt already exists: pass its path
# as --ckpt and submit_with_preflights does a 2-step chain:
#     preflight -> train
#
# For FRESH training of a never-trained combo (no baseline ckpt): use
# --needs-overfit. submit_with_preflights does a 3-step chain:
#     overfit-pass -> preflight -> train
# The overfit-pass produces a tiny ckpt at OVERFIT_CKPT_OUT (default
# <out-dir>/overfit.ckpt), heavy preflights uses it, then real training
# runs on green light. The overfit-pass artifact is reusable across full
# training attempts on the same (model, dataset, task, scheme) combo;
# pass --ckpt <path-to-prior-overfit.ckpt> on subsequent attempts to
# skip the overfit step.
#
# --needs-overfit additional args (all required when set):
#   --of-task-config <hydra group>   e.g. default_task / default_task_wristcam
#   --of-task-name   <env id>        e.g. PickMeat-rf
#   --of-zarr        <abs path>      training zarr path
#   --of-agent-id    <int>           per-arm id (decentralised) or 0
#   --of-exp-name    <wandb suffix>  exp_name base; '-overfit' is appended
#   --of-rgb-weights <str|null>      'IMAGENET1K_V1' or 'null'
#   --of-config-name <str>           default robot_dp.yaml; joint_dp.yaml for cent multi-arm
#   --of-num-epochs  <int>           default 200
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
NEEDS_OVERFIT=0
OF_TASK_CONFIG=""
OF_TASK_NAME=""
OF_ZARR=""
OF_AGENT_ID=""
OF_EXP_NAME=""
OF_RGB_WEIGHTS=""
OF_CONFIG_NAME="robot_dp.yaml"
OF_NUM_EPOCHS="200"

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
        --needs-overfit)     NEEDS_OVERFIT=1; shift ;;
        --of-task-config)    OF_TASK_CONFIG="$2"; shift 2 ;;
        --of-task-name)      OF_TASK_NAME="$2"; shift 2 ;;
        --of-zarr)           OF_ZARR="$2"; shift 2 ;;
        --of-agent-id)       OF_AGENT_ID="$2"; shift 2 ;;
        --of-exp-name)       OF_EXP_NAME="$2"; shift 2 ;;
        --of-rgb-weights)    OF_RGB_WEIGHTS="$2"; shift 2 ;;
        --of-config-name)    OF_CONFIG_NAME="$2"; shift 2 ;;
        --of-num-epochs)     OF_NUM_EPOCHS="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

[ -n "$TRAIN_LAUNCHER" ] || { echo "ERROR: --train-launcher required" >&2; usage; }
[ -f "$TRAIN_LAUNCHER" ] || { echo "ERROR: $TRAIN_LAUNCHER not found" >&2; exit 1; }
[ -n "$DATASET_PATH" ]   || { echo "ERROR: --dataset required" >&2; usage; }
[ -n "$SCENE_CONFIG" ]   || { echo "ERROR: --scene-config required" >&2; usage; }

if [ "$NEEDS_OVERFIT" -eq 1 ]; then
    [ -n "$OF_TASK_CONFIG" ]  || { echo "ERROR: --of-task-config required with --needs-overfit" >&2; exit 1; }
    [ -n "$OF_TASK_NAME" ]    || { echo "ERROR: --of-task-name required with --needs-overfit" >&2; exit 1; }
    [ -n "$OF_ZARR" ]         || { echo "ERROR: --of-zarr required with --needs-overfit" >&2; exit 1; }
    [ -n "$OF_AGENT_ID" ]     || { echo "ERROR: --of-agent-id required with --needs-overfit" >&2; exit 1; }
    [ -n "$OF_EXP_NAME" ]     || { echo "ERROR: --of-exp-name required with --needs-overfit" >&2; exit 1; }
    [ -n "$OF_RGB_WEIGHTS" ]  || { echo "ERROR: --of-rgb-weights required with --needs-overfit" >&2; exit 1; }
else
    [ -n "$CKPT_PATH" ] || { echo "ERROR: --ckpt required (or pass --needs-overfit to bootstrap one)" >&2; usage; }
fi

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
OVERFIT_SCRIPT="/iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/preflight/slurm_overfit_pass.sh"
if [ "$NEEDS_OVERFIT" -eq 1 ]; then
    [ -f "$OVERFIT_SCRIPT" ] || { echo "ERROR: $OVERFIT_SCRIPT not found" >&2; exit 1; }
fi

# DRYRUN=1 echoes the sbatch commands instead of submitting. Useful for CI
# parse-checks and for showing the user what will happen before they commit.
DRYRUN="${DRYRUN:-0}"

# When --needs-overfit, run a 1-episode overfit pass first and feed its
# ckpt as CKPT_PATH into heavy preflights. The overfit ckpt is published
# to OUT_DIR/overfit.ckpt; downstream attempts can skip the overfit step
# by passing --ckpt OUT_DIR/overfit.ckpt directly.
JID_OF=""
PRE_DEP=""
if [ "$NEEDS_OVERFIT" -eq 1 ]; then
    CKPT_PATH="${OUT_DIR}/overfit.ckpt"
    OVERFIT_EXPORT="ALL,TASK_CONFIG=${OF_TASK_CONFIG},TASK_NAME=${OF_TASK_NAME},ZARR_PATH=${OF_ZARR},AGENT_ID=${OF_AGENT_ID},EXP_NAME=${OF_EXP_NAME},RGB_WEIGHTS=${OF_RGB_WEIGHTS},OVERFIT_CKPT_OUT=${CKPT_PATH},NUM_EPOCHS=${OF_NUM_EPOCHS},CONFIG_NAME=${OF_CONFIG_NAME}"
    echo "[submit_with_preflights] Submitting overfit-pass (bootstrap)..."
    if [ "$DRYRUN" = "1" ]; then
        echo "  DRYRUN: sbatch --parsable --export=\"${OVERFIT_EXPORT}\" \"$OVERFIT_SCRIPT\""
        JID_OF="<DRY_OF>"
    else
        JID_OF=$(sbatch --parsable --export="${OVERFIT_EXPORT}" "$OVERFIT_SCRIPT")
    fi
    echo "  overfit JID:   ${JID_OF}"
    echo "  produces:      ${CKPT_PATH}"
    PRE_DEP="--dependency=afterok:${JID_OF} --kill-on-invalid-dep=yes"
fi

# Submit heavy preflights with the inputs as env vars.
PREFLIGHT_EXPORT="ALL,CKPT_PATH=${CKPT_PATH},DATASET_PATH=${DATASET_PATH},SCENE_CONFIG=${SCENE_CONFIG},OUT_DIR=${OUT_DIR}${EXTRA_ENV}"

echo "[submit_with_preflights] Submitting heavy preflights..."
if [ "$DRYRUN" = "1" ]; then
    echo "  DRYRUN: sbatch --parsable ${PRE_DEP} --export=\"${PREFLIGHT_EXPORT}\" \"$PREFLIGHT_SCRIPT\""
    JID_PRE="<DRY_PRE>"
else
    JID_PRE=$(sbatch --parsable ${PRE_DEP} --export="${PREFLIGHT_EXPORT}" "$PREFLIGHT_SCRIPT")
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
EOF
if [ -n "$JID_OF" ]; then
    echo "  ${JID_OF}  overfit-pass (bootstrap)"
fi
cat <<EOF
  ${JID_PRE}  heavy preflights${JID_OF:+ (waits afterok overfit)}
  ${JID_TRAIN}  training (waits afterok preflight)
EOF
if [ -n "$JID_CALIB" ]; then
    echo "  ${JID_CALIB}  calibration capture (waits afterok train)"
fi
cat <<EOF

Cancel all:
  scancel ${JID_OF:+${JID_OF} }${JID_PRE} ${JID_TRAIN}${JID_CALIB:+ ${JID_CALIB}}

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
