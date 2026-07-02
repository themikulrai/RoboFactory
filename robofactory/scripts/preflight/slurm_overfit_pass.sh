#!/bin/bash
#SBATCH --job-name=overfit_pass
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --error=/iris/u/mikulrai/slurm/%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2
#
# Plan workflow#2 caveat — bootstrap step for fresh-train heavy preflights.
#
# Heavy preflights (slurm_heavy_preflights.sh) require a freshly-overfit
# ckpt to feed check_overfit_replay_sanity.py. For the FIRST training of
# a new (model, dataset, task, scheme) combination, no such ckpt exists.
# This driver produces one by training on max_train_episodes=1 for a small
# number of epochs, then copying the resulting ckpt to OVERFIT_CKPT_OUT.
#
# The temp cwd dance: train.py saves ckpts at `checkpoints/<zarr_stem>/N.ckpt`
# RELATIVE to cwd. The canonical robofactory dir has `checkpoints/` as a
# symlink into /iris/u/mikulrai/checkpoints/RoboFactory/, which would
# clobber real training output. So we stage a temp work dir with symlinks
# into the source tree, an *unsymlinked* `checkpoints/` (real fresh dir),
# run train.py from there, then copy the produced ckpt out and prune the
# work dir.
#
# Required env vars:
#   TASK_CONFIG       - hydra task group (e.g. default_task, default_task_wristcam)
#   TASK_NAME         - env id (e.g. PickMeat-rf, ThreeRobotsStackCube-rf)
#   ZARR_PATH         - absolute path to training zarr
#   AGENT_ID          - 0 / 1 / 2 (decentralised: per-arm)
#   EXP_NAME          - wandb exp_name (suffix '-overfit' is added)
#   RGB_WEIGHTS       - 'IMAGENET1K_V1' (str) or 'null' (no quotes; passed verbatim)
#   OVERFIT_CKPT_OUT  - absolute path where the produced ckpt is copied
#
# Optional env vars:
#   NUM_EPOCHS        - default 200; tune up if overfit MSE doesn't reach 0
#   CONFIG_NAME       - default robot_dp.yaml (use joint_dp.yaml for centralised multi-arm)
#   WANDB_PROJECT     - default diffusion-robofactory
#
# Usage (typical, called by submit_with_preflights.sh --needs-overfit):
#   sbatch --export=ALL,TASK_CONFIG=default_task,TASK_NAME=PickMeat-rf,\
#                 ZARR_PATH=/iris/.../PickMeat-rf_150.zarr,AGENT_ID=0,\
#                 EXP_NAME=pm-d1-ep300-in1k,RGB_WEIGHTS=IMAGENET1K_V1,\
#                 OVERFIT_CKPT_OUT=/iris/u/mikulrai/runs/preflight/<run_id>/overfit.ckpt \
#          scripts/preflight/slurm_overfit_pass.sh

set -euxo pipefail

: "${TASK_CONFIG:?TASK_CONFIG must be set}"
: "${TASK_NAME:?TASK_NAME must be set}"
: "${ZARR_PATH:?ZARR_PATH must be set}"
: "${AGENT_ID:?AGENT_ID must be set}"
: "${EXP_NAME:?EXP_NAME must be set}"
: "${RGB_WEIGHTS:?RGB_WEIGHTS must be set}"
: "${OVERFIT_CKPT_OUT:?OVERFIT_CKPT_OUT must be set}"

NUM_EPOCHS="${NUM_EPOCHS:-200}"
CONFIG_NAME="${CONFIG_NAME:-robot_dp.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-diffusion-robofactory}"

[ -f "$ZARR_PATH" ] || [ -d "$ZARR_PATH" ] || {
    echo "ERROR: ZARR_PATH does not exist: $ZARR_PATH" >&2
    exit 1
}

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
mkdir -p $HOME/.r3m

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export CUDA_VISIBLE_DEVICES=0

ROBOFACTORY=/iris/u/mikulrai/projects/RoboFactory/robofactory
WORK_DIR=/iris/u/mikulrai/tmp/overfit_pass/${SLURM_JOB_ID:-$$}_${EXP_NAME}
ZARR_STEM=$(basename "$ZARR_PATH" .zarr)

echo "[overfit] WORK_DIR=$WORK_DIR"
echo "[overfit] ZARR_STEM=$ZARR_STEM"
echo "[overfit] OVERFIT_CKPT_OUT=$OVERFIT_CKPT_OUT"
echo "[overfit] NUM_EPOCHS=$NUM_EPOCHS  AGENT_ID=$AGENT_ID  RGB_WEIGHTS=$RGB_WEIGHTS"

# Stage a clean work dir whose `checkpoints/` is a real, fresh directory
# (NOT the symlink into the canonical store). Symlink the rest of the
# tree so train.py imports + configs resolve as normal.
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/checkpoints"
mkdir -p "$WORK_DIR/data"
ln -sfn "$ROBOFACTORY/policy"      "$WORK_DIR/policy"
ln -sfn "$ROBOFACTORY/configs"     "$WORK_DIR/configs"
ln -sfn "$ROBOFACTORY/robofactory" "$WORK_DIR/robofactory"
# data/zarr_data is read by the dataset loader; symlink the canonical
# path so the existing absolute ZARR_PATH stays valid AND the relative
# data/* hydra defaults still resolve.
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
ln -sfn "${RF_DATA_ROOT:?RF_DATA_ROOT unset — source ~/.config/dataroots.sh}/zarr_data" "$WORK_DIR/data/zarr_data"
ln -sfn "${RF_DATA_ROOT:?}/h5_data"   "$WORK_DIR/data/h5_data" 2>/dev/null || true

cd "$WORK_DIR"

OVERFIT_OUT_DIR=$(dirname "$OVERFIT_CKPT_OUT")
mkdir -p "$OVERFIT_OUT_DIR"

# Train. Overrides:
#   max_train_episodes=1            -> overfit on a single demo
#   num_epochs=$NUM_EPOCHS           -> small, fixed budget
#   checkpoint_every=$NUM_EPOCHS     -> only save at end (avoid wasted IO)
#   val_every=99999                  -> skip val (no signal in overfit-of-1)
#   capture_calibration=False        -> don't pollute /runs/calibration/
#   resume=False                     -> never resume (fresh overfit each call)
python ./policy/Diffusion-Policy/train.py \
  --config-name="$CONFIG_NAME" \
  task="$TASK_CONFIG" \
  task.name="$TASK_NAME" \
  task.dataset.zarr_path="$ZARR_PATH" \
  task.dataset.max_train_episodes=1 \
  current_agent_id="$AGENT_ID" \
  policy.obs_encoder.rgb_model.weights="$RGB_WEIGHTS" \
  training.debug=False \
  training.resume=False \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs="$NUM_EPOCHS" \
  training.checkpoint_every="$NUM_EPOCHS" \
  training.val_every=99999 \
  training.rollout_every=99999 \
  training.capture_calibration=False \
  exp_name="${EXP_NAME}-overfit" \
  logging.mode=online \
  logging.project="$WANDB_PROJECT" \
  logging.group=overfit_pass \
  "logging.tags=[overfit-pass,${EXP_NAME},agent${AGENT_ID}]" \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8

# train.py saved the final ckpt at <WORK_DIR>/checkpoints/<ZARR_STEM>/<NUM_EPOCHS>.ckpt.
PRODUCED="$WORK_DIR/checkpoints/$ZARR_STEM/${NUM_EPOCHS}.ckpt"
[ -f "$PRODUCED" ] || {
    echo "ERROR: expected overfit ckpt not found at $PRODUCED" >&2
    ls -la "$WORK_DIR/checkpoints/$ZARR_STEM/" >&2 || true
    exit 1
}

# Atomic publish to the public path (so heavy preflights afterok-dep finds it).
cp "$PRODUCED" "${OVERFIT_CKPT_OUT}.tmp"
mv "${OVERFIT_CKPT_OUT}.tmp" "$OVERFIT_CKPT_OUT"
echo "[overfit] published ckpt: $OVERFIT_CKPT_OUT  ($(stat -c %s "$OVERFIT_CKPT_OUT") bytes)"

# Free the temp work tree (ckpt is safe in OVERFIT_CKPT_OUT).
rm -rf "$WORK_DIR"
echo "[overfit] DONE"
