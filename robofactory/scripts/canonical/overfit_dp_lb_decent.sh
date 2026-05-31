#!/bin/bash
# // not migrated: env-var-driven zarr aliasing (ARM 0..3/EXP_TAG) + per-arm runtime symlink; not statically resolvable
#SBATCH --job-name=dp_lb_overfit_quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/iris/u/mikulrai/logs/overfit_lb/dp_lb_overfit_%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/overfit_lb/dp_lb_overfit_%x_%j.err

# Quick overfit test for LongPipelineDelivery (LB) task: can DP memorize 1 demo?
# User's hypothesis: cube tasks (2SC TSC) fail overfit because of cube-specific
# bug; LB works at 51.9-91% SR normally so its overfit should succeed.
#
# Quick mode: 500 epochs (not 2000), batch 32, single GPU per arm.
#
# Env vars (required): ARM (0..3)

set -euxo pipefail
mkdir -p /iris/u/mikulrai/logs/overfit_lb

: "${ARM:?ARM env var required (0..3)}"
CAM_FAMILY="${CAM_FAMILY:-workspace}"
OVERFIT_N="${OVERFIT_N:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXP_TAG="${EXP_TAG:-overfit1}"

TASK_CONFIG="default_task_wristcam"
ZARR_DIR="/iris/u/mikulrai/data/RoboFactory/zarr_data"
ZARR_SRC_NAME="LongPipelineDelivery_${CAM_FAMILY}_agent${ARM}"
ZARR_NEW_NAME="LongPipelineDelivery_${CAM_FAMILY}_${EXP_TAG}_agent${ARM}_150"
ZARR_SRC="${ZARR_DIR}/${ZARR_SRC_NAME}.zarr"
ZARR_PATH="${ZARR_DIR}/${ZARR_NEW_NAME}.zarr"

[ -e "$ZARR_SRC" ] || { echo "MISSING source zarr: $ZARR_SRC"; exit 1; }
[ -e "$ZARR_PATH" ] || ln -s "${ZARR_SRC_NAME}.zarr" "$ZARR_PATH"

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
mkdir -p "$HOME/.r3m"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
export CUDA_VISIBLE_DEVICES=0

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

python ./policy/Diffusion-Policy/train.py \
  --config-name=robot_dp.yaml \
  task=$TASK_CONFIG \
  task.name=LongPipelineDelivery-rf \
  task.dataset.zarr_path="$ZARR_PATH" \
  task.dataset.max_train_episodes=$OVERFIT_N \
  task.dataset.val_ratio=0 \
  current_agent_id=$ARM \
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  training.debug=False \
  training.resume=True \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=$NUM_EPOCHS \
  training.checkpoint_every=$CHECKPOINT_EVERY \
  training.val_every=10000 \
  training.rollout_every=10000 \
  training.sample_every=10000 \
  training.capture_calibration=False \
  exp_name=lb-${CAM_FAMILY}-${EXP_TAG}-a${ARM} \
  logging.mode=online \
  logging.project=LB-DP \
  logging.group=dp_lb_${CAM_FAMILY}_${EXP_TAG} \
  logging.name="Train LB WS DP ${EXP_TAG} A${ARM}" \
  "logging.tags=[robot_dp,lb,${EXP_TAG},${CAM_FAMILY},decent,quick-overfit,n${OVERFIT_N}]" \
  dataloader.batch_size=$BATCH_SIZE \
  val_dataloader.batch_size=$BATCH_SIZE

CKPT_ROOT="/iris/u/mikulrai/checkpoints/RoboFactory"
EVAL_LINK="${CKPT_ROOT}/LongPipelineDelivery-rf_agent${ARM}_${CAM_FAMILY}_${EXP_TAG}_150"
[ -e "$EVAL_LINK" ] || ln -s "${ZARR_NEW_NAME}" "$EVAL_LINK"

echo "Train done at $(date). Ckpts at ${CKPT_ROOT}/${ZARR_NEW_NAME}/ (symlinked from $EVAL_LINK)"
echo "Eval: --ckpt-suffix=${CAM_FAMILY}_${EXP_TAG} --data-num=150 --checkpoint-num=$NUM_EPOCHS"
