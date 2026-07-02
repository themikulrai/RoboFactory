#!/bin/bash
# // not migrated: env-var-driven zarr aliasing + one-off optimizer.lr/weight_decay/ema.max_value/lr_warmup_steps/use_bf16/compile maxfit knobs; runtime-templated
#SBATCH --job-name=lp_overfit_maxfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --error=/iris/u/mikulrai/slurm/%j.err

# LongPipelineDelivery overfit-1 training tuned for max-memorization:
#   - lr=1e-3, weight_decay=0, warmup=20 (fast LR ramp + no anti-overfit reg)
#   - random_crop=False (image aug off — augmentation kills 1-demo memorization)
#   - ema.max_value=0.99 (faster EMA tracking on ~4.8k grad steps)
#   - bf16 AMP + torch.compile (speed)
#   - 200 epochs, checkpoint_every=50 (preempt-resilient)
#
# Env vars (required): ARM (0..3)

set -euxo pipefail

: "${ARM:?ARM env var required (0..3)}"
CAM_FAMILY="${CAM_FAMILY:-workspace}"
OVERFIT_N="${OVERFIT_N:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXP_TAG="${EXP_TAG:-overfit1_maxfit}"
LR="${LR:-1e-3}"
WARMUP="${WARMUP:-20}"
SEED="${SEED:-100}"

TASK_CONFIG="default_task_wristcam"
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
ZARR_DIR="${RF_DATA_ROOT:?RF_DATA_ROOT unset — source ~/.config/dataroots.sh}/zarr_data"
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
  policy.obs_encoder.random_crop=False \
  optimizer.lr=$LR \
  optimizer.weight_decay=0 \
  ema.max_value=0.99 \
  training.debug=False \
  training.resume=True \
  training.seed=$SEED \
  training.device=cuda:0 \
  training.num_epochs=$NUM_EPOCHS \
  training.checkpoint_every=$CHECKPOINT_EVERY \
  training.lr_warmup_steps=$WARMUP \
  training.val_every=10000 \
  training.rollout_every=10000 \
  training.sample_every=10000 \
  training.capture_calibration=False \
  training.use_bf16=True \
  training.compile=True \
  exp_name=lp-${CAM_FAMILY}-${EXP_TAG}-a${ARM} \
  logging.mode=online \
  logging.project=LP-DP \
  logging.group=dp_lp_${CAM_FAMILY}_${EXP_TAG} \
  logging.name="Train LP WS DP ${EXP_TAG} A${ARM}" \
  "logging.tags=[robot_dp,lp,${EXP_TAG},${CAM_FAMILY},decent,quick-overfit,n${OVERFIT_N},maxfit]" \
  dataloader.batch_size=$BATCH_SIZE \
  val_dataloader.batch_size=$BATCH_SIZE

CKPT_ROOT="/iris/u/mikulrai/checkpoints/RoboFactory"
EVAL_LINK="${CKPT_ROOT}/LongPipelineDelivery-rf_agent${ARM}_${CAM_FAMILY}_${EXP_TAG}_150"
[ -e "$EVAL_LINK" ] || ln -s "${ZARR_NEW_NAME}" "$EVAL_LINK"

echo "Train done at $(date). Ckpts at ${CKPT_ROOT}/${ZARR_NEW_NAME}/ (symlinked from $EVAL_LINK)"
echo "Eval: --ckpt-suffix=${CAM_FAMILY}_${EXP_TAG} --data-num=150 --checkpoint-num=$NUM_EPOCHS"
