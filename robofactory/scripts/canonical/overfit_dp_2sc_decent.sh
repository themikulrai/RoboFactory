#!/bin/bash
# // not migrated: env-var-driven zarr aliasing (ARM/CAM_FAMILY/EXP_TAG/OVERFIT_N) + runtime symlink scheme; no static manifest resolution
#SBATCH --job-name=dp_2sc_overfit_decent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/iris/u/mikulrai/logs/overfit_2026-05-11/dp_2sc_overfit_%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/overfit_2026-05-11/dp_2sc_overfit_%x_%j.err

# Diagnostic overfit run for 2SC DP decentralised — sanity check for the same
# encoder-collapse / shared-obs pathology seen in TSC. N=1 demo overfit; if SR
# on the source seed is still ~0%, pipeline has a bug deeper than at-scale
# ambiguity. If SR ~100%, pipeline works and the 0% from 150-demo training is
# a data-ambiguity / encoder-shortcut issue at scale.
#
# Env vars (required): ARM (0 or 1)
# Env vars (optional): CAM_FAMILY=workspace, OVERFIT_N=1, NUM_EPOCHS=2000,
#                      CHECKPOINT_EVERY=400, BATCH_SIZE=32, EXP_TAG=overfit1,
#                      WITH_GLOBAL=true (set false for no-head_cam_global ablation)

set -euxo pipefail
mkdir -p /iris/u/mikulrai/logs/overfit_2026-05-11

: "${ARM:?ARM env var required (0 or 1)}"
CAM_FAMILY="${CAM_FAMILY:-workspace}"
OVERFIT_N="${OVERFIT_N:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WITH_GLOBAL="${WITH_GLOBAL:-true}"
EXP_TAG="${EXP_TAG:-overfit${OVERFIT_N}}"

if [ "$WITH_GLOBAL" = "false" ]; then
    TASK_CONFIG="default_task_wristcam_no_global"
else
    TASK_CONFIG="default_task_wristcam"
fi

ZARR_DIR="/iris/u/mikulrai/data/RoboFactory/zarr_data"
ZARR_SRC_NAME="TwoRobotsStackCube-rf_${CAM_FAMILY}_decent_agent${ARM}_150"
ZARR_NEW_NAME="TwoRobotsStackCube-rf_${CAM_FAMILY}_${EXP_TAG}_agent${ARM}_150"
ZARR_SRC="${ZARR_DIR}/${ZARR_SRC_NAME}.zarr"
ZARR_PATH="${ZARR_DIR}/${ZARR_NEW_NAME}.zarr"

# Symlink so ckpts go to a distinct dir (don't clobber the 'decent' 300.ckpt)
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
  task.name=TwoRobotsStackCube-rf \
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
  exp_name=2sc-${CAM_FAMILY}-${EXP_TAG}-a${ARM} \
  logging.mode=online \
  logging.project=2SC-DP \
  logging.group=dp_2sc_${CAM_FAMILY}_${EXP_TAG} \
  logging.name="Train 2SC WS DP ${EXP_TAG} A${ARM}" \
  "logging.tags=[robot_dp,2sc,${EXP_TAG},${CAM_FAMILY},decent,encoder-debug,n${OVERFIT_N},with_global_${WITH_GLOBAL}]" \
  dataloader.batch_size=$BATCH_SIZE \
  val_dataloader.batch_size=$BATCH_SIZE

# Make canonical eval-name symlink so eval_multi_dp.py can find the ckpts.
CKPT_ROOT="/iris/u/mikulrai/checkpoints/RoboFactory"
EVAL_LINK="${CKPT_ROOT}/TwoRobotsStackCube-rf_agent${ARM}_${CAM_FAMILY}_${EXP_TAG}_150"
[ -e "$EVAL_LINK" ] || ln -s "${ZARR_NEW_NAME}" "$EVAL_LINK"

echo "Train done at $(date). Ckpts at ${CKPT_ROOT}/${ZARR_NEW_NAME}/ (symlinked from $EVAL_LINK)"
echo "Eval: --ckpt-suffix=${CAM_FAMILY}_${EXP_TAG} --data-num=150 --checkpoint-num=$NUM_EPOCHS"
