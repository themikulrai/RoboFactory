#!/bin/bash
#SBATCH --job-name=dp_2sc_overfit_qpos_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/iris/u/mikulrai/logs/overfit_qpos_long_2026-05-13/dp_%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/overfit_qpos_long_2026-05-13/dp_%x_%j.err

# Tier H — extended state=qpos retrain to 8000 epochs total (resume from 2000).
# Per-step L2 at 2000 ep was 0.016 rad mean. Need to drive it below ~5 mrad to
# avoid trajectory drift over 250+ eval steps. checkpoint_every=1000.

set -euxo pipefail
mkdir -p /iris/u/mikulrai/logs/overfit_qpos_long_2026-05-13

: "${ARM:?ARM env var required (0 or 1)}"
NUM_EPOCHS="${NUM_EPOCHS:-8000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXP_TAG="${EXP_TAG:-overfit1qpos}"

ZARR_PATH="/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_${EXP_TAG}_agent${ARM}_150.zarr"
[ -e "$ZARR_PATH" ] || { echo "MISSING zarr: $ZARR_PATH"; exit 1; }

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
mkdir -p "$HOME/.r3m"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
export CUDA_VISIBLE_DEVICES=0

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# training.resume=True will pick up from the latest checkpoint in the run_uuid dir
python ./policy/Diffusion-Policy/train.py \
  --config-name=robot_dp.yaml \
  task=default_task_wristcam \
  task.name=TwoRobotsStackCube-rf \
  task.dataset.zarr_path="$ZARR_PATH" \
  task.dataset.max_train_episodes=1 \
  task.dataset.val_ratio=0 \
  current_agent_id=$ARM \
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  training.debug=False \
  training.resume=True \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=$NUM_EPOCHS \
  training.checkpoint_every=$CHECKPOINT_EVERY \
  training.val_every=500 \
  training.rollout_every=10000 \
  training.sample_every=10000 \
  training.capture_calibration=False \
  exp_name=2sc-workspace-${EXP_TAG}-long-a${ARM} \
  logging.mode=online \
  logging.project=2SC-DP \
  logging.group=dp_2sc_workspace_${EXP_TAG}_long \
  logging.name="Train 2SC WS DP ${EXP_TAG} long A${ARM}" \
  "logging.tags=[robot_dp,2sc,${EXP_TAG},workspace,decent,h3-fix-long,n1,with_global_true]" \
  dataloader.batch_size=$BATCH_SIZE \
  val_dataloader.batch_size=$BATCH_SIZE

# Symlink any new ckpts under the canonical eval path
CKPT_ROOT="/iris/u/mikulrai/checkpoints/RoboFactory"
ZARR_NEW_NAME="TwoRobotsStackCube-rf_workspace_${EXP_TAG}_agent${ARM}_150"
RUN_UUID=$(cat "${CKPT_ROOT}/${ZARR_NEW_NAME}/run_uuid.txt" 2>/dev/null || echo "")
if [ -n "$RUN_UUID" ]; then
  for f in "${CKPT_ROOT}/${RUN_UUID}"/*.ckpt; do
    bn=$(basename "$f")
    [ -e "${CKPT_ROOT}/${ZARR_NEW_NAME}/${bn}" ] || ln -s "../${RUN_UUID}/${bn}" "${CKPT_ROOT}/${ZARR_NEW_NAME}/${bn}"
  done
  echo "Symlinked new ckpts from ${CKPT_ROOT}/${RUN_UUID}"
fi
echo "Train done at $(date)"
ls -la ${CKPT_ROOT}/${ZARR_NEW_NAME}/ | head -20
