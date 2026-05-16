#!/bin/bash
#SBATCH --job-name=eval_lp_overfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/iris/u/mikulrai/logs/eval_lp_overfit/eval_lp_%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_lp_overfit/eval_lp_%x_%j.err

# LP overfit eval: does the 1-demo overfit ckpt reproduce success at the
# training seed (=14)? Three retries (DDPM stochastic).
#
# Eval-time inputs:
#   CAM_FAMILY      : default workspace (alt: wristcam) — drives CKPT_SUFFIX + --obs-cam-family
#   CKPT_SUFFIX     : default ${CAM_FAMILY}_overfit1_maxfit
#   CHECKPOINT_NUM  : default 200
#   SEED            : default 14 (training-distribution episode seed for ep0)
#   N_RETRIES       : default 3 (DDPM sampler is stochastic)
#   MAX_STEPS       : default 900 (LP demos are 808 steps; +margin for success check)
#   QUIET           : default yes (set to "no" to drop --quiet → per-step prints + tqdm)

set -euxo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_lp_overfit

CAM_FAMILY="${CAM_FAMILY:-workspace}"
CKPT_SUFFIX="${CKPT_SUFFIX:-${CAM_FAMILY}_overfit1_maxfit}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-200}"
DATA_NUM="${DATA_NUM:-150}"
SEED="${SEED:-14}"
N_RETRIES="${N_RETRIES:-3}"
MAX_STEPS="${MAX_STEPS:-900}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"
QUIET="${QUIET:-yes}"
SKIP_PROBE="${SKIP_PROBE:-no}"

QUIET_FLAG=""
[ "$QUIET" = "yes" ] && QUIET_FLAG="--quiet"
SKIP_PROBE_FLAG=""
[ "$SKIP_PROBE" = "yes" ] && SKIP_PROBE_FLAG="--skip-collapse-probe"

case "$CAM_FAMILY" in
  workspace) CAM_TAG=WS ;;
  wristcam)  CAM_TAG=WC ;;
  *) CAM_TAG="$CAM_FAMILY" ;;
esac

# Build seed list as N_RETRIES copies of SEED
SEEDS=""
for ((i=0; i<N_RETRIES; i++)); do SEEDS="$SEEDS $SEED"; done

# 4 arms, all wristcam-multi
ROBOT_UIDS="panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi"

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
export HYDRA_FULL_ERROR=1
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

for id in 0 1 2 3; do
  ckpt="/iris/u/mikulrai/checkpoints/RoboFactory/LongPipelineDelivery-rf_agent${id}_${CKPT_SUFFIX}_${DATA_NUM}/${CHECKPOINT_NUM}.ckpt"
  [ -f "$ckpt" ] || { echo "MISSING ckpt: $ckpt"; exit 1; }
done

python -u ./policy/Diffusion-Policy/eval_multi_dp.py \
  --config=configs/table/long_pipeline_delivery.yaml \
  --robot-uids "$ROBOT_UIDS" \
  --data-num="$DATA_NUM" \
  --checkpoint-num="$CHECKPOINT_NUM" \
  --ckpt-suffix="$CKPT_SUFFIX" \
  --obs-cam-family="$CAM_FAMILY" \
  --include-global \
  --img-height=224 \
  --img-width=224 \
  -o rgb -b "$SIM_BACKEND" -n 1 \
  --render-mode=sensors \
  -s $SEEDS \
  $QUIET_FLAG \
  $SKIP_PROBE_FLAG \
  --max-steps="$MAX_STEPS" \
  --wandb \
  --wandb-project=LP-DP \
  --wandb-name="Eval LP ${CAM_TAG} DP Overfit1 Maxfit ckpt${CHECKPOINT_NUM} s${SEED}x${N_RETRIES}" \
  --wandb-tags="eval,lp,decent,${CAM_FAMILY},overfit1,maxfit,seed${SEED},ckpt${CHECKPOINT_NUM}"

echo "Eval done at $(date)"
