#!/bin/bash
#SBATCH --job-name=smk_tsc_dec_wc_dp
#SBATCH --output=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_decent_wristcam_dp_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_decent_wristcam_dp_%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=orion
#SBATCH --partition=orion

# Coverage-matrix smoke: TSC × decent × wristcam × DP (1 seed)
# Ckpts in trash; we created temp symlinks in checkpoints/ pointing at the trashed dirs.
# Source: tsc_d2_wristcam_table_60seeds_reeval.sh

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

python -u -m robofactory.utils.preflight_eval \
  --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_multi_dp.py \
  --config=configs/table/three_robots_stack_cube.yaml \
  --data-num=150 --checkpoint-num=300 \
  --ckpt-suffix=d2_wristcam \
  --obs-cam-family=wristcam \
  --robot-uids=panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi \
  --include-global \
  --img-height=224 --img-width=224 \
  --render-mode=sensors -o=rgb -b=gpu -n 1 \
  -s 100 \
  --max-steps=400 \
  --wandb \
  --wandb-tags='eval,smoke,coverage-matrix,orion,tsc,decent,wristcam,dp' \
  --quiet
