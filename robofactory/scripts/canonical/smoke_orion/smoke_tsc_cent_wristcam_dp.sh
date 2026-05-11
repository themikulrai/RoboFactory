#!/bin/bash
#SBATCH --job-name=smk_tsc_cent_wc_dp
#SBATCH --output=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_cent_wristcam_dp_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_cent_wristcam_dp_%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=orion
#SBATCH --partition=orion

# Coverage-matrix smoke: TSC × cent × wristcam × DP (1 seed)
# Ckpt in trash (Tier 3 purge 2026-05-08) but smoke runs anyway to verify pipeline.
# Source: tsc_d2_wristcam_cent_table_60seeds.sh

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

CKPT_REL=/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_joint_d2_wristcam_150.trash_1778245319_2287959/300.ckpt

python -u -m robofactory.utils.preflight_eval \
  --ckpt-path "$CKPT_REL" \
  --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_joint_dp.py \
  --ckpt-path "$CKPT_REL" \
  --config configs/table/three_robots_stack_cube.yaml \
  --env-id ThreeRobotsStackCube-rf \
  --camera-family wristcam \
  --n-agents 3 \
  --robot-uids panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi \
  --max-steps 400 \
  --quiet \
  --wandb \
  --wandb-tags 'eval,smoke,coverage-matrix,orion,tsc,cent,wristcam,dp' \
  --seed 100
