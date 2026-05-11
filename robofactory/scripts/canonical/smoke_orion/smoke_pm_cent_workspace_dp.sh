#!/bin/bash
#SBATCH --job-name=smk_pm_cent_ws_dp
#SBATCH --output=/iris/u/mikulrai/logs/coverage_matrix_smoke/pm_cent_workspace_dp_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/coverage_matrix_smoke/pm_cent_workspace_dp_%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=orion
#SBATCH --partition=orion

# Coverage-matrix smoke: PM × cent × workspace × DP (1 seed)
# Source: pm_eval_in1k_60seeds.sh — adapted for orion + 1 seed.

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# Smoke: skip the strict 60-seed sha-checked preflight guards because the seed
# file is a 60-seed file but we run with 1 seed; only run the lightweight
# scene/path-exists guards.
python -u -m robofactory.utils.preflight_eval \
  --ckpt-path /iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt \
  --scene-config configs/table/pick_meat.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_dp.py \
  --config=configs/table/pick_meat.yaml \
  --ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt \
  --data-num=150 \
  --checkpoint-num=300 \
  -o rgb -b cpu -n 1 \
  --render-mode=sensors \
  -s 100 \
  --quiet \
  --max-steps=200 \
  --wandb \
  --wandb-tags='eval,smoke,coverage-matrix,orion,pm,cent,workspace,dp'
