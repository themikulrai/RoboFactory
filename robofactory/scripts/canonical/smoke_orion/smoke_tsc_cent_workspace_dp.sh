#!/bin/bash
#SBATCH --job-name=smk_tsc_cent_ws_dp
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --error=/iris/u/mikulrai/slurm/%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=orion
#SBATCH --partition=orion

# Coverage-matrix smoke: TSC × cent × workspace × DP (1 seed)
# Ckpt is in trash (Tier 3 purge 2026-05-08) but smoke runs anyway to verify pipeline.
# Source: tsc_d1_workspace_cent_table_60seeds.sh

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# eval_joint_dp.py opens './' + ckpt_path -- needs path relative to project root.
# The original symlink under checkpoints/... points at the same physical file
# now in the .trash dir; resolve through the trashed dir directly.
CKPT_REL=/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_joint_d1_workspace_150_in1k.trash_1778245319_2287959/300.ckpt

python -u -m robofactory.utils.preflight_eval \
  --ckpt-path "$CKPT_REL" \
  --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_joint_dp.py \
  --ckpt-path "$CKPT_REL" \
  --config configs/table/three_robots_stack_cube.yaml \
  --env-id ThreeRobotsStackCube-rf \
  --camera-family workspace \
  --n-agents 3 \
  --max-steps 400 \
  --quiet \
  --wandb \
  --wandb-tags 'eval,smoke,coverage-matrix,orion,tsc,cent,workspace,dp' \
  --seed 100
