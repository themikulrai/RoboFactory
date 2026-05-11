#!/bin/bash
#SBATCH --job-name=tsc_d1_freezeenc_decent_60s
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/tsc_d1_freezeenc_decent_60s_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/tsc_d1_freezeenc_decent_60s_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2

# Tier 2.1 — Decentralised TSC eval, D1 workspace, FROZEN ImageNet ResNet18 encoder.
# Ckpts (3 arms, evaluated jointly by eval_multi_dp.py):
#   ThreeRobotsStackCube-rf_agent{0,1,2}_freezeenc_150/300.ckpt
# Trained by .slurm_scripts/retrain_dp_tsc_d1_freezeenc.sh:
#   - cameras: workspace (no head_cam_global), img 240x320
#   - encoder: IMAGENET1K_V1, training.freeze_encoder=True
#   - epochs: 300, batch 64, lr 2e-4 (defaults)
# Goal: if any arm reaches >=30% SR, it becomes a calibration-candidate
# good-ref ckpt for S1 (encoder-collapse probe).

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# Stage-3 preflight guards — refuses to run on scene/camera/seed/wandb mismatch.
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/three_robots_stack_cube.yaml}"
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent0_freezeenc_150/300.ckpt}"
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/canonical/_resolve_train_cfg.sh
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
SEEDS=$(paste -sd' ' /iris/u/mikulrai/runs/eval_seeds_60.txt)
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds.sha256 \
    --argv-seeds "$SEEDS"
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

# Stage-3 per-eval guards (plan v2 C1#10). eval_multi_dp.py computes per-arm
# ckpt paths internally from --data-num/--checkpoint-num/--ckpt-suffix.
python -u -m robofactory.utils.preflight_eval \
  --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_multi_dp.py \
  --config=configs/table/three_robots_stack_cube.yaml \
  --data-num=150 \
  --checkpoint-num=300 \
  --ckpt-suffix=freezeenc \
  --obs-cam-family=workspace \
  --no-include-global \
  --img-height=240 \
  --img-width=320 \
  -o rgb -b cpu -n 1 \
  --render-mode=sensors \
  -s $SEEDS \
  --quiet \
  --max-steps=400 \
  --wandb \
  --wandb-tags='eval,dp,decent,freezeenc,tsc,table-scene,tier2.1,calibration-candidate'
