#!/bin/bash
#SBATCH --job-name=tsc_d2_wc_cent_60s
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/tsc_d2_wc_cent_60s_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/tsc_d2_wc_cent_60s_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2

# Tier 2.2 Job B — Centralised TSC re-eval on TABLE scene, D2 wristcam.
# Ckpt: ThreeRobotsStackCube-rf_joint_d2_wristcam_150/300.ckpt
# Train: .slurm_scripts/retrain_dp_tsc_d2_ep300_in1k_central.sh
#   - cameras: wristcam, IMAGENET1K_V1, share_rgb_model=True
#   - n_agents=3, robot_uids=panda_wristcam_multi x3
# Wandb run id: 9nkouo1r (per user). Confirms the cent variant has the
# same encoder-collapse fate as the decent d2_wristcam ckpts.
# Joint cent ckpts must use eval_joint_dp.py (eval_dp.py is single-arm).

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
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_joint_d2_wristcam_150/300.ckpt}"
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

# NOTE: eval_joint_dp.py does `open('./' + ckpt_path)` (line 91), so the
# ckpt path MUST be relative to the project root (cwd above) — an absolute
# path produces './/iris/...' and 404s. The path below is symlinked from
# /iris/u/mikulrai/checkpoints/RoboFactory/... — same physical file.
CKPT_REL=checkpoints/ThreeRobotsStackCube-rf_joint_d2_wristcam_150/300.ckpt

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
  --wandb-tags 'eval,dp,cent,tsc,d2-wristcam,table-scene,tier2.2,reeval-encoder-fix' \
  --seed $SEEDS
