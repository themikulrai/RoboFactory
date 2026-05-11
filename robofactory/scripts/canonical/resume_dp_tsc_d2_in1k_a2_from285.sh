#!/bin/bash
#SBATCH --job-name=resume_tsc_d2_in1k_a2
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --time=06:00:00
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/resume_tsc_d2_in1k_a2_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/resume_tsc_d2_in1k_a2_%j.err

# Resume of gz1g5q1o (24h walltime kill at epoch 285 of 300). Explicit load.
#
# IMPORTANT — submit via the heavy-preflight orchestrator, not raw `sbatch`:
#
#   bash scripts/canonical/submit_with_preflights.sh \
#     --train-launcher scripts/canonical/resume_dp_tsc_d2_in1k_a2_from285.sh \
#     --ckpt /iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150/285.ckpt \
#     --dataset /iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150.zarr \
#     --scene-config configs/table/three_robots_stack_cube.yaml
#
# This gates training on slurm_heavy_preflights.sh via SLURM
# --dependency=afterok:<heavy>. If the preflights fail (init-pose drift,
# vulkan render mismatch, or overfit-replay regression) the training job
# is auto-cancelled before any GPU time is consumed. Workflow plan ref:
# "Workflow improvements (compounding stress-savers)" #2.

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export CUDA_VISIBLE_DEVICES=0

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# Cheap-suite preflight (Plan v2 C2 S5, belt+suspenders to heavy preflights):
# config-only checks for control_rate / action / image / camera parity. Runs
# on the compute node so a config edit between submit and start is still
# caught. Heavy preflights (init-pose, overfit-replay, vulkan) are gated at
# submit time via submit_with_preflights.sh.
PREFLIGHT_RUN_DIR="/iris/u/mikulrai/runs/preflight/${SLURM_JOB_ID:-local}"
mkdir -p "$PREFLIGHT_RUN_DIR"
python -m robofactory.scripts.preflight.dump_train_cfg \
    --task-config default_task_wristcam \
    --scene-config configs/table/three_robots_stack_cube.yaml \
    --agent-id 2 \
    --out-train "$PREFLIGHT_RUN_DIR/train_cheap.yaml" \
    --out-eval  "$PREFLIGHT_RUN_DIR/eval_cheap.yaml"
python -m robofactory.scripts.preflight.train_eval_consistency \
    --train-cfg "$PREFLIGHT_RUN_DIR/train_cheap.yaml" \
    --eval-cfg  "$PREFLIGHT_RUN_DIR/eval_cheap.yaml" \
    --out       "$PREFLIGHT_RUN_DIR/preflight_consistency.json" \
  || { echo "preflight failed; aborting training"; exit 1; }

python ./policy/Diffusion-Policy/train.py \
  --config-name=robot_dp.yaml \
  task=default_task_wristcam \
  task.name=ThreeRobotsStackCube-rf \
  task.dataset.zarr_path=/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150.zarr \
  task.dataset.max_train_episodes=150 \
  current_agent_id=2 \
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  training.debug=False \
  training.resume=False \
  +training.load_ckpt=/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150/285.ckpt \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=tsc-d2-ep300-in1k-a2 \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d2_tsc_encoder_fix \
  'logging.tags=[robot_dp,default_task_wristcam,tsc-d2-ep300-in1k-a2,encoder-fix,imagenet-pretrained,decentralised,d2-wristcam,resume-of-gz1g5q1o]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64
