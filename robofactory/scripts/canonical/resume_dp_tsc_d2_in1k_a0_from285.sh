#!/bin/bash
#SBATCH --job-name=resume_tsc_d2_in1k_a0
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
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/resume_tsc_d2_in1k_a0_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/resume_tsc_d2_in1k_a0_%j.err

# Resume of dn5uqhrs (24h walltime kill at epoch 285 of 300). Explicit load
# from epoch 285 to avoid auto-resume picking encoder-incompatible
# 290/295/300.ckpt left over from pre-encoder-fix sister runs.

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=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
export CUDA_VISIBLE_DEVICES=0

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

python ./policy/Diffusion-Policy/train.py \
  --config-name=robot_dp.yaml \
  task=default_task_wristcam \
  task.name=ThreeRobotsStackCube-rf \
  task.dataset.zarr_path=/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent0_d2_wristcam_150.zarr \
  task.dataset.max_train_episodes=150 \
  current_agent_id=0 \
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  training.debug=False \
  training.resume=False \
  +training.load_ckpt=/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent0_d2_wristcam_150/285.ckpt \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=tsc-d2-ep300-in1k-a0 \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d2_tsc_encoder_fix \
  'logging.tags=[robot_dp,default_task_wristcam,tsc-d2-ep300-in1k-a0,encoder-fix,imagenet-pretrained,decentralised,d2-wristcam,resume-of-dn5uqhrs]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64
