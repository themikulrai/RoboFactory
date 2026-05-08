#!/bin/bash
#SBATCH --job-name=tsc_d2_in1k_a1
#SBATCH --partition=orion
#SBATCH --account=orion
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/tsc_d2_in1k_a1_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/tsc_d2_in1k_a1_%j.err

# D2 wristcam decentralised TSC — Agent 1, ImageNet ResNet18.
# Ckpt dir: checkpoints/ThreeRobotsStackCube-rf_agent1_d2_wristcam_150/ — distinct from D1.

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
  task.dataset.zarr_path=/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent1_d2_wristcam_150.zarr \
  task.dataset.max_train_episodes=150 \
  current_agent_id=1 \
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  training.debug=False \
  training.resume=False \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=tsc-d2-ep300-in1k-a1 \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d2_tsc_encoder_fix \
  'logging.tags=[robot_dp,default_task_wristcam,tsc-d2-ep300-in1k-a1,encoder-fix,imagenet-pretrained,decentralised,d2-wristcam]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64
