#!/bin/bash
#SBATCH --job-name=pm_retrain_dino_spatch
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_dino_spatch_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_dino_spatch_%j.err

# Track B B.5.3b: DINOv2 ViT-S/14 frozen + Perceiver-style cross-attn over patch tokens.
# Tests "preserve spatial info, don't pool to CLS" hypothesis (CAGE-style).
# Reads 256 patch tokens at 224x224 input, compresses via 64 learnable latents.

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HF_HOME=$HOME/.cache/huggingface

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=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
export CUDA_VISIBLE_DEVICES=0

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

python ./policy/Diffusion-Policy/train.py \
  --config-name=robot_dp.yaml \
  task=default_task \
  task.name=PickMeat-rf \
  task.dataset.zarr_path=/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150.zarr \
  task.dataset.max_train_episodes=150 \
  current_agent_id=0 \
  policy.obs_encoder.rgb_model._target_=diffusion_policy.model.vision.model_getter.get_dinov2_patchattn \
  policy.obs_encoder.rgb_model.name=vit_small_patch14_dinov2 \
  policy.obs_encoder.resize_shape=[224,224] \
  policy.obs_encoder.use_group_norm=False \
  training.debug=False \
  training.resume=False \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=pm-d1-ep300-dino-spatch \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d1_encoder_fix \
  'logging.tags=[robot_dp,default_task,pm-d1-ep300-dino-spatch,track-b,dinov2-vits14,patch-attn,perceiver]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64
