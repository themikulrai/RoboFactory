#!/bin/bash
#SBATCH --job-name=pm_retrain_crop
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_crop_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_crop_%j.err

# Track B B.5.1: crop ablation. Random-init ResNet18 (matches paper config),
# but with crop_shape=[216,288] enabled (90% of 240x320, mirrors Chi et al's
# robomimic 76/84 ratio). Tests whether the encoder's "feature collapse" is
# really an absent-augmentation symptom rather than an absent-prior failure.
# Target: SR > 25% would prove crop alone fixes the policy.

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
# vulkan_icd.sh from conda activate.d already sets VK_ICD_FILENAMES; skip stale _vulkan_env.sh sourcing.

# Redirect HOME off the AFS quota onto IRIS scratch (defensive: random-init still touches torch hub).
export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch

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
  policy.obs_encoder.crop_shape=[216,288] \
  training.debug=False \
  training.resume=False \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=pm-d1-ep300-crop \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d1_encoder_fix \
  'logging.tags=[robot_dp,default_task,pm-d1-ep300-crop,track-b,crop-ablation,random-init]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64
