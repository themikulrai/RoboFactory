#!/bin/bash
#SBATCH --job-name=pm_retrain_in1k_crop
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_in1k_crop_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_in1k_crop_%j.err

# Track B follow-up: ImageNet-ResNet18 (same encoder that recovered 58.33% SR baseline)
# + random-crop augmentation on top. Tests whether crop adds further gain over the
# already-fixed encoder. Crop_shape=[216,288] = 90% of 240x320, mirrors Chi et al's
# robomimic 76/84 ratio. The earlier crop ablation used random-init weights (lost).
#
# Save-dir collision avoidance: workspace derives ckpt path from zarr stem
# (checkpoints/{zarr_stem}/{epoch}.ckpt). A symlinked zarr alias gives this run
# a private dir at checkpoints/PickMeat-rf_150_in1k_crop/.

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

# Redirect HOME off the AFS quota onto IRIS scratch so torchvision caches succeed.
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
  task.dataset.zarr_path=/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_in1k_crop.zarr \
  task.dataset.max_train_episodes=150 \
  current_agent_id=0 \
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  policy.obs_encoder.crop_shape=[216,288] \
  training.debug=False \
  training.resume=False \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=pm-d1-ep300-in1k-crop \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d1_encoder_fix \
  'logging.tags=[robot_dp,default_task,pm-d1-ep300-in1k-crop,track-b,crop-on-imagenet,encoder-fix]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64

# Sanity: verify the ckpt landed in the dedicated dir (not the shared PickMeat-rf_150 dir).
ls -la /iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150_in1k_crop/300.ckpt
