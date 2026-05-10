#!/bin/bash
#SBATCH --job-name=pm_retrain_in1k
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --time=24:00:00
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_in1k_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pm_retrain_in1k_%j.err

# Encoder fix: ImageNet-pretrained ResNet18 (was rgb_model.weights=null → trained from scratch).
# Retrain target: PM ep300 SR ≥ 40% (vs true baseline ~10%).
# Disk-quota fix: redirect torchvision and r3m caches off AFS home onto IRIS scratch.
#
# Heavy-preflight chain (workflow#2): submit this launcher gated on the
# overfit-pass + heavy-preflight battery via submit_with_preflights.sh
# rather than calling sbatch directly. From-scratch retrains use
# --needs-overfit so a 1-episode overfit ckpt is produced first; heavy
# preflights then validate train/eval pipeline parity against that ckpt
# before the real 24h retrain is allowed to launch.
#
# Example (from-scratch PM in1k, 3-step chain):
#   bash scripts/canonical/submit_with_preflights.sh \
#       --train-launcher scripts/canonical/retrain_dp_pm_d1_ep300_in1k.sh \
#       --dataset /iris/u/mikulrai/data/RoboFactory/zarr_data/PickMeat-rf_150.zarr \
#       --scene-config configs/table/pick_meat.yaml \
#       --needs-overfit \
#       --of-task-config default_task --of-task-name PickMeat-rf \
#       --of-zarr /iris/u/mikulrai/data/RoboFactory/zarr_data/PickMeat-rf_150.zarr \
#       --of-agent-id 0 --of-exp-name pm-d1-ep300-in1k \
#       --of-rgb-weights IMAGENET1K_V1
#
# On subsequent attempts (overfit ckpt already exists), drop --needs-overfit
# and pass --ckpt <prior-overfit.ckpt> for the 2-step chain.

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
# vulkan_icd.sh from conda activate.d already sets VK_ICD_FILENAMES; skip stale _vulkan_env.sh sourcing.

# Redirect HOME off the AFS quota onto IRIS scratch so torchvision/r3m caches succeed.
export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
mkdir -p $HOME/.r3m

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
  policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 \
  training.debug=False \
  training.resume=False \
  training.seed=100 \
  training.device=cuda:0 \
  training.num_epochs=300 \
  training.rollout_every=10000 \
  exp_name=pm-d1-ep300-in1k \
  logging.mode=online \
  logging.project=diffusion-robofactory \
  logging.group=dp_d1_encoder_fix \
  'logging.tags=[robot_dp,default_task,pm-d1-ep300-in1k,encoder-fix,imagenet-pretrained]' \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64
