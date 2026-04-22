#!/bin/bash
# Centralised (joint) diffusion policy -- baseline trainer.
# Usage:
#   bash train_joint.sh TakePhoto-rf 150 0 0
#   bash train_joint.sh LongPipelineDelivery-rf 150 0 0
#
# Positional args:
#   $1 task (TakePhoto-rf | LongPipelineDelivery-rf | other 4-arm task)
#   $2 load_num (number of training trajectories, e.g. 150)
#   $3 seed
#   $4 gpu_id

set -eu

task_name=${1}
load_num=${2}
seed=${3}
gpu_id=${4}

DEBUG=False
alg_name=joint_dp
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-${alg_name}-${addition_info}

case "${task_name}" in
  TakePhoto-rf)              cfg_rel="robocasa/take_photo.yaml" ;;
  LongPipelineDelivery-rf)   cfg_rel="robocasa/long_pipeline_delivery.yaml" ;;
  *)  cfg_rel="" ;;   # leave null so the env runner errors loudly
esac

if [ "${DEBUG}" = "True" ]; then
  wandb_mode=offline
else
  wandb_mode=online
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

extra_env_runner=""
if [ -n "${cfg_rel}" ]; then
  # Hydra needs the absolute path; RoboFactory's CONFIG_DIR is picked up at runtime,
  # so we pass it via env var and build the full path inside the runner fallback.
  extra_env_runner="task.env_runner.env_id=${task_name}"
fi

python ./policy/Diffusion-Policy/train.py \
  --config-name=${config_name}.yaml \
  task.name=${task_name} \
  task.dataset.zarr_path="data/zarr_data/${task_name}_joint_${load_num}.zarr" \
  eval.data_num=${load_num} \
  training.debug=${DEBUG} \
  training.seed=${seed} \
  training.device="cuda:0" \
  exp_name=${exp_name} \
  logging.mode=${wandb_mode} \
  ${extra_env_runner} \
  "${@:5}"
