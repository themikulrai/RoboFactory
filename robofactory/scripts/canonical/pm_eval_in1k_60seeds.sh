#!/bin/bash
#SBATCH --job-name=pm_eval_in1k_60s
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pm_eval_in1k_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pm_eval_in1k_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=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
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# Stage-3 preflight guards — refuses to run on scene/camera/seed/wandb mismatch.
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/pick_meat.yaml}"
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt}"
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/canonical/_resolve_train_cfg.sh
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
SEEDS=$(paste -sd' ' /iris/u/mikulrai/runs/eval_seeds_60_dp.txt)
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60_dp.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds_60_dp.sha256 \
    --argv-seeds "$SEEDS"
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

# Stage-3 per-eval guards (plan v2 C1#10): scene-match + path-exists +
# wandb-online. Crashes the job loudly before any GPU work if a guard fails.
python -u -m robofactory.utils.preflight_eval \
  --ckpt-path /iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt \
  --scene-config configs/table/pick_meat.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_dp.py \
  --config=configs/table/pick_meat.yaml \
  --ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt \
  --data-num=150 \
  --checkpoint-num=300 \
  -o rgb -b cpu -n 1 \
  --render-mode=sensors \
  -s $SEEDS \
  --quiet \
  --max-steps=200 \
  --wandb \
  --wandb-tags='eval,pm,track-b,encoder-in1k,60seeds,xbucket,seedset-dp-xbucket'
