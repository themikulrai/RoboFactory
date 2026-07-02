#!/bin/bash
#SBATCH --job-name=pm_eval_dino_spatch_60s
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --error=/iris/u/mikulrai/slurm/%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=iris-hp-z8,iris-hgx-1

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task PickMeat-rf --cam dino_spatch --method dp --category eval --variant ablation

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HF_HOME=$HOME/.cache/huggingface

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# Stage-3 preflight guards — refuses to run on scene/camera/seed/wandb mismatch.
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/pick_meat.yaml}"
TRAIN_CFG_PATH="${TRAIN_CFG_PATH:-${EVAL_CFG_PATH}}"
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
SEEDS=$(paste -sd' ' /iris/u/mikulrai/runs/eval_seeds_60_dp.txt)
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60_dp.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds_60_dp.sha256 \
    --argv-seeds "$SEEDS"
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

# Stage-3 per-eval guards (plan v2 C1#10).
python -u -m robofactory.utils.preflight_eval \
  --ckpt-path /iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_dino_spatch.ckpt \
  --scene-config configs/table/pick_meat.yaml || exit 1

RESULT_FILE="$RUN_LOG_DIR/pm_eval_dino_spatch_${SLURM_JOB_ID:-manual}.jsonl"
python -u ./policy/Diffusion-Policy/eval_dp.py \
  --config=configs/table/pick_meat.yaml \
  --ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_dino_spatch.ckpt \
  --data-num=150 \
  --checkpoint-num=300 \
  -o rgb -b cpu -n 1 \
  --render-mode=sensors \
  -s $SEEDS \
  --quiet \
  --max-steps=200 \
  --jsonl-path "$RESULT_FILE" \
  --wandb \
  --wandb-tags='eval,pm,track-b,encoder-dino-spatch,60seeds,xbucket,seedset-dp-xbucket'

# A8 loop-killer: auto-post a field-notes cell (job id + result jsonl + SR) so the
# new-cell-per-run convention is mechanical, not memory-dependent (WEEK1 §A8).
python scripts/log_eval.py \
    --jsonl "$RESULT_FILE" \
    --job "${SLURM_JOB_ID:-manual}" \
    --title "Eval PickMeat WC DP [dino-spatch enc, 60seeds]" || true

logrun_finish --status done --config "" --cmd "$0"
