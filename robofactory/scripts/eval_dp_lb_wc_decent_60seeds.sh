#!/bin/bash
# 60-seed TABLE rollout eval of Diffusion-Policy LiftBarrier (LB) DECENTRALISED
# wristcam (per-arm DP models loaded in one process on cuda:0). Reconstructed
# from ad-hoc slurm job 15420463 (SR 55/60). Re-run uses shader_pack=default AND
# the new always-on tiled multi-view video. DP seed pool (x-bucket), max_steps=150.
#
# partition / gpu / account are set at submit time via the iris-mcp tool args.
#
#SBATCH --job-name=eval_dp_lb_wc_dec
#SBATCH --output=/iris/u/mikulrai/logs/eval_dp_lb_wc_decent/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_dp_lb_wc_decent/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_dp_lb_wc_decent

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$TMPDIR"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"

# Eval always runs against the canonical main checkout (committed multiview code).
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

# DP decent ckpts resolve via ckpt_suffix: checkpoints/{task}_agent{id}_{suffix}_{data_num}/{ckpt}.ckpt
# checkpoints/ is a symlink -> /iris/u/mikulrai/checkpoints/RoboFactory
for id in 0 1; do
    p="checkpoints/LiftBarrier-rf_agent${id}_wristcam_decent_150/300.ckpt"
    [ -f "$p" ] || { echo "missing DP ckpt: $p"; exit 1; }
done

VIDEO_DIR=/iris/u/mikulrai/logs/eval_dp_lb_wc_decent/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/eval_dp_lb_wc_decent
mkdir -p "$VIDEO_DIR" "$OUT_DIR"
# DP seed pool (x-bucket 10000-10029 + 1000-1029), passed space-separated.
SEEDS=$(tr '\n' ' ' < /iris/u/mikulrai/runs/eval_seeds_60_dp.txt)

"$PY" -u ./policy/Diffusion-Policy/eval_multi_dp.py \
    --env-id LiftBarrier-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml \
    --robot-uids "panda_wristcam_multi,panda_wristcam_multi" \
    --control-mode pd_joint_pos \
    --obs-mode rgb \
    --sim-backend auto \
    --ckpt-suffix wristcam_decent \
    --data-num 150 \
    --checkpoint-num 300 \
    --obs-cam-family wristcam \
    --include-global \
    --img-height 224 \
    --img-width 224 \
    --max-steps 150 \
    --shader default \
    --quiet \
    --skip-collapse-probe \
    --video-dir "$VIDEO_DIR" \
    --jsonl-path "${OUT_DIR}/lb_wc_decent_ckpt300_60seeds_${SLURM_JOB_ID}.jsonl" \
    --wandb \
    --wandb-project LB-DP \
    --wandb-name "Eval LB WC DP Decent ckpt300 60seeds" \
    --wandb-tags "eval,dp,decent,lb,wristcam,table-scene,60seeds,ckpt300,shaderfix,multiview" \
    --seed $SEEDS

echo "Done at $(date)"
echo "Videos: $VIDEO_DIR"
echo "Results JSONL: ${OUT_DIR}/lb_wc_decent_ckpt300_60seeds_${SLURM_JOB_ID}.jsonl"
