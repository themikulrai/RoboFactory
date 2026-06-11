#!/bin/bash
# 60-seed TABLE rollout eval of joint/CENTRALISED Diffusion-Policy LiftBarrier
# (LB) wristcam (one joint DP checkpoint). Reconstructed from ad-hoc slurm job
# 15484502 (SR 57/60). Re-run uses shader_pack=default (the runner now hardcodes
# it) AND the new always-on tiled multi-view video. DP seed pool, max_steps=150.
#
# partition / gpu / account are set at submit time via the iris-mcp tool args.
#
#SBATCH --job-name=eval_dp_lb_wc_cent
#SBATCH --output=/iris/u/mikulrai/logs/eval_dp_lb_wc_cent/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_dp_lb_wc_cent/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_dp_lb_wc_cent

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

CKPT=/iris/u/mikulrai/checkpoints/RoboFactory/58ea4af29c21/300.ckpt
[ -f "$CKPT" ] || { echo "missing DP joint ckpt: $CKPT"; exit 1; }

RECORD_DIR=/iris/u/mikulrai/logs/eval_dp_lb_wc_cent/joint_${SLURM_JOB_ID}_{env_id}
mkdir -p /iris/u/mikulrai/logs/eval_dp_lb_wc_cent
# DP seed pool (x-bucket), passed space-separated.
SEEDS=$(tr '\n' ' ' < /iris/u/mikulrai/runs/eval_seeds_60_dp.txt)

RESULT_FILE="/iris/u/mikulrai/logs/eval_dp_lb_wc_cent/lb_wc_cent_joint_60seeds_${SLURM_JOB_ID:-manual}.jsonl"
"$PY" -u ./policy/Diffusion-Policy/eval_joint_dp.py \
    --ckpt-path "$CKPT" \
    --config configs/table/lift_barrier.yaml \
    --camera-family wristcam \
    --env-id LiftBarrier-rf \
    --n-agents 2 \
    --robot-uids "panda_wristcam_multi,panda_wristcam_multi" \
    --max-steps 150 \
    --record-dir "$RECORD_DIR" \
    --jsonl-path "$RESULT_FILE" \
    --seed $SEEDS

echo "Done at $(date)"
echo "Record dir: $RECORD_DIR"
echo "Results JSONL: $RESULT_FILE"
# A8 loop-killer: auto-post a field-notes cell (job id + result jsonl + SR) so the
# new-cell-per-run convention is mechanical, not memory-dependent (WEEK1 §A8).
python scripts/log_eval.py \
    --jsonl "$RESULT_FILE" \
    --job "${SLURM_JOB_ID:-manual}" \
    --title "Eval LB wc DP [Cent joint, 60seeds]" || true
