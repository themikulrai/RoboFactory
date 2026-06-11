#!/usr/bin/env bash
# SLURM wrapper: FAIR workspace hierarchical Lift-Barrier eval (HEADLINE number).
# FULLY-TRAINED models, workspace head cameras, Lift-Barrier pi0.5 project seed convention
# (env seed = base*100000; bases 100..159 = 60 trials). Runs HL Qwen + 2x pi0.5 LL + SAPIEN
# in one compute-node job (Gate-G2 SLURM_JOB_ID guard). DualHLPolicy (3 head-cam images + dual).
set -euo pipefail

echo "[sbatch] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

LAUNCHER=/iris/u/mikulrai/projects/RoboFactory/scripts/run_hierarchical_lift_barrier_eval.sh

# FULLY-TRAINED checkpoints.
HL_MODEL=/iris/u/mikulrai/data/memer/ckpts/lb_dual_r64/checkpoint-3000
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm0/lb_ws_decent_arm0_v1/18000
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm1/lb_ws_decent_arm1_v1/18000

LIVE_JSON=/iris/u/mikulrai/data/memer/eval/ws_hier_fair_live.json
RESULTS_DIR=/iris/u/mikulrai/projects/RoboFactory/eval_results/ws_hier_fair_${SLURM_JOB_ID:-manual}
VIDEO_DIR=$RESULTS_DIR/videos

mkdir -p "$RESULTS_DIR" "$VIDEO_DIR" "$(dirname "$LIVE_JSON")"

# Workspace head cameras; bases 100..159 (60 trials) at stride 100000 -> env seeds
# 10_000_000.. (project convention); live JSON records the BASE seed. max_env_steps=400,
# replan_after=8, K=25.
bash "$LAUNCHER" \
  --hl-model "$HL_MODEL" \
  --ll-ckpt-arm0 "$LL_CKPT_ARM0" \
  --ll-ckpt-arm1 "$LL_CKPT_ARM1" \
  --camera-family workspace \
  --hl-query-interval 25 \
  --n-episodes 60 \
  --base-seed 100 \
  --seed-stride 100000 \
  --max-env-steps 400 \
  --replan-after 8 \
  --live-json "$LIVE_JSON" \
  --results-dir "$RESULTS_DIR" \
  --video-dir "$VIDEO_DIR"

echo "[sbatch] DONE. results in $RESULTS_DIR ; live json at $LIVE_JSON"
ls -la "$RESULTS_DIR" || true
