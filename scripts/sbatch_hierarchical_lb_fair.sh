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
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
HL_MODEL="${MR_CKPT_ROOT:?MR_CKPT_ROOT unset — source ~/.config/dataroots.sh}/memer/ckpts/lb_dual_r64/checkpoint-3000"
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm0/lb_ws_decent_arm0_v1/18000
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm1/lb_ws_decent_arm1_v1/18000

LIVE_JSON=/iris/u/mikulrai/logs/MemER/eval/ws_hier_fair_live.json
source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam workspace --method pi05 --category eval --variant hier_fair
RESULTS_DIR="$RUN_LOG_DIR"
VIDEO_DIR="$RUN_VIDEO_DIR"

mkdir -p "$(dirname "$LIVE_JSON")"

# Workspace head cameras; bases 100..159 (60 trials) at stride 100000 -> env seeds
# 10_000_000.. (project convention); live JSON records the BASE seed. max_env_steps=400,
# replan_after=8, K=25.
bash "$LAUNCHER" \
  --hl-model "$HL_MODEL" \
  --ll-ckpt-arm0 "$LL_CKPT_ARM0" \
  --ll-ckpt-arm1 "$LL_CKPT_ARM1" \
  --camera-family workspace \
  --hl-query-interval 25 \
  --seed-pool canonical_env_60 \
  --max-env-steps 400 \
  --replan-after 8 \
  --live-json "$LIVE_JSON" \
  --results-dir "$RESULTS_DIR" \
  --video-dir "$VIDEO_DIR"

echo "[sbatch] DONE. results in $RESULTS_DIR ; live json at $LIVE_JSON"
ls -la "$RESULTS_DIR" || true
logrun_finish --status done --config "" --cmd "$0"
