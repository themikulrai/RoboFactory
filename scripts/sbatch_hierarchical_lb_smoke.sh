#!/usr/bin/env bash
# SLURM wrapper for the FIRST closed-loop hierarchical Lift-Barrier eval (Task #9).
# PLUMBING/SANITY on REAL (~30%-trained) checkpoints -- success rate is THROWAWAY.
# Runs HL Qwen + 2x pi0.5 LL + SAPIEN eval all inside this single compute-node job, so the
# eval's Gate-G2 guard (requires SLURM_JOB_ID) is satisfied. Submitted to iris-hi.
set -euo pipefail

echo "[sbatch] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

LAUNCHER=/iris/u/mikulrai/projects/RoboFactory/scripts/run_hierarchical_lift_barrier_eval.sh

source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
HL_MODEL="${MR_CKPT_ROOT:?MR_CKPT_ROOT unset — source ~/.config/dataroots.sh}/memer/ckpts/lb_dual_r64/checkpoint-1000"
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm0/lb_ws_decent_arm0_v1/6000
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm1/lb_ws_decent_arm1_v1/6000
source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam wristcam --method pi05 --category misc --variant hier_smoke
RESULTS_DIR="$RUN_LOG_DIR"
VIDEO_DIR="$RUN_VIDEO_DIR"

# All three model processes share the single allocated GPU (HL_GPU=LL_GPU0=LL_GPU1=0 default).
# LL JAX servers run at XLA mem-fraction 0.30, prealloc off -> coexist with torch HL + SAPIEN.
# hierarchical mode (NOT --flat-baseline): HL queried every K steps -> per-arm subtasks -> LL.
bash "$LAUNCHER" \
  --hl-model "$HL_MODEL" \
  --ll-ckpt-arm0 "$LL_CKPT_ARM0" \
  --ll-ckpt-arm1 "$LL_CKPT_ARM1" \
  --hl-query-interval 25 \
  --n-episodes 5 \
  --base-seed 1000 \
  --results-dir "$RESULTS_DIR" \
  --video-dir "$VIDEO_DIR"

echo "[sbatch] DONE. results in $RESULTS_DIR"
ls -la "$RESULTS_DIR" || true
logrun_finish --status done --config "" --cmd "$0"
