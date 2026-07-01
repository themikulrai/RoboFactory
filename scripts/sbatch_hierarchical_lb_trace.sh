#!/usr/bin/env bash
# Re-run of the FAIR workspace hierarchical Lift-Barrier eval, IDENTICAL config
# (same checkpoints/seeds/params => reproduces the same 57/60 rollouts deterministically),
# but with the patched eval that (a) records the per-step subtask trace into the JSON and
# (b) burns the active subtask + step onto every video frame. Used to build the per-episode
# subtask-distribution figure + subtitled failure/success clips.
set -euo pipefail

echo "[sbatch] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

LAUNCHER=/iris/u/mikulrai/projects/RoboFactory/scripts/run_hierarchical_lift_barrier_eval.sh

source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
HL_MODEL="${MR_CKPT_ROOT:?MR_CKPT_ROOT unset — source ~/.config/dataroots.sh}/memer/ckpts/lb_dual_r64/checkpoint-3000"
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm0/lb_ws_decent_arm0_v1/18000
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm1/lb_ws_decent_arm1_v1/18000

LIVE_JSON=/iris/u/mikulrai/data/memer/eval/ws_hier_trace_live.json
source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam workspace --method pi05 --category misc --variant hier_trace
RESULTS_DIR="$RUN_LOG_DIR"
VIDEO_DIR="$RUN_VIDEO_DIR"

mkdir -p "$(dirname "$LIVE_JSON")"

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

echo "[sbatch] DONE. results in $RESULTS_DIR"
ls -la "$RESULTS_DIR" || true
logrun_finish --status done --config "" --cmd "$0"
