#!/usr/bin/env bash
# GRASP PROBE: short wristcam hierarchical eval (30 seeds) with per-arm grasp-contact
# logging (does each gripper actually hold the barrier, or lift empty?). Reuses the wc
# pipeline; instrumentation lives in run_episode (grasp_trace / barz_trace in results JSON).
set -euo pipefail
echo "[sbatch] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
nvidia-smi -L || true

export TMPDIR=/iris/u/mikulrai/tmp; mkdir -p "$TMPDIR"
export OPENPI_DATA_HOME=/iris/u/mikulrai/data/openpi
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache

LAUNCHER=/iris/u/mikulrai/projects/RoboFactory/scripts/run_hierarchical_lift_barrier_eval.sh
HL_MODEL=/iris/u/mikulrai/data/memer/ckpts/lb_wc_dual_r64/checkpoint-3000
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm0/lb_wc_decent_arm0_v1/19999
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm1/lb_wc_decent_arm1_v1/19999
LIVE_JSON=/iris/u/mikulrai/data/memer/eval/wc_grasp_live.json
source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam wristcam --method pi05 --category probe --variant grasp
RESULTS_DIR="$RUN_LOG_DIR"
mkdir -p "$(dirname "$LIVE_JSON")"

bash "$LAUNCHER" \
  --hl-model "$HL_MODEL" \
  --ll-config-arm0 pi05_robofactory_lb_wc_decent_arm0 --ll-ckpt-arm0 "$LL_CKPT_ARM0" \
  --ll-config-arm1 pi05_robofactory_lb_wc_decent_arm1 --ll-ckpt-arm1 "$LL_CKPT_ARM1" \
  --camera-family wristcam --hl-query-interval 25 \
  --env-seeds 10000000,10100000,10200000,10300000,10400000,10500000,10600000,10700000,10800000,10900000,11000000,11100000,11200000,11300000,11400000,11500000,11600000,11700000,11800000,11900000,12000000,12100000,12200000,12300000,12400000,12500000,12600000,12700000,12800000,12900000 \
  --max-env-steps 400 --replan-after 8 \
  --live-json "$LIVE_JSON" --results-dir "$RESULTS_DIR"

echo "[sbatch] DONE. results in $RESULTS_DIR"; ls -la "$RESULTS_DIR" || true
logrun_finish --status done --config "" --cmd "$0"
