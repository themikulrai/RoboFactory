#!/usr/bin/env bash
# COMPREHENSIVE wristcam (egocentric) hierarchical Lift-Barrier eval.
# HL Qwen ckpt-3000 + 2x pi0.5 wristcam LL @19999, EGOCENTRIC hand cameras.
# Same seed convention as the workspace fair run (bases 100..159, env=base*100000)
# so wc-vs-ws is a fair head-to-head. Patched eval records per-step subtask trace
# (into the JSON + live JSON) and burns the active subtask onto every video frame.
set -euo pipefail
echo "[sbatch] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} CUDA=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

# Put server logs on the SHARED filesystem (default TMPDIR is node-local /tmp,
# unreadable from elsewhere) so LL/HL server failures are diagnosable.
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$TMPDIR"

# CRITICAL: point openpi at the WARM asset cache the training jobs used, so the LL
# serve_policy processes find the 12GB pi05_base locally instead of re-downloading it
# from gs://openpi-assets to the cold NFS home cache (that 11.6GB download blew past the
# launcher's 6-min LL health window and killed the first two attempts). These are
# inherited by the launcher's LL serve subshells (they only override CUDA/XLA).
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/iris/u/mikulrai/.cache/openpi}"
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache

LAUNCHER=/iris/u/mikulrai/projects/RoboFactory/scripts/run_hierarchical_lift_barrier_eval.sh

HL_MODEL="${MR_CKPT_ROOT:?MR_CKPT_ROOT unset — source ~/.config/dataroots.sh}/memer/ckpts/lb_wc_dual_r64/checkpoint-3000"
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm0/lb_wc_decent_arm0_v1/19999
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm1/lb_wc_decent_arm1_v1/19999

LIVE_JSON=/iris/u/mikulrai/logs/MemER/eval/wc_hier_live.json
source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam wristcam --method pi05 --category eval --variant hier
RESULTS_DIR="$RUN_LOG_DIR"
VIDEO_DIR="$RUN_VIDEO_DIR"
mkdir -p "$(dirname "$LIVE_JSON")"

bash "$LAUNCHER" \
  --hl-model "$HL_MODEL" \
  --ll-config-arm0 pi05_robofactory_lb_wc_decent_arm0 --ll-ckpt-arm0 "$LL_CKPT_ARM0" \
  --ll-config-arm1 pi05_robofactory_lb_wc_decent_arm1 --ll-ckpt-arm1 "$LL_CKPT_ARM1" \
  --camera-family wristcam \
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
