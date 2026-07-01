#!/usr/bin/env bash
# SLURM wrapper: CAMERA-CORRECT (workspace head cameras) hierarchical Lift-Barrier eval.
# Real, camera-matched number on partially-trained checkpoints. Boss + helpers both see the
# WORKSPACE head cameras (head_camera_global / agent0 / agent1) they trained on -- unlike the
# throwaway sanity run which used egocentric hand cameras. Runs HL Qwen + 2x pi0.5 LL + SAPIEN
# in one compute-node job (satisfies Gate-G2 SLURM_JOB_ID guard). DualHLPolicy fix (54e0521).
set -euo pipefail

echo "[sbatch] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

LAUNCHER=/iris/u/mikulrai/projects/RoboFactory/scripts/run_hierarchical_lift_barrier_eval.sh

# LATEST complete checkpoints at launch time.
HL_MODEL=/iris/u/mikulrai/data/memer/ckpts/lb_dual_r64/checkpoint-1000
LL_CKPT_ARM0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm0/lb_ws_decent_arm0_v1/6000
LL_CKPT_ARM1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_decent_arm1/lb_ws_decent_arm1_v1/6000

LIVE_JSON=/iris/u/mikulrai/data/memer/eval/ws_hier_live.json
source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam workspace --method pi05 --category eval --variant hier
RESULTS_DIR="$RUN_LOG_DIR"
VIDEO_DIR="$RUN_VIDEO_DIR"

mkdir -p "$(dirname "$LIVE_JSON")"

# Workspace camera family; seeds 20000-20029 (30, OOD) == the ws flat baseline for comparison;
# max_env_steps=400, replan_after=8 to match the baseline. K=25 HL query interval.
bash "$LAUNCHER" \
  --hl-model "$HL_MODEL" \
  --ll-ckpt-arm0 "$LL_CKPT_ARM0" \
  --ll-ckpt-arm1 "$LL_CKPT_ARM1" \
  --camera-family workspace \
  --hl-query-interval 25 \
  --n-episodes 30 \
  --base-seed 20000 \
  --max-env-steps 400 \
  --replan-after 8 \
  --live-json "$LIVE_JSON" \
  --results-dir "$RESULTS_DIR" \
  --video-dir "$VIDEO_DIR"

echo "[sbatch] DONE. results in $RESULTS_DIR ; live json at $LIVE_JSON"
ls -la "$RESULTS_DIR" || true
logrun_finish --status done --config "" --cmd "$0"
