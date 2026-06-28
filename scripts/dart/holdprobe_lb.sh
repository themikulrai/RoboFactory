#!/bin/bash
# 2-episode HOLD-PROBE for the LB data-gen plateau fix:
#   - LB_HOLD_FRAMES_K=300  -> env terminates ONLY after a ~300-frame sustained hold
#     (instead of the 8-frame default), so the demo doesn't get chopped mid-lift.
#   - --settle-steps 340    -> the program keeps holding the bar (frozen_qpos) past the
#     lift so the env can count those 300 held frames -> the recorded demo plateaus.
#   - 1 recovery (+shove) + 1 clean, with video, so we can SEE the lift->decelerate->hold-still.
set -euo pipefail
export HOME=/iris/u/mikulrai
RF_WT=/iris/u/mikulrai/RoboFactory-subtask-wt
export PYTHONPATH="$RF_WT${PYTHONPATH:+:$PYTHONPATH}"
export LB_HOLD_FRAMES_K=300
PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
cd "$RF_WT"

OUT=/iris/u/mikulrai/data/RoboFactory/subtask_holdprobe_lb
rm -rf "$OUT" 2>/dev/null || true
mkdir -p "$OUT"

echo "=== hold-probe datagen on $(hostname) at $(date) ==="
echo "LB_HOLD_FRAMES_K=$LB_HOLD_FRAMES_K  settle=340  max-steps=600  mix=recovery+clean  save-video"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" -u scripts/dart/run_subtask_rollouts.py \
  --task LiftBarrier \
  --num 2 \
  --mix recovery=0.5,clean=0.5 \
  --dart-sigma 0.1 --inject-floor 0.05 \
  --settle-steps 340 \
  --max-steps 600 \
  --per-attempt-timeout 900 \
  --config-suffix _aug \
  --save-video \
  --record-dir "$OUT"

echo "=== done; outputs ==="
ls -la "$OUT" || true
echo "--- mp4s ---"; find "$OUT" -name "*.mp4" 2>/dev/null
echo "--- h5s ---"; find "$OUT" -name "*.h5" 2>/dev/null
