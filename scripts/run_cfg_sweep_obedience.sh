#!/usr/bin/env bash
# =============================================================================
# CFG SWEEP over the VALID subtask-obedience probe (LB ws decent arm0).
#
# For each guidance scale w in W_LIST, boot the 2 per-arm pi0.5 LL servers with
# OPENPI_CFG_W=w (via run_subtask_obedience_eval.sh --cfg-w) and run the VALID
# verdict (--mode valid: waitact + grasp-vs-lift, NO out-of-distribution
# left-vs-right). Each w writes its own result JSON + video subdir. A final
# combined summary JSON aggregates the per-w numbers.
#
# Runs all w SERIALLY on ONE L40S (each launcher boots+tears-down its own
# servers, so only one pair is resident at a time).
# =============================================================================
set -euo pipefail

RF_WT=/iris/u/mikulrai/RoboFactory-subtask-wt
LAUNCH="$RF_WT/scripts/run_subtask_obedience_eval.sh"
RF_PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

CK=/iris/u/mikulrai/checkpoints/openpi
# Checkpoints derive from $CAM (ws|wc) so this script serves either camera family.
CAM="${CAM:-ws}"
STEP="${STEP:-14999}"
CKPT_ARM0="${CKPT_ARM0:-$CK/pi05_robofactory_lb_${CAM}_subtaskdart_decent_arm0/lb_${CAM}_subtaskdart_decent_arm0_15k_v1/$STEP}"
CKPT_ARM1="${CKPT_ARM1:-$CK/pi05_robofactory_lb_${CAM}_subtaskdart_decent_arm1/lb_${CAM}_subtaskdart_decent_arm1_15k_v1/$STEP}"

W_LIST=(${W_LIST:-1.0 1.5 2.0 3.0 5.0})
N_SEEDS="${N_SEEDS:-12}"
ARM="${ARM:-0}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${OUT_ROOT:-$RF_WT/eval_results/cfg_sweep_${STAMP}}"
mkdir -p "$OUT_ROOT"
echo "[sweep] out root: $OUT_ROOT"
echo "[sweep] W_LIST=${W_LIST[*]} n_seeds=$N_SEEDS cam=$CAM arm=$ARM"

# Shared OPENPI data home (avoid 11.6GB pi05_base re-download per server).
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/iris/u/mikulrai/data/openpi}"

SUMMARY="$OUT_ROOT/summary.jsonl"
: > "$SUMMARY"

for w in "${W_LIST[@]}"; do
  echo "============================================================"
  echo "[sweep] === w=$w ==="
  echo "============================================================"
  OUT_JSON="$OUT_ROOT/valid_w${w}.json"
  VIDEO_DIR="$OUT_ROOT/videos_w${w}"
  bash "$LAUNCH" \
    --cam "$CAM" --arm "$ARM" --n-seeds "$N_SEEDS" --mode valid \
    --cfg-w "$w" \
    --ll-xla-mem-fraction 0.40 \
    --ckpt-arm0 "$CKPT_ARM0" --ckpt-arm1 "$CKPT_ARM1" \
    --out "$OUT_JSON" --video-dir "$VIDEO_DIR" \
    || { echo "[sweep] w=$w FAILED" >&2; continue; }

  # Extract the headline numbers into the summary jsonl.
  "$RF_PY" - "$w" "$OUT_JSON" "$SUMMARY" <<'PY'
import json, sys
w, path, summ = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
wa = d.get("waitact", {}); gl = d.get("grasplift", {}); v = d.get("verdict", {})
row = {
    "w": float(w),
    "waitact_ratio": v.get("motion_ratio"),
    "mean_wait_motion": wa.get("mean_wait_motion"),
    "mean_act_motion": wa.get("mean_act_motion"),
    "waitact_pass_rate": wa.get("pass_rate"),
    "grasplift_gap": gl.get("mean_grasplift_gap"),
    "grasplift_pass_rate": gl.get("pass_rate"),
    "mean_dz_grasp": gl.get("mean_dz_grasp"),
    "mean_dz_lift": gl.get("mean_dz_lift"),
    "verdict": v.get("verdict"),
}
with open(summ, "a") as f:
    f.write(json.dumps(row) + "\n")
print("[sweep] w=%s -> ratio=%.2f wait=%.4f gap=%.4f verdict=%s" % (
    w, row["waitact_ratio"] or -1, row["mean_wait_motion"] or -1,
    row["grasplift_gap"] or -1, row["verdict"]))
PY
done

echo "============================================================"
echo "[sweep] DONE. Per-w JSON + videos in $OUT_ROOT"
echo "[sweep] summary table:"
column -t -s$'\t' < <(
  echo -e "w\tratio\twait_mot\tgap\tgl_rate\tverdict"
  "$RF_PY" - "$SUMMARY" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print("%s\t%.2f\t%.4f\t%.4f\t%.2f\t%s" % (
        r["w"], r["waitact_ratio"] or -1, r["mean_wait_motion"] or -1,
        r["grasplift_gap"] or -1, r["grasplift_pass_rate"] or -1, r["verdict"]))
PY
)
