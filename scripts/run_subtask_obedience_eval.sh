#!/usr/bin/env bash
# =============================================================================
# SUBTASK-OBEDIENCE eval launcher (the downstream-VLM GO/NO-GO GATE).
#
# Boots the 2 per-arm pi0.5 LL servers (openpi env) and runs the obedience eval
# (RoboFactory subtask worktree env, SAPIEN) as a websocket client. The eval
# measures whether the trained LL actually FOLLOWS its per-arm subtask language
# channel (vs ignoring it and running on proprioception):
#
#   probe   left-vs-right: arm gets "grasp the left end" vs "grasp the right end";
#           obedience = fraction of seeds the TCP approaches the COMMANDED end.
#   waitact wait-vs-act:   arm gets "wait" vs "grasp the left end"; obedience =>
#           wait-motion ~ 0, act-motion clearly larger (per-seed ratio).
#   gate    probe + waitact + a single PASS/FAIL verdict (the headline). PASS iff
#           left-right obedience >= 0.7 AND mean act/wait motion ratio >= 3.0 AND
#           the 'wait' rollout barely moves.
#
# TWO-ENV DESIGN (incompatible numpy versions never share a process):
#   1. TWO LL pi0.5 servers (OPENPI env, numpy<2)  -> openpi websocket on LL_PORT0/1
#   2. Obedience eval (ROBOFACTORY SUBTASK worktree env, numpy<2, SAPIEN)
#
# Ports are job-unique free ports (no hardcoded 8000/8001 collision for
# co-scheduled pi0.5 evals; see memory/project_pi05_eval_port_collision).
#
# Usage:
#   run_subtask_obedience_eval.sh \
#       --cam ws --arm 0 --n-seeds 15 \
#       --ckpt-arm0 <dir/15000> --ckpt-arm1 <dir/15000> \
#       [--config-arm0 NAME --config-arm1 NAME] [--mode gate] \
#       [--baseline-ckpt-arm0 <dir> --baseline-ckpt-arm1 <dir>]
#
#   --mock   no LL servers; zero-action stand-in (plumbing smoke; probe ~chance).
# =============================================================================
set -euo pipefail

# ---- env pythons ----
OPENPI_PY=/iris/u/mikulrai/projects/openpi/.venv/bin/python
RF_PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

# ---- repos ----
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
# The subtask code (subtask_* modules + this eval) lives in the worktree. PYTHONPATH
# below forces `robofactory` to resolve to it, NOT the plain projects/RoboFactory.
RF_WT=/iris/u/mikulrai/RoboFactory-subtask-wt
EVAL="$RF_WT/robofactory/policy/openpi_pi05/eval_subtask_obedience.py"

# ---- defaults (override via flags) ----
CAM="ws"                 # ws|wc -> camera-family workspace|wristcam
ARM=0                    # which arm is PROBED (0 or 1)
N_SEEDS=15
PROBE_BASE_SEED=1000
MODE="gate"              # schedule|probe|waitact|gate|both
SCHED_VARIANT=""         # simultaneous|arm0_leads|arm1_leads (staggered-entry test)
SCHED_SEEDS=""           # CSV env seeds for schedule mode
CKPT_ARM0=""
CKPT_ARM1=""
CONFIG_ARM0=""           # default derived from --cam below
CONFIG_ARM1=""
BASELINE_CKPT_ARM0=""
BASELINE_CKPT_ARM1=""
BASELINE_CONFIG_ARM0=""
BASELINE_CONFIG_ARM1=""
LL_PORT0=""
LL_PORT1=""
BL_PORT0=""
BL_PORT1=""
LL_GPU0=0
LL_GPU1=0
LL_XLA_MEM_FRACTION=0.30
CFG_W=""                 # classifier-free-guidance scale w (empty => unset => w=1 default)
PROBE_MAX_STEPS=80
REPLAN_AFTER=8
RESULTS_DIR="$RF_WT/eval_results"
VIDEO_DIR=""
OUT_JSON=""
MOCK=0
EXPECT_CONFIG=1          # PR2 handshake: hard-fail if served config_name differs

LOG_DIR="${TMPDIR:-/tmp}/subtask_obedience_$$"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cam) CAM="$2"; shift 2;;
    --arm) ARM="$2"; shift 2;;
    --n-seeds) N_SEEDS="$2"; shift 2;;
    --probe-base-seed) PROBE_BASE_SEED="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --schedule-variant) SCHED_VARIANT="$2"; shift 2;;
    --schedule-seeds) SCHED_SEEDS="$2"; shift 2;;
    --ckpt-arm0) CKPT_ARM0="$2"; shift 2;;
    --ckpt-arm1) CKPT_ARM1="$2"; shift 2;;
    --config-arm0) CONFIG_ARM0="$2"; shift 2;;
    --config-arm1) CONFIG_ARM1="$2"; shift 2;;
    --baseline-ckpt-arm0) BASELINE_CKPT_ARM0="$2"; shift 2;;
    --baseline-ckpt-arm1) BASELINE_CKPT_ARM1="$2"; shift 2;;
    --baseline-config-arm0) BASELINE_CONFIG_ARM0="$2"; shift 2;;
    --baseline-config-arm1) BASELINE_CONFIG_ARM1="$2"; shift 2;;
    --ll-port0) LL_PORT0="$2"; shift 2;;
    --ll-port1) LL_PORT1="$2"; shift 2;;
    --ll-gpu0) LL_GPU0="$2"; shift 2;;
    --ll-gpu1) LL_GPU1="$2"; shift 2;;
    --ll-xla-mem-fraction) LL_XLA_MEM_FRACTION="$2"; shift 2;;
    --cfg-w) CFG_W="$2"; shift 2;;
    --probe-max-steps) PROBE_MAX_STEPS="$2"; shift 2;;
    --replan-after) REPLAN_AFTER="$2"; shift 2;;
    --results-dir) RESULTS_DIR="$2"; shift 2;;
    --video-dir) VIDEO_DIR="$2"; shift 2;;
    --out) OUT_JSON="$2"; shift 2;;
    --no-expect-config) EXPECT_CONFIG=0; shift;;
    --mock) MOCK=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# ---- derive camera-family + default config names from --cam ----
case "$CAM" in
  ws) CAMERA_FAMILY="workspace";;
  wc) CAMERA_FAMILY="wristcam";;
  *) echo "--cam must be ws|wc, got $CAM" >&2; exit 2;;
esac
CONFIG_ARM0="${CONFIG_ARM0:-pi05_robofactory_lb_${CAM}_subtaskdart_decent_arm0}"
CONFIG_ARM1="${CONFIG_ARM1:-pi05_robofactory_lb_${CAM}_subtaskdart_decent_arm1}"

# ---- job-unique free ports (avoid the 8000/8001 collision) ----
source "$RF_WT/robofactory/scripts/_lib/free_ports.sh"
read -r _FP0 _FP1 _FP2 _FP3 <<<"$(free_ports 4)"
LL_PORT0="${LL_PORT0:-$_FP0}"
LL_PORT1="${LL_PORT1:-$_FP1}"
BL_PORT0="${BL_PORT0:-$_FP2}"
BL_PORT1="${BL_PORT1:-$_FP3}"
echo "[launcher] cam=$CAM ($CAMERA_FAMILY) arm=$ARM mode=$MODE ports: LL0=$LL_PORT0 LL1=$LL_PORT1"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
PIDS=()

cleanup() {
  echo "[launcher] tearing down servers: ${PIDS[*]:-none}"
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_ws_health() {  # host port retries -- openpi /healthz
  local host="$1" port="$2" retries="${3:-300}"
  for ((i=0; i<retries; i++)); do
    if "$RF_PY" - "$host" "$port" <<'PY' 2>/dev/null
import sys, http.client
h, p = sys.argv[1], int(sys.argv[2])
try:
    c = http.client.HTTPConnection(h, p, timeout=3); c.request("GET", "/healthz")
    r = c.getresponse(); r.read(); c.close()
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    then return 0; fi
    sleep 2
  done
  echo "[launcher] FATAL: LL ws server $host:$port never became healthy" >&2
  return 1
}

serve_ll() {  # config dir port gpu logname [cfg_w]
  # cfg_w (6th arg, optional): classifier-free-guidance scale exported as OPENPI_CFG_W to
  # the openpi server. Empty/unset => default w=1 (byte-identical to original sampler).
  local cfg="$1" dir="$2" port="$3" gpu="$4" logname="$5" cfg_w="${6:-}"
  echo "[launcher] serving $cfg @ $dir on :$port (GPU=$gpu) cfg_w=${cfg_w:-1.0(default)}"
  (
    cd "$OPENPI_DIR" || exit 1
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$LL_XLA_MEM_FRACTION"
    [[ -n "$cfg_w" ]] && export OPENPI_CFG_W="$cfg_w"
    exec "$OPENPI_PY" "$OPENPI_DIR/scripts/serve_policy.py" \
        --port "$port" policy:checkpoint \
        --policy.config="$cfg" --policy.dir="$dir"
  ) >"$LOG_DIR/$logname.log" 2>&1 &
  PIDS+=("$!")
}

# ---------------------------------------------------------------- LL servers
if [[ "$MOCK" -eq 0 ]]; then
  [[ -z "$CKPT_ARM0" || -z "$CKPT_ARM1" ]] && { echo "--ckpt-arm0/1 required (or --mock)" >&2; exit 2; }
  serve_ll "$CONFIG_ARM0" "$CKPT_ARM0" "$LL_PORT0" "$LL_GPU0" "ll_arm0" "$CFG_W"
  serve_ll "$CONFIG_ARM1" "$CKPT_ARM1" "$LL_PORT1" "$LL_GPU1" "ll_arm1" "$CFG_W"
  if [[ -n "$BASELINE_CKPT_ARM0" ]]; then
    BASELINE_CONFIG_ARM0="${BASELINE_CONFIG_ARM0:-pi05_robofactory_lb_${CAM}_flatbaseline_decent_arm0}"
    BASELINE_CONFIG_ARM1="${BASELINE_CONFIG_ARM1:-pi05_robofactory_lb_${CAM}_flatbaseline_decent_arm1}"
    serve_ll "$BASELINE_CONFIG_ARM0" "$BASELINE_CKPT_ARM0" "$BL_PORT0" "$LL_GPU0" "bl_arm0"
    serve_ll "$BASELINE_CONFIG_ARM1" "$BASELINE_CKPT_ARM1" "$BL_PORT1" "$LL_GPU1" "bl_arm1"
  fi
  wait_ws_health 127.0.0.1 "$LL_PORT0" || { cat "$LOG_DIR/ll_arm0.log"; exit 1; }
  wait_ws_health 127.0.0.1 "$LL_PORT1" || { cat "$LOG_DIR/ll_arm1.log"; exit 1; }
  if [[ -n "$BASELINE_CKPT_ARM0" ]]; then
    wait_ws_health 127.0.0.1 "$BL_PORT0" || { cat "$LOG_DIR/bl_arm0.log"; exit 1; }
    wait_ws_health 127.0.0.1 "$BL_PORT1" || { cat "$LOG_DIR/bl_arm1.log"; exit 1; }
  fi
  echo "[launcher] LL servers healthy"
fi

# ---------------------------------------------------------------- eval
EVAL_ARGS=(
  --mode "$MODE"
  --camera-family "$CAMERA_FAMILY"
  --probe-arm "$ARM"
  --num-seeds "$N_SEEDS"
  --probe-base-seed "$PROBE_BASE_SEED"
  --probe-max-steps "$PROBE_MAX_STEPS"
  --replan-after "$REPLAN_AFTER"
  --ports "$LL_PORT0,$LL_PORT1"
  --results-dir "$RESULTS_DIR"
  --checkpoint "$CKPT_ARM0"
  --config-name "$CONFIG_ARM0"
)
[[ "$EXPECT_CONFIG" -eq 1 && "$MOCK" -eq 0 ]] && EVAL_ARGS+=(--expect-config "$CONFIG_ARM0,$CONFIG_ARM1")
[[ -n "$SCHED_VARIANT" ]] && EVAL_ARGS+=(--schedule-variant "$SCHED_VARIANT")
[[ -n "$SCHED_SEEDS" ]] && EVAL_ARGS+=(--schedule-seeds "$SCHED_SEEDS")
[[ -n "${CONTACT_DIR:-}" ]] && EVAL_ARGS+=(--schedule-log-contact --schedule-no-terminate --contact-dir "$CONTACT_DIR")
[[ -n "$VIDEO_DIR" ]] && EVAL_ARGS+=(--video-dir "$VIDEO_DIR")
[[ -n "$OUT_JSON" ]] && EVAL_ARGS+=(--out "$OUT_JSON")
[[ "$MOCK" -eq 1 ]] && EVAL_ARGS+=(--mock-ll)
if [[ -n "$BASELINE_CKPT_ARM0" && "$MOCK" -eq 0 ]]; then
  EVAL_ARGS+=(--baseline-ports "$BL_PORT0,$BL_PORT1" --baseline-checkpoint "$BASELINE_CKPT_ARM0")
  [[ "$EXPECT_CONFIG" -eq 1 ]] && EVAL_ARGS+=(--baseline-expect-config "$BASELINE_CONFIG_ARM0,$BASELINE_CONFIG_ARM1")
fi

echo "[launcher] running eval: ${EVAL_ARGS[*]}"
# PYTHONPATH forces `robofactory` -> the subtask worktree (subtask_* modules).
cd "$RF_WT"
PYTHONPATH="$RF_WT" "$RF_PY" "$EVAL" "${EVAL_ARGS[@]}"
echo "[launcher] eval done; results in $RESULTS_DIR (server logs in $LOG_DIR)"
