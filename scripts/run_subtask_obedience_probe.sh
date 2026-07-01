#!/usr/bin/env bash
# =============================================================================
# Per-arm SUBTASK-OBEDIENCE probe launcher (Lift-Barrier, decentralized pi0.5).
#
# Modeled on run_hierarchical_lift_barrier_eval.sh, MINUS the high-level Qwen
# server: the probe injects fixed per-arm subtask prompts itself. It serves the
# TWO LL pi0.5 arm policies (OPENPI env, numpy<2, openpi websocket), health-checks
# them, then runs the probe (ROBOFACTORY env, SAPIEN) as a websocket client.
#
# Per target arm, from the SAME reset, runs THREE short rollouts: A=grasp-left,
# B=grasp-right, W=wait. PRIMARY metric = in-distribution ENGAGEMENT contrast
# (own-end grasp vs wait); SECONDARY = OOD direction A/B obedience. See
#   robofactory/policy/openpi_pi05/eval_subtask_obedience.py  (full design + OOD caveat).
#
# MUST run on a GPU COMPUTE node (SAPIEN renders black on the iris login node).
# Submit via sbatch / the iris-mcp submit_job; do NOT run on the login node.
#
# Usage:
#   run_subtask_obedience_probe.sh \
#       --config-arm0 pi05_robofactory_lb_ws_subtaskdart_decent_arm0 --ckpt-arm0 <dir> \
#       --config-arm1 pi05_robofactory_lb_ws_subtaskdart_decent_arm1 --ckpt-arm1 <dir> \
#       --camera-family ws --target-arm both --probe-steps 24 \
#       [--seed-pool fresh_ood_60 | --env-seeds 20000,20001,...] --n-seeds 24 \
#       [--min-approach-dist 0.15] [--min-net-displacement 0.05] \
#       [--out-json <path>] [--results-dir <dir>] [--mock]
#
#   --mock : skip serving; probe runs --mock-ll (zero-action, plumbing smoke only).
# =============================================================================
set -euo pipefail

# ---- mandatory env (OPENPI_DATA_HOME: else serve re-downloads 11.6GB pi05_base and
#      times out the health check -- memory project_hier_eval_openpi_data_home). ----
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/iris/u/mikulrai/.cache/openpi}"
export TMPDIR="${TMPDIR:-/iris/u/mikulrai/tmp}"; mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-/iris/u/mikulrai/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/iris/u/mikulrai/.cache}"

# ---- env pythons / repos ----
OPENPI_PY=/iris/u/mikulrai/projects/openpi/.venv/bin/python
RF_PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
RF_DIR=/iris/u/mikulrai/projects/RoboFactory
PROBE="$RF_DIR/robofactory/policy/openpi_pi05/eval_subtask_obedience.py"

# ---- defaults (override via flags) ----
LL_CONFIG_ARM0="pi05_robofactory_lb_ws_subtaskdart_decent_arm0"
LL_CONFIG_ARM1="pi05_robofactory_lb_ws_subtaskdart_decent_arm1"
LL_CKPT_ARM0=""
LL_CKPT_ARM1=""
LL_PORT0=""             # "" -> job-unique free port allocated after arg parsing
LL_PORT1=""
LL_GPU0=0
LL_GPU1=0
LL_XLA_MEM_FRACTION=0.30
CAMERA_FAMILY="ws"
TARGET_ARM="both"
PROBE_STEPS=24
NONTARGET_PROMPT="wait"
SCENARIO="obedience"     # {obedience, hold-after-grasp}
GRASP_STEPS=25           # hold-after-grasp Phase-1 grasp steps
CLOSE_STEPS=20           # hold-after-grasp Phase-1 "close the gripper" steps
LIFT_STEPS=50            # hold-after-grasp Phase-1 lift steps
HOLD_STEPS=50            # hold-after-grasp Phase-2 ("wait") steps (M)
HOLD_LIFT_DZ=""          # "" -> probe default (0.08 m)
HOLD_DROP_TOL=""         # "" -> probe default (0.05 m)
HOLD_GRASP_FRAC=""       # "" -> probe default (0.5)
VOCAB_JSON=""           # "" -> probe uses its default subtask_vocab.json
MIN_APPROACH_DIST=""    # "" -> probe default (0.15 m); <0 disables the near-check
MIN_NET_DISPLACEMENT="" # "" -> probe default (0.05 m); <0 disables the moved-check
SEED_POOL=""
ENV_SEEDS=""
N_SEEDS=24
MAX_ENV_STEPS=500
REPLAN_AFTER=8
RESULTS_DIR="$RF_DIR/eval_results/subtask_obedience_${SLURM_JOB_ID:-manual}"
OUT_JSON=""
VIDEO_DIR=""
MOCK=0
LOG_DIR="${TMPDIR}/subtask_obedience_$$"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-arm0) LL_CONFIG_ARM0="$2"; shift 2;;
    --config-arm1) LL_CONFIG_ARM1="$2"; shift 2;;
    --ckpt-arm0) LL_CKPT_ARM0="$2"; shift 2;;
    --ckpt-arm1) LL_CKPT_ARM1="$2"; shift 2;;
    --ll-port0) LL_PORT0="$2"; shift 2;;
    --ll-port1) LL_PORT1="$2"; shift 2;;
    --ll-gpu0) LL_GPU0="$2"; shift 2;;
    --ll-gpu1) LL_GPU1="$2"; shift 2;;
    --ll-xla-mem-fraction) LL_XLA_MEM_FRACTION="$2"; shift 2;;
    --camera-family) CAMERA_FAMILY="$2"; shift 2;;
    --target-arm) TARGET_ARM="$2"; shift 2;;
    --probe-steps) PROBE_STEPS="$2"; shift 2;;
    --nontarget-prompt) NONTARGET_PROMPT="$2"; shift 2;;
    --scenario) SCENARIO="$2"; shift 2;;
    --grasp-steps) GRASP_STEPS="$2"; shift 2;;
    --close-steps) CLOSE_STEPS="$2"; shift 2;;
    --lift-steps) LIFT_STEPS="$2"; shift 2;;
    --hold-steps) HOLD_STEPS="$2"; shift 2;;
    --hold-lift-dz) HOLD_LIFT_DZ="$2"; shift 2;;
    --hold-drop-tol) HOLD_DROP_TOL="$2"; shift 2;;
    --hold-grasp-frac) HOLD_GRASP_FRAC="$2"; shift 2;;
    --min-approach-dist) MIN_APPROACH_DIST="$2"; shift 2;;
    --min-net-displacement) MIN_NET_DISPLACEMENT="$2"; shift 2;;
    --vocab-json) VOCAB_JSON="$2"; shift 2;;
    --seed-pool) SEED_POOL="$2"; shift 2;;
    --env-seeds) ENV_SEEDS="$2"; shift 2;;
    --n-seeds) N_SEEDS="$2"; shift 2;;
    --max-env-steps) MAX_ENV_STEPS="$2"; shift 2;;
    --replan-after) REPLAN_AFTER="$2"; shift 2;;
    --results-dir) RESULTS_DIR="$2"; shift 2;;
    --out-json) OUT_JSON="$2"; shift 2;;
    --video-dir) VIDEO_DIR="$2"; shift 2;;
    --mock) MOCK=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

echo "[probe] host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
nvidia-smi -L || true

# Allocate job-unique free ports for any LL port not pinned (avoids the 8000/8001 collision
# that silently routes a client to the WRONG policy server -- memory pi05_eval_port_collision).
source "$RF_DIR/robofactory/scripts/_lib/free_ports.sh"
read -r _FP0 _FP1 <<<"$(free_ports 2)"
LL_PORT0="${LL_PORT0:-$_FP0}"
LL_PORT1="${LL_PORT1:-$_FP1}"
echo "[probe] ports: LL0=$LL_PORT0 LL1=$LL_PORT1"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
PIDS=()

cleanup() {
  echo "[probe] tearing down servers: ${PIDS[*]:-none}"
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_ws_health() {  # host port retries -- openpi /healthz over the plain HTTP upgrade port
  local host="$1" port="$2" retries="${3:-450}"   # 15min: 2 servers load 6.2GB pi05_base concurrently (I/O contention) -> slow startup; 6min was too tight
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
  echo "[probe] FATAL: LL ws server $host:$port never became healthy" >&2
  return 1
}

# ---------------------------------------------------------------- serve 2 LL arms
if [[ "$MOCK" -eq 0 ]]; then
  [[ -z "$LL_CKPT_ARM0" || -z "$LL_CKPT_ARM1" ]] && { echo "--ckpt-arm0/1 required (or --mock)" >&2; exit 2; }
  echo "[probe] serving LL arm0 ($LL_CONFIG_ARM0) on :$LL_PORT0 (GPU=$LL_GPU0)"
  ( cd "$OPENPI_DIR" && \
    CUDA_VISIBLE_DEVICES="$LL_GPU0" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$LL_XLA_MEM_FRACTION" \
    "$OPENPI_PY" "$OPENPI_DIR/scripts/serve_policy.py" \
        --port "$LL_PORT0" policy:checkpoint \
        --policy.config="$LL_CONFIG_ARM0" --policy.dir="$LL_CKPT_ARM0" ) \
      >"$LOG_DIR/ll_arm0.log" 2>&1 &
  PIDS+=("$!")
  echo "[probe] serving LL arm1 ($LL_CONFIG_ARM1) on :$LL_PORT1 (GPU=$LL_GPU1)"
  ( cd "$OPENPI_DIR" && \
    CUDA_VISIBLE_DEVICES="$LL_GPU1" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$LL_XLA_MEM_FRACTION" \
    "$OPENPI_PY" "$OPENPI_DIR/scripts/serve_policy.py" \
        --port "$LL_PORT1" policy:checkpoint \
        --policy.config="$LL_CONFIG_ARM1" --policy.dir="$LL_CKPT_ARM1" ) \
      >"$LOG_DIR/ll_arm1.log" 2>&1 &
  PIDS+=("$!")
  wait_ws_health 127.0.0.1 "$LL_PORT0" || { cat "$LOG_DIR/ll_arm0.log"; exit 1; }
  wait_ws_health 127.0.0.1 "$LL_PORT1" || { cat "$LOG_DIR/ll_arm1.log"; exit 1; }
  echo "[probe] both LL servers healthy on :$LL_PORT0 :$LL_PORT1"
fi

# ---------------------------------------------------------------- run the probe
PROBE_ARGS=(
  --host 127.0.0.1 --ports "$LL_PORT0,$LL_PORT1"
  --config-arm0 "$LL_CONFIG_ARM0" --config-arm1 "$LL_CONFIG_ARM1"
  --ckpt-arm0 "$LL_CKPT_ARM0" --ckpt-arm1 "$LL_CKPT_ARM1"
  --camera-family "$CAMERA_FAMILY"
  --target-arm "$TARGET_ARM"
  --probe-steps "$PROBE_STEPS"
  --nontarget-prompt "$NONTARGET_PROMPT"
  --scenario "$SCENARIO"
  --grasp-steps "$GRASP_STEPS"
  --close-steps "$CLOSE_STEPS"
  --lift-steps "$LIFT_STEPS"
  --hold-steps "$HOLD_STEPS"
  --max-env-steps "$MAX_ENV_STEPS"
  --replan-after "$REPLAN_AFTER"
  --n-seeds "$N_SEEDS"
  --results-dir "$RESULTS_DIR"
)
[[ -n "$HOLD_LIFT_DZ"    ]] && PROBE_ARGS+=(--hold-lift-dz "$HOLD_LIFT_DZ")
[[ -n "$HOLD_DROP_TOL"   ]] && PROBE_ARGS+=(--hold-drop-tol "$HOLD_DROP_TOL")
[[ -n "$HOLD_GRASP_FRAC" ]] && PROBE_ARGS+=(--hold-grasp-frac "$HOLD_GRASP_FRAC")
[[ -n "$VOCAB_JSON" ]] && PROBE_ARGS+=(--vocab-json "$VOCAB_JSON")
[[ -n "$MIN_APPROACH_DIST" ]] && PROBE_ARGS+=(--min-approach-dist "$MIN_APPROACH_DIST")
[[ -n "$MIN_NET_DISPLACEMENT" ]] && PROBE_ARGS+=(--min-net-displacement "$MIN_NET_DISPLACEMENT")
[[ -n "$SEED_POOL"  ]] && PROBE_ARGS+=(--seed-pool "$SEED_POOL")
[[ -n "$ENV_SEEDS"  ]] && PROBE_ARGS+=(--env-seeds "$ENV_SEEDS")
[[ -n "$OUT_JSON"   ]] && PROBE_ARGS+=(--out-json "$OUT_JSON")
[[ -n "$VIDEO_DIR"  ]] && PROBE_ARGS+=(--video-dir "$VIDEO_DIR")
[[ "$MOCK" -eq 1    ]] && PROBE_ARGS+=(--mock-ll)

echo "[probe] running: ${PROBE_ARGS[*]}"
"$RF_PY" "$PROBE" "${PROBE_ARGS[@]}"
echo "[probe] DONE. results in $RESULTS_DIR (server logs in $LOG_DIR)"
