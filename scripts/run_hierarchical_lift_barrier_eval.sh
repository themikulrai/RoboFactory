#!/usr/bin/env bash
# =============================================================================
# Closed-loop HIERARCHICAL Lift-Barrier eval launcher.
#
# THREE-ENV DESIGN (each component runs in its OWN python venv; they talk over
# network sockets so their incompatible numpy versions never share a process):
#
#   1. HL Qwen subtask server  (MEMER env, numpy 2.x)
#         memer/scripts/hl_server.py  -> stdlib HTTP on HL_PORT
#   2. TWO LL pi0.5 servers     (OPENPI env, numpy<2)
#         openpi/scripts/serve_policy.py -> openpi websocket on LL_PORT0 / LL_PORT1
#   3. Hierarchical eval        (ROBOFACTORY env, numpy<2, SAPIEN)
#         RoboFactory .../eval_hierarchical_lift_barrier.py
#         (HTTP client to HL + websocket client to the 2 LL servers)
#
# The launcher boots 1+2, health-checks all three, runs 3, and tears everything
# down on exit via a trap (covers normal exit, Ctrl-C, and errors).
#
# Usage:
#   run_hierarchical_lift_barrier_eval.sh \
#       --hl-model <qwen_ckpt> \
#       --ll-config-arm0 pi05_robofactory_lb_ws_decent_arm0 --ll-ckpt-arm0 <dir> \
#       --ll-config-arm1 pi05_robofactory_lb_ws_decent_arm1 --ll-ckpt-arm1 <dir> \
#       [--n-episodes 20] [--hl-query-interval 25] [--flat-baseline] [--mock]
#
#   --mock           : run HL in --mock mode AND eval in --mock-ll (no checkpoints needed;
#                      pure plumbing smoke). Skips booting the openpi LL servers.
#   --flat-baseline  : skip HL; fixed prompt for both arms (LL-only sanity baseline).
# =============================================================================
set -euo pipefail

# ---- env pythons ----
MEMER_PY=/iris/u/mikulrai/envs/memer/bin/python
OPENPI_PY=/iris/u/mikulrai/projects/openpi/.venv/bin/python
RF_PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

# ---- repos ----
MEMER_DIR=/iris/u/mikulrai/projects/memer
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
RF_DIR=/iris/u/mikulrai/projects/RoboFactory

HL_SERVER="$MEMER_DIR/scripts/hl_server.py"
EVAL="$RF_DIR/robofactory/policy/openpi_pi05/eval_hierarchical_lift_barrier.py"

# ---- defaults (override via flags) ----
HL_MODEL=""
HL_PORT=8200
HL_QUERY_INTERVAL=25
LL_CONFIG_ARM0="pi05_robofactory_lb_ws_decent_arm0"
LL_CONFIG_ARM1="pi05_robofactory_lb_ws_decent_arm1"
LL_CKPT_ARM0=""
LL_CKPT_ARM1=""
LL_PORT0=8000
LL_PORT1=8001
N_EPISODES=20
BASE_SEED=1000
RESULTS_DIR="$RF_DIR/eval_results"
VIDEO_DIR=""
FLAT_BASELINE=0
MOCK=0
LOG_DIR="${TMPDIR:-/tmp}/hl_lb_eval_$$"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hl-model) HL_MODEL="$2"; shift 2;;
    --hl-port) HL_PORT="$2"; shift 2;;
    --hl-query-interval) HL_QUERY_INTERVAL="$2"; shift 2;;
    --ll-config-arm0) LL_CONFIG_ARM0="$2"; shift 2;;
    --ll-config-arm1) LL_CONFIG_ARM1="$2"; shift 2;;
    --ll-ckpt-arm0) LL_CKPT_ARM0="$2"; shift 2;;
    --ll-ckpt-arm1) LL_CKPT_ARM1="$2"; shift 2;;
    --ll-port0) LL_PORT0="$2"; shift 2;;
    --ll-port1) LL_PORT1="$2"; shift 2;;
    --n-episodes) N_EPISODES="$2"; shift 2;;
    --base-seed) BASE_SEED="$2"; shift 2;;
    --results-dir) RESULTS_DIR="$2"; shift 2;;
    --video-dir) VIDEO_DIR="$2"; shift 2;;
    --flat-baseline) FLAT_BASELINE=1; shift;;
    --mock) MOCK=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
PIDS=()

cleanup() {
  echo "[launcher] tearing down servers: ${PIDS[*]:-none}"
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  # give them a moment, then hard-kill stragglers
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_http_health() {  # host port retries
  local host="$1" port="$2" retries="${3:-120}"
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
  echo "[launcher] FATAL: $host:$port never became healthy" >&2
  return 1
}

wait_ws_health() {  # host port retries -- openpi /healthz over plain HTTP upgrade port
  local host="$1" port="$2" retries="${3:-180}"
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

# ---------------------------------------------------------------- 1) HL server
if [[ "$FLAT_BASELINE" -eq 0 ]]; then
  if [[ "$MOCK" -eq 1 ]]; then
    echo "[launcher] starting HL server (MOCK) on :$HL_PORT"
    "$MEMER_PY" "$HL_SERVER" --mock --port "$HL_PORT" >"$LOG_DIR/hl.log" 2>&1 &
  else
    [[ -z "$HL_MODEL" ]] && { echo "--hl-model required (or use --mock / --flat-baseline)" >&2; exit 2; }
    echo "[launcher] starting HL server on :$HL_PORT (model=$HL_MODEL)"
    "$MEMER_PY" "$HL_SERVER" --model-path "$HL_MODEL" --port "$HL_PORT" \
        >"$LOG_DIR/hl.log" 2>&1 &
  fi
  PIDS+=("$!")
  wait_http_health 127.0.0.1 "$HL_PORT" || { cat "$LOG_DIR/hl.log"; exit 1; }
  echo "[launcher] HL healthy on :$HL_PORT"
fi

# ---------------------------------------------------------------- 2) LL servers
if [[ "$MOCK" -eq 0 ]]; then
  [[ -z "$LL_CKPT_ARM0" || -z "$LL_CKPT_ARM1" ]] && { echo "--ll-ckpt-arm0/1 required (or --mock)" >&2; exit 2; }
  echo "[launcher] starting LL arm0 ($LL_CONFIG_ARM0) on :$LL_PORT0"
  "$OPENPI_PY" "$OPENPI_DIR/scripts/serve_policy.py" \
      --port "$LL_PORT0" policy:checkpoint \
      --policy.config="$LL_CONFIG_ARM0" --policy.dir="$LL_CKPT_ARM0" \
      >"$LOG_DIR/ll_arm0.log" 2>&1 &
  PIDS+=("$!")
  echo "[launcher] starting LL arm1 ($LL_CONFIG_ARM1) on :$LL_PORT1"
  "$OPENPI_PY" "$OPENPI_DIR/scripts/serve_policy.py" \
      --port "$LL_PORT1" policy:checkpoint \
      --policy.config="$LL_CONFIG_ARM1" --policy.dir="$LL_CKPT_ARM1" \
      >"$LOG_DIR/ll_arm1.log" 2>&1 &
  PIDS+=("$!")
  wait_ws_health 127.0.0.1 "$LL_PORT0" || { cat "$LOG_DIR/ll_arm0.log"; exit 1; }
  wait_ws_health 127.0.0.1 "$LL_PORT1" || { cat "$LOG_DIR/ll_arm1.log"; exit 1; }
  echo "[launcher] both LL servers healthy on :$LL_PORT0 :$LL_PORT1"
fi

# ---------------------------------------------------------------- 3) eval
EVAL_ARGS=(
  --ports "$LL_PORT0,$LL_PORT1"
  --hl-host 127.0.0.1 --hl-port "$HL_PORT"
  --hl-query-interval "$HL_QUERY_INTERVAL"
  --max-episodes "$N_EPISODES"
  --seed "$BASE_SEED"
  --results-dir "$RESULTS_DIR"
)
[[ -n "$VIDEO_DIR" ]] && EVAL_ARGS+=(--video-dir "$VIDEO_DIR")
[[ "$FLAT_BASELINE" -eq 1 ]] && EVAL_ARGS+=(--flat-baseline)
[[ "$MOCK" -eq 1 ]] && EVAL_ARGS+=(--mock-ll)

echo "[launcher] running eval: ${EVAL_ARGS[*]}"
"$RF_PY" "$EVAL" "${EVAL_ARGS[@]}"
echo "[launcher] eval done; results in $RESULTS_DIR (server logs in $LOG_DIR)"
