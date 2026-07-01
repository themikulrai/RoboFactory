#!/bin/bash
# 100-seed TABLE rollout eval of Pi0.5 2SC (TwoRobotsStackCube) decentralized,
# WORKSPACE cameras, FLAT baseline (no subtask channel). Two per-arm policy
# servers (openpi venv, one a40 each) + eval_decent_pi05.py client.
#
# Cube spawn = 15x15cm box (eval15 yaml, centered, in-distribution vs 25cm datagen).
# CKPT_STEP selects step (default 14999). partition/gpu/mem/time set via iris-mcp args.
set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_pi05_2sc_ws_flat_decent

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${RF_LEROBOT_HOME:?RF_LEROBOT_HOME unset — source ~/.config/dataroots.sh}}"
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache
export JAX_COMPILATION_CACHE_DIR=/iris/u/mikulrai/.cache/jax
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_STEP="${CKPT_STEP:-14999}"

source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/_lib/free_ports.sh
read -r PORT0 PORT1 <<<"$(free_ports 2)"
echo "[server] 2sc ws flat decent ports ${PORT0} ${PORT1}"

ARM0_CFG=pi05_robofactory_2sc_ws_flatbaseline_decent_arm0
ARM1_CFG=pi05_robofactory_2sc_ws_flatbaseline_decent_arm1
CKPT_WS=/iris/u/mikulrai/checkpoints/openpi/2SC/Non-Subtask/Workspace
ARM0_DIR=${CKPT_WS}/2sc_ws_flatbaseline_decent_arm0_anvil/${CKPT_STEP}
ARM1_DIR=${CKPT_WS}/2sc_ws_flatbaseline_decent_arm1_anvil/${CKPT_STEP}
for d in "$ARM0_DIR" "$ARM1_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done
echo "Evaluating step ${CKPT_STEP}: arm0=${ARM0_DIR} arm1=${ARM1_DIR}"

SERVER_LOG_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_ws_flat_decent/${SLURM_JOB_ID:-local}_servers
mkdir -p "$SERVER_LOG_DIR"

start_server () {
    local gpu_idx=$1 port=$2 cfg=$3 dir=$4 logname=$5
    echo "[server] arm GPU=${gpu_idx} port=${port} cfg=${cfg}"
    (cd "$OPENPI_DIR" && \
     CUDA_VISIBLE_DEVICES=${gpu_idx} \
     XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
     HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
     "$OPENPI_PY" scripts/serve_policy.py \
         --port "${port}" \
         policy:checkpoint \
         --policy.config="${cfg}" \
         --policy.dir="${dir}") \
        > "${SERVER_LOG_DIR}/${logname}.log" 2>&1 &
    echo $! > "${SERVER_LOG_DIR}/${logname}.pid"
}

start_server 0 "$PORT0" "$ARM0_CFG" "$ARM0_DIR" arm0
start_server 1 "$PORT1" "$ARM1_CFG" "$ARM1_DIR" arm1

cleanup () {
    echo "[cleanup] stopping policy servers"
    for f in "${SERVER_LOG_DIR}"/*.pid; do
        [ -f "$f" ] || continue
        kill -TERM "$(cat "$f")" 2>/dev/null || true
    done
    sleep 2
    for f in "${SERVER_LOG_DIR}"/*.pid; do
        [ -f "$f" ] || continue
        kill -KILL "$(cat "$f")" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
echo "[wait] probing ports ${PORT0}/${PORT1} (deadline 1800s; lora_only restores 11.6GB pi05_base) ..."
for port in "$PORT0" "$PORT1"; do
    deadline=$((SECONDS + 1800))
    while ! "$PREFLIGHT_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',${port}))==0 else 1)" 2>/dev/null; do
        if [ $SECONDS -ge $deadline ]; then
            echo "[wait] timeout port ${port}"; tail -n 80 "${SERVER_LOG_DIR}"/*.log; exit 1
        fi
        sleep 5
    done
    echo "[wait] port ${port} up"
done

VIDEO_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_ws_flat_decent/videos_${SLURM_JOB_ID:-local}
OUT_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_ws_flat_decent
mkdir -p "$VIDEO_DIR" "$OUT_DIR"
# Match the wc flat-baseline eval protocol exactly for a clean ws-vs-wc A/B:
# same held-out seed pool (paired cube spawns) AND same step budget.
SEEDS=$(seq -s, 20000 20099)   # 100 HELD-OUT seeds, identical to the wc flat eval (paired spawns)
# MAX_ENV_STEPS: FLAT-baseline evals use 600 (NOT the 400 from the wc *decent/propriodrop*
# template — that mismatch undercounts SR since every fail hits the cap). Keep 600 here.
MAX_ENV_STEPS=600

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task TwoRobotsStackCube-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/two_robots_stack_cube_eval15.yaml \
    --host 127.0.0.1 \
    --ports "${PORT0},${PORT1}" \
    --expect-config "${ARM0_CFG},${ARM1_CFG}" \
    --num-arms 2 \
    --num-episodes 1 \
    --seeds "$SEEDS" \
    --max-env-steps "$MAX_ENV_STEPS" \
    --replan-after 8 \
    --prompt "stack the two cubes with the two robot arms" \
    --allow-oov-prompt \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/two_robots_stack_cube.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-frame-stride 2 \
    --run-id "2sc_ws_flat_${SLURM_JOB_ID:-local}"

echo "Done at $(date)"
RESULT_FILE="$(ls -t $OUT_DIR/eval_decent_TwoRobotsStackCube-rf_*.json 2>/dev/null | head -1)"
echo "Results JSON: $RESULT_FILE"
