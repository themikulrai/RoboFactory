#!/bin/bash
# 60-seed TABLE rollout eval of Pi0.5 2SC (TwoRobotsStackCube) decentralized
# wristcam, with proprioceptive modality dropout. Two per-arm policy servers
# (openpi venv, one a40 each) + eval_decent_pi05.py client (RoboFactory env).
#
# CKPT_STEP env var selects the checkpoint step (default 19999; only 10000 and
# 19999 exist on disk for this run). partition / gpu / account are set at
# submit time via the iris-mcp tool args.
#
#SBATCH --job-name=eval_pi05_2sc_wc_dec
#SBATCH --output=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:2

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export OPENPI_DATA_HOME=/iris/u/mikulrai/data/openpi
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot
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
CKPT_BASE=/iris/u/mikulrai/checkpoints/openpi
CKPT_STEP="${CKPT_STEP:-19999}"

# Job-unique free ports (one per arm) so co-scheduled evals on the same node
# never collide on a shared hardcoded port. See _lib/free_ports.sh.
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/_lib/free_ports.sh
read -r PORT0 PORT1 <<<"$(free_ports 2)"
echo "[server] 2sc decent will use ports ${PORT0} ${PORT1}"

ARM0_CFG=pi05_robofactory_2sc_wc_decent_propriodrop_arm0
ARM1_CFG=pi05_robofactory_2sc_wc_decent_propriodrop_arm1
ARM0_DIR=${CKPT_BASE}/${ARM0_CFG}/2sc_wc_decent_propriodrop_arm0_v3/${CKPT_STEP}
ARM1_DIR=${CKPT_BASE}/${ARM1_CFG}/2sc_wc_decent_propriodrop_arm1_v3/${CKPT_STEP}
for d in "$ARM0_DIR" "$ARM1_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done
echo "Evaluating step ${CKPT_STEP}: arm0=${ARM0_DIR} arm1=${ARM1_DIR}"

SERVER_LOG_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

start_server () {
    local gpu_idx=$1 port=$2 cfg=$3 dir=$4 logname=$5
    echo "[server] arm GPU=${gpu_idx} port=${port} cfg=${cfg}"
    (cd "$OPENPI_DIR" && \
     CUDA_VISIBLE_DEVICES=${gpu_idx} \
     XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
     HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot \
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
echo "[wait] probing ports ${PORT0}/${PORT1} (deadline 600s) ..."
for port in "$PORT0" "$PORT1"; do
    deadline=$((SECONDS + 600))
    while ! "$PREFLIGHT_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',${port}))==0 else 1)" 2>/dev/null; do
        if [ $SECONDS -ge $deadline ]; then
            echo "[wait] timeout port ${port}"; tail -n 80 "${SERVER_LOG_DIR}"/*.log; exit 1
        fi
        sleep 5
    done
    echo "[wait] port ${port} up"
done

VIDEO_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent
mkdir -p "$VIDEO_DIR" "$OUT_DIR"
SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_60.txt)

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task TwoRobotsStackCube-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/two_robots_stack_cube.yaml \
    --host 127.0.0.1 \
    --ports "${PORT0},${PORT1}" \
    --num-arms 2 \
    --num-episodes 1 \
    --seeds "$SEEDS" \
    --max-env-steps 400 \
    --replan-after 8 \
    --prompt "stack the two cubes in collaboration with other robot arm" \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/two_robots_stack_cube_wristcam.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-all \
    --trajectory-log-path "${OUT_DIR}/trajectory_${SLURM_JOB_ID}.jsonl" \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project 2SC-Proprio-Dropout \
    --wandb-tags "eval,pi05,decent,2sc,wristcam,table-scene,60seeds,step-${CKPT_STEP},propriodrop"

echo "Done at $(date)"
echo "Results JSON: $(ls -t $OUT_DIR/eval_decent_TwoRobotsStackCube-rf_*.json 2>/dev/null | head -1)"
