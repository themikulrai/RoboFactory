#!/bin/bash
# 60-seed TABLE rollout eval of Pi0.5 LiftBarrier (LB) DECENTRALISED wristcam.
# Two per-arm policy servers (openpi venv, one a40 each) + eval_decent_pi05.py
# client (RoboFactory env). Reconstructed from ad-hoc slurm job 15676266
# (sky-fixed SR 30/60). Re-run uses shader_pack=default (sky-fix, hardcoded in
# eval_decent_pi05.py) AND the new always-on tiled multi-view video.
#
# partition / gpu / account are set at submit time via the iris-mcp tool args.
#
#SBATCH --job-name=eval_pi05_lb_wc_dec
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --error=/iris/u/mikulrai/slurm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:2

set -euo pipefail

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

source /iris/u/mikulrai/bin/log-run-paths.sh
logrun_init --task LiftBarrier-rf --cam wc --method pi05 --category eval --variant decent

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

# Eval always runs against the canonical main checkout (committed multiview code).
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
CKPT_BASE=/iris/u/mikulrai/checkpoints/openpi

# Pick two job-unique free ports so co-scheduled eval jobs on the same node
# never collide on the hardcoded 8000/8001 (a same-node port clash silently
# routes one client to the wrong policy server).
read -r PORT0 PORT1 <<<"$("$PREFLIGHT_PYTHON" -c "
import socket
ss=[socket.socket() for _ in range(2)]
for s in ss: s.bind(('127.0.0.1',0))
print(*[s.getsockname()[1] for s in ss])
for s in ss: s.close()
")"
echo "[server] decent will use ports ${PORT0} ${PORT1}"

ARM0_CFG=pi05_robofactory_lb_wc_decent_arm0
ARM1_CFG=pi05_robofactory_lb_wc_decent_arm1
ARM0_DIR=${CKPT_BASE}/${ARM0_CFG}/lb-wc-dec-a0-bs16-20k-2026-05-16-fp/19999
ARM1_DIR=${CKPT_BASE}/${ARM1_CFG}/lb-wc-dec-a1-bs16-20k-2026-05-16-fp/19999
for d in "$ARM0_DIR" "$ARM1_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done
echo "Evaluating decent: arm0=${ARM0_DIR} arm1=${ARM1_DIR}"

SERVER_LOG_DIR="$RUN_LOG_DIR/servers"
mkdir -p "$SERVER_LOG_DIR"

start_server () {
    local gpu_idx=$1 port=$2 cfg=$3 dir=$4 logname=$5
    echo "[server] starting arm GPU=${gpu_idx} port=${port} cfg=${cfg} dir=${dir}"
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

VIDEO_DIR="$RUN_VIDEO_DIR"
OUT_DIR="$RUN_LOG_DIR"
mkdir -p "$VIDEO_DIR" "$OUT_DIR"
SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_60.txt)

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task LiftBarrier-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml \
    --host 127.0.0.1 \
    --ports "${PORT0},${PORT1}" \
    --expect-config "${ARM0_CFG},${ARM1_CFG}" \
    --num-arms 2 \
    --num-episodes 1 \
    --seeds "$SEEDS" \
    --max-env-steps 400 \
    --replan-after 8 \
    --prompt "lift the steel barrier using two robot arms" \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/lift_barrier_wristcam.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project LB-Pi05 \
    --wandb-tags "eval,pi05,decent,lb,wristcam,table-scene,60seeds,step-19999,shaderfix,multiview"

echo "Done at $(date)"
RESULT_FILE="$(ls -t $OUT_DIR/eval_decent_LiftBarrier-rf_*.json 2>/dev/null | head -1)"
echo "Results JSON: $RESULT_FILE"
echo "Videos: $VIDEO_DIR"
# A8 loop-killer: auto-post a field-notes cell (job id + result jsonl + SR) so the
# new-cell-per-run convention is mechanical, not memory-dependent (WEEK1 §A8).
python scripts/log_eval.py \
    --jsonl "$RESULT_FILE" \
    --job "${SLURM_JOB_ID:-manual}" \
    --title "Eval LB wc Pi0.5 [Decent, 60seeds]" || true

logrun_finish --status done --config "" --cmd "$0"
