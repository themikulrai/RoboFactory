#!/bin/bash
# 60-seed TABLE rollout eval of Pi0.5 LiftBarrier (LB) CENTRALISED wristcam.
# One centralised policy server (openpi venv, one a40) + eval_pi05.py client
# (RoboFactory env). Reconstructed faithfully from ad-hoc slurm job 15475619
# (original SR 46/60, black-sky). Re-run now uses shader_pack=default (sky-fix,
# hardcoded in eval_pi05.py) AND the new always-on tiled multi-view video.
#
# partition / gpu / account are set at submit time via the iris-mcp tool args.
#
#SBATCH --job-name=eval_pi05_lb_wc_cent
#SBATCH --output=/iris/u/mikulrai/logs/eval_pi05_lb_wc_cent/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_pi05_lb_wc_cent/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_pi05_lb_wc_cent

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

# Eval always runs against the canonical main checkout (which carries the
# committed multiview-video eval code), NOT a transient worktree.
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
CENT_CFG=pi05_robofactory_lb_wc_cent
CENT_DIR=/iris/u/mikulrai/checkpoints/openpi/${CENT_CFG}/lb-wc-cent-bs16-20k-2026-05-16-fp/19999
[ -d "$CENT_DIR" ] || { echo "missing ckpt dir: $CENT_DIR"; exit 1; }
echo "Evaluating cent ckpt: $CENT_DIR"

# Pick a job-unique free port so co-scheduled eval jobs on the same node never
# collide (the 8000 hardcode let a decent job's arm0 server hijack this client).
PORT=$("$PREFLIGHT_PYTHON" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "[server] cent will use port ${PORT}"

SERVER_LOG_DIR=/iris/u/mikulrai/logs/eval_pi05_lb_wc_cent/${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

echo "[server] cent GPU=0 port=${PORT} cfg=${CENT_CFG}"
(cd "$OPENPI_DIR" && \
 CUDA_VISIBLE_DEVICES=0 \
 XLA_PYTHON_CLIENT_PREALLOCATE=false \
 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
 HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot \
 "$OPENPI_PY" scripts/serve_policy.py \
     --port "${PORT}" \
     policy:checkpoint \
     --policy.config="${CENT_CFG}" \
     --policy.dir="${CENT_DIR}") \
    > "${SERVER_LOG_DIR}/cent.log" 2>&1 &
echo $! > "${SERVER_LOG_DIR}/cent.pid"

cleanup () {
    echo "[cleanup] stopping policy server"
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

echo "[wait] probing port ${PORT} (deadline 600s) ..."
deadline=$((SECONDS + 600))
while ! "$PREFLIGHT_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',${PORT}))==0 else 1)" 2>/dev/null; do
    if [ $SECONDS -ge $deadline ]; then
        echo "[wait] timeout port ${PORT}"; tail -n 80 "${SERVER_LOG_DIR}"/*.log; exit 1
    fi
    sleep 5
done
echo "[wait] port ${PORT} up"

VIDEO_DIR=/iris/u/mikulrai/logs/eval_pi05_lb_wc_cent/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/eval_pi05_lb_wc_cent
mkdir -p "$VIDEO_DIR" "$OUT_DIR"
SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_60.txt)

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_pi05.py \
    --task LiftBarrier-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/lift_barrier_wristcam.json \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
    --num-arms 2 \
    --seeds "$SEEDS" \
    --num-episodes 1 \
    --max-env-steps 400 \
    --replan-after 8 \
    --prompt "lift the steel barrier using two robot arms" \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project LB-Pi05 \
    --wandb-tags "eval,pi05,cent,lb,wristcam,table-scene,60seeds,step-19999,shaderfix,multiview"

echo "Done at $(date)"
echo "Results JSON: $(ls -t $OUT_DIR/eval_LiftBarrier-rf_*.json 2>/dev/null | head -1)"
echo "Videos: $VIDEO_DIR"
