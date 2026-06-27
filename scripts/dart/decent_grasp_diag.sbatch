#!/bin/bash
# DART DECENT GRASP DIAG — does arm1 (RIGHT) have a systematic grasp-phase
# prediction offset vs arm0 (LEFT) for the LB wc DECENT pi0.5 policy?
#
# Serves the TWO per-arm decent checkpoints (one server each, free runtime ports)
# and runs eval_decent_pi05_diag.py in OPEN-LOOP mode (--inject-h5-images) with
# --log-pred-vs-demo. Open-loop: images + demo target both come from H5 traj_0, so
# per-step pred-vs-demo error is NOT confounded by closed-loop visual drift. We run
# several seeds (known arm1-failure + success) for live-state variation; the core
# metric is per-arm error along the demo grasp window (steps ~25-50).
#
#SBATCH --job-name=dart_decent_grasp_diag
#SBATCH --output=/iris/u/mikulrai/logs/dart_swap_eval/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/dart_swap_eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a40:1

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/dart_swap_eval

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
export SAPIEN_HEADLESS=1

WT=/iris/u/mikulrai/RoboFactory-swap-wt
cd "$WT/robofactory"
export PYTHONPATH="$WT:${PYTHONPATH:-}"

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

A0_CFG=pi05_robofactory_lb_wc_decent_arm0
A1_CFG=pi05_robofactory_lb_wc_decent_arm1
A0_DIR=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm0/lb-wc-dec-a0-bs16-20k-2026-05-16-fp/19999
A1_DIR=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm1/lb-wc-dec-a1-bs16-20k-2026-05-16-fp/19999
[ -d "$A0_DIR" ] || { echo "missing arm0 ckpt: $A0_DIR"; exit 1; }
[ -d "$A1_DIR" ] || { echo "missing arm1 ckpt: $A1_DIR"; exit 1; }

pick_port () { "$PREFLIGHT_PYTHON" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"; }
P0=$(pick_port); P1=$(pick_port)
echo "[server] arm0 port=$P0 cfg=$A0_CFG"
echo "[server] arm1 port=$P1 cfg=$A1_CFG"

SERVER_LOG_DIR=/iris/u/mikulrai/logs/dart_swap_eval/${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

start_server () {
    local gpu="$1" port="$2" cfg="$3" dir="$4" tag="$5"
    (cd "$OPENPI_DIR" && \
     CUDA_VISIBLE_DEVICES="$gpu" \
     XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
     HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot \
     "$OPENPI_PY" scripts/serve_policy.py \
         --port "$port" \
         policy:checkpoint \
         --policy.config="$cfg" \
         --policy.dir="$dir") \
        > "${SERVER_LOG_DIR}/${tag}.log" 2>&1 &
    echo $! > "${SERVER_LOG_DIR}/${tag}.pid"
}

# Both servers share the single GPU (each ~6.3GB params, 0.30 mem frac).
start_server 0 "$P0" "$A0_CFG" "$A0_DIR" arm0
start_server 0 "$P1" "$A1_CFG" "$A1_DIR" arm1

cleanup () {
    echo "[cleanup] stopping policy servers"
    for f in "${SERVER_LOG_DIR}"/*.pid; do [ -f "$f" ] || continue; kill -TERM "$(cat "$f")" 2>/dev/null || true; done
    sleep 2
    for f in "${SERVER_LOG_DIR}"/*.pid; do [ -f "$f" ] || continue; kill -KILL "$(cat "$f")" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_port () {
    local port="$1"
    echo "[wait] probing port $port (deadline 700s) ..."
    local deadline=$((SECONDS + 700))
    while ! "$PREFLIGHT_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; do
        if [ $SECONDS -ge $deadline ]; then echo "[wait] TIMEOUT port $port"; tail -n 60 "${SERVER_LOG_DIR}"/*.log; exit 1; fi
        sleep 5
    done
    echo "[wait] port $port up"
}
wait_port "$P0"
wait_port "$P1"

OUT=/iris/u/mikulrai/logs/dart_swap_eval/${SLURM_JOB_ID}/preds
mkdir -p "$OUT"
LB_CFG=/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml
PROMPT="lift the steel barrier using two robot arms"

# Seeds: 4 known RIGHT-arm-failure + 2 known success (per 15870012 rep0 SR map).
SEEDS="10200000 10500000 10800000 11100000 10000000 10300000"

for S in $SEEDS; do
    echo "=== seed $S (open-loop, inject-h5-images) ==="
    "$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05_diag.py \
        --task LiftBarrier-rf \
        --config "$LB_CFG" \
        --host 127.0.0.1 \
        --ports "${P0},${P1}" \
        --seed "$S" \
        --max-env-steps 110 \
        --replan-after 8 \
        --prompt "$PROMPT" \
        --robot-uid panda_wristcam_multi \
        --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
        --num-arms 2 \
        --inject-h5-images \
        --log-pred-vs-demo "${OUT}/preds_seed${S}.jsonl" || echo "[warn] seed $S failed"
done

echo "Done at $(date)"
echo "pred logs: $OUT"
ls -la "$OUT"
