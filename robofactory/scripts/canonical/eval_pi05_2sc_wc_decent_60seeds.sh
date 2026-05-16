#!/bin/bash
#SBATCH --job-name=pi05_2sc_wc_dec_60s
#SBATCH --output=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=orion
#SBATCH --partition=orion
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com

# Decentralised pi0.5 eval, 2SC WC ckpts (step 19999), TABLE scene, 60 seeds.
# Two per-arm LoRA servers (one a6000 each) on ports 8000/8001, routed by
# eval_decent_pi05.py. Requires the null-slot zero-fill patch in
# eval_decent_pi05.py (extra_0_rgb_raw is null in 2SC wristcam mapping).
#
# Ckpts (verified on disk; final step 19999):
#   /iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_2sc_decent_wristcam_lora_finetune_arm{0,1}/
#       pi05_2sc_wc_decent_a{0,1}/19999
#
# Training jobs: a0=15391100 (COMPLETED 24h on orion a6000),
#                a1=15433702 RESUME (COMPLETED 5h20m on iris-hi a40,
#                                    seeded from failed 15394891 at 22h).

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export OPENPI_DATA_HOME=/iris/u/mikulrai/data/openpi
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache
export JAX_COMPILATION_CACHE_DIR=/iris/u/mikulrai/.cache/jax
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR" /iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# -----------------------------------------------------------------------------
# Stage-3 preflight (same pattern as pi05_d1_decent_table_60seeds.sh).
# Resolves TRAIN_CFG_PATH from trainer-dumped .hydra_config.yaml when present,
# falls back to eval cfg with soft-warn otherwise.
# -----------------------------------------------------------------------------
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/two_robots_stack_cube.yaml}"
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_2sc_decent_wristcam_lora_finetune_arm0/pi05_2sc_wc_decent_a0/19999}"
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/canonical/_resolve_train_cfg.sh
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_60.txt)
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds.sha256 \
    --argv-seeds "$SEEDS"
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

$PREFLIGHT_PYTHON -u -m robofactory.utils.preflight_eval \
    --scene-config configs/table/two_robots_stack_cube.yaml || exit 1

# -----------------------------------------------------------------------------
# Bring up 2 per-arm Pi0.5 policy servers in the background, one per GPU.
# -----------------------------------------------------------------------------
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_BASE=/iris/u/mikulrai/checkpoints/openpi
CKPT_STEP=19999

ARM0_CFG=pi05_robofactory_2sc_decent_wristcam_lora_finetune_arm0
ARM1_CFG=pi05_robofactory_2sc_decent_wristcam_lora_finetune_arm1

ARM0_DIR=${CKPT_BASE}/${ARM0_CFG}/pi05_2sc_wc_decent_a0/${CKPT_STEP}
ARM1_DIR=${CKPT_BASE}/${ARM1_CFG}/pi05_2sc_wc_decent_a1/${CKPT_STEP}

for d in "$ARM0_DIR" "$ARM1_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done

SERVER_LOG_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

start_server () {
    local gpu_idx=$1 port=$2 cfg=$3 dir=$4 logname=$5
    echo "[server] starting arm GPU=${gpu_idx} port=${port} cfg=${cfg} dir=${dir}"
    (cd "$OPENPI_DIR" && \
     CUDA_VISIBLE_DEVICES=${gpu_idx} \
     XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
     "$OPENPI_PY" scripts/serve_policy.py \
         --port "${port}" \
         policy:checkpoint \
         --policy.config="${cfg}" \
         --policy.dir="${dir}") \
        > "${SERVER_LOG_DIR}/${logname}.log" 2>&1 &
    echo $! > "${SERVER_LOG_DIR}/${logname}.pid"
}

start_server 0 8000 "$ARM0_CFG" "$ARM0_DIR" arm0
start_server 1 8001 "$ARM1_CFG" "$ARM1_DIR" arm1

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

echo "[wait] probing ports 8000/8001 ..."
for port in 8000 8001; do
    deadline=$((SECONDS + 600))
    while ! "$PREFLIGHT_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',${port}))==0 else 1)" 2>/dev/null; do
        if [ $SECONDS -ge $deadline ]; then
            echo "[wait] timeout waiting for port ${port}; tail of server logs:"
            for f in "${SERVER_LOG_DIR}"/*.log; do echo "=== $f ==="; tail -n 60 "$f"; done
            exit 1
        fi
        sleep 5
    done
    echo "[wait] port ${port} up"
done

# -----------------------------------------------------------------------------
# Eval driver. num_arms=2, wristcam mapping has extra_0_rgb_raw=null which
# eval_decent_pi05.py must zero-fill (see required patch).
# -----------------------------------------------------------------------------
VIDEO_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task TwoRobotsStackCube-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/two_robots_stack_cube.yaml \
    --host 127.0.0.1 \
    --ports 8000,8001 \
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
    --video-max 3 \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-tags 'eval,pi05,decent,2sc,wristcam,table-scene,first-eval'
