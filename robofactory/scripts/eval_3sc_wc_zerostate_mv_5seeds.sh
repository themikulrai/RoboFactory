#!/bin/bash
#SBATCH --job-name=eval_3sc_wc_zerostate_freshseed20k
#SBATCH --output=/iris/u/mikulrai/logs/eval_pi05_decent/zerostate_freshseed20k_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_pi05_decent/zerostate_freshseed20k_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Fresh-seed OOD eval of Pi0.5 3SC wristcam-decent — POST-SEEDFIX, ZERO-STATE @ step 19999.
# Mirrors the noaug freshseed20k eval script but swaps:
#   1. Ckpt paths -> the zerostate-trained models
#   2. Sets OPENPI_ZERO_STATE=1 so the policy server's RoboFactoryDecentInputs
#      transform zeros the proprio at INFERENCE too (must match training).
# Same 60 OOD seeds 20000..20059 for direct head-to-head vs IMPORTANT (25%) / noaug (15%).

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_pi05_decent

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export OPENPI_DATA_HOME=/iris/u/mikulrai/data/openpi
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache
export JAX_COMPILATION_CACHE_DIR=/iris/u/mikulrai/.cache/jax
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

# CRITICAL: inference must zero state the same way training did.
export OPENPI_ZERO_STATE=1

export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set" >&2
    exit 1
fi
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

echo "=== eval zerostate_freshseed20k job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "OPENPI_ZERO_STATE=${OPENPI_ZERO_STATE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /iris/u/mikulrai/projects/RoboFactory/robofactory
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

$PREFLIGHT_PYTHON -u -m robofactory.utils.preflight_eval \
    --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_BASE=/iris/u/mikulrai/checkpoints/openpi
CKPT_STEP=19999

ARM0_CFG=pi05_robofactory_tsc_wc_decent_arm0
ARM1_CFG=pi05_robofactory_tsc_wc_decent_arm1
ARM2_CFG=pi05_robofactory_tsc_wc_decent_arm2

# Zerostate ckpts (post-seedfix, aug ON, state zeroed at training & inference).
ARM0_DIR=${CKPT_BASE}/${ARM0_CFG}/pi05_3sc_wc_decent_postseedfix_zerostate_arm0_v1/${CKPT_STEP}
ARM1_DIR=${CKPT_BASE}/${ARM1_CFG}/pi05_3sc_wc_decent_postseedfix_zerostate_arm1_v1/${CKPT_STEP}
ARM2_DIR=${CKPT_BASE}/${ARM2_CFG}/pi05_3sc_wc_decent_postseedfix_zerostate_arm2_v1/${CKPT_STEP}

for d in "$ARM0_DIR" "$ARM1_DIR" "$ARM2_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done

SERVER_LOG_DIR=/iris/u/mikulrai/logs/eval_pi05_decent/zerostate_freshseed20k_${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

PORT_BASE=$(( 8000 + ((SLURM_JOB_ID % 600) * 3) ))
PORT0=$PORT_BASE
PORT1=$((PORT_BASE + 1))
PORT2=$((PORT_BASE + 2))
echo "Using policy server ports ${PORT0}/${PORT1}/${PORT2}"

start_server () {
    local gpu_idx=$1 port=$2 cfg=$3 dir=$4 logname=$5
    echo "[server] starting arm GPU=${gpu_idx} port=${port} cfg=${cfg} dir=${dir}"
    (cd "$OPENPI_DIR" && \
     CUDA_VISIBLE_DEVICES=${gpu_idx} \
     XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
     OPENPI_ZERO_STATE=1 \
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
start_server 2 "$PORT2" "$ARM2_CFG" "$ARM2_DIR" arm2

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

echo "[wait] probing policy server ports..."
for port in "$PORT0" "$PORT1" "$PORT2"; do
    deadline=$((SECONDS + 1200))
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

OUT_DIR=/iris/u/mikulrai/logs/eval_pi05_decent/zerostate_freshseed20k_${SLURM_JOB_ID}
VIDEO_DIR=${OUT_DIR}/videos
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

# Same 60 fresh OOD seeds as IMPORTANT + noaug freshseed evals.
SEEDS="20000,20001,20002,20003,20004"

WANDB_NAME="Eval 3SC wristcam Pi0.5 [Decent OOD freshseed20k Postseedfix ZeroState]"

echo "=== Driver: eval_decent_pi05.py, num_seeds=60 ==="
"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task ThreeRobotsStackCube-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml \
    --host 127.0.0.1 \
    --ports "${PORT0},${PORT1},${PORT2}" \
    --num-arms 3 \
    --num-episodes 1 \
    --seeds "$SEEDS" \
    --max-env-steps 800 \
    --replan-after 8 \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/three_robots_stack_cube_wristcam.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-max 5 \
    --run-id "zerostate_freshseed20k_${SLURM_JOB_ID}" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-name "$WANDB_NAME" \
    --wandb-tags 'eval,pi05,decent,3sc,wristcam,table-scene,freshseed,ood-audit,zerostate,postseedfix'

echo "=== Done $(date) ==="
