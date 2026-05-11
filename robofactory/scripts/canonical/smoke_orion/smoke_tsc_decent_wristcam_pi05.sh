#!/bin/bash
#SBATCH --job-name=smk_tsc_dec_wc_pi05
#SBATCH --output=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_decent_wristcam_pi05_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_decent_wristcam_pi05_%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=orion
#SBATCH --partition=orion

# Coverage-matrix smoke: TSC × decent × wristcam × Pi0.5 (1 seed)
# Already has 30% baseline; smoke confirms launcher pipeline on orion.

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
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR" /iris/u/mikulrai/logs/coverage_matrix_smoke
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

$PREFLIGHT_PYTHON -u -m robofactory.utils.preflight_eval \
    --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_BASE=/iris/u/mikulrai/checkpoints/openpi
CKPT_STEP=19999

ARM0_CFG=pi05_robofactory_decent_wristcam_lora_finetune_arm0
ARM1_CFG=pi05_robofactory_decent_wristcam_lora_finetune_arm1
ARM2_CFG=pi05_robofactory_decent_wristcam_lora_finetune_arm2

ARM0_DIR=${CKPT_BASE}/${ARM0_CFG}/pi05_wristcam_decent_arm0_v1/${CKPT_STEP}
ARM1_DIR=${CKPT_BASE}/${ARM1_CFG}/pi05_wristcam_decent_arm1_v1/${CKPT_STEP}
ARM2_DIR=${CKPT_BASE}/${ARM2_CFG}/pi05_wristcam_decent_arm2_v1/${CKPT_STEP}

for d in "$ARM0_DIR" "$ARM1_DIR" "$ARM2_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done

SERVER_LOG_DIR=/iris/u/mikulrai/logs/coverage_matrix_smoke/tsc_decent_wristcam_pi05_${SLURM_JOB_ID}_servers
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
start_server 2 8002 "$ARM2_CFG" "$ARM2_DIR" arm2

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

echo "[wait] probing ports 8000/8001/8002 ..."
for port in 8000 8001 8002; do
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

VIDEO_DIR=/iris/u/mikulrai/logs/coverage_matrix_smoke/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/coverage_matrix_smoke
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task ThreeRobotsStackCube-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml \
    --host 127.0.0.1 \
    --ports 8000,8001,8002 \
    --num-arms 3 \
    --num-episodes 1 \
    --seeds "100" \
    --max-env-steps 400 \
    --replan-after 8 \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/three_robots_stack_cube_wristcam.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-max 1 \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-tags 'eval,smoke,coverage-matrix,orion,tsc,decent,wristcam,pi05'
