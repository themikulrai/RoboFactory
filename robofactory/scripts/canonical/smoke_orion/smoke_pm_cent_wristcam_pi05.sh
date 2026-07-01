#!/bin/bash
#SBATCH --job-name=smk_pm_cent_wc_pi05
#SBATCH --output=/iris/u/mikulrai/logs/coverage_matrix_smoke/pm_cent_wristcam_pi05_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/coverage_matrix_smoke/pm_cent_wristcam_pi05_%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=orion
#SBATCH --partition=orion

# Coverage-matrix smoke: PM × cent × wristcam × Pi0.5 (1 seed)
# Source: pi05_pm_eval_table_60seeds.sh — adapted for orion, wristcam ckpt, 1 seed.

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/iris/u/mikulrai/.cache/openpi}"
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache
export JAX_COMPILATION_CACHE_DIR=/iris/u/mikulrai/.cache/jax
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR" /iris/u/mikulrai/logs/coverage_matrix_smoke
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

$PREFLIGHT_PYTHON -u -m robofactory.utils.preflight_eval \
    --scene-config configs/table/pick_meat.yaml || exit 1

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_DIR=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_pm_wristcam_lora_finetune/pi05_pm_wristcam_v1/19999
PM_CFG=pi05_robofactory_pm_wristcam_lora_finetune

[ -d "$CKPT_DIR" ] || { echo "missing ckpt dir: $CKPT_DIR"; exit 1; }

SERVER_LOG_DIR=/iris/u/mikulrai/logs/coverage_matrix_smoke/pm_cent_wristcam_pi05_${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

echo "[server] starting PM wristcam policy GPU=0 port=8000 cfg=${PM_CFG} dir=${CKPT_DIR}"
(cd "$OPENPI_DIR" && \
 CUDA_VISIBLE_DEVICES=0 \
 XLA_PYTHON_CLIENT_PREALLOCATE=false \
 XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
 "$OPENPI_PY" scripts/serve_policy.py \
     --port 8000 \
     policy:checkpoint \
     --policy.config="${PM_CFG}" \
     --policy.dir="${CKPT_DIR}") \
    > "${SERVER_LOG_DIR}/pm.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "${SERVER_LOG_DIR}/pm.pid"

cleanup () {
    echo "[cleanup] stopping policy server"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[wait] probing port 8000 ..."
deadline=$((SECONDS + 600))
while ! "$PREFLIGHT_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)" 2>/dev/null; do
    if [ $SECONDS -ge $deadline ]; then
        echo "[wait] timeout waiting for port 8000; tail of server log:"
        tail -n 80 "${SERVER_LOG_DIR}/pm.log"
        exit 1
    fi
    sleep 5
done
echo "[wait] port 8000 up"

VIDEO_DIR=/iris/u/mikulrai/logs/coverage_matrix_smoke/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/coverage_matrix_smoke
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_pi05.py \
    --task PickMeat-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/pick_meat.yaml \
    --host 127.0.0.1 \
    --port 8000 \
    --num-arms 1 \
    --num-episodes 1 \
    --seeds "100" \
    --max-env-steps 400 \
    --replan-after 8 \
    --robot-uid panda \
    --robot-uids-csv "panda_wristcam" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/pick_meat_wristcam.json \
    --prompt "pick the meat with the gripper" \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-max 1 \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-tags 'eval,smoke,coverage-matrix,orion,pm,cent,wristcam,pi05'
