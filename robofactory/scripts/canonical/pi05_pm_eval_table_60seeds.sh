#!/bin/bash
#SBATCH --job-name=pi05_pm_eval_60s
#SBATCH --output=/iris/u/mikulrai/logs/eval_pi05/pi05_pm_eval_60s_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_pi05/pi05_pm_eval_60s_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2

# First-ever eval of Pi0.5 PickMeat (workspace, single-arm) on TABLE scene.
# Ckpt: /iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_pm_lora_finetune/pi05_pm_d1_v1/19999
#   - openpi TrainConfig: pi05_robofactory_pm_lora_finetune (config.py:1743)
#   - wandb training run: h9lcxoip
#   - 1 image slot populated (head_camera), other 3 IMAGE_SLOTS null in
#     pick_meat.json — eval_pi05.py was patched today to zero-fill nulls
#     at obs-build time (mirrors RoboFactoryInputs train-side behavior).
#
# Hypothesis: matrix says PM × workspace × Pi0.5 = T (trained, never eval'd).
# Compare to DP PM in1k baseline of 58.3% (S6 reproduced 58.33% on 2026-05-08).

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
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR" /iris/u/mikulrai/logs/eval_pi05
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export WANDB_API_KEY=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
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# -----------------------------------------------------------------------------
# Stage-3 preflight guards (scene-match + camera-family + seed hash + wandb-online).
# Pi0.5 PM ckpt was trained pre-9710f9a so no .hydra_config.yaml dump exists;
# helper falls back to eval-cfg with soft-warn (legacy-safe).
# -----------------------------------------------------------------------------
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/pick_meat.yaml}"
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_pm_lora_finetune/pi05_pm_d1_v1/19999}"
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/canonical/_resolve_train_cfg.sh
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_60_dp.txt)
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60_dp.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds_60_dp.sha256 \
    --argv-seeds "$SEEDS"
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

$PREFLIGHT_PYTHON -u -m robofactory.utils.preflight_eval \
    --scene-config configs/table/pick_meat.yaml || exit 1

# -----------------------------------------------------------------------------
# Bring up 1 Pi0.5 policy server on port 8000.
# -----------------------------------------------------------------------------
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_DIR=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_pm_lora_finetune/pi05_pm_d1_v1/19999
PM_CFG=pi05_robofactory_pm_lora_finetune

[ -d "$CKPT_DIR" ] || { echo "missing ckpt dir: $CKPT_DIR"; exit 1; }

SERVER_LOG_DIR=/iris/u/mikulrai/logs/eval_pi05/pi05_pm_eval_60s_${SLURM_JOB_ID}_servers
mkdir -p "$SERVER_LOG_DIR"

echo "[server] starting PM policy GPU=0 port=8000 cfg=${PM_CFG} dir=${CKPT_DIR}"
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

# Wait for port 8000 to bind.
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

# -----------------------------------------------------------------------------
# Eval driver. PickMeat is single-arm, single-cam (head_camera). Other 3
# IMAGE_SLOTS in pick_meat.json are null and zero-filled by patched eval_pi05.py.
# $SEEDS was set above preflight so the cross-check sees the same list.
# -----------------------------------------------------------------------------
VIDEO_DIR=/iris/u/mikulrai/logs/eval_pi05/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/eval_pi05
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

UNIX_TS=$(date +%s)
RUN_NAME=pi05_pm_eval_table_${UNIX_TS}

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_pi05.py \
    --task PickMeat-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/pick_meat.yaml \
    --host 127.0.0.1 \
    --port 8000 \
    --num-arms 1 \
    --num-episodes 1 \
    --seeds "$SEEDS" \
    --max-env-steps 400 \
    --replan-after 8 \
    --robot-uid panda \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/pick_meat.json \
    --prompt "pick the meat with the gripper" \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-max 3 \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-tags 'eval,pi05,cent,pm,d1-workspace,table-scene,seedset-dp-xbucket,paired-vs-dp'
