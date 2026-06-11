#!/bin/bash
# Pi0.5 2SC wristcam decentralised eval with PROMPT SWAPPED to the (wrong)
# 3-arm training prompt. Tests whether wrong-prompt training is what zeros SR.
#
# Eval prompt: "stack the three cubes using three robot arms"
#   (task_index 1 from training tasks.jsonl, which Pi0.5 prompt sampler uses)
# Original 2SC eval used the 2-arm prompt and scored 5/60 IID (slurm 15437235).
# If promptswap > 5/60 -> wrong-prompt training was confounding (visual policy
# learned underneath). If still ~5/60 or 0 -> weights themselves are poisoned.
#
# Same ckpts/seeds/scene as original 2SC IID eval (apples-to-apples).
# Ckpts (step 19999):
#   /iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_2sc_decent_wristcam_lora_finetune_arm{0,1}/
#       pi05_2sc_wc_decent_a{0,1}/19999

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
LOG_BASE=/iris/u/mikulrai/logs/eval_pi05_decent/promptswap_${SLURM_JOB_ID}
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR" "$LOG_BASE"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set in environment before submitting}"
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

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

SERVER_LOG_DIR=${LOG_BASE}/servers
mkdir -p "$SERVER_LOG_DIR"

# Job-unique free ports (one per arm) so co-scheduled evals never collide on a
# shared hardcoded port. See _lib/free_ports.sh.
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/_lib/free_ports.sh
read -r PORT0 PORT1 <<<"$(free_ports 2)"
echo "[server] promptswap decent will use ports ${PORT0} ${PORT1}"

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

echo "[wait] probing ports ${PORT0}/${PORT1} ..."
for port in "$PORT0" "$PORT1"; do
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

VIDEO_DIR=${LOG_BASE}/videos
OUT_DIR=${LOG_BASE}
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

# PROMPT SWAP: use the (wrong) 3-arm training-side prompt to test if
# wrong-prompt-during-training is the cause of 0% SR.
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
    --prompt "stack the three cubes using three robot arms" \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/two_robots_stack_cube_wristcam.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-max 3 \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-name 'Eval 2SC wc Pi0.5 [Decent IID promptswap]' \
    --wandb-tags 'eval,pi05,decent,2sc,wristcam,table-scene,promptswap,ood-audit'
