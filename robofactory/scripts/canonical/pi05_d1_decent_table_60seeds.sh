#!/bin/bash
#SBATCH --job-name=pi05_d1_decent_60s
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pi05_d1_decent_60s_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pi05_d1_decent_60s_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a40:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2
# gres=gpu:a40:3 forces 48 GB cards — Pi0.5 LoRA OOM'd on 8 GB RTX 2070 (job 15366954).

# Tier 2.3 — Decentralised Pi0.5 eval, D1 workspace ckpts, TABLE scene, 60 seeds.
# Three per-arm LoRA servers (one GPU each) on ports 8000/8001/8002, all routed
# by eval_decent_pi05.py.
#
# Ckpts (verified on disk; final step 19999):
#   /iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_decent_lora_finetune_arm{0,1,2}/
#       pi05_d1_decent_arm{0,1,2}_v0/19999/
#
# Training:
#   - openpi TrainConfig: pi05_robofactory_decent_lora_finetune_arm{0,1,2}
#   - dataset: robofactory_three_arm_stack (D1, no wristcam, workspace cams)
#   - 4 image slots: base_0_rgb_raw=head_camera_global,
#     left_wrist_0_rgb_raw=head_camera_agent0, right_wrist_0_rgb_raw=head_camera_agent1,
#     extra_0_rgb_raw=head_camera_agent2  (RoboFactoryDecentInputs masks all but
#     base + this-agent's "wrist" slot)
#   - per-arm wandb training runs: i80mhtcu (a0), fk3eef21 (a1), 0xdaxfw1 (a2)
#
# Goal: confirm whether D1 workspace decent reaches the same ~30% as the
# wristcam decent baseline, or whether it's a different result entirely. Eval
# logs into a fresh wandb run; does NOT resume training runs.

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
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR" /iris/u/mikulrai/logs/phase2_debug
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export WANDB_API_KEY=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
export WANDB_PROJECT=openpi-robofactory
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# -----------------------------------------------------------------------------
# Stage-3 preflight guards. C1#10 (commit 9710f9a) added a trainer-side
# .hydra_config.yaml dump so the camera-family check becomes a real
# workspace-vs-wristcam assertion. _resolve_train_cfg.sh prefers that file
# when present; legacy ckpts (trained pre-9710f9a — like the d1 decent v0
# set below) lack the dump and the helper falls back to eval-cfg with
# soft-warn. Future Pi0.5 retrains will surface a hard error here.
# -----------------------------------------------------------------------------
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/three_robots_stack_cube.yaml}"
# Pi0.5 ckpt dirs hold an `assets/` symlink + `params/` weights tree; point
# CKPT_FOR_PREFLIGHT at the per-arm step dir so dirname() lands on the run
# dir where a future trainer-side dump would live.
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_decent_lora_finetune_arm0/pi05_d1_decent_arm0_v0/19999}"
source "$(dirname "$0")/_resolve_train_cfg.sh"
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds.sha256
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

$PREFLIGHT_PYTHON -u -m robofactory.utils.preflight_eval \
    --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

# -----------------------------------------------------------------------------
# Bring up 3 per-arm Pi0.5 policy servers in the background, one per GPU,
# each on its own port. serve_policy.py lives in the openpi project; its
# venv (uv-managed, py3.11) is at /iris/u/mikulrai/projects/openpi/.venv.
# -----------------------------------------------------------------------------
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
CKPT_BASE=/iris/u/mikulrai/checkpoints/openpi
CKPT_STEP=19999

ARM0_CFG=pi05_robofactory_decent_lora_finetune_arm0
ARM1_CFG=pi05_robofactory_decent_lora_finetune_arm1
ARM2_CFG=pi05_robofactory_decent_lora_finetune_arm2

ARM0_DIR=${CKPT_BASE}/${ARM0_CFG}/pi05_d1_decent_arm0_v0/${CKPT_STEP}
ARM1_DIR=${CKPT_BASE}/${ARM1_CFG}/pi05_d1_decent_arm1_v0/${CKPT_STEP}
ARM2_DIR=${CKPT_BASE}/${ARM2_CFG}/pi05_d1_decent_arm2_v0/${CKPT_STEP}

for d in "$ARM0_DIR" "$ARM1_DIR" "$ARM2_DIR"; do
    [ -d "$d" ] || { echo "missing ckpt dir: $d"; exit 1; }
done

SERVER_LOG_DIR=/iris/u/mikulrai/logs/phase2_debug/pi05_d1_decent_60s_${SLURM_JOB_ID}_servers
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

# Cleanup hook: tear servers down on exit/abort.
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

# Wait for all 3 servers to bind their ports (TCP probe).
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

# -----------------------------------------------------------------------------
# Eval driver. Uses workspace camera mapping (D1 ckpts saw head_camera_agent{0..2}
# as the per-agent "wrist" slot, NOT hand_camera_X).
# -----------------------------------------------------------------------------
SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_60.txt)

VIDEO_DIR=/iris/u/mikulrai/logs/eval_pi05_decent/videos_${SLURM_JOB_ID}
OUT_DIR=/iris/u/mikulrai/logs/eval_pi05_decent
mkdir -p "$VIDEO_DIR" "$OUT_DIR"

UNIX_TS=$(date +%s)
RUN_NAME=pi05_d1_decent_eval_table_${UNIX_TS}

"$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_decent_pi05.py \
    --task ThreeRobotsStackCube-rf \
    --config /iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml \
    --host 127.0.0.1 \
    --ports 8000,8001,8002 \
    --num-arms 3 \
    --num-episodes 1 \
    --seeds "$SEEDS" \
    --max-env-steps 400 \
    --replan-after 8 \
    --robot-uid panda_wristcam_multi \
    --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi" \
    --camera-mapping /iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/three_robots_stack_cube.json \
    --out-dir "$OUT_DIR" \
    --video-dir "$VIDEO_DIR" \
    --video-max 3 \
    --run-id "$SLURM_JOB_ID" \
    --wandb \
    --wandb-project openpi-robofactory \
    --wandb-tags 'eval,pi05,decent,d1-workspace,table-scene,tier2.3,encoder-fix'
