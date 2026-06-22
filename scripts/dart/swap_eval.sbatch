#!/bin/bash
# DART SWAP DIAGNOSTIC — Pi0.5 LiftBarrier centralised wristcam.
#
# GOAL: determine whether the right-arm grasp failure follows the PHYSICAL right
# side or the policy's ACTION SLOT. We run the SAME ~14 failing base seeds twice
# against ONE centralised Pi0.5 policy server:
#   SWAP OFF -> normal eval (baseline, reproduces the failure set)
#   SWAP ON  -> --swap-arms: the policy sees physical-arm1 in slot0 and
#               physical-arm0 in slot1 (state + wrist image), and its slot0
#               output drives physical arm1 (input + output consistently swapped).
# A SYMMETRIC policy would give identical SR. If the failures MOVE to the other
# physical side under swap, the failure is tied to the ACTION SLOT (policy/data
# asymmetry). If they STAY on the same physical side, it follows the PHYSICAL side
# (sim / geometry asymmetry).
#
# Orchestration (server start / free-port pick / health-wait / cleanup) is reused
# verbatim from robofactory/scripts/eval_pi05_lb_wc_cent_60seeds.sh — only the
# client invocation differs (two runs, fixed failing-seed list, swap off/on).
#
# ISOLATION: runs from the dart-swap-diag WORKTREE (shared RF main checkout is
# branch-switched by other jobs mid-run). cd worktree + PYTHONPATH=worktree.
#
# partition / gpu / account are set at submit time via the iris-mcp tool args.
#
#SBATCH --job-name=dart_swap_eval_lb_wc_cent
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
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
export SAPIEN_HEADLESS=1

# ISOLATION: run the eval CLIENT from the dart-swap-diag worktree (carries the
# --swap-arms code). PYTHONPATH points at the worktree so `import robofactory`
# resolves there, NOT the shared main checkout that other jobs branch-switch.
WT=/iris/u/mikulrai/RoboFactory-swap-wt
cd "$WT/robofactory"
export PYTHONPATH="$WT:${PYTHONPATH:-}"

OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
CENT_CFG=pi05_robofactory_lb_wc_cent
CENT_DIR=/iris/u/mikulrai/checkpoints/openpi/${CENT_CFG}/lb-wc-cent-bs16-20k-2026-05-16-fp/19999
[ -d "$CENT_DIR" ] || { echo "missing ckpt dir: $CENT_DIR"; exit 1; }
echo "Evaluating cent ckpt: $CENT_DIR"

# Job-unique free port so co-scheduled evals on the same node never collide on
# the hardcoded 8000 (see _lib/free_ports.sh rationale).
PORT=$("$PREFLIGHT_PYTHON" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()")
echo "[server] cent will use port ${PORT}"

SERVER_LOG_DIR=/iris/u/mikulrai/logs/dart_swap_eval/${SLURM_JOB_ID}_servers
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

# FINAL env seeds (passed via --env-seeds). eval_seeds.canonical_env_60 =
# [b*100_000 for b in 100..159]; assert_no_train_overlap REFUSES seeds 0..182, so
# we MUST use the x100_000 values (all > 182), NOT the base numbers. No x100_000
# transform is applied inside the eval anymore — these are the final env seeds.
#
# 14 FAILING seeds from the recent representative centralized LB run (SR 46/60):
FAILING_SEEDS="10700000,11000000,11800000,11900000,12900000,13000000,13200000,13500000,14300000,14500000,14600000,14700000,15200000,15900000"
# 4 known-SUCCESS control seeds (confirm the swap doesn't break working episodes):
CONTROL_SEEDS="10000000,10200000,10300000,10400000"
# Same 18 env seeds for BOTH swap-off and swap-on so every seed is directly
# comparable across the two runs (same seed => same env init => matching video).
ENV_SEEDS="${FAILING_SEEDS},${CONTROL_SEEDS}"
echo "[env-seeds] $ENV_SEEDS"

CAM_MAP=/iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/lift_barrier_wristcam.json
LB_CFG=/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml
PROMPT="lift the steel barrier using two robot arms"

OFF_DIR=/iris/u/mikulrai/logs/dart_swap_eval/${SLURM_JOB_ID}/swap_off
ON_DIR=/iris/u/mikulrai/logs/dart_swap_eval/${SLURM_JOB_ID}/swap_on
mkdir -p "$OFF_DIR/videos" "$ON_DIR/videos"

run_eval () {
    # $1 = out dir, $2 = run-id suffix, rest = extra flags (e.g. --swap-arms)
    local out_dir="$1"; shift
    local rid="$1"; shift
    "$PREFLIGHT_PYTHON" -u ./policy/openpi_pi05/eval_pi05.py \
        --task LiftBarrier-rf \
        --config "$LB_CFG" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --expect-config "${CENT_CFG}" \
        --camera-mapping "$CAM_MAP" \
        --robot-uid panda_wristcam_multi \
        --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" \
        --num-arms 2 \
        --num-envs 1 \
        --env-seeds "$ENV_SEEDS" \
        --num-episodes 1 \
        --max-env-steps 400 \
        --replan-after 8 \
        --prompt "$PROMPT" \
        --out-dir "$out_dir" \
        --video-dir "$out_dir/videos" \
        --run-id "${SLURM_JOB_ID}_${rid}" \
        --wandb \
        --wandb-project LB-Pi05 \
        --wandb-tags "eval,pi05,cent,lb,wristcam,table-scene,dart-swap,${rid}" \
        "$@"
}

echo "=== SWAP OFF ==="
run_eval "$OFF_DIR" "swap_off" || true

echo "=== SWAP ON ==="
run_eval "$ON_DIR"  "swap_on" --swap-arms || true

# server is stopped by the EXIT trap (cleanup).
echo "Done at $(date)"
echo "swap_off JSON: $(ls -t $OFF_DIR/eval_LiftBarrier-rf_*.json 2>/dev/null | head -1)"
echo "swap_on  JSON: $(ls -t $ON_DIR/eval_LiftBarrier-rf_*.json 2>/dev/null | head -1)"
echo "videos: $OFF_DIR/videos  |  $ON_DIR/videos"
