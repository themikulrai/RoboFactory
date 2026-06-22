#!/bin/bash
# Egocentric (multi-view) re-render of the Pi0.5 LB-wc-decent OVERFIT ep0 probe:
# seed-0 rollout at step 10000 and step 49999, each tiled (global + both wrist),
# then hstacked left=10k | right=49999 -> one comparison clip that shows the
# egocentric input (replaces the old global-only tile). 2 serve+rollout cycles.
#
#SBATCH --job-name=overfit_ego_pi05
#SBATCH --output=/iris/u/mikulrai/logs/eval_overfit_egocentric/%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_overfit_egocentric/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1

set -euo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_overfit_egocentric
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
export HOME=/iris/u/mikulrai TORCH_HOME=$HOME/.cache/torch OPENPI_DATA_HOME=/iris/u/mikulrai/data/openpi
export HF_HOME=/iris/u/mikulrai/.cache/huggingface HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache JAX_COMPILATION_CACHE_DIR=/iris/u/mikulrai/.cache/jax TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 HYDRA_FULL_ERROR=1
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"

cd /iris/u/mikulrai/projects/RoboFactory/robofactory
OPENPI_DIR=/iris/u/mikulrai/projects/openpi
OPENPI_PY=${OPENPI_DIR}/.venv/bin/python
PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
A0=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm0/overfit_ep0_arm0
A1=/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm1/overfit_ep0_arm1
OUT=/iris/u/mikulrai/logs/eval_overfit_egocentric/${SLURM_JOB_ID}
mkdir -p "$OUT"
CAMJSON=/iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/lift_barrier_wristcam.json
CFG=/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml

run_step () {
    local step=$1 gpu0=$2 gpu1=$3
    local d0="$A0/$step" d1="$A1/$step"
    [ -d "$d0" ] && [ -d "$d1" ] || { echo "missing overfit ckpt step $step"; exit 1; }
    read -r P0 P1 <<<"$("$PY" -c "import socket
ss=[socket.socket() for _ in range(2)]
for s in ss: s.bind(('127.0.0.1',0))
print(*[s.getsockname()[1] for s in ss])
for s in ss: s.close()")"
    echo "[step $step] ports $P0 $P1"
    local LOG="$OUT/servers_$step"; mkdir -p "$LOG"
    (cd "$OPENPI_DIR" && CUDA_VISIBLE_DEVICES=$gpu0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot \
       "$OPENPI_PY" scripts/serve_policy.py --port "$P0" policy:checkpoint --policy.config=pi05_robofactory_lb_wc_decent_arm0 --policy.dir="$d0") >"$LOG/a0.log" 2>&1 &
    local pid0=$!
    (cd "$OPENPI_DIR" && CUDA_VISIBLE_DEVICES=$gpu1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 HF_LEROBOT_HOME=/iris/u/mikulrai/data/RoboFactory/lerobot \
       "$OPENPI_PY" scripts/serve_policy.py --port "$P1" policy:checkpoint --policy.config=pi05_robofactory_lb_wc_decent_arm1 --policy.dir="$d1") >"$LOG/a1.log" 2>&1 &
    local pid1=$!
    for port in "$P0" "$P1"; do
        local dl=$((SECONDS+600))
        while ! "$PY" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; do
            [ $SECONDS -ge $dl ] && { echo "timeout $port"; tail -40 "$LOG"/*.log; kill $pid0 $pid1 2>/dev/null; exit 1; }
            sleep 5
        done
    done
    echo "[step $step] servers up; rolling out seed 0"
    "$PY" -u ./policy/openpi_pi05/eval_decent_pi05.py --task LiftBarrier-rf --config "$CFG" --host 127.0.0.1 \
        --ports "$P0,$P1" --num-arms 2 --num-episodes 1 --seeds 0 --max-env-steps 400 --replan-after 8 \
        --prompt "lift the steel barrier using two robot arms" --robot-uid panda_wristcam_multi \
        --robot-uids-csv "panda_wristcam_multi,panda_wristcam_multi" --camera-mapping "$CAMJSON" \
        --out-dir "$OUT/step$step" --video-dir "$OUT/step$step/videos" --run-id "ovf$step" || true
    kill -TERM $pid0 $pid1 2>/dev/null || true; sleep 3; kill -KILL $pid0 $pid1 2>/dev/null || true
}

run_step 10000 0 0
run_step 49999 0 0

V10=$(ls "$OUT"/step10000/videos/*.mp4 2>/dev/null | head -1)
V50=$(ls "$OUT"/step49999/videos/*.mp4 2>/dev/null | head -1)
echo "10k video: $V10"; echo "50k video: $V50"
[ -f "$V10" ] && [ -f "$V50" ] || { echo "FATAL: missing rollout video(s)"; exit 1; }
# Left = 10k, Right = 49999. Pad to equal length so hstack works on differing durations.
ffmpeg -nostdin -loglevel error -y -i "$V10" -i "$V50" -filter_complex \
  "[0:v]drawtext=text='10k steps':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[a];\
   [1:v]drawtext=text='49999 steps':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[b];\
   [a][b]hstack=inputs=2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "$OUT/pi05_overfit_egocentric.mp4"
echo "DONE -> $OUT/pi05_overfit_egocentric.mp4"
ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$OUT/pi05_overfit_egocentric.mp4" || true
