#!/bin/bash
#SBATCH --job-name=heavy_preflights
#SBATCH --output=/iris/u/mikulrai/logs/preflight/heavy_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/preflight/heavy_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris-hgx-2
#
# Plan v2 Phase C2 S5 — single-node half of the heavy-preflight suite.
#
# Runs the two single-node heavy checks sequentially and aborts on the
# first failure. The third heavy check (vulkan render-distribution) is
# cross-node and has its own driver: scripts/preflight/slurm_vulkan_render_check.sh.
#
# Why split: vulkan needs the SAME scene+seed rendered on TWO different
# compute nodes and a third compare job that depends on both. Wrapping
# that into one sbatch would deadlock on its own --dependency.
#
# Required env vars:
#   CKPT_PATH       — path to the freshly-overfit ckpt (for overfit replay)
#   DATASET_PATH    — zarr or LeRobot dataset (for both checks)
#   SCENE_CONFIG    — eval YAML (for both checks)
#   OUT_DIR         — where to drop the JSON reports
#
# Optional env vars:
#   EPISODE_IDX=0
#   MAX_STEPS=50
#   MSE_TOLERANCE=0.01
#   N_POS_DIMS=3
#   N_SAMPLES=50
#   POS_TOLERANCE_CM=0.5
#   ROT_TOLERANCE_DEG=0.5
#
# Usage (typical):
#   OUT_DIR=/iris/u/mikulrai/runs/preflight/<run_id> mkdir -p "$OUT_DIR"
#   sbatch --export=ALL,CKPT_PATH=...,DATASET_PATH=...,SCENE_CONFIG=...,OUT_DIR=$OUT_DIR \
#          scripts/preflight/slurm_heavy_preflights.sh
#
# To chain with a training job that runs only on green light:
#   J1=$(sbatch --parsable ... slurm_heavy_preflights.sh)
#   sbatch --dependency=afterok:$J1 ... <train_launcher>
#
# To also run the cross-node vulkan check (recommended when changing the
# render path or moving to a new node generation):
#   J1=$(sbatch --parsable ... slurm_heavy_preflights.sh)
#   J2=$(sbatch --parsable ... slurm_vulkan_render_check.sh render <a.npy>)
#   J3=$(sbatch --parsable ... slurm_vulkan_render_check.sh render <b.npy>)
#   J4=$(sbatch --dependency=afterok:$J2:$J3 ... slurm_vulkan_render_check.sh compare ...)
#   sbatch --dependency=afterok:$J1:$J4 ... <train_launcher>
#
set -euo pipefail

: "${CKPT_PATH:?CKPT_PATH must be set}"
: "${DATASET_PATH:?DATASET_PATH must be set}"
: "${SCENE_CONFIG:?SCENE_CONFIG must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"

EPISODE_IDX="${EPISODE_IDX:-0}"
MAX_STEPS="${MAX_STEPS:-50}"
MSE_TOLERANCE="${MSE_TOLERANCE:-0.01}"
N_POS_DIMS="${N_POS_DIMS:-3}"
N_SAMPLES="${N_SAMPLES:-50}"
POS_TOLERANCE_CM="${POS_TOLERANCE_CM:-0.5}"
ROT_TOLERANCE_DEG="${ROT_TOLERANCE_DEG:-0.5}"

mkdir -p "$OUT_DIR"

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory
PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(ts)] heavy preflights starting"
echo "[$(ts)]   CKPT_PATH=${CKPT_PATH}"
echo "[$(ts)]   DATASET_PATH=${DATASET_PATH}"
echo "[$(ts)]   SCENE_CONFIG=${SCENE_CONFIG}"
echo "[$(ts)]   OUT_DIR=${OUT_DIR}"

# ---------------------------------------------------------------------------
# Check 1/2 — init-pose Wasserstein (cheap-ish, ~1 min)
# ---------------------------------------------------------------------------
echo "[$(ts)] [1/2] init_pose_wasserstein"
"$PY" scripts/preflight/check_init_pose_wasserstein.py \
    --train-data-path "$DATASET_PATH" \
    --eval-config "$SCENE_CONFIG" \
    --n-samples "$N_SAMPLES" \
    --n-pos-dims "$N_POS_DIMS" \
    --pos-tolerance-cm "$POS_TOLERANCE_CM" \
    --rot-tolerance-deg "$ROT_TOLERANCE_DEG" \
    --out-json "$OUT_DIR/init_pose_wasserstein.json"
echo "[$(ts)] [1/2] init_pose_wasserstein PASSED"

# ---------------------------------------------------------------------------
# Check 2/2 — overfit replay sanity (heavier, depends on policy load)
# ---------------------------------------------------------------------------
echo "[$(ts)] [2/2] overfit_replay_sanity"
"$PY" scripts/preflight/check_overfit_replay_sanity.py \
    --ckpt-path "$CKPT_PATH" \
    --dataset-path "$DATASET_PATH" \
    --scene-config "$SCENE_CONFIG" \
    --episode-idx "$EPISODE_IDX" \
    --max-steps "$MAX_STEPS" \
    --mse-tolerance "$MSE_TOLERANCE" \
    --mode full \
    --output-json "$OUT_DIR/overfit_replay.json"
echo "[$(ts)] [2/2] overfit_replay_sanity PASSED"

echo "[$(ts)] all single-node heavy preflights PASSED"
echo "[$(ts)] reports in: $OUT_DIR"
echo "[$(ts)] reminder: cross-node vulkan check is a separate dispatch — see header comment"
exit 0
