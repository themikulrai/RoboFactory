#!/bin/bash
#SBATCH --job-name=pm_in1k_calib
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/iris/u/mikulrai/runs/calibration/slurm-%j.out

# C2 S0 — capture PM in1k healthy-reference features for the encoder-collapse probe.
# Output: /iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz
# 30 fixed seeds (10000..10029) so future re-captures are bit-comparable.

set -euo pipefail

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

mkdir -p /iris/u/mikulrai/runs/calibration

SEEDS=$(seq 10000 10029 | tr '\n' ' ')

python script/debug/probe_pm_in1k_calibration.py \
    --config=configs/table/pick_meat.yaml \
    --ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt \
    --img-height=240 --img-width=320 \
    --seeds ${SEEDS} \
    --out /iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz

echo "PM in1k calibration capture complete."
