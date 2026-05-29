#!/bin/bash
#SBATCH --job-name=g2_fidelity
#SBATCH --output=/iris/u/mikulrai/slurm/g2_fidelity_%j.out
#SBATCH --time=00:20:00
set -euo pipefail

echo "=== node info ==="
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
nvidia-smi -L || true
echo "=================="

cd /iris/u/mikulrai/projects/RoboFactory

# Vulkan / SAPIEN offscreen rendering on the compute node.
export DISPLAY=""
export PYTHONPATH=/iris/u/mikulrai/projects/RoboFactory:${PYTHONPATH:-}

/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
    robofactory/policy/openpi_pi05/run_g2_fidelity_check.py \
    --episode-index 0 \
    --rmse-threshold 0.1

echo "=== exit code $? ==="
