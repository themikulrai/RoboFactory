#!/bin/bash
#SBATCH --job-name=spawn_diagnostics
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/spawn_diagnostics_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/spawn_diagnostics_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=iris-hp-z8,iris-hgx-1

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

echo "HOST: $(hostname)"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
nvidia-smi -L

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

echo "========================================================"
echo "STEP 1: Reconstruct PM training meat positions from zarr images"
echo "Hypothesis: training positions cluster near eval success centroid (0.063, 0.041)"
echo "========================================================"
python script/debug/reconstruct_train_spawns_pm.py

echo ""
echo "========================================================"
echo "STEP 2: Probe TSC cube positions (training from images + eval from env)"
echo "========================================================"
python script/debug/probe_tsc_spawn.py

echo ""
echo "========================================================"
echo "STEP 3: DDPM stochasticity and conditioning test"
echo "Q1: Does the DDPM vary with torch seed (stochastic)?"
echo "Q2: Do different images (success vs failure seed) produce different predictions?"
echo "Q3: Do the ResNet features actually differ for different meat positions?"
echo "========================================================"
python script/debug/check_diffusion_stochasticity.py

echo "DONE"
