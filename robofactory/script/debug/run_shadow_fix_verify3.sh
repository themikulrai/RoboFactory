#!/bin/bash
#SBATCH --job-name=shadow_verify3
#SBATCH --output=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/debug/shadow_verify3_%j.log
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=iris-hp-z8,iris-hgx-1

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

CKPT=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt
H5=/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5
CFG=configs/table/pick_meat.yaml

echo "HOST: $(hostname)"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
nvidia-smi -L
echo "============================================================"
echo "GOAL: confirm env obs mean ~0.484 with shader_pack=default + shadow=False"
echo "============================================================"

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

echo ""
echo "--- forward_one train (baseline) ---"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source train --h5 "$H5" --traj 0

echo ""
echo "--- forward_one env seed=0 (shadow=False + shader_pack=default) ---"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 0

echo "DONE"
