#!/bin/bash
#SBATCH --job-name=shadow_verify2
#SBATCH --output=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/debug/shadow_verify2_%j.log
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

CKPT=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt
H5=/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5
CFG=configs/table/pick_meat.yaml

echo "HOST: $(hostname)"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
nvidia-smi -L
echo "============================================================"
echo "EXPECT: env obs raw mean ~0.484 (matching training), norm mean ~-0.03"
echo "============================================================"

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

echo ""
echo "--- forward_one train ---"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source train --h5 "$H5" --traj 0

echo ""
echo "--- forward_one env seed=0 shadow=False ---"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 0

echo "DONE"
