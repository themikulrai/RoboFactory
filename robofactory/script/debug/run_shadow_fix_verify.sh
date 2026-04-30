#!/bin/bash
#SBATCH --job-name=shadow_fix_verify
#SBATCH --output=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/debug/shadow_fix_verify_%j.log
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

CKPT=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt
H5=/iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5
CFG=configs/table/pick_meat.yaml

echo "HOST: $(hostname)"
echo "============================================================"
echo "VERIFYING: enable_shadow=False closes the train-vs-env image gap"
echo "============================================================"

echo ""
echo "############ forward_one --obs-source train ############"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source train --h5 "$H5" --traj 0

echo ""
echo "############ forward_one --obs-source env (seed 0, shadow=False) ############"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 0

echo ""
echo "############ DONE ############"
