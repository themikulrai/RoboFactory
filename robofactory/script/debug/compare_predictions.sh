#!/bin/bash
#SBATCH --job-name=cmp_preds
#SBATCH --output=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/debug/compare_preds_%j.log
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=iris-hp-z8,iris-hgx-1

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

CKPT=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt
CFG=configs/table/pick_meat.yaml

echo "HOST: $(hostname)"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
echo "============================================================"
echo "GOAL: compare policy predictions for success seed 10011 vs failure seed 10000"
echo "Meat positions: 10011=(0.0613,0.0211)  10000=(0.0591,-0.0156)  diff-y=0.037m"
echo "If policy predicts SAME first action for both -> not image-conditioned on meat pos"
echo "If policy predicts DIFFERENT first actions   -> image conditioning works, bug is execution"
echo "============================================================"

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

echo ""
echo "=== SEED 10011 (SUCCESS, meat_y=+0.021) ==="
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 10011

echo ""
echo "=== SEED 10000 (FAILURE, meat_y=-0.016) ==="
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 10000

echo ""
echo "=== SEED 10005 (FAILURE, meat_y=-0.044, most negative y) ==="
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 10005

echo "DONE"
