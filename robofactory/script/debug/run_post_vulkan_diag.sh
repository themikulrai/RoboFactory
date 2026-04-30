#!/bin/bash
#SBATCH --job-name=post_vk_diag
#SBATCH --output=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/debug/post_vk_diag_%j.log
#SBATCH --time=00:20:00
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

echo "============================================================"
echo "HOST: $(hostname)"
nvidia-smi -L
echo "----- Vulkan ICD check -----"
ls -la /etc/vulkan/icd.d/ 2>/dev/null || echo "no /etc/vulkan/icd.d"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "no /usr/share/vulkan/icd.d"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>}"
echo "SAPIEN_VULKAN_LIBRARY_PATH=${SAPIEN_VULKAN_LIBRARY_PATH:-<unset>}"
echo "============================================================"

echo ""
echo "############ STAGE 0: brightness vs paper PNG ############"
python script/debug/test_render_brightness.py || true

echo ""
echo "############ STAGE 1: forward_one --obs-source train (traj 0) ############"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source train --h5 "$H5" --traj 0

echo ""
echo "############ STAGE 2: forward_one --obs-source env (seed 0) ############"
python script/debug/forward_one.py --ckpt "$CKPT" --obs-source env --config "$CFG" --seed 0

echo ""
echo "############ STAGE 3: replay_h5 trajs 0 1 2 ############"
python script/debug/replay_h5.py --h5 "$H5" --config "$CFG" --traj 0 1 2

echo ""
echo "############ DONE ############"
