#!/bin/bash
#SBATCH --job-name=vk_brightness
#SBATCH --output=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/debug/vk_brightness_%j.log
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

echo "============================================================"
echo "HOST: $(hostname)"
nvidia-smi -L
echo "----- /etc/vulkan/icd.d -----"
ls -la /etc/vulkan/icd.d/ 2>/dev/null
echo "----- /usr/share/vulkan/icd.d -----"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null
echo "----- libGLX_nvidia -----"
ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia* 2>/dev/null
echo "----- vulkaninfo (summary) -----"
which vulkaninfo && vulkaninfo --summary 2>&1 | head -40 || echo "no vulkaninfo"
echo "============================================================"

echo ""
echo "############ ATTEMPT A: baseline (no env vars) ############"
unset VK_ICD_FILENAMES
python script/debug/test_render_brightness.py
ECODE_A=$?
echo "ATTEMPT A exit code: $ECODE_A"

echo ""
echo "############ ATTEMPT B: VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json ############"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
python script/debug/test_render_brightness.py
ECODE_B=$?
echo "ATTEMPT B exit code: $ECODE_B"

echo ""
echo "############ ATTEMPT C: patched SAPIEN, no manual env var ############"
unset VK_ICD_FILENAMES
echo "--- vulkaninfo --summary BEFORE python ---"
vulkaninfo --summary 2>/dev/null | grep -E "apiVersion|driverName|driverInfo|GPU id" | head -10 || echo "no vulkaninfo"
python script/debug/test_render_brightness.py
ECODE_C=$?
echo "ATTEMPT C exit code: $ECODE_C"

echo ""
echo "############ DONE — A=$ECODE_A B=$ECODE_B C=$ECODE_C ############"
