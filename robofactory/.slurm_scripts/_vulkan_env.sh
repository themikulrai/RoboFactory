#!/usr/bin/env bash
# Source at the top of every RoboFactory SLURM script (after `conda activate`).
# Belt-and-suspenders: even if the SAPIEN patch is lost (reinstall), this still
# sets VK_ICD_FILENAMES so SAPIEN's user-override branch wins.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
        : # canonical path present; loader uses it without override
    elif [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
        export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
        echo "[vulkan_env] using /etc/vulkan/icd.d/nvidia_icd.json on $(hostname -s)"
    else
        echo "[vulkan_env] WARNING: no nvidia_icd.json found on $(hostname -s);" >&2
        echo "  SAPIEN will use its stale bundled ICD. Consider adding this host to BAD_VULKAN_NODES." >&2
    fi
fi
