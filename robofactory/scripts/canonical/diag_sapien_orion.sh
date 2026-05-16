#!/bin/bash
#SBATCH --job-name=diag_sapien
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/iris/u/mikulrai/logs/diag/diag_sapien_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/diag/diag_sapien_%j.err

# Timed import + env-build probe to find WHERE LP eval hangs on orion.
# Single-arm task too (PickMeat) for control: does LP-specific 4-arm scene cause the hang?

set -euxo pipefail
mkdir -p /iris/u/mikulrai/logs/diag

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

echo "=== node + vulkan ICDs ==="
hostname
ls -la /etc/vulkan/icd.d/ 2>/dev/null || echo "no /etc/vulkan/icd.d"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "no /usr/share/vulkan/icd.d"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
nvidia-smi -L 2>&1 | head -5 || echo "no nvidia-smi"

python -u <<'PYEOF'
import time, sys
T0 = time.time()
def stamp(msg):
    print(f"[{time.time()-T0:6.2f}s] {msg}", flush=True)

stamp("start")
import torch; stamp(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
import sapien; stamp(f"sapien {sapien.__version__ if hasattr(sapien,'__version__') else '?'}")
import gymnasium as gym; stamp("gymnasium")
import mani_skill; stamp("mani_skill")
import robofactory.tasks; stamp("robofactory.tasks")
import robofactory.agents; stamp("robofactory.agents (panda_wristcam_multi registered)")

stamp("make LP env (4-arm, gpu)...")
try:
    env = gym.make(
        "LongPipelineDelivery-rf",
        robot_uids=("panda_wristcam_multi",)*4,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="gpu",
        render_mode="sensors",
    )
    stamp("env_make OK")
except Exception as e:
    stamp(f"env_make FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

stamp("env.reset(seed=14)...")
try:
    obs, info = env.reset(seed=14)
    stamp(f"env.reset OK; obs keys = {list(obs.keys())[:5]}")
except Exception as e:
    stamp(f"env.reset FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

stamp("env.step zeros action...")
import numpy as np
act = np.zeros(env.action_space.shape, dtype=np.float32)
try:
    obs, rew, term, trunc, info = env.step(act)
    stamp("env.step OK")
except Exception as e:
    stamp(f"env.step FAILED: {type(e).__name__}: {e}")

env.close()
stamp("DONE")
PYEOF
