#!/bin/bash
#SBATCH --job-name=regen_seedfix_ws
#SBATCH --output=/iris/u/mikulrai/logs/datagen_post_seedfix/regen_ws_%A_%a.out
#SBATCH --error=/iris/u/mikulrai/logs/datagen_post_seedfix/regen_ws_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#
# Regenerate the WORKSPACE half of RoboFactory training data on the post-
# seedfix codepath (commit 69fcf5d9). Sister of regen_all_data_post_seedfix.sh
# (which uses panda_wristcam_multi via run_wristcam_rollouts.py).
#
# This launcher uses the canonical workspace planner:
#   python -m robofactory.planner.run -c configs/table/<task>.yaml -s 0 -n 150
# which uses the YAML's robot_uid (panda-0) — the original "_d1_workspace"
# lineage. Trajectories differ from the wristcam half because the robot
# URDF/dynamics differ, even with the same env seed.
#
# Output: /iris/u/mikulrai/datasets/multi_robot/RoboFactory/hf_download_post_seedfix_workspace/
#         <ENV_ID>/motionplanning/<traj-name>.h5 (planner.run convention)

set -euo pipefail

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export SAPIEN_HEADLESS=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# (env_id, yaml_path, n_agents)
TASKS=(
  "PickMeat-rf:configs/table/pick_meat.yaml:1"
  "TwoRobotsStackCube-rf:configs/table/two_robots_stack_cube.yaml:2"
  "ThreeRobotsStackCube-rf:configs/table/three_robots_stack_cube.yaml:3"
  "LongPipelineDelivery-rf:configs/table/long_pipeline_delivery.yaml:4"
  "LiftBarrier-rf:configs/table/lift_barrier.yaml:2"
  "PlaceFood-rf:configs/table/place_food.yaml:2"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID unset; submit via sbatch with --array" >&2
  exit 2
fi

IFS=':' read -r ENV_ID YAML N_AGENTS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
RECORD_ROOT="${RF_DATA_ROOT:?RF_DATA_ROOT unset — source ~/.config/dataroots.sh}/hf_download_post_seedfix_workspace"
OUT_DIR="${RECORD_ROOT}/${ENV_ID}/motionplanning"

echo "[regen-ws] array_id=${SLURM_ARRAY_TASK_ID}  env_id=${ENV_ID}  yaml=${YAML}  n_agents=${N_AGENTS}"
echo "[regen-ws] output_dir=${OUT_DIR}"
echo "[regen-ws] git sha: $(git -C /iris/u/mikulrai/projects/RoboFactory rev-parse HEAD)"

# Refuse to clobber an existing post-seedfix workspace dataset.
if [[ -d "${OUT_DIR}" ]] && [[ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
  echo "[ERROR] ${OUT_DIR} already populated. Move it aside before re-running this index." >&2
  exit 3
fi

mkdir -p "${OUT_DIR}"

python -m robofactory.planner.run \
  -e "${ENV_ID}" \
  -c "${YAML}" \
  -s 0 \
  -n 150 \
  --record-dir "${RECORD_ROOT}" \
  --traj-name "${ENV_ID}_workspace_seed0_n150"

# Sentinel for downstream auditors.
SENTINEL="${OUT_DIR}/.seedfix_regenerated"
{
  echo "regenerated_utc=$(date -u +%FT%TZ)"
  echo "git_sha=$(git -C /iris/u/mikulrai/projects/RoboFactory rev-parse HEAD)"
  echo "seedfix_commit=69fcf5d9"
  echo "env_id=${ENV_ID}"
  echo "yaml=${YAML}"
  echo "n_agents=${N_AGENTS}"
  echo "n_traj=150"
  echo "robot=panda-0  (workspace cam, no wrist)"
  echo "slurm_jobid=${SLURM_JOB_ID}"
  echo "slurm_array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "host=$(hostname)"
} > "${SENTINEL}"

echo "[regen-ws] DONE  env_id=${ENV_ID}  sentinel=${SENTINEL}"
