#!/bin/bash
#SBATCH --job-name=regen_seedfix
#SBATCH --output=/iris/u/mikulrai/logs/datagen_post_seedfix/regen_%A_%a.out
#SBATCH --error=/iris/u/mikulrai/logs/datagen_post_seedfix/regen_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#
# Regenerate RoboFactory motion-planner training data on the post-seedfix
# codepath (commit 69fcf5d9, Apr 29 2026), which switched np.random.rand ->
# self.env._episode_rng.rand in 6 places in RFSceneBuilder.initialize()
# (scene primitives, asset objects, agent qpos). Pre-fix datasets are not
# seed-reproducible from --seed; post-fix data is.
#
# One array task = one (task, seed=0, n=150) planner run with
# panda_wristcam_multi agents, which records BOTH workspace head_camera_agent*
# AND wristcam hand_camera_* into a single h5. The "_d1_workspace" vs
# "_d2_wristcam" distinction is downstream (parse_h5_to_zarr_unified.py
# --camera-family {workspace,wristcam}); we do NOT need 12 separate planner
# runs.
#
# Output: /iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/<Task>/<Task>.h5
#         + .json + .seedfix_regenerated sentinel
#
# Mapping (idx -> task):
#   0 PickMeat            (1 agent, ~5 min)
#   1 TwoRobotsStackCube  (2 agents)
#   2 ThreeRobotsStackCube(3 agents, ~15 min)
#   3 LongPipelineDelivery(4 agents, ~30 min)
#   4 LiftBarrier         (2 agents, ~10 min)
#   5 PlaceFood           (2 agents)

set -euo pipefail

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export SAPIEN_HEADLESS=1

cd /iris/u/mikulrai/projects/RoboFactory

TASKS=(
  "PickMeat"
  "TwoRobotsStackCube"
  "ThreeRobotsStackCube"
  "LongPipelineDelivery"
  "LiftBarrier"
  "PlaceFood"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID unset; submit via sbatch with --array" >&2
  exit 2
fi

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
RECORD_ROOT=/iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix
OUT_DIR="${RECORD_ROOT}/${TASK}"
OUT_H5="${OUT_DIR}/${TASK}.h5"

echo "[regen] array_id=${SLURM_ARRAY_TASK_ID}  task=${TASK}"
echo "[regen] output=${OUT_H5}"
echo "[regen] git sha: $(git -C /iris/u/mikulrai/projects/RoboFactory rev-parse HEAD)"

# Refuse to overwrite an existing post-seedfix dataset; this script is one-shot.
if [[ -f "${OUT_H5}" ]] && (( $(stat -c '%s' "${OUT_H5}" 2>/dev/null || echo 0) > 1024 )); then
  echo "[ERROR] ${OUT_H5} already exists. Move it aside before re-running this index." >&2
  exit 3
fi

# run_wristcam_rollouts.py uses panda_wristcam_multi agents (records BOTH
# workspace head_camera_agent* AND wristcam hand_camera_* per agent). It
# pulls seeds from the legacy hf_download/<Task>/<Task>.json so traj_i in
# the new dataset corresponds to the same seed as the legacy traj_i.
# (It always starts seed numbering from those JSON values; effectively -s 0
# in the post-seedfix sense since the codepath is now deterministic.)
python scripts/feasibility_wrist_replay/run_wristcam_rollouts.py \
  --task "${TASK}" \
  --num 150 \
  --record-dir "${RECORD_ROOT}"

# Sentinel for downstream auditors: post-seedfix dataset, captured git sha.
SENTINEL="${OUT_DIR}/.seedfix_regenerated"
{
  echo "regenerated_utc=$(date -u +%FT%TZ)"
  echo "git_sha=$(git -C /iris/u/mikulrai/projects/RoboFactory rev-parse HEAD)"
  echo "seedfix_commit=69fcf5d9"
  echo "task=${TASK}"
  echo "n_traj=150"
  echo "slurm_jobid=${SLURM_JOB_ID}"
  echo "slurm_array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "host=$(hostname)"
} > "${SENTINEL}"

echo "[regen] DONE  task=${TASK}  sentinel=${SENTINEL}"
