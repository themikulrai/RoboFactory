#!/bin/bash
#SBATCH --job-name=conv_post_seedfix
#SBATCH --account=orion
#SBATCH --partition=orion
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/mikulrai/slurm/%j.out
#SBATCH --error=/iris/u/mikulrai/slurm/%j.err
#
# Idempotent orchestrator: convert every post-seedfix wristcam H5 into both
#   (a) zarr  (Diffusion Policy)   via parse_h5_to_zarr_unified.py
#   (b) parquet (LeRobot / Pi0.5)  via convert_robofactory_data_to_lerobot.py
# for both camera families (workspace head_camera_agent*, wristcam hand_camera_*).
#
# IMPORTANT DESIGN NOTE: The single wristcam-launcher H5 in hf_download_post_seedfix
# records BOTH head_camera_agent* AND hand_camera_* (panda_wristcam_multi agent).
# The separate hf_download_post_seedfix_workspace/* H5s only contain action arrays
# (robofactory.planner.run does not record observations) and are NOT usable here.
# So we drive every cell off hf_download_post_seedfix/<Task>/<Task>.h5.
#
# Each array index = one (task, cam_family, kind[joint|agent<i>|lerobot]) cell.
# Idempotent: each cell checks for its output + .done_sentinel before running.

set -euo pipefail

# --- environment plumbing -----------------------------------------------------
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
export HOME=/iris/u/mikulrai
export SAPIEN_HEADLESS=1

RF_PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
OPENPI_PY=/iris/u/mikulrai/projects/openpi/.venv/bin/python

ZARR_SCRIPT=/iris/u/mikulrai/projects/RoboFactory/robofactory/script/parse_h5_to_zarr_unified.py
LEROBOT_SCRIPT=/iris/u/mikulrai/projects/openpi/examples/robofactory/convert_robofactory_data_to_lerobot.py

source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
H5_ROOT="${RF_DATA_ROOT:?RF_DATA_ROOT unset — source ~/.config/dataroots.sh}/hf_download_post_seedfix"
ZARR_OUT_ROOT="${RF_DATA_ROOT:?}/zarr_data"
LEROBOT_OUT_ROOT="${RF_LEROBOT_HOME:?RF_LEROBOT_HOME unset — source ~/.config/dataroots.sh}"
CAM_MAP_DIR=/iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings

mkdir -p "${ZARR_OUT_ROOT}" "${LEROBOT_OUT_ROOT}"

# --- per-task metadata: NAME:N_AGENTS:TASK_SLUG ------------------------------
declare -A N_AGENTS=(
  [PickMeat]=1
  [TwoRobotsStackCube]=2
  [ThreeRobotsStackCube]=3
  [LongPipelineDelivery]=4
  [LiftBarrier]=2
  [PlaceFood]=2
)
# task_slug for LeRobot repo_id and camera-mapping filename stems
declare -A SLUG=(
  [PickMeat]=pick_meat
  [TwoRobotsStackCube]=two_robots_stack_cube
  [ThreeRobotsStackCube]=three_robots_stack_cube
  [LongPipelineDelivery]=long_pipeline_delivery
  [LiftBarrier]=lift_barrier
  [PlaceFood]=place_food
)
TASKS_ORDER=(PickMeat TwoRobotsStackCube ThreeRobotsStackCube LongPipelineDelivery LiftBarrier PlaceFood)

# --- build the cell list ------------------------------------------------------
# Each entry: "TASK:CAM:KIND" where KIND in {joint, agent0..agent{N-1}, lerobot}.
CELLS=()
for TASK in "${TASKS_ORDER[@]}"; do
  N=${N_AGENTS[$TASK]}
  for CAM in workspace wristcam; do
    CELLS+=("${TASK}:${CAM}:joint")
    if (( N > 1 )); then
      for (( i=0; i<N; i++ )); do
        CELLS+=("${TASK}:${CAM}:agent${i}")
      done
    fi
    CELLS+=("${TASK}:${CAM}:lerobot")
  done
done

NCELLS=${#CELLS[@]}

# --- listing mode: print all cells and exit (no SLURM_ARRAY_TASK_ID needed) --
if [[ "${1:-}" == "--list" ]]; then
  echo "Total cells: ${NCELLS}"
  for idx in "${!CELLS[@]}"; do
    cell="${CELLS[$idx]}"
    IFS=':' read -r TASK CAM KIND <<< "$cell"
    H5="${H5_ROOT}/${TASK}/${TASK}.h5"
    SENTINEL_H5="${H5_ROOT}/${TASK}/.seedfix_regenerated"
    if [[ -f "${SENTINEL_H5}" ]] && [[ -f "${H5}" ]] && (( $(stat -c '%s' "${H5}") > 1024 )); then
      h5_state="READY"
    else
      h5_state="IN_FLIGHT_OR_MISSING"
    fi
    printf "%3d  %-25s  %-10s  %-10s  h5=%s\n" "$idx" "$TASK" "$CAM" "$KIND" "$h5_state"
  done
  exit 0
fi

# --- dispatch one cell --------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID unset; submit via sbatch with --array=0-$((NCELLS-1))" >&2
  exit 2
fi
IDX=${SLURM_ARRAY_TASK_ID}
if (( IDX < 0 || IDX >= NCELLS )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${IDX} out of range [0, ${NCELLS})" >&2
  exit 2
fi

CELL="${CELLS[$IDX]}"
IFS=':' read -r TASK CAM KIND <<< "$CELL"
N=${N_AGENTS[$TASK]}
SLUG_LC=${SLUG[$TASK]}

H5="${H5_ROOT}/${TASK}/${TASK}.h5"
SENTINEL_H5="${H5_ROOT}/${TASK}/.seedfix_regenerated"

echo "[conv] array_id=${IDX}  task=${TASK}  cam=${CAM}  kind=${KIND}  n_agents=${N}"
echo "[conv] h5=${H5}"
echo "[conv] host=$(hostname)  date=$(date -u +%FT%TZ)"

# --- gate on H5 readiness -----------------------------------------------------
if [[ ! -f "${SENTINEL_H5}" ]]; then
  echo "[conv] SKIP  no .seedfix_regenerated sentinel for ${TASK}; data-gen still in-flight."
  exit 0
fi
if [[ ! -f "${H5}" ]] || (( $(stat -c '%s' "${H5}") <= 1024 )); then
  echo "[conv] SKIP  H5 missing or placeholder-sized for ${TASK}."
  exit 0
fi
# Refuse if mtime is within last 10 minutes (still being written) - defensive.
NOW=$(date +%s)
MTIME=$(stat -c '%Y' "${H5}")
AGE=$(( NOW - MTIME ))
if (( AGE < 600 )); then
  echo "[conv] SKIP  H5 mtime within 10 min (${AGE}s) - likely still writing."
  exit 0
fi

# Sanity: the JSON sidecar
JSON="${H5%.h5}.json"
if [[ ! -f "${JSON}" ]]; then
  echo "[conv] ERROR  missing JSON sidecar ${JSON}"; exit 1
fi

# --- dispatch by kind ---------------------------------------------------------
case "$KIND" in
  joint|agent*)
    # zarr conversion
    if [[ "$KIND" == "joint" ]]; then
      OUT="${ZARR_OUT_ROOT}/${TASK}_${CAM}_joint.zarr"
      MODE_ARGS=(--mode joint --n-agents "$N")
    else
      AGENT_I=${KIND#agent}
      OUT="${ZARR_OUT_ROOT}/${TASK}_${CAM}_agent${AGENT_I}.zarr"
      MODE_ARGS=(--mode per_agent --agent-id "$AGENT_I")
    fi
    DONE="${OUT}.done_sentinel"
    if [[ -f "${DONE}" ]]; then
      echo "[conv] SKIP  zarr cell already done: ${OUT}"; exit 0
    fi
    if [[ -d "${OUT}" ]] && [[ ! -f "${DONE}" ]]; then
      echo "[conv] WARN  zarr exists without sentinel; removing partial ${OUT}"
      rm -rf "${OUT}"
    fi
    echo "[conv] writing zarr -> ${OUT}  (with --include-global)"
    "${RF_PY}" "${ZARR_SCRIPT}" \
      --h5-path "${H5}" \
      --out-zarr "${OUT}" \
      --camera-family "${CAM}" \
      --include-global \
      "${MODE_ARGS[@]}" \
      --resize 224
    {
      echo "done_utc=$(date -u +%FT%TZ)"
      echo "task=${TASK}"; echo "cam=${CAM}"; echo "kind=${KIND}"
      echo "h5=${H5}"
      echo "slurm_jobid=${SLURM_JOB_ID}"
      echo "host=$(hostname)"
    } > "${DONE}"
    echo "[conv] DONE  ${OUT}"
    ;;

  lerobot)
    # pi0.5 LeRobot conversion (uses openpi venv, not RoboFactory env)
    REPO_ID="robofactory_${SLUG_LC}_${CAM}"
    OUT="${LEROBOT_OUT_ROOT}/${REPO_ID}"
    DONE="${OUT}.done_sentinel"
    if [[ -f "${DONE}" ]]; then
      echo "[conv] SKIP  lerobot cell already done: ${OUT}"; exit 0
    fi

    # Camera-mapping selection
    CAM_MAP=""
    case "$CAM" in
      wristcam) CAM_MAP="${CAM_MAP_DIR}/${SLUG_LC}_wristcam.json" ;;
      workspace) CAM_MAP="${CAM_MAP_DIR}/${SLUG_LC}.json" ;;
    esac
    EXTRA_ARGS=()
    if [[ -n "${CAM_MAP}" ]] && [[ -f "${CAM_MAP}" ]]; then
      EXTRA_ARGS+=(--camera_mapping "${CAM_MAP}")
    fi
    # Prompts: PickMeat uses dedicated prompts; others rely on script default.
    if [[ "${TASK}" == "PickMeat" ]]; then
      EXTRA_ARGS+=(--prompts_path "${CAM_MAP_DIR}/pick_meat_prompts.json")
    fi
    # Agent name in H5: single-arm uses 'panda_wristcam_multi' for multi-arm tasks too.
    AGENT_NAME=panda_wristcam_multi

    echo "[conv] writing LeRobot repo=${REPO_ID}  cam_map=${CAM_MAP:-default}"
    "${OPENPI_PY}" "${LEROBOT_SCRIPT}" \
      --h5_path "${H5}" \
      --out_root "${LEROBOT_OUT_ROOT}" \
      --repo_id "${REPO_ID}" \
      --num_arms "${N}" \
      --agent_name "${AGENT_NAME}" \
      "${EXTRA_ARGS[@]}" \
      --overwrite

    # symlink for HF cache discovery (mirrors slurm_convert_robofactory_wristcam.sh)
    HF_CACHE_DIR=/iris/u/mikulrai/.cache/huggingface/lerobot
    mkdir -p "${HF_CACHE_DIR}"
    ln -sfn "${OUT}" "${HF_CACHE_DIR}/${REPO_ID}"

    {
      echo "done_utc=$(date -u +%FT%TZ)"
      echo "task=${TASK}"; echo "cam=${CAM}"; echo "kind=${KIND}"
      echo "repo_id=${REPO_ID}"; echo "h5=${H5}"
      echo "slurm_jobid=${SLURM_JOB_ID}"
      echo "host=$(hostname)"
    } > "${DONE}"
    echo "[conv] DONE  ${OUT}"
    ;;

  *)
    echo "[conv] ERROR  unknown KIND=${KIND}"; exit 1 ;;
esac
