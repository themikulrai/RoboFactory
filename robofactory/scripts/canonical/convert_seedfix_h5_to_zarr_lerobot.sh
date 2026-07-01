#!/bin/bash
#SBATCH --job-name=conv_seedfix_h5
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/mikulrai/logs/conversion_post_seedfix/conv_%A_%a.out
#SBATCH --error=/iris/u/mikulrai/logs/conversion_post_seedfix/conv_%A_%a.err
#
# Convert post-seedfix H5 demonstrations into:
#   (a) zarr        (Diffusion Policy training)        via parse_h5_to_zarr_unified.py
#   (b) parquet     (Pi0.5 LeRobot fine-tuning)       via convert_robofactory_data_to_lerobot.py
#
# IMPORTANT - DATA SOURCE NOTE
# ----------------------------
# The "workspace" H5s under hf_download_post_seedfix_workspace/<Task>-rf/motionplanning/
# contain ONLY action arrays (robofactory.planner.run did not record observations).
# They cannot be converted directly.
#
# The "wristcam" H5s under hf_download_post_seedfix/<Task>/<Task>.h5 were recorded with
# the panda_wristcam_multi agent and contain BOTH camera families in one file:
#   - workspace cams:  obs/sensor_data/head_camera_agent{i}/rgb  (multi-arm)
#                      obs/sensor_data/head_camera/rgb           (single-arm fallback)
#   - wristcam      :  obs/sensor_data/hand_camera_{i}/rgb       (multi-arm)
#                      obs/sensor_data/hand_camera/rgb           (single-arm fallback)
# So every (task, cam_family) cell is driven off the single WC H5.
#
# Array index = (TASK, CAM, KIND) tuple where KIND in {cent, decent<i>, lerobot}.
# PM has N=1 so it only gets {cent, lerobot} per cam family (cent == decent).
#
# Idempotent invariants:
#   - Refuse if input H5 missing / placeholder / too-recent mtime (avoid mid-write reads).
#   - Refuse if output exists without sentinel (no silent overwrite of partial data).
#   - Write a .done_sentinel only after successful conversion.
#
# Usage:
#   sbatch --array=<csv> convert_seedfix_h5_to_zarr_lerobot.sh
#   bash convert_seedfix_h5_to_zarr_lerobot.sh --list           # dump cell table
#   bash convert_seedfix_h5_to_zarr_lerobot.sh --ready-indices  # csv of READY indices

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

H5_ROOT=/iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix
ZARR_OUT_ROOT=/iris/u/mikulrai/data/RoboFactory/zarr_data
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
LEROBOT_OUT_ROOT="${RF_LEROBOT_HOME:?RF_LEROBOT_HOME unset — source ~/.config/dataroots.sh}"
CAM_MAP_DIR=/iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings

mkdir -p "${ZARR_OUT_ROOT}" "${LEROBOT_OUT_ROOT}" /iris/u/mikulrai/logs/conversion_post_seedfix

# --- per-task metadata --------------------------------------------------------
declare -A N_AGENTS=(
  [PickMeat]=1
  [TwoRobotsStackCube]=2
  [ThreeRobotsStackCube]=3
  [LongPipelineDelivery]=4
  [LiftBarrier]=2
  [PlaceFood]=2
)
# task_slug = camera_mapping JSON stem + LeRobot repo_id stem
declare -A SLUG=(
  [PickMeat]=pick_meat
  [TwoRobotsStackCube]=two_robots_stack_cube
  [ThreeRobotsStackCube]=three_robots_stack_cube
  [LongPipelineDelivery]=long_pipeline_delivery
  [LiftBarrier]=lift_barrier
  [PlaceFood]=place_food
)
TASKS_ORDER=(PickMeat TwoRobotsStackCube ThreeRobotsStackCube LongPipelineDelivery LiftBarrier PlaceFood)

# --- build cell list ----------------------------------------------------------
# (TASK:CAM:KIND) where KIND in {cent, decent<i>, lerobot}.
# Order: per task, [workspace then wristcam]; within cam: cent, decent0..decent{N-1}, lerobot
# PM (N=1) skips decent (cent already serves the single arm).
CELLS=()
for TASK in "${TASKS_ORDER[@]}"; do
  N=${N_AGENTS[$TASK]}
  for CAM in workspace wristcam; do
    CELLS+=("${TASK}:${CAM}:cent")
    if (( N > 1 )); then
      for (( i=0; i<N; i++ )); do
        CELLS+=("${TASK}:${CAM}:decent${i}")
      done
    fi
    CELLS+=("${TASK}:${CAM}:lerobot")
  done
done
NCELLS=${#CELLS[@]}

# --- helper: classify readiness for a cell ------------------------------------
# echoes "READY" or "IN_FLIGHT_OR_MISSING"
classify_cell() {
  local TASK=$1
  local H5="${H5_ROOT}/${TASK}/${TASK}.h5"
  if [[ -f "${H5}" ]] && (( $(stat -c '%s' "${H5}") > 1048576 )); then
    echo READY
  else
    echo IN_FLIGHT_OR_MISSING
  fi
}

# --- listing modes ------------------------------------------------------------
if [[ "${1:-}" == "--list" ]]; then
  echo "Total cells: ${NCELLS}"
  printf "%4s  %-22s  %-10s  %-12s  %-20s\n" idx task cam kind status
  for idx in "${!CELLS[@]}"; do
    IFS=':' read -r TASK CAM KIND <<< "${CELLS[$idx]}"
    state=$(classify_cell "$TASK")
    printf "%4d  %-22s  %-10s  %-12s  %-20s\n" "$idx" "$TASK" "$CAM" "$KIND" "$state"
  done
  exit 0
fi
if [[ "${1:-}" == "--ready-indices" ]]; then
  ready=()
  for idx in "${!CELLS[@]}"; do
    IFS=':' read -r TASK CAM KIND <<< "${CELLS[$idx]}"
    if [[ "$(classify_cell "$TASK")" == "READY" ]]; then
      ready+=("$idx")
    fi
  done
  ( IFS=, ; echo "${ready[*]}" )
  exit 0
fi
if [[ "${1:-}" == "--inflight-indices" ]]; then
  inflight=()
  for idx in "${!CELLS[@]}"; do
    IFS=':' read -r TASK CAM KIND <<< "${CELLS[$idx]}"
    if [[ "$(classify_cell "$TASK")" != "READY" ]]; then
      inflight+=("$idx")
    fi
  done
  ( IFS=, ; echo "${inflight[*]}" )
  exit 0
fi

# --- dispatch one cell --------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID unset; submit via sbatch with --array=...  (or run --list)" >&2
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
JSON="${H5%.h5}.json"

echo "[conv] array_id=${IDX}  task=${TASK}  cam=${CAM}  kind=${KIND}  n_agents=${N}"
echo "[conv] h5=${H5}"
echo "[conv] host=$(hostname)  date=$(date -u +%FT%TZ)"

# --- gate: input H5 must exist, be real-sized, and not be mid-write -----------
if [[ ! -f "${H5}" ]]; then
  echo "[conv] ERROR  input H5 missing: ${H5}"; exit 1
fi
SZ=$(stat -c '%s' "${H5}")
if (( SZ <= 1048576 )); then
  echo "[conv] ERROR  H5 placeholder-sized (${SZ} bytes): ${H5}"; exit 1
fi
NOW=$(date +%s)
MTIME=$(stat -c '%Y' "${H5}")
AGE=$(( NOW - MTIME ))
if (( AGE < 600 )); then
  echo "[conv] ERROR  H5 mtime within 10 min (${AGE}s) - likely still writing."; exit 1
fi
if [[ ! -f "${JSON}" ]]; then
  echo "[conv] ERROR  missing JSON sidecar: ${JSON}"; exit 1
fi

# --- dispatch by kind ---------------------------------------------------------
case "$KIND" in
  cent|decent*)
    # zarr conversion
    if [[ "$KIND" == "cent" ]]; then
      OUT="${ZARR_OUT_ROOT}/${TASK}-rf_${CAM}_cent_150.zarr"
      # PM has N=1 - the script's _stream_joint hard-codes actions/panda-{i} which
      # does not exist for single-agent H5s. Route N=1 through per_agent (the
      # _action_key fallback covers the flat 'actions' dataset).
      if (( N == 1 )); then
        MODE_ARGS=(--mode per_agent --agent-id 0)
      else
        MODE_ARGS=(--mode joint --n-agents "$N")
      fi
    else
      AGENT_I=${KIND#decent}
      OUT="${ZARR_OUT_ROOT}/${TASK}-rf_${CAM}_decent_agent${AGENT_I}_150.zarr"
      MODE_ARGS=(--mode per_agent --agent-id "$AGENT_I")
    fi
    DONE="${OUT}.done_sentinel"

    # Fail fast if output exists (with or without sentinel) - never silently overwrite
    if [[ -e "${OUT}" || -e "${DONE}" ]]; then
      echo "[conv] ERROR  output already exists (refuse to overwrite): ${OUT}"
      echo "[conv]   remove manually or move aside to .trash if intentional."
      exit 1
    fi

    echo "[conv] writing zarr -> ${OUT}  (include-global=1, matches Pi0.5 LeRobot base cam)"
    "${RF_PY}" "${ZARR_SCRIPT}" \
      --h5-path "${H5}" \
      --out-zarr "${OUT}" \
      --camera-family "${CAM}" \
      "${MODE_ARGS[@]}" \
      --include-global \
      --resize 224
    {
      echo "done_utc=$(date -u +%FT%TZ)"
      echo "task=${TASK}"; echo "cam=${CAM}"; echo "kind=${KIND}"
      echo "h5=${H5}"
      echo "slurm_jobid=${SLURM_JOB_ID:-local}"
      echo "host=$(hostname)"
    } > "${DONE}"
    echo "[conv] DONE  ${OUT}"
    ;;

  lerobot)
    REPO_ID="robofactory_${SLUG_LC}_${CAM}_seedfix_v1"
    OUT="${LEROBOT_OUT_ROOT}/${REPO_ID}"
    DONE="${OUT}.done_sentinel"

    if [[ -e "${OUT}" || -e "${DONE}" ]]; then
      echo "[conv] ERROR  output already exists (refuse to overwrite): ${OUT}"
      echo "[conv]   remove manually or move aside to .trash if intentional."
      exit 1
    fi

    # Camera-mapping selection
    case "$CAM" in
      wristcam)  CAM_MAP="${CAM_MAP_DIR}/${SLUG_LC}_wristcam.json" ;;
      workspace) CAM_MAP="${CAM_MAP_DIR}/${SLUG_LC}.json" ;;
      *) echo "[conv] ERROR  unknown CAM=${CAM}"; exit 1 ;;
    esac
    if [[ ! -f "${CAM_MAP}" ]]; then
      echo "[conv] ERROR  missing camera_mapping JSON: ${CAM_MAP}"; exit 1
    fi
    # Every task gets its own per-task prompts (else converter falls back to the
    # hardcoded TSC "stack the three cubes" DEFAULT_PROMPTS -> wrong language baked in).
    PROMPTS="${CAM_MAP_DIR}/${SLUG_LC}_prompts.json"
    if [[ ! -f "${PROMPTS}" ]]; then
      echo "[conv] ERROR  missing prompts JSON: ${PROMPTS}"
      echo "[conv]   refusing to fall back to converter DEFAULT_PROMPTS (wrong task language)."
      exit 1
    fi
    EXTRA_ARGS=(--camera_mapping "${CAM_MAP}" --prompts_path "${PROMPTS}")
    AGENT_NAME=panda_wristcam_multi

    echo "[conv] writing LeRobot repo=${REPO_ID}  cam_map=${CAM_MAP}"
    "${OPENPI_PY}" "${LEROBOT_SCRIPT}" \
      --h5_path "${H5}" \
      --out_root "${LEROBOT_OUT_ROOT}" \
      --repo_id "${REPO_ID}" \
      --num_arms "${N}" \
      --agent_name "${AGENT_NAME}" \
      "${EXTRA_ARGS[@]}"

    # Symlink for HF cache discovery
    HF_CACHE_DIR=/iris/u/mikulrai/.cache/huggingface/lerobot
    mkdir -p "${HF_CACHE_DIR}"
    ln -sfn "${OUT}" "${HF_CACHE_DIR}/${REPO_ID}"

    {
      echo "done_utc=$(date -u +%FT%TZ)"
      echo "task=${TASK}"; echo "cam=${CAM}"; echo "kind=${KIND}"
      echo "repo_id=${REPO_ID}"; echo "h5=${H5}"
      echo "slurm_jobid=${SLURM_JOB_ID:-local}"
      echo "host=$(hostname)"
    } > "${DONE}"
    echo "[conv] DONE  ${OUT}"
    ;;

  *)
    echo "[conv] ERROR  unknown KIND=${KIND}"; exit 1 ;;
esac
