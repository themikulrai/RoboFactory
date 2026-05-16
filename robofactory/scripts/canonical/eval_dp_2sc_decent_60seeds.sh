#!/bin/bash
#SBATCH --job-name=eval_dp_2sc_decent
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/iris/u/mikulrai/logs/eval_2026-05-11/eval_dp_2sc_%x_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/eval_2026-05-11/eval_dp_2sc_%x_%j.err

# Parameterized 2SC decent DP eval (60-seed TABLE protocol by default).
# Env vars (required):
#   CAM_FAMILY      : 'workspace' or 'wristcam' — all other knobs are derived from this
# Env vars (optional):
#   MAX_STEPS       : default 250  (was 400 — 250 trades small SR-bias for ~halved eval time on low-SR tasks)
#   SIM_BACKEND     : default 'gpu' (use SAPIEN GPU rendering; falls back to cpu in driver if unsupported)
#   CHECKPOINT_NUM  : default 300
#   DATA_NUM        : default 150
#   WANDB_PROJECT   : default 2SC-DP
#   CKPT_SUFFIX_OVERRIDE : default empty (uses ${CAM_FAMILY}_decent). Set e.g. 'workspace_noglobal5' for ablation ckpts.
#   WITH_GLOBAL          : default true (pass --include-global). Set false for no-global ablation evals.
#   SEEDS_LIST           : default 60-seed protocol. Colon-separated override for custom subset (e.g. "1013:1064:1065:1096:1113"). Use ':' not ',' (sbatch --export treats commas as var separators).
#   WANDB_NAME_SUFFIX    : default ''. Appended to auto-derived name. Avoid spaces; use underscores.
#   ROBOT_TYPE_OVERRIDE  : default '' (auto from CAM_FAMILY). Set a single robot uid (e.g. 'panda_wristcam_multi') to override BOTH arms. Use single-value form to dodge sbatch --export comma-parsing.
#   GRIPPER_SNAP         : default 'false'. Set 'true' to enable np.sign() snap on dim-7 (gripper) — diagnostic for gripper-mode-averaging failure.

set -euxo pipefail
mkdir -p /iris/u/mikulrai/logs/eval_2026-05-11

: "${CAM_FAMILY:?CAM_FAMILY env var required (workspace|wristcam)}"
case "$CAM_FAMILY" in
    workspace)
        CKPT_SUFFIX="workspace_decent"
        ROBOT_UIDS="panda,panda"
        WANDB_NAME="Eval 2SC WS DP Decent"
        ;;
    wristcam)
        CKPT_SUFFIX="wristcam_decent"
        ROBOT_UIDS="panda_wristcam_multi,panda_wristcam_multi"
        WANDB_NAME="Eval 2SC WC DP Decent"
        ;;
    *) echo "Unknown CAM_FAMILY=$CAM_FAMILY"; exit 1 ;;
esac
MAX_STEPS="${MAX_STEPS:-250}"
SIM_BACKEND="${SIM_BACKEND:-gpu}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-300}"
DATA_NUM="${DATA_NUM:-150}"
WANDB_PROJECT="${WANDB_PROJECT:-2SC-DP}"
WITH_GLOBAL="${WITH_GLOBAL:-true}"
CKPT_SUFFIX_FINAL="${CKPT_SUFFIX_OVERRIDE:-$CKPT_SUFFIX}"

DEFAULT_SEEDS_LIST="10000:10001:10002:10003:10004:10005:10006:10007:10008:10009:10010:10011:10012:10013:10014:10015:10016:10017:10018:10019:10020:10021:10022:10023:10024:10025:10026:10027:10028:10029:1000:1001:1002:1003:1004:1005:1006:1007:1008:1009:1010:1011:1012:1013:1014:1015:1016:1017:1018:1019:1020:1021:1022:1023:1024:1025:1026:1027:1028:1029"
SEEDS_LIST="${SEEDS_LIST:-$DEFAULT_SEEDS_LIST}"
SEEDS=$(echo "$SEEDS_LIST" | tr ':' ' ')

INCLUDE_GLOBAL_FLAG="--include-global"
if [ "$WITH_GLOBAL" = "false" ]; then
    INCLUDE_GLOBAL_FLAG="--no-include-global"
fi

# Allow forcing a robot embodiment (e.g., to match h5 generation robot)
if [ -n "${ROBOT_TYPE_OVERRIDE:-}" ]; then
    ROBOT_UIDS="${ROBOT_TYPE_OVERRIDE},${ROBOT_TYPE_OVERRIDE}"
fi

GRIPPER_SNAP_FLAG=""
if [ "${GRIPPER_SNAP:-false}" = "true" ]; then
    GRIPPER_SNAP_FLAG="--gripper-snap"
fi

WANDB_NAME="${WANDB_NAME} ${WANDB_NAME_SUFFIX:-}"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "CAM_FAMILY=$CAM_FAMILY CKPT_SUFFIX=$CKPT_SUFFIX MAX_STEPS=$MAX_STEPS SIM_BACKEND=$SIM_BACKEND"

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory
export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /iris/u/mikulrai/.wandb_api_key 2>/dev/null)}"
export HYDRA_FULL_ERROR=1
cd /iris/u/mikulrai/projects/RoboFactory/robofactory

for id in 0 1; do
  ckpt="/iris/u/mikulrai/checkpoints/RoboFactory/TwoRobotsStackCube-rf_agent${id}_${CKPT_SUFFIX_FINAL}_${DATA_NUM}/${CHECKPOINT_NUM}.ckpt"
  [ -f "$ckpt" ] || { echo "MISSING ckpt: $ckpt"; exit 1; }
done

python -u ./policy/Diffusion-Policy/eval_multi_dp.py \
  --config=configs/table/two_robots_stack_cube.yaml \
  --robot-uids "$ROBOT_UIDS" \
  --data-num="$DATA_NUM" \
  --checkpoint-num="$CHECKPOINT_NUM" \
  --ckpt-suffix="$CKPT_SUFFIX_FINAL" \
  --obs-cam-family="$CAM_FAMILY" \
  $INCLUDE_GLOBAL_FLAG \
  --img-height=224 \
  --img-width=224 \
  -o rgb -b "$SIM_BACKEND" -n 1 \
  --render-mode=sensors \
  -s $SEEDS \
  --quiet \
  --max-steps="$MAX_STEPS" \
  $GRIPPER_SNAP_FLAG \
  --wandb \
  --wandb-project="$WANDB_PROJECT" \
  --wandb-name="$WANDB_NAME" \
  --wandb-tags="eval,2sc,decent,${CAM_FAMILY},encoder-in1k,${CKPT_SUFFIX_FINAL},seedfix-v1,max${MAX_STEPS},sim-${SIM_BACKEND},with_global_${WITH_GLOBAL}"

echo "Eval done at $(date)"
