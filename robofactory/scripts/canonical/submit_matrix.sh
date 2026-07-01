#!/bin/bash
# submit_matrix.sh — single-entry-point matrix submitter for DP and Pi0.5
# training runs across (task, scheme, cam, arm) cells.
#
# Replaces the copy-paste-per-combo retrain_dp_*.sh scripts. Each invocation:
#   1. validates a (task, scheme, cam[, arm]) tuple,
#   2. resolves derived values (env_id, scene yaml, zarr path, exp_name, ...),
#   3. runs pre-flight assertions that catch known foot-guns
#      (missing zarr, workspace_cameras_identical_bug on ws-decent, etc.),
#   4. emits a one-shot SLURM script to a versioned temp dir,
#   5. submits via submit_with_preflights.sh (DP) or sbatch (Pi0.5).
#
# Usage:
#   bash scripts/canonical/submit_matrix.sh \
#       --policy {dp|pi05} \
#       --task   {pm|2sc|tsc|lp|lb|pf} \
#       --scheme {cent|decent} \
#       --cam    {ws|wc} \
#       [--arm N]                     # required for scheme=decent (0-indexed)
#       [--epochs N]                  # DP epochs (default per-task)
#       [--num-train-steps N]         # Pi0.5 steps (default per-task)
#       [--batch-size N]              # default per-task
#       [--rgb-weights {null,IMAGENET1K_V1}]   # DP-only
#       [--cluster {iris-hi|iris|orion}]
#       [--gpu-type {a40|a5000|a6000}]
#       [--time HH:MM:SS]
#       [--allow-broken-symlink]      # bypass legacy-symlink-broken check (Pi0.5)
#       [--allow-duplicate-name]      # bypass wandb-name collision check
#       [--no-preflights]             # DP only: skip submit_with_preflights chain
#
# Environment:
#   DRYRUN=1   echo the emitted sbatch script + train command, do not submit.
#              Use this to verify wiring before committing real GPU time.
#
# Examples:
#   # 2SC decent workspace, arm 0, DP, on iris-hi
#   bash scripts/canonical/submit_matrix.sh --policy dp \
#       --task 2sc --scheme decent --cam ws --arm 0
#
#   # TSC cent wristcam, Pi0.5
#   bash scripts/canonical/submit_matrix.sh --policy pi05 \
#       --task tsc --scheme cent --cam wc
#
#   # Dryrun (no submit)
#   DRYRUN=1 bash scripts/canonical/submit_matrix.sh --policy dp \
#       --task tsc --scheme decent --cam ws --arm 0

set -euo pipefail

# ----- Defaults
POLICY=""
TASK=""
SCHEME=""
CAM=""
ARM=""
EPOCHS=""
NUM_TRAIN_STEPS=""
BATCH_SIZE=""
RGB_WEIGHTS="IMAGENET1K_V1"
CLUSTER="iris-hi"
GPU_TYPE="a40"
TIME_LIMIT="24:00:00"
ALLOW_BROKEN_SYMLINK=0
ALLOW_DUPLICATE_NAME=0
NO_PREFLIGHTS=0
DRYRUN="${DRYRUN:-0}"

ROBOFACTORY_ROOT="/iris/u/mikulrai/projects/RoboFactory/robofactory"
OPENPI_ROOT="/iris/u/mikulrai/projects/openpi"
OPENPI_PY="${OPENPI_PY:-$OPENPI_ROOT/.venv/bin/python}"
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
ZARR_DIR="${RF_DATA_ROOT:?RF_DATA_ROOT unset — source ~/.config/dataroots.sh}/zarr_data"
HF_LEROBOT_DIR="${HF_HOME:-/iris/u/mikulrai/.cache/huggingface}/lerobot"
LOG_DIR="/iris/u/mikulrai/logs/matrix"
TEMP_DIR="/iris/u/mikulrai/runs/submit_matrix"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_key}"

# ----- Per-task spec (parallel to _RF_TASKS in openpi config.py)
#   env_id: SAPIEN env name passed as task.name=
#   scene_config: configs/table/*.yaml file
#   n_arms: 1 (PM), 2 (2SC/LB/PF), 3 (TSC), 4 (LP)
declare -A TASK_ENV_ID=(
    [pm]="PickMeat-rf"
    [2sc]="TwoRobotsStackCube-rf"
    [tsc]="ThreeRobotsStackCube-rf"
    [lp]="LongPipelineDelivery-rf"
    [lb]="LiftBarrier-rf"
    [pf]="PlaceFood-rf"
)
declare -A TASK_SCENE=(
    [pm]="pick_meat.yaml"
    [2sc]="two_robots_stack_cube.yaml"
    [tsc]="three_robots_stack_cube.yaml"
    [lp]="long_pipeline_delivery.yaml"
    [lb]="lift_barrier.yaml"
    [pf]="place_food.yaml"
)
declare -A TASK_N_ARMS=(
    [pm]=1 [2sc]=2 [tsc]=3 [lp]=4 [lb]=2 [pf]=2
)
declare -A TASK_DP_EPOCHS=(
    [pm]=300 [2sc]=300 [tsc]=300 [lp]=400 [lb]=300 [pf]=300
)
declare -A TASK_PI05_STEPS=(
    [pm]=20000 [2sc]=20000 [tsc]=20000 [lp]=30000 [lb]=30000 [pf]=30000
)
declare -A TASK_PI05_BATCH=(
    [pm]=16 [2sc]=16 [tsc]=16 [lp]=8 [lb]=8 [pf]=8
)
# Zarr naming template per task. Tokens: {ENV} {CAM} {SCHEME}.
# Default (seedfix-v1): "{ENV}_{CAM}_{SCHEME}_150.zarr".
# LP wasn't migrated to seedfix-v1; uses its own pattern: "LongPipelineDelivery_{CAM}_{joint|agent<i>}.zarr".
declare -A TASK_ZARR_TPL=(
    [pm]="{ENV}_{CAM}_{SCHEME}_150.zarr"
    [2sc]="{ENV}_{CAM}_{SCHEME}_150.zarr"
    [tsc]="{ENV}_{CAM}_{SCHEME}_150.zarr"
    [lp]="LongPipelineDelivery_{CAM}_{LP_SCHEME}.zarr"
    [lb]="{ENV}_{CAM}_{SCHEME}_150.zarr"
    [pf]="{ENV}_{CAM}_{SCHEME}_150.zarr"
)

# ----- Arg parse
usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# //;s/^#//' >&2; exit 2; }
die()   { echo "submit_matrix: $*" >&2; exit 1; }
warn()  { echo "submit_matrix: WARN: $*" >&2; }
info()  { echo "submit_matrix: $*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy)              POLICY="$2"; shift 2 ;;
        --task)                TASK="$2"; shift 2 ;;
        --scheme)              SCHEME="$2"; shift 2 ;;
        --cam)                 CAM="$2"; shift 2 ;;
        --arm)                 ARM="$2"; shift 2 ;;
        --epochs)              EPOCHS="$2"; shift 2 ;;
        --num-train-steps)     NUM_TRAIN_STEPS="$2"; shift 2 ;;
        --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
        --rgb-weights)         RGB_WEIGHTS="$2"; shift 2 ;;
        --cluster)             CLUSTER="$2"; shift 2 ;;
        --gpu-type)            GPU_TYPE="$2"; shift 2 ;;
        --time)                TIME_LIMIT="$2"; shift 2 ;;
        --allow-broken-symlink) ALLOW_BROKEN_SYMLINK=1; shift ;;
        --allow-duplicate-name) ALLOW_DUPLICATE_NAME=1; shift ;;
        --no-preflights)       NO_PREFLIGHTS=1; shift ;;
        -h|--help)             usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

# ----- Validate args
[[ -z "$POLICY" ]] && die "missing --policy {dp|pi05}"
[[ -z "$TASK"   ]] && die "missing --task"
[[ -z "$SCHEME" ]] && die "missing --scheme {cent|decent}"
[[ -z "$CAM"    ]] && die "missing --cam {ws|wc}"

case "$POLICY" in dp|pi05) ;; *) die "--policy must be 'dp' or 'pi05', got '$POLICY'";; esac
case "$SCHEME" in cent|decent) ;; *) die "--scheme must be 'cent' or 'decent', got '$SCHEME'";; esac
case "$CAM"    in ws|wc) ;; *) die "--cam must be 'ws' or 'wc', got '$CAM'";; esac
[[ -z "${TASK_ENV_ID[$TASK]:-}" ]] && die "--task '$TASK' unknown; valid: ${!TASK_ENV_ID[*]}"

N_ARMS="${TASK_N_ARMS[$TASK]}"
if [[ "$SCHEME" == "decent" ]]; then
    [[ -z "$ARM" ]] && die "scheme=decent requires --arm <i>"
    [[ "$ARM" -ge "$N_ARMS" ]] && die "--arm $ARM out of range; task=$TASK has n_arms=$N_ARMS"
    [[ "$TASK" == "pm" ]] && die "PM is single-arm; scheme=decent is meaningless"
else
    [[ -n "$ARM" ]] && die "scheme=cent does not take --arm (got $ARM)"
fi

# ----- Resolve derived values
ENV_ID="${TASK_ENV_ID[$TASK]}"
SCENE_CONFIG_PATH="$ROBOFACTORY_ROOT/configs/table/${TASK_SCENE[$TASK]}"
SCENE_CONFIG_REL="configs/table/${TASK_SCENE[$TASK]}"

# Zarr path: resolve via TASK_ZARR_TPL with token substitution.
ZARR_CAM=$([[ "$CAM" == "ws" ]] && echo "workspace" || echo "wristcam")
if [[ "$SCHEME" == "cent" ]]; then
    ZARR_SCHEME="cent"
    LP_SCHEME="joint"
else
    ZARR_SCHEME="decent_agent$ARM"
    LP_SCHEME="agent$ARM"
fi
ZARR_FILE="${TASK_ZARR_TPL[$TASK]}"
ZARR_FILE="${ZARR_FILE//\{ENV\}/$ENV_ID}"
ZARR_FILE="${ZARR_FILE//\{CAM\}/$ZARR_CAM}"
ZARR_FILE="${ZARR_FILE//\{SCHEME\}/$ZARR_SCHEME}"
ZARR_FILE="${ZARR_FILE//\{LP_SCHEME\}/$LP_SCHEME}"
ZARR_PATH="$ZARR_DIR/$ZARR_FILE"

# DP Hydra task config
if [[ "$CAM" == "ws" ]]; then
    TASK_CONFIG="default_task"
else
    TASK_CONFIG="default_task_wristcam"
fi
# DP --config-name selection: cent multi-arm uses joint_dp.yaml; everything else robot_dp.yaml.
if [[ "$SCHEME" == "cent" && "$N_ARMS" -gt 1 ]]; then
    CONFIG_NAME="joint_dp.yaml"
    DP_TASK_OVERRIDE="task=joint_task_${N_ARMS}arm"
else
    CONFIG_NAME="robot_dp.yaml"
    DP_TASK_OVERRIDE="task=$TASK_CONFIG"
fi

# Exp / wandb names. Decent gets _arm<i>, cent doesn't.
SUFFIX=""
[[ "$SCHEME" == "decent" ]] && SUFFIX="_arm$ARM"
EXP_NAME="${TASK}-${CAM}-${SCHEME}${SUFFIX}-${POLICY}-v0"
# Wandb run name convention from dp_wandb_run_naming.md:
WANDB_NAME_PREFIX="Train ${TASK^^} ${CAM^^} $([[ "$POLICY" == "dp" ]] && echo DP || echo Pi0.5)"
if [[ "$SCHEME" == "decent" ]]; then
    WANDB_NAME="$WANDB_NAME_PREFIX Decent A$ARM"
else
    WANDB_NAME="$WANDB_NAME_PREFIX"
fi
JOB_NAME="${POLICY}_${TASK}_${CAM}_${SCHEME}${SUFFIX}"

# Pi0.5 canonical config name (matches make_robofactory_pi05_config naming)
PI05_CFG="pi05_robofactory_${TASK}_${CAM}_${SCHEME}${SUFFIX}"

# ----- Pre-flight assertions

# (1) Zarr exists (DP only — Pi0.5 reads LeRobot HF cache)
if [[ "$POLICY" == "dp" ]]; then
    if [[ ! -e "$ZARR_PATH/.zgroup" && ! -e "$ZARR_PATH/group.json" ]]; then
        die "zarr not found or empty: $ZARR_PATH (run regen_all_data_post_seedfix.sh first)"
    fi
    SENTINEL="${ZARR_PATH}.done_sentinel"
    [[ ! -f "$SENTINEL" ]] && warn "no .done_sentinel for $ZARR_PATH — datagen may be partial"
fi

# (2) Scene yaml exists; workspace + decent on multi-arm hits the identical-cam bug
if [[ ! -f "$SCENE_CONFIG_PATH" ]]; then
    die "scene config not found: $SCENE_CONFIG_PATH"
fi
if [[ "$CAM" == "ws" && "$SCHEME" == "decent" && "$N_ARMS" -gt 1 ]]; then
    # Bug: per-arm decent ws training needs distinct head_cameras per arm.
    # Memory project_workspace_cameras_identical_bug.md — scene yamls give all
    # head_camera_agentN the same `params: [[...], [...]]`, so per-arm decent has
    # no visual signal → 0/60 SR.
    PARAMS_LINES=$(awk '
        /uid: head_camera_agent/ {found=1; next}
        found && /params:/ {print; found=0}
    ' "$SCENE_CONFIG_PATH")
    N_TOTAL_AGENT_CAMS=$(echo "$PARAMS_LINES" | grep -c "params:" || true)
    N_UNIQUE_AGENT_CAMS=$(echo "$PARAMS_LINES" | sort -u | grep -c "params:" || true)
    if [[ "$N_TOTAL_AGENT_CAMS" -gt 1 && "$N_UNIQUE_AGENT_CAMS" -le 1 ]]; then
        warn "scene yaml has $N_TOTAL_AGENT_CAMS head_camera_agent* entries but only $N_UNIQUE_AGENT_CAMS unique pose →"
        warn "this is the workspace-cameras-identical bug. Per-arm decent will be 0/60 SR."
        warn "fix $SCENE_CONFIG_REL (give each head_camera_agentN its own params) OR re-run with --cam wc"
        [[ "$ALLOW_BROKEN_SYMLINK" -eq 0 ]] && die "aborting; pass --allow-broken-symlink to override"
    fi
fi

# (3) Pi0.5 legacy-symlink check (only if legacy alias path will be used; we
# use canonical names so this is mostly defensive). For canonical names, the
# LeRobot dataset should be at $HF_LEROBOT_DIR/<seedfix_v1_name>.
if [[ "$POLICY" == "pi05" ]]; then
    REPO_ID_WS_PM="robofactory_pick_meat_workspace_seedfix_v1"
    REPO_ID_WC_PM="robofactory_pick_meat_wristcam_seedfix_v1"
    REPO_ID_WS_2SC="robofactory_two_robots_stack_cube_workspace_seedfix_v1"
    REPO_ID_WC_2SC="robofactory_two_robots_stack_cube_wristcam_seedfix_v1"
    REPO_ID_WS_TSC="robofactory_three_robots_stack_cube_workspace_seedfix_v1"
    REPO_ID_WC_TSC="robofactory_three_robots_stack_cube_wristcam_seedfix_v1"
    REPO_ID_WS_LP="robofactory_long_pipeline_delivery_workspace"
    REPO_ID_WC_LP="robofactory_long_pipeline_delivery_wristcam"
    case "${TASK}_${CAM}" in
        pm_ws) REPO="$REPO_ID_WS_PM";;
        pm_wc) REPO="$REPO_ID_WC_PM";;
        2sc_ws) REPO="$REPO_ID_WS_2SC";;
        2sc_wc) REPO="$REPO_ID_WC_2SC";;
        tsc_ws) REPO="$REPO_ID_WS_TSC";;
        tsc_wc) REPO="$REPO_ID_WC_TSC";;
        lp_ws) REPO="$REPO_ID_WS_LP";;
        lp_wc) REPO="$REPO_ID_WC_LP";;
        *) REPO="";;
    esac
    if [[ -n "$REPO" ]]; then
        DATA_PATH="$HF_LEROBOT_DIR/$REPO"
        if [[ ! -e "$DATA_PATH" ]]; then
            warn "LeRobot dataset not found: $DATA_PATH"
            warn "create symlink: ln -s /iris/u/mikulrai/datasets/multi_robot/RoboFactory/lerobot/$REPO $DATA_PATH"
            [[ "$ALLOW_BROKEN_SYMLINK" -eq 0 ]] && die "aborting; pass --allow-broken-symlink to override"
        fi
    fi
fi

# (4) iris-hi quota soft-check (≥6 jobs already queued → suggest switching)
if [[ "$CLUSTER" == "iris-hi" ]] && command -v squeue &>/dev/null; then
    N_RUNNING=$(squeue -u "$USER" -h -p iris-hi 2>/dev/null | wc -l || echo 0)
    if [[ "$N_RUNNING" -ge 6 ]]; then
        warn "iris-hi already has $N_RUNNING jobs (quota=6). Consider --cluster orion."
    fi
fi

# ----- Emit one-shot SLURM script
RUN_ID="${POLICY}_${TASK}_${CAM}_${SCHEME}${SUFFIX}_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$TEMP_DIR/$RUN_ID"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"
SCRIPT_PATH="$SCRIPT_DIR/train.sh"

# Resolve cluster sbatch directives
case "$CLUSTER" in
    iris-hi)
        SBATCH_PARTITION="iris-hi"
        SBATCH_ACCOUNT="iris"
        SBATCH_NICE=""
        ;;
    iris)
        SBATCH_PARTITION="iris"
        SBATCH_ACCOUNT="iris"
        SBATCH_NICE=""
        ;;
    orion)
        SBATCH_PARTITION="orion"
        SBATCH_ACCOUNT="orion"
        SBATCH_NICE="#SBATCH --nice=10000"
        ;;
    *) die "unknown --cluster '$CLUSTER'";;
esac

# Resolve defaults from per-task tables
if [[ "$POLICY" == "dp" ]]; then
    [[ -z "$EPOCHS" ]] && EPOCHS="${TASK_DP_EPOCHS[$TASK]}"
    [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE=64  # DP standard
elif [[ "$POLICY" == "pi05" ]]; then
    [[ -z "$NUM_TRAIN_STEPS" ]] && NUM_TRAIN_STEPS="${TASK_PI05_STEPS[$TASK]}"
    [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE="${TASK_PI05_BATCH[$TASK]}"
fi

# Load wandb key (never commit it inline — memory wandb_online.md says never offline)
if [[ -f "$WANDB_KEY_FILE" ]]; then
    WANDB_API_KEY=$(< "$WANDB_KEY_FILE")
else
    warn "no wandb key at $WANDB_KEY_FILE; copy from CLAUDE.md if training online"
    WANDB_API_KEY=""
fi

# Determine current_agent_id (decent) — keep 0 for cent for hydra parity with existing scripts
CURRENT_AGENT_ID=$([[ "$SCHEME" == "decent" ]] && echo "$ARM" || echo "0")

cat > "$SCRIPT_PATH" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$SBATCH_PARTITION
#SBATCH --account=$SBATCH_ACCOUNT
$SBATCH_NICE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:$GPU_TYPE:1
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err

# Generated by submit_matrix.sh on $(date -Iseconds)
# Matrix cell: policy=$POLICY task=$TASK scheme=$SCHEME cam=$CAM arm=${ARM:-N/A}

set -euxo pipefail
source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh

export HOME=/iris/u/mikulrai
export TORCH_HOME=\$HOME/.cache/torch
mkdir -p \$HOME/.r3m
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="$WANDB_API_KEY"
export CUDA_VISIBLE_DEVICES=0
EOF

if [[ "$POLICY" == "dp" ]]; then
    cat >> "$SCRIPT_PATH" <<EOF
conda activate RoboFactory
cd $ROBOFACTORY_ROOT

# Cheap-suite preflight (config-only checks for control_rate / action / image / camera parity)
PREFLIGHT_RUN_DIR="/iris/u/mikulrai/runs/preflight/\${SLURM_JOB_ID:-local}"
mkdir -p "\$PREFLIGHT_RUN_DIR"
python -m robofactory.scripts.preflight.dump_train_cfg \\
    --task-config $TASK_CONFIG \\
    --scene-config $SCENE_CONFIG_REL \\
    --agent-id $CURRENT_AGENT_ID \\
    --out-train "\$PREFLIGHT_RUN_DIR/train_cheap.yaml" \\
    --out-eval  "\$PREFLIGHT_RUN_DIR/eval_cheap.yaml"
python -m robofactory.scripts.preflight.train_eval_consistency \\
    --train-cfg "\$PREFLIGHT_RUN_DIR/train_cheap.yaml" \\
    --eval-cfg  "\$PREFLIGHT_RUN_DIR/eval_cheap.yaml" \\
    --out       "\$PREFLIGHT_RUN_DIR/preflight_consistency.json" \\
  || { echo "preflight failed; aborting training"; exit 1; }

python ./policy/Diffusion-Policy/train.py \\
  --config-name=$CONFIG_NAME \\
  $DP_TASK_OVERRIDE \\
  task.name=$ENV_ID \\
  task.dataset.zarr_path=$ZARR_PATH \\
  task.dataset.max_train_episodes=150 \\
  current_agent_id=$CURRENT_AGENT_ID \\
  policy.obs_encoder.rgb_model.weights=$RGB_WEIGHTS \\
  training.debug=False \\
  training.resume=False \\
  training.seed=100 \\
  training.device=cuda:0 \\
  training.num_epochs=$EPOCHS \\
  training.rollout_every=10000 \\
  exp_name=$EXP_NAME \\
  logging.mode=online \\
  logging.project=diffusion-robofactory \\
  logging.name="$WANDB_NAME" \\
  'logging.tags=[robot_dp,$TASK_CONFIG,$TASK,$CAM,$SCHEME,matrix-submit]' \\
  dataloader.batch_size=$BATCH_SIZE \\
  val_dataloader.batch_size=$BATCH_SIZE
EOF
else
    # Pi0.5 path
    cat >> "$SCRIPT_PATH" <<EOF
conda activate openpi 2>/dev/null || true   # openpi may use .venv instead
cd $OPENPI_ROOT

CHECKPOINT_DIR="/iris/u/mikulrai/checkpoints/openpi/$PI05_CFG/$EXP_NAME"
mkdir -p "\$(dirname "\$CHECKPOINT_DIR")"

$OPENPI_PY scripts/train.py $PI05_CFG \\
    --$PI05_CFG.name="$EXP_NAME" \\
    --$PI05_CFG.exp_name="$EXP_NAME" \\
    --$PI05_CFG.checkpoint_dir="\$CHECKPOINT_DIR" \\
    --$PI05_CFG.project_name="openpi-robofactory" \\
    --$PI05_CFG.batch_size=$BATCH_SIZE \\
    --$PI05_CFG.num_train_steps=$NUM_TRAIN_STEPS \\
    --$PI05_CFG.wandb_enabled=True
EOF
fi

chmod +x "$SCRIPT_PATH"

# ----- Submit (or dryrun)
info "emitted train script: $SCRIPT_PATH"
info "  policy=$POLICY task=$TASK scheme=$SCHEME cam=$CAM arm=${ARM:-N/A}"
info "  env=$ENV_ID exp=$EXP_NAME"
info "  $(if [[ "$POLICY" == "dp" ]]; then echo "epochs=$EPOCHS zarr=$ZARR_PATH"; else echo "pi05_cfg=$PI05_CFG steps=$NUM_TRAIN_STEPS"; fi)"

if [[ "$DRYRUN" == "1" ]]; then
    info "DRYRUN=1 → not submitting. Script contents:"
    cat "$SCRIPT_PATH"
    exit 0
fi

if [[ "$POLICY" == "dp" && "$NO_PREFLIGHTS" -eq 0 ]]; then
    # Use submit_with_preflights.sh for the heavy-preflight chain.
    bash "$ROBOFACTORY_ROOT/scripts/canonical/submit_with_preflights.sh" \
        --train-launcher "$SCRIPT_PATH" \
        --dataset "$ZARR_PATH" \
        --scene-config "$SCENE_CONFIG_REL" \
        --needs-overfit \
        --of-task-config "$TASK_CONFIG" \
        --of-task-name "$ENV_ID" \
        --of-zarr "$ZARR_PATH" \
        --of-agent-id "$CURRENT_AGENT_ID" \
        --of-exp-name "$EXP_NAME" \
        --of-rgb-weights "$RGB_WEIGHTS" \
        --of-config-name "$CONFIG_NAME"
else
    # Direct sbatch (Pi0.5 or DP with --no-preflights).
    if ! command -v sbatch &>/dev/null; then
        die "sbatch not on PATH (login node has no SLURM binaries — submit from a node that has them or via iris-mcp)"
    fi
    sbatch "$SCRIPT_PATH"
fi
