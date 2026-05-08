#!/usr/bin/env bash
# Bash mirror of robofactory/utils/paths.py — keep them in sync.
# Source this from SLURM templates or interactive shells:
#   source /iris/u/mikulrai/projects/RoboFactory/robofactory/utils/paths.sh

USER_HOME=/iris/u/mikulrai
HOME_="$USER_HOME"  # avoid clobbering shell builtin $HOME; new code can prefer this
RUN_ROOT="$USER_HOME/runs"
DEBUG_OUTPUT_ROOT="$USER_HOME/debug_output"
MANIFEST_PATH="$RUN_ROOT/manifest.csv"

CHECKPOINT_ROOT="$USER_HOME/checkpoints"
LOGS_ROOT="$USER_HOME/logs"
DATA_ROOT="$USER_HOME/data"
SCRATCH_ROOT="$USER_HOME/scratch"

CALIB_DIR="$RUN_ROOT/calibration"
CKPT_INDEX_PATH="$RUN_ROOT/ckpt_index.jsonl"

# Usage: debug_output_dir <script-stem> [YYYYMMDD]
debug_output_dir() {
    local stem="$1"
    local date_str="${2:-$(date +%Y%m%d)}"
    echo "$DEBUG_OUTPUT_ROOT/${stem}_${date_str}"
}
