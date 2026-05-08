#!/bin/bash
# Plan v2 C2 S4 — prune Pi0.5 checkpoint trash dirs older than 14 days.
#
# Pi0.5's `--overwrite` was patched to `mv <dir> <dir>.trash_<unix_ts>_<pid>`
# (src/openpi/training/checkpoints.py:38) instead of `shutil.rmtree`.
# This script reaps those trash dirs once they age out — recovery window for
# accidental clobber, then space reclaimed.
#
# Run hourly via crontab; see install_prune_trash_cron.sh for setup.
# Idempotent; safe to re-run; --dry-run prints actions without executing.

set -euo pipefail

ROOTS=(
    /iris/u/mikulrai/checkpoints/openpi
    /iris/u/mikulrai/checkpoints/RoboFactory
)
MAX_AGE_DAYS=14
LOG=/iris/u/mikulrai/logs/prune_trash.log
DRY_RUN=0

usage() {
    echo "Usage: $0 [--dry-run] [--max-age-days N] [--root PATH]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --max-age-days) MAX_AGE_DAYS="$2"; shift 2 ;;
        --root) ROOTS=("$2"); shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

mkdir -p "$(dirname "$LOG")"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

NOW=$(date +%s)
THRESHOLD=$((MAX_AGE_DAYS * 86400))

reaped=0
kept=0
errors=0

for ROOT in "${ROOTS[@]}"; do
    [ -d "$ROOT" ] || { echo "$(ts) skip: $ROOT not a directory" >> "$LOG"; continue; }

    # Pi0.5 trash dirs are named `<orig>.trash_<unix_ts>_<pid>`. Extract the
    # ts from the dir name (not mtime — mtime changes on rename, ts encodes
    # creation). Globbing depth=4 covers `<root>/<config>/<exp>.trash_*` and
    # `<root>/<config>/<exp>/<wandb_id>.trash_*` layouts.
    while IFS= read -r -d '' dir; do
        bn=$(basename "$dir")
        # Match `*.trash_<digits>_<digits>`. Refuse anything else (paranoid).
        if [[ ! "$bn" =~ \.trash_([0-9]+)_([0-9]+)$ ]]; then
            echo "$(ts) WARN unexpected name, skipping: $dir" >> "$LOG"
            errors=$((errors + 1))
            continue
        fi
        created_ts="${BASH_REMATCH[1]}"
        age=$((NOW - created_ts))
        if [ "$age" -ge "$THRESHOLD" ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "$(ts) DRY-RUN would rm -rf $dir (age=${age}s)"
                echo "$(ts) DRY-RUN would rm -rf $dir (age=${age}s)" >> "$LOG"
            else
                echo "$(ts) rm -rf $dir (age=${age}s)" >> "$LOG"
                rm -rf -- "$dir" || { errors=$((errors + 1)); continue; }
            fi
            reaped=$((reaped + 1))
        else
            kept=$((kept + 1))
        fi
    done < <(find "$ROOT" -maxdepth 4 -type d -name "*.trash_*" -print0)
done

echo "$(ts) summary: reaped=$reaped kept=$kept errors=$errors threshold=${MAX_AGE_DAYS}d dry_run=$DRY_RUN" >> "$LOG"
exit 0
