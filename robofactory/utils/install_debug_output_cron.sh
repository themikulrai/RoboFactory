#!/usr/bin/env bash
# Idempotently installs a daily cron sweep that removes files under
# /iris/u/mikulrai/debug_output/ older than 30 days, then prunes empty dirs.
# Re-running is a no-op once installed.
set -euo pipefail

MARKER="# robofactory: debug_output 30-day sweep"
SWEEP_FILES="find /iris/u/mikulrai/debug_output -type f -mtime +30 -delete 2>/dev/null"
SWEEP_DIRS="find /iris/u/mikulrai/debug_output -mindepth 1 -type d -empty -mtime +30 -delete 2>/dev/null"
LINE="0 4 * * * ${SWEEP_FILES}; ${SWEEP_DIRS}"

if crontab -l 2>/dev/null | grep -qF "$MARKER"; then
    echo "[ok] debug_output cron sweep already installed"
    crontab -l 2>/dev/null | grep -A1 -F "$MARKER"
    exit 0
fi

(crontab -l 2>/dev/null; echo ""; echo "$MARKER"; echo "$LINE") | crontab -
echo "[ok] installed:"
echo "  $LINE"
