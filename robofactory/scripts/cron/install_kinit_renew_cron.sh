#!/bin/bash
# Install the kinit_renew.sh cron job (hourly at :07 to spread load).
# Idempotent: safe to re-run; checks for existing line before appending.
set -euo pipefail

CRON_LINE="7 * * * * /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/cron/kinit_renew.sh >/dev/null 2>&1"

current=$(crontab -l 2>/dev/null || true)
if echo "$current" | grep -Fq "kinit_renew.sh"; then
    echo "kinit_renew.sh cron entry already present — nothing to do."
    exit 0
fi

(echo "$current"; echo "$CRON_LINE") | crontab -
echo "Installed cron line:"
echo "  $CRON_LINE"
echo
echo "Verify with: crontab -l | grep kinit_renew"
echo "Logs:        /iris/u/mikulrai/logs/kinit_renew.log"
