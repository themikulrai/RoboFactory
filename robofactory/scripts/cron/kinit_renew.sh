#!/bin/bash
# Plan v2 workflow improvement #8 — automatic Kerberos ticket renewal.
#
# Stanford CS Kerberos tickets default to 24 h lifetime and a 7-day
# renewable window. While inside the renewable window, `kinit -R`
# refreshes the lifetime without a password. After the renewable
# window expires, a fresh password-based `kinit` is required.
#
# Why this exists: iris-mcp (and other SSH-via-MCP tools) authenticate
# via the local krb5cc. When the ticket expires mid-session, every
# `submit_job` / `tail_output` / `cancel_job` call fails until the user
# manually re-authenticates. Hourly `kinit -R` keeps the ticket fresh
# inside the renewable window so the MCP stays usable.
#
# Run hourly via crontab; install via install_kinit_renew_cron.sh.
# Idempotent; safe to re-run.

set -uo pipefail

LOG=/iris/u/mikulrai/logs/kinit_renew.log
mkdir -p "$(dirname "$LOG")"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# klist -s returns 0 iff a valid TGT exists.
if ! klist -s 2>/dev/null; then
    echo "$(ts) WARN: no valid Kerberos ticket; manual kinit required (renewable lifetime likely exhausted)" >> "$LOG"
    exit 0  # don't spam cron with non-zero exits
fi

if kinit -R 2>>"$LOG"; then
    echo "$(ts) renewed OK" >> "$LOG"
else
    echo "$(ts) WARN: kinit -R failed (renewable lifetime exhausted?); run \`kinit mikulrai@CS.STANFORD.EDU\` manually" >> "$LOG"
fi
exit 0
