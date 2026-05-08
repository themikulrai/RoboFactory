#!/bin/bash
# Move a plan file from ~/.claude/plans/ into docs/plans/ and symlink it
# back so the Claude harness still finds it at the expected path.
#
# Workflow improvement #10. Reversible: rm the symlink + mv the file back
# to ~/.claude/plans/ to undo.
#
# Usage:
#   bash docs/plans/migrate.sh <plan_filename>
#   bash docs/plans/migrate.sh the-new-feature.md
set -euo pipefail

PLAN_NAME="${1:-}"
if [ -z "$PLAN_NAME" ]; then
    echo "Usage: $0 <plan_filename>" >&2
    exit 2
fi

REPO_PLANS="/iris/u/mikulrai/projects/RoboFactory/robofactory/docs/plans"
CLAUDE_PLANS="/iris/u/mikulrai/.claude/plans"

SRC="${CLAUDE_PLANS}/${PLAN_NAME}"
DST="${REPO_PLANS}/${PLAN_NAME}"

if [ ! -f "$SRC" ] || [ -L "$SRC" ]; then
    echo "ERROR: ${SRC} is not a regular file (already migrated, or doesn't exist)." >&2
    exit 1
fi
if [ -e "$DST" ]; then
    echo "ERROR: ${DST} already exists. Inspect/resolve manually." >&2
    exit 1
fi

mkdir -p "$REPO_PLANS"
mv "$SRC" "$DST"
ln -s "$DST" "$SRC"

echo "Migrated: $PLAN_NAME"
echo "  Real:    $DST"
echo "  Symlink: $SRC -> $DST"
echo
echo "Next: stage with 'git add docs/plans/${PLAN_NAME}'"
