#!/bin/bash
# Thin wrapper that delegates to chain_train_eval.py: submit train launcher(s),
# then submit eval launcher(s) gated on them via --dependency=afterok.
#
# Usage:
#     ./chain_train_eval.sh --train <train_id> [--train <id2>] \
#         --eval <eval_id> [--shards 4] [--after <jobid>] [--dry-run]
set -euo pipefail

# `robofactory` is editable-installed against the MAIN checkout, so a bare
# `python -m robofactory...` resolves to THAT (possibly stale) tree no matter the
# cwd. cd to THIS script's own repo root first: `python -m` puts cwd on sys.path[0],
# so cwd's `robofactory/` package shadows the editable install and we always run the
# code shipped alongside this wrapper (this worktree), regardless of where it was
# invoked from. Repo root is three levels up from scripts/canonical/ (unlike
# submit_eval.sh/run_eval.sh, which rely on cwd / a hardcoded main-checkout path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

exec /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
    -m robofactory.scripts.canonical.chain_train_eval "$@"
