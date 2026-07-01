#!/bin/bash
# Thin wrapper that delegates to chain_train_eval.py: submit train launcher(s),
# then submit eval launcher(s) gated on them via --dependency=afterok.
#
# Usage:
#     ./chain_train_eval.sh --train <train_id> [--train <id2>] \
#         --eval <eval_id> [--shards 4] [--after <jobid>] [--dry-run]
set -euo pipefail
exec /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
    -m robofactory.scripts.canonical.chain_train_eval "$@"
