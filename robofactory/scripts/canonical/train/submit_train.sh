#!/bin/bash
# Thin wrapper that delegates to submit_helper.py (which builds the sbatch CLI
# from the manifest and execs it).
#
# Usage:
#     ./submit_train.sh <launcher_id>             # submit
#     ./submit_train.sh <launcher_id> --print     # print sbatch CLI only
#     ./submit_train.sh <launcher_id> -- --dry-run  # forward `--dry-run` to run_train.sh
set -euo pipefail
exec /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
    -m robofactory.scripts.canonical.train.submit_helper "$@"
