#!/bin/bash
# Universal eval launcher wrapper.
#
# Activates conda + exports the cache/wandb env vars shared by every legacy
# launcher, then execs the Python dispatcher. SBATCH directives intentionally
# absent — `submit_eval.sh` passes resources via `sbatch` CLI flags so they
# can come from the manifest.
#
# Usage:
#     ./run_eval.sh <launcher_id> [--dry-run]
#     sbatch <resource_flags> ./run_eval.sh <launcher_id>
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <launcher_id> [--dry-run]"
    exit 1
fi

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch
export OPENPI_DATA_HOME=/iris/u/mikulrai/data/openpi
export HF_HOME=/iris/u/mikulrai/.cache/huggingface
export XDG_CACHE_HOME=/iris/u/mikulrai/.cache
export JAX_COMPILATION_CACHE_DIR=/iris/u/mikulrai/.cache/jax
export TMPDIR=/iris/u/mikulrai/tmp
mkdir -p "$XDG_CACHE_HOME/jax/xla_autotune" "$TMPDIR"
export XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=$XDG_CACHE_HOME/jax/xla_autotune ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HYDRA_FULL_ERROR=1
export WANDB_PROJECT="${WANDB_PROJECT:-openpi-robofactory}"
export WANDB_MODE="${WANDB_MODE:-online}"

# WANDB_API_KEY: prefer existing env, then ~/.wandb_api_key.
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.wandb_api_key" ]; then
    WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
    export WANDB_API_KEY
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[run_eval.sh] WANDB_API_KEY not set and ~/.wandb_api_key missing." >&2
    exit 1
fi

cd /iris/u/mikulrai/projects/RoboFactory/robofactory
exec /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
    -m robofactory.scripts.canonical.eval.run_eval "$@"
