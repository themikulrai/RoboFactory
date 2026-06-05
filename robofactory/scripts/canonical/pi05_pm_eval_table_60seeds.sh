#!/bin/bash
# Thin wrapper -> manifest launcher pm_pi05_paired_dp_seeds (see eval/manifest.yaml).
exec "$(dirname "$0")/eval/submit_eval.sh" pm_pi05_paired_dp_seeds "$@"
