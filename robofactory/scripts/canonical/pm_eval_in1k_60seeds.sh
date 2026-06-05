#!/bin/bash
# Thin wrapper -> manifest launcher pm_dp_in1k (see eval/manifest.yaml).
exec "$(dirname "$0")/eval/submit_eval.sh" pm_dp_in1k "$@"
