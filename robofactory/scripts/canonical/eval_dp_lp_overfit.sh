#!/bin/bash
# Thin wrapper -> manifest launcher lp_dp_overfit1_maxfit (see eval/manifest.yaml).
exec "$(dirname "$0")/eval/submit_eval.sh" lp_dp_overfit1_maxfit "$@"
