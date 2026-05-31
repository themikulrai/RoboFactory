#!/bin/bash
# Thin wrapper -> manifest launcher tsc_dp_d1_freezeenc_decent (see eval/manifest.yaml).
exec "$(dirname "$0")/eval/submit_eval.sh" tsc_dp_d1_freezeenc_decent "$@"
