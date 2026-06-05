#!/bin/bash
# Thin wrapper -> manifest launcher tsc_pi05_d1_decent (see eval/manifest.yaml).
exec "$(dirname "$0")/eval/submit_eval.sh" tsc_pi05_d1_decent "$@"
