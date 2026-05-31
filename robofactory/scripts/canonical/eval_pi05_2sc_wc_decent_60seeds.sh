#!/bin/bash
# Thin wrapper -> manifest launcher 2sc_pi05_wc_decent (see eval/manifest.yaml).
exec "$(dirname "$0")/eval/submit_eval.sh" 2sc_pi05_wc_decent "$@"
