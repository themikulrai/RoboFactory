#!/bin/bash
# MIGRATED to the train manifest+dispatcher. This is now a thin wrapper.
# Launcher id: resume_tsc_d2_in1k_a2_from285  (see scripts/canonical/train/manifest.yaml)
# Override-dict parity with the pre-migration script verified by
# scripts/canonical/train/test_run_train.py::TestParity.
#
# Usage:  ./resume_dp_tsc_d2_in1k_a2_from285.sh              # submit via manifest
#         ./resume_dp_tsc_d2_in1k_a2_from285.sh --print      # print sbatch CLI
#         ./resume_dp_tsc_d2_in1k_a2_from285.sh -- --dry-run # forward --dry-run to run_train.sh
exec "$(dirname "$0")/train/submit_train.sh" resume_tsc_d2_in1k_a2_from285 "$@"
