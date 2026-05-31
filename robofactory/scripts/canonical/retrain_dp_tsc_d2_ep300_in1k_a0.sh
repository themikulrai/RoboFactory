#!/bin/bash
# MIGRATED to the train manifest+dispatcher. This is now a thin wrapper.
# Launcher id: tsc_d2_ep300_in1k_a0  (see scripts/canonical/train/manifest.yaml)
# Override-dict parity with the pre-migration script verified by
# scripts/canonical/train/test_run_train.py::TestParity.
#
# Usage:  ./retrain_dp_tsc_d2_ep300_in1k_a0.sh              # submit via manifest
#         ./retrain_dp_tsc_d2_ep300_in1k_a0.sh --print      # print sbatch CLI
#         ./retrain_dp_tsc_d2_ep300_in1k_a0.sh -- --dry-run # forward --dry-run to run_train.sh
exec "$(dirname "$0")/train/submit_train.sh" tsc_d2_ep300_in1k_a0 "$@"
