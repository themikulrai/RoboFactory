#!/bin/bash
# MIGRATED to the train manifest+dispatcher. This is now a thin wrapper.
# Launcher id: pm_d1_ep300_dinov2_spatch  (see scripts/canonical/train/manifest.yaml)
# Override-dict parity with the pre-migration script verified by
# scripts/canonical/train/test_run_train.py::TestParity.
#
# Usage:  ./retrain_dp_pm_d1_ep300_dinov2_spatch.sh              # submit via manifest
#         ./retrain_dp_pm_d1_ep300_dinov2_spatch.sh --print      # print sbatch CLI
#         ./retrain_dp_pm_d1_ep300_dinov2_spatch.sh -- --dry-run # forward --dry-run to run_train.sh
exec "$(dirname "$0")/../canonical/train/submit_train.sh" pm_d1_ep300_dinov2_spatch "$@"
