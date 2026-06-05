#!/bin/bash
# Thin wrapper -> manifest launchers 2sc_dp_decent_{wristcam,workspace} (see eval/manifest.yaml).
# Legacy script was CAM_FAMILY-parameterized; pick the launcher from $CAM_FAMILY (default wristcam).
case "${CAM_FAMILY:-wristcam}" in
  workspace) LID=2sc_dp_decent_workspace ;;
  wristcam)  LID=2sc_dp_decent_wristcam ;;
  *) echo "Unknown CAM_FAMILY=${CAM_FAMILY} (workspace|wristcam)"; exit 1 ;;
esac
exec "$(dirname "$0")/eval/submit_eval.sh" "$LID" "$@"
