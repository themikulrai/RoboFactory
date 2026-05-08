#!/bin/bash
# Plan v2 C1#10 hardening helper.
#
# Resolve TRAIN_CFG_PATH for the Stage-3 preflight guards. Caller sets
# CKPT_FOR_PREFLIGHT to one ckpt path (or one arm's ckpt for decent),
# and EVAL_CFG_PATH to the eval YAML. We prefer the trainer-dumped
# `<ckpt_dir>/.hydra_config.yaml` (commit 9710f9a) so the camera-family
# guard becomes a real workspace-vs-wristcam assertion. Legacy ckpts
# (trained pre-9710f9a) lack the dump; fall back to EVAL_CFG_PATH and
# the camera-family check stays soft-warn — same as before this patch.
#
# Usage:
#   CKPT_FOR_PREFLIGHT=<path> EVAL_CFG_PATH=<path> source _resolve_train_cfg.sh

if [ -z "${TRAIN_CFG_PATH:-}" ]; then
    if [ -z "${CKPT_FOR_PREFLIGHT:-}" ]; then
        TRAIN_CFG_PATH="${EVAL_CFG_PATH}"
        echo "[resolve_train_cfg] no CKPT_FOR_PREFLIGHT set; using eval cfg as train-cfg"
    else
        _CAND_DIR="$(dirname "${CKPT_FOR_PREFLIGHT}")"
        _CAND="${_CAND_DIR}/.hydra_config.yaml"
        if [ -f "${_CAND}" ]; then
            TRAIN_CFG_PATH="${_CAND}"
            echo "[resolve_train_cfg] using trainer-dumped: ${TRAIN_CFG_PATH}"
        else
            TRAIN_CFG_PATH="${EVAL_CFG_PATH}"
            echo "[resolve_train_cfg] no .hydra_config.yaml at ${_CAND}; falling back to eval cfg (legacy ckpt)"
        fi
        unset _CAND_DIR _CAND
    fi
fi
export TRAIN_CFG_PATH
