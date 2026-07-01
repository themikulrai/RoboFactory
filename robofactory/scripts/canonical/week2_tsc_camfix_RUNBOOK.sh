#!/bin/bash
# =============================================================================
# WEEK-2 / paper-claim CL3 — TSC WORKSPACE camera-fix ablation RUNBOOK.
#
# QUESTION: does fixing the TSC identical-camera bug rescue the TSC workspace
# DP baseline off its 0% floor, under the benchmark's own protocol?
#
# SINGLE variable vs the released DP workspace baseline: the per-agent head
# cameras (head_camera_agent0/1/2) are now DISTINCT
# (configs/table/three_robots_stack_cube.yaml, commit e3ae765, render-verified
# MAD 25-35). Everything else is upstream parity: workspace cam family,
# state=action (DP default — NOT qpos, which would confound), 240x320 RGB,
# resnet18 IN1k, 300 epochs, seed 100, 150 demos, DP (not Pi0.5).
#
# THIS FILE IS A GATED RUNBOOK, NOT A SUBMITTER. Each STEP is a copy-paste
# command block. GPU steps (datagen / convert / train / eval) are GATED:
# the human submits them after the prior gate PASSES. DO NOT `bash` this file.
#
# Branch: week2/tsc-camfix   Worktree: .claude/worktrees/week2-tsc
# =============================================================================
set -euo pipefail

WT=/iris/u/mikulrai/projects/RoboFactory/.claude/worktrees/week2-tsc
RF=$WT/robofactory
PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
source /iris/u/mikulrai/.config/dataroots.sh 2>/dev/null || true
H5_ROOT="${RF_DATA_ROOT:?RF_DATA_ROOT unset — source ~/.config/dataroots.sh}/hf_download_camfix_workspace"
ZARR=/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data
ENV=ThreeRobotsStackCube-rf
YAML=configs/table/three_robots_stack_cube.yaml
H5=$H5_ROOT/$ENV/motionplanning/${ENV}_workspace_camfix_seed0_n150.h5

echo "This is a runbook. Read the STEP blocks; do not execute the whole file." && exit 0

# -----------------------------------------------------------------------------
# STEP 0 — RENDER-VERIFY GATE (run BEFORE any mass datagen). ~5 min, 1 GPU.
#   Confirms the 3 per-agent head cameras are now distinct in the FIXED yaml.
#   Compute node ONLY (NOT login; exclude bad-Vulkan nodes). Must print PASS.
# -----------------------------------------------------------------------------
# srun --partition=iris-hi-interactive --account=iris --gres=gpu:1 \
#      --exclude=iris-hp-z8,iris-hgx-1 --cpus-per-task=8 --mem=32G \
#      --time=00:30:00 --pty bash -lc '
#   source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
#   conda activate RoboFactory
#   export HOME=/iris/u/mikulrai SAPIEN_HEADLESS=1
#   PYTHONPATH="'"$RF"':'"$RF"'/policy/Diffusion-Policy" \
#     '"$PY"' "'"$WT"'/scripts/verify_tsc_cameras.py" --scene table
# '
#   GATE: exit 0 + "RESULT: PASS  (per-agent cameras are distinct)". If FAIL,
#   STOP — the camfix yaml is wrong and the whole ablation is void.

# -----------------------------------------------------------------------------
# STEP 1 — DATAGEN (GATED; expensive ~4 h, 8 CPU, no GPU strictly needed for
#   cpu sim-backend but a GPU node is used for SAPIEN render). 150 demos.
#   NOTE obs-mode=rgb (-o rgb): the H5 MUST carry head_camera_agent{i}/rgb for
#   the converter; default obs-mode=none would write NO RGB. shader=default.
#   TERM-not-KILL on cancel (H5 B-tree corruption risk — memory note).
# -----------------------------------------------------------------------------
# sbatch script body (compute node, exclude bad-Vulkan):
#   #SBATCH --partition=iris-hi --account=iris --gres=gpu:a40:1
#   #SBATCH --exclude=iris-hp-z8,iris-hgx-1 --cpus-per-task=8 --mem=32G
#   #SBATCH --time=06:00:00 --signal=B:TERM@120   # TERM not KILL
#   source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
#   conda activate RoboFactory
#   export HOME=/iris/u/mikulrai SAPIEN_HEADLESS=1
#   cd '"$RF"'
#   '"$PY"' -m robofactory.planner.run \
#     -e '"$ENV"' -c '"$YAML"' -o rgb --shader default -b cpu \
#     -s 0 -n 150 \
#     --record-dir '"$H5_ROOT"' \
#     --traj-name '"${ENV}"'_workspace_camfix_seed0_n150
#   GATE before convert: probe the H5 has head_camera_agent0/1/2/rgb shaped
#   (T,240,320,3) AND that agent0 vs agent1 RGB differ (re-confirm camfix
#   survived into the data, not just the live env):
#   '"$PY"' -c "
# import h5py,numpy as np
# f=h5py.File('$H5','r'); tk=list(f.keys())[0]
# a=f[tk+'/obs/sensor_data/head_camera_agent0/rgb'][0].astype(float)
# b=f[tk+'/obs/sensor_data/head_camera_agent1/rgb'][0].astype(float)
# print('agent0 shape',a.shape,'MAD(a0,a1)=',np.mean(np.abs(a-b)))
# assert np.mean(np.abs(a-b))>10, 'cameras IDENTICAL in H5 — camfix did not land'
# print('DATA-GATE PASS')"

# -----------------------------------------------------------------------------
# STEP 2 — H5 -> ZARR (GATED; ~20 min CPU, no GPU). state=action parity
#   (DEFAULT --state-source action; do NOT pass qpos). 4 zarrs:
#   3 decent (per_agent, single head_camera_agent{i}, native 240x320) +
#   1 cent (joint, head_camera_0/1/2 + global, 224x224 to match joint_task_3arm).
# -----------------------------------------------------------------------------
# CONV=$RF/script/parse_h5_to_zarr_unified.py
# # decent a0/a1/a2  (240x320, single per-agent cam, no --include-global):
# for i in 0 1 2; do
#   $PY $CONV --h5-path $H5 \
#     --out-zarr $ZARR/${ENV}_workspace_camfix_agent${i}_150.zarr \
#     --camera-family workspace --mode per_agent --agent-id $i \
#     --resize-h 240 --resize-w 320 --state-source action --load-num 150
# done
# # cent joint (224x224, 3 head cams + global):
# $PY $CONV --h5-path $H5 \
#   --out-zarr $ZARR/${ENV}_workspace_camfix_joint_150.zarr \
#   --camera-family workspace --include-global --mode joint --n-agents 3 \
#   --resize 224 --state-source action --load-num 150
#   GATE: each zarr dir exists; data/state == data/action (parity); image keys
#   present; decent imgs (3,240,320), cent imgs (3,224,224).

# -----------------------------------------------------------------------------
# STEP 3 — TRAIN x4 (GATED; ~16-22 h each, 1x a6000, 96G, orion low-profile).
#   Launchers live in scripts/canonical/train/manifest.yaml (validated).
#   Dispatch each via run_train (it builds the sbatch + Hydra overrides).
#   Decent uses robot_dp.yaml + default_task (reads head_camera_agent{i});
#   cent uses joint_dp.yaml + joint_task_3arm + camera_family=workspace (auto).
# -----------------------------------------------------------------------------
# M=$RF/scripts/canonical/train/manifest.yaml
# # dry-run first to eyeball overrides (no GPU):
# for L in tsc_ws_camfix_a0 tsc_ws_camfix_a1 tsc_ws_camfix_a2 tsc_ws_camfix_cent; do
#   $PY -m robofactory.scripts.canonical.train.run_train $L --manifest $M --dry-run
# done
# # then submit (human gates): use the manifest sbatch block per launcher, or
# # the submit_train.sh wrapper. ckpt dirs:
# #   checkpoints/ThreeRobotsStackCube-rf_workspace_camfix_agent{0,1,2}_150/300.ckpt
# #   checkpoints/ThreeRobotsStackCube-rf_workspace_camfix_joint_150/300.ckpt
#   GATE: confirm finite loss through ~100 steps + wandb online + correct zarr
#   before calling each job launched (memory note: watch startup).

# -----------------------------------------------------------------------------
# STEP 4 — EVAL (GATED; ~2-4 h, 1x a40, 64G). Released DP protocol, 60 seeds,
#   shadows off, shader=default, partial-credit ON (patched eval scripts log
#   stage_* flags). Decent = eval_multi_dp (3 ckpts); cent = eval_joint_dp.
# -----------------------------------------------------------------------------
# # SEED-POOL DECISION (ASK THE USER — this picks the headline denominator):
# #   --seed-pool upstream_100  = the PAPER protocol set (range 1000..1099, 100
# #       seeds). Use this if "released protocol" == the published RoboFactory
# #       eval. 100 seeds => the >=7/60 gate rescales to >=12/100.
# #   --seed-pool dp_legacy_60  = the old DP 60-seed convention (DEPRECATED but
# #       what the prior "0/60" floors were measured on; apples-to-apples with
# #       the failing baseline). Keeps the gate at >=7/60.
# #   The pre-registered gate below is written for the 60-seed pool. If you switch
# #   to upstream_100, rescale (>=12/100). DEFAULT BELOW = dp_legacy_60 for
# #   apples-to-apples with the documented 0/60 floor.
# # DECENT (eval_multi_dp.py):
# cd $RF
# $PY -u ./policy/Diffusion-Policy/eval_multi_dp.py \
#   --env-id $ENV --config $YAML \
#   --control-mode pd_joint_pos \
#   --obs-mode rgb --sim-backend auto \
#   --ckpt-suffix workspace_camfix --data-num 150 --checkpoint-num 300 \
#   --obs-cam-family workspace --img-height 240 --img-width 320 \
#   --max-steps 400 --shader default --quiet --skip-collapse-probe \
#   --seed-pool dp_legacy_60 \
#   --video-dir /iris/u/mikulrai/eval_results/tsc_ws_camfix/decent \
#   --jsonl-path /iris/u/mikulrai/eval_results/tsc_ws_camfix/decent_ckpt300_60seeds.jsonl \
#   --wandb --wandb-project TSC-DP-camfix \
#   --wandb-name "Eval TSC WS DP Decent camfix ckpt300 60seeds" \
#   --wandb-tags "eval,dp,decent,tsc,workspace,camfix,60seeds,ckpt300,week2-cl3"
#
# # CENT (eval_joint_dp.py):
# $PY -u ./policy/Diffusion-Policy/eval_joint_dp.py \
#   --ckpt-path /iris/u/mikulrai/checkpoints/RoboFactory/${ENV}_workspace_camfix_joint_150/300.ckpt \
#   --config $YAML --camera-family workspace --env-id $ENV --n-agents 3 \
#   --max-steps 400 --quiet \
#   --seed-pool dp_legacy_60 \
#   --record-dir /iris/u/mikulrai/eval_results/tsc_ws_camfix/cent \
#   --jsonl-path /iris/u/mikulrai/eval_results/tsc_ws_camfix/cent_ckpt300_60seeds.jsonl \
#   --wandb --wandb-tags "eval,dp,cent,tsc,workspace,camfix,60seeds,ckpt300,week2-cl3"
#
#   The eval scripts hard-refuse the login node + non-default shader. They also
#   now emit per-episode stage_* partial-credit flags (is_cubeB_on_cubeA etc).

# -----------------------------------------------------------------------------
# PRE-REGISTERED OUTCOME (decide on the per-task headline = best of decent SR
# and cent SR over the 60-seed dp_legacy pool, AFTER render-verify PASS):
#   >= 7/60 (>=11.7%) : CL3 CAUSAL — camera fix lifts TSC ws off the 0% floor;
#                       "corrected-baseline headline". (Note: still likely
#                       BELOW the paper's ~22% band; prior P(lift)~0.35.)
#   1..6/60           : WEAK / equivocal — fix moves the needle but underpowered;
#                       report as suggestive, fold into partial-credit analysis.
#   0/60 (render-verify PASS) : NECESSITY-NOT-SUFFICIENCY (pre-registered
#                       downgrade) — the camera bug is REAL but not the whole
#                       story; other factors (undertraining @300ep, action
#                       encoding, success-criterion) gate TSC ws. Partial-credit
#                       stage_* flags then quantify HOW FAR the policy gets.
#   render-verify FAIL: experiment VOID — do not interpret any SR.
# =============================================================================
