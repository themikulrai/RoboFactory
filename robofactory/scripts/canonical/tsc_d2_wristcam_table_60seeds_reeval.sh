#!/bin/bash
#SBATCH --job-name=tsc_d2_table_60s_reeval
#SBATCH --output=/iris/u/mikulrai/logs/tsc_d2_wristcam_table_60s_reeval_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/tsc_d2_wristcam_table_60s_reeval_%j.err
#SBATCH --partition=iris
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Re-eval of the 0%-SR DP TSC d2_wristcam ckpts on the *correct* TABLE scene.
# Hypothesis under test: prior 0% SR was scene-mismatch (eval ran on robocasa kitchen,
# but ckpts were trained for TABLE). See user memory `project_robofactory_tsc_scene`.
# Smoking gun: .slurm_scripts/job_1komhuu7.sh is a 10-seed smoke that used
# configs/robocasa/three_robots_stack_cube.yaml against the same ckpt set. This
# launcher is identical EXCEPT the scene config and seed count.

set -e

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

export CUDA_VISIBLE_DEVICES=0
# Explicit HOME + WANDB_API_KEY: prior 15341094 hit HTTP 401 because ~/.netrc
# resolution depended on the compute node's HOME (which differs from
# /iris/u/mikulrai on iris* nodes).
export HOME=/iris/u/mikulrai
export WANDB_API_KEY=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

# Stage-3 preflight guards — refuses to run on scene/camera/seed/wandb mismatch.
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/three_robots_stack_cube.yaml}"
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent0_d2_wristcam_150/300.ckpt}"
source "$(dirname "$0")/_resolve_train_cfg.sh"
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60.txt \
    --expected-sha256-file /iris/u/mikulrai/runs/eval_seeds.sha256
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

# Stage-3 per-eval guards (plan v2 C1#10). eval_multi_dp.py computes per-arm
# ckpt paths internally, so we verify wandb-online + scene-config-exists only.
python -u -m robofactory.utils.preflight_eval \
  --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_multi_dp.py \
  --config=configs/table/three_robots_stack_cube.yaml \
  --data-num=150 --checkpoint-num=300 \
  --ckpt-suffix=d2_wristcam \
  --obs-cam-family=wristcam \
  --robot-uids=panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi \
  --include-global \
  --img-height=224 --img-width=224 \
  --render-mode=sensors -o=rgb -b=gpu -n 1 \
  -s 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
     10010 10011 10012 10013 10014 10015 10016 10017 10018 10019 \
     10020 10021 10022 10023 10024 10025 10026 10027 10028 10029 \
     1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 \
     1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 \
     1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 \
  --max-steps=400 \
  --wandb \
  --wandb-tags='eval,dp,tsc,d2_wristcam,table,60seeds,reeval_2026-05,scene_match_hypothesis_test' \
  --quiet
