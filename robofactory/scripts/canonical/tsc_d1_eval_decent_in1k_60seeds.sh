#!/bin/bash
#SBATCH --job-name=tsc_d1_eval_decent_in1k
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/tsc_d1_eval_decent_in1k_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/tsc_d1_eval_decent_in1k_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulrai+spam@gmail.com
#SBATCH --exclude=iris-hp-z8,iris-hgx-1

# D1 workspace decentralised TSC eval — ImageNet ResNet18 ckpts at epoch 300.
# ckpts: ThreeRobotsStackCube-rf_Agent{0,1,2}_150/300.ckpt (default _Agent_ naming, capital A).
# D1 was trained with workspace cams + img_height=240 + NO head_cam_global.

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
export TORCH_HOME=$HOME/.cache/torch

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=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
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

# Stage-3 preflight guards — refuses to run on scene/camera/seed/wandb mismatch.
EVAL_CFG_PATH="${EVAL_CFG_PATH:-configs/table/three_robots_stack_cube.yaml}"
# Decent eval: arm0 ckpt as the canonical .hydra_config.yaml source — all
# 3 arms share train pipeline; if any arm dumped one, this picks it up.
CKPT_FOR_PREFLIGHT="${CKPT_FOR_PREFLIGHT:-/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_Agent0_150/300.ckpt}"
source /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/canonical/_resolve_train_cfg.sh
PREFLIGHT_PYTHON=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
$PREFLIGHT_PYTHON -m robofactory.utils.preflight_eval_guards \
    --train-cfg "${TRAIN_CFG_PATH}" \
    --eval-cfg "${EVAL_CFG_PATH}" \
    --seed-file /iris/u/mikulrai/runs/eval_seeds_60.txt \
    --expected-sha256 "${EVAL_SEEDS_SHA256:-}"
if [ $? -ne 0 ]; then echo "Preflight failed; aborting."; exit 1; fi

# Stage-3 per-eval guards (plan v2 C1#10). eval_multi_dp.py computes per-arm
# ckpt paths internally from --data-num/--checkpoint-num, so we only verify
# wandb-online and scene-config-exists here; per-arm ckpt-existence is the
# entrypoint's responsibility.
python -u -m robofactory.utils.preflight_eval \
  --scene-config configs/table/three_robots_stack_cube.yaml || exit 1

python -u ./policy/Diffusion-Policy/eval_multi_dp.py \
  --config=configs/table/three_robots_stack_cube.yaml \
  --data-num=150 \
  --checkpoint-num=300 \
  --ckpt-suffix="" \
  --obs-cam-family=workspace \
  --no-include-global \
  --img-height=240 \
  --img-width=320 \
  -o rgb -b cpu -n 1 \
  --render-mode=sensors \
  -s 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
     10010 10011 10012 10013 10014 10015 10016 10017 10018 10019 \
     10020 10021 10022 10023 10024 10025 10026 10027 10028 10029 \
     1000  1001  1002  1003  1004  1005  1006  1007  1008  1009 \
     1010  1011  1012  1013  1014  1015  1016  1017  1018  1019 \
     1020  1021  1022  1023  1024  1025  1026  1027  1028  1029 \
  --quiet \
  --max-steps=400 \
  --wandb \
  --wandb-tags='eval,tsc,d1,decentralised-dp,encoder-in1k,60seeds,xbucket'
