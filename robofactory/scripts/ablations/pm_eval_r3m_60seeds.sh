#!/bin/bash
#SBATCH --job-name=pm_eval_r3m_60s
#SBATCH --output=/iris/u/mikulrai/logs/phase2_debug/pm_eval_r3m_%j.out
#SBATCH --error=/iris/u/mikulrai/logs/phase2_debug/pm_eval_r3m_%j.err
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=iris-hp-z8,iris-hgx-1

source /iris/u/mikulrai/data/miniforge3/etc/profile.d/conda.sh
conda activate RoboFactory

export HOME=/iris/u/mikulrai
[ -L "$HOME/.r3m" ] || ln -sfn /iris/u/mikulrai/.cache/r3m "$HOME/.r3m"

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=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
export HYDRA_FULL_ERROR=1

cd /iris/u/mikulrai/projects/RoboFactory/robofactory

python -u ./policy/Diffusion-Policy/eval_dp.py \
  --config=configs/table/pick_meat.yaml \
  --ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_r3m.ckpt \
  --data-num=150 \
  --checkpoint-num=300 \
  -o rgb -b cpu -n 1 \
  --render-mode=sensors \
  -s 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 \
     10010 10011 10012 10013 10014 10015 10016 10017 10018 10019 \
     10020 10021 10022 10023 10024 10025 10026 10027 10028 10029 \
     1000  1001  1002  1003  1004  1005  1006  1007  1008  1009 \
     1010  1011  1012  1013  1014  1015  1016  1017  1018  1019 \
     1020  1021  1022  1023  1024  1025  1026  1027  1028  1029 \
  --quiet \
  --max-steps=200 \
  --wandb \
  --wandb-tags='eval,pm,track-b,encoder-r3m,60seeds,xbucket'
