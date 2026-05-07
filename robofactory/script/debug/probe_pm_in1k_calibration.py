"""Capture PM in1k healthy-reference features for the encoder-collapse probe.

Runs the same forward path as `script/debug/probe_decent_dp_features.py` but
loads via `eval_dp.DP` (which supports `--ckpt-path`) so the non-standard PM
in1k ckpt at `PickMeat-rf_150/backup/300_in1k.ckpt` works without symlinks.

Output is a .npz compatible with the existing analysis tooling and the
`utils.preflight_collapse` probe (after wrapping into (obs_dict, gt_action)
pairs at load time).

Usage::

    python script/debug/probe_pm_in1k_calibration.py \\
        --config=configs/table/pick_meat.yaml \\
        --ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt \\
        --img-height=240 --img-width=320 \\
        --seeds 10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 10010 10011 10012 10013 10014 10015 10016 10017 10018 10019 10020 10021 10022 10023 10024 10025 10026 10027 10028 10029 \\
        --out /iris/u/mikulrai/runs/calibration/pm_in1k_goodref.npz
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import numpy as np
import torch
import yaml
import gymnasium as gym

from robofactory.tasks import *  # noqa: F401, F403
import robofactory.agents  # noqa: F401

from eval_dp import DP, get_model_input
from script.debug.probe_decent_dp_features import (
    encoder_features,
    predict_action_np,
    cube_xyz,
)


OPEN = 0.04


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument('--img-height', type=int, default=240)
    ap.add_argument('--img-width', type=int, default=320)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    env = gym.make(
        env_id,
        config=args.config,
        obs_mode='rgb',
        control_mode='pd_joint_pos',
        render_mode='sensors',
        num_envs=1,
        sim_backend='cpu',
        enable_shadow=False,
        sensor_configs=dict(shader_pack='default'),
    )
    raw_obs, _ = env.reset(seed=args.seeds[0])
    print(f"env_id={env_id} ckpt_path={args.ckpt_path}", flush=True)

    dp_model = DP(env_id, checkpoint_num=0, data_num=0, ckpt_path=args.ckpt_path)

    n_seeds = len(args.seeds)
    feats_buf = None
    actions_buf = None
    images_buf = None
    qpos_buf = None
    feat_dim = None

    for si, seed in enumerate(args.seeds):
        raw_obs, _ = env.reset(seed=int(seed))
        qpos = raw_obs['agent']['qpos'].squeeze(0)[:-2].cpu().numpy()
        qpos_with_grip = np.append(qpos, OPEN)
        obs_d = get_model_input(raw_obs, qpos_with_grip, img_h=args.img_height, img_w=args.img_width)

        feats = encoder_features(dp_model.policy, obs_d)
        action = predict_action_np(dp_model.policy, obs_d)

        if feats_buf is None:
            feat_dim = feats.shape[-1]
            Ta, Da = action.shape
            qpos_dim = qpos_with_grip.shape[0]
            feats_buf = np.zeros((n_seeds, 1, feat_dim), dtype=np.float32)
            actions_buf = np.zeros((n_seeds, 1, Ta, Da), dtype=np.float32)
            images_buf = np.zeros((n_seeds, 1, 3, 60, 80), dtype=np.float32)
            qpos_buf = np.zeros((n_seeds, 1, qpos_dim), dtype=np.float32)
            print(f"feat_dim={feat_dim} action_chunk={action.shape}", flush=True)

        feats_buf[si, 0] = feats[0]
        actions_buf[si, 0] = action
        qpos_buf[si, 0] = qpos_with_grip

        import cv2
        img_hwc = np.moveaxis(obs_d['head_cam'], 0, -1)
        img_small = cv2.resize(img_hwc, (80, 60), interpolation=cv2.INTER_AREA)
        images_buf[si, 0] = np.moveaxis(img_small, -1, 0)

        if (si + 1) % 5 == 0 or si == n_seeds - 1:
            print(f"  seed {seed} done ({si+1}/{n_seeds})", flush=True)

    env.close()

    np.savez(
        args.out,
        seeds=np.array(args.seeds, dtype=np.int64),
        features=feats_buf,
        actions=actions_buf,
        images=images_buf,
        qpos=qpos_buf,
        ckpt_path=args.ckpt_path,
        config=args.config,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    print(f"\nSaved {args.out}", flush=True)
    F = feats_buf[:, 0]  # (N, F)
    cs = (F @ F.T) / (np.linalg.norm(F, axis=1, keepdims=True)
                      @ np.linalg.norm(F, axis=1, keepdims=True).T + 1e-9)
    off = cs[np.triu_indices(n_seeds, k=1)]
    A = actions_buf[:, 0].reshape(n_seeds, -1)
    a_pair = np.linalg.norm(A[:, None] - A[None, :], axis=-1)
    a_off = a_pair[np.triu_indices(n_seeds, k=1)]
    print(f"  feat cos-sim mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}",
          flush=True)
    print(f"  action L2 mean={a_off.mean():.4f}  min={a_off.min():.4f}  max={a_off.max():.4f}",
          flush=True)


if __name__ == '__main__':
    main()
