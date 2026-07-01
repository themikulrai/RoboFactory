"""Tier B' — teacher-force inference for the qpos-retrained ckpt.

Mirror of teacher_force_overfit_inference.py but loads the qpos ckpt and
qpos zarr. Verifies the new ckpt actually memorized its training data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import zarr
import dill

sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import hydra  # noqa
from diffusion_policy.workspace.robotworkspace import RobotWorkspace  # noqa


def load_policy(ckpt_path: str, use_ema: bool = True, device: str = 'cuda:0'):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def episode_slice(zr, ep_idx: int):
    ee = np.asarray(zr['meta']['episode_ends']).astype(np.int64)
    s = int(ee[ep_idx - 1]) if ep_idx > 0 else 0
    e = int(ee[ep_idx])
    return s, e


@torch.no_grad()
def teacher_force(policy, zr, ep_start: int, ep_end: int, n_obs_steps: int,
                  include_global: bool, n_obs_index: int, device: str = 'cuda:0'):
    """For each t, feed training obs window and record chunk[n_obs_index] vs action[t]."""
    head = np.asarray(zr['data/head_camera'][ep_start:ep_end])
    if include_global:
        glob = np.asarray(zr['data/head_camera_global'][ep_start:ep_end])
    state = np.asarray(zr['data/state'][ep_start:ep_end])
    action = np.asarray(zr['data/action'][ep_start:ep_end])
    T = head.shape[0]

    preds, tgts = [], []
    for t in range(n_obs_steps - 1, T):
        idx_lo = t - n_obs_steps + 1
        head_w = head[idx_lo:t + 1].astype(np.float32) / 255.0
        state_w = state[idx_lo:t + 1].astype(np.float32)
        obs_dict = {
            'head_cam': torch.from_numpy(head_w).unsqueeze(0).to(device=device),
            'agent_pos': torch.from_numpy(state_w).unsqueeze(0).to(device=device),
        }
        if include_global:
            glob_w = glob[idx_lo:t + 1].astype(np.float32) / 255.0
            obs_dict['head_cam_global'] = torch.from_numpy(glob_w).unsqueeze(0).to(device=device)
        action_dict = policy.predict_action(obs_dict)
        chunk = action_dict['action'].detach().cpu().numpy()
        # chunk[0] is for oldest obs position; chunk[n_obs_index] is the first executed action
        preds.append(chunk[0, n_obs_index])
        tgts.append(action[t])
    return np.asarray(preds), np.asarray(tgts)


def summarize(preds, tgts):
    err = preds - tgts
    per_dim_rms = np.sqrt((err ** 2).mean(axis=0))
    per_step_l2 = np.linalg.norm(err, axis=1)
    return dict(
        n_steps=int(preds.shape[0]),
        per_dim_rms_rad=per_dim_rms.tolist(),
        per_dim_max_abs=np.abs(err).max(axis=0).tolist(),
        mean_step_l2_rad=float(per_step_l2.mean()),
        max_step_l2_rad=float(per_step_l2.max()),
        worst_step_idx=int(np.argmax(per_step_l2)),
        chunk_shape=list(preds.shape),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-suffix', default='workspace_overfit1qpos')
    ap.add_argument('--zarr-stem', default='workspace_overfit1qpos')
    ap.add_argument('--ckpt-root', default='/iris/u/mikulrai/checkpoints/RoboFactory')
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=2000)
    ap.add_argument('--task', default='TwoRobotsStackCube-rf')
    ap.add_argument('--zarr-root', default='/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data')
    ap.add_argument('--ep-idx', type=int, default=13)
    ap.add_argument('--n-obs-steps', type=int, default=3)
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.add_argument('--out-json', default='/iris/u/mikulrai/runs/diagnostics/overfit_h3_tierB_qpos/summary.json')
    args = ap.parse_args()

    out = dict(args=vars(args), arms={})
    for arm in (0, 1):
        ckpt_dir = f'{args.ckpt_root}/{args.task}_agent{arm}_{args.ckpt_suffix}_{args.data_num}'
        ckpt = f'{ckpt_dir}/{args.checkpoint_num}.ckpt'
        zarr_path = f'{args.zarr_root}/{args.task}_{args.zarr_stem}_agent{arm}_{args.data_num}.zarr'
        print(f'\n=== arm {arm} ===')
        print(f'ckpt: {ckpt}')
        print(f'zarr: {zarr_path}')

        zr = zarr.open(zarr_path, mode='r')
        s, e = episode_slice(zr, args.ep_idx)
        print(f'episode {args.ep_idx}: [{s}, {e}), length={e-s}')

        arm_results = {}
        # Probe chunk[0] AND chunk[n_obs_steps-1] (the first executed action)
        for use_ema in (True, False):
            policy, cfg = load_policy(ckpt, use_ema=use_ema)
            for chunk_idx in (0, args.n_obs_steps - 1):  # 0 and 2
                preds, tgts = teacher_force(policy, zr, s, e,
                                              n_obs_steps=args.n_obs_steps,
                                              include_global=args.include_global,
                                              n_obs_index=chunk_idx)
                stats = summarize(preds, tgts)
                tag = f'{"ema" if use_ema else "raw"}_chunk{chunk_idx}'
                arm_results[tag] = stats
                print(f'-- arm {arm} {tag} --')
                print(f'   mean step L2 = {stats["mean_step_l2_rad"]:.4f} rad   '
                      f'max = {stats["max_step_l2_rad"]:.4f}   '
                      f'worst t = {stats["worst_step_idx"]}')
                print(f'   per-dim RMS = {[round(x, 4) for x in stats["per_dim_rms_rad"]]}')
            del policy
            torch.cuda.empty_cache()
        out['arms'][arm] = arm_results

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nFull summary: {args.out_json}')


if __name__ == '__main__':
    main()
