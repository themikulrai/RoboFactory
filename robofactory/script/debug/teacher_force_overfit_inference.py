"""Tier B — Teacher-force inference on the overfit zarr.

Question: does the trained ckpt reproduce the recorded action when fed the
EXACT training-time observation (zarr image + zarr state)? If yes, the model
memorised the (image, state=action) mapping and the eval-time failure is
downstream — i.e., the train/eval obs mismatch quantified in Tier A. If no,
memorisation itself failed despite 2000 epochs.

Loads both per-arm overfit ckpts. For each, iterates episode 13 of its zarr
(the one selected by max_train_episodes=1, downsample_mask seed=42 → idx 13)
and computes:
  pred_chunk[0, 0, :] = policy.predict_action({head_cam, head_cam_global, agent_pos})['action'][0,0]
  err[t] = pred_chunk[0,0,:] - zarr.data.action[start+t]

Aggregates per-dim MSE / RMS / max-abs over the episode, both EMA and raw
weights.
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
                  include_global: bool, device: str = 'cuda:0'):
    """For each t in [n_obs_steps-1, ep_end-ep_start-1], feed the training obs
    window and record chunk[0, 0, :]. Returns (preds[T-n_obs_steps+1, 8],
    tgts[T-n_obs_steps+1, 8])."""
    head = np.asarray(zr['data/head_camera'][ep_start:ep_end])         # (T, 3, 224, 224) uint8
    if include_global:
        glob = np.asarray(zr['data/head_camera_global'][ep_start:ep_end])
    state = np.asarray(zr['data/state'][ep_start:ep_end])              # (T, 8)
    action = np.asarray(zr['data/action'][ep_start:ep_end])            # (T, 8)
    T = head.shape[0]

    preds, tgts = [], []
    for t in range(n_obs_steps - 1, T):
        # Build obs window [t - n_obs_steps + 1, t] (inclusive of t)
        idx_lo = t - n_obs_steps + 1
        head_w = head[idx_lo:t + 1].astype(np.float32) / 255.0          # (To, 3, 224, 224)
        state_w = state[idx_lo:t + 1].astype(np.float32)                # (To, 8)
        obs_dict = {
            'head_cam': torch.from_numpy(head_w).unsqueeze(0).to(device=device),    # (1, To, 3, H, W)
            'agent_pos': torch.from_numpy(state_w).unsqueeze(0).to(device=device),  # (1, To, 8)
        }
        if include_global:
            glob_w = glob[idx_lo:t + 1].astype(np.float32) / 255.0
            obs_dict['head_cam_global'] = torch.from_numpy(glob_w).unsqueeze(0).to(device=device)
        action_dict = policy.predict_action(obs_dict)
        chunk = action_dict['action'].detach().cpu().numpy()            # (1, horizon, 8)
        preds.append(chunk[0, 0])                                       # predicted action at step t
        tgts.append(action[t])
    return np.asarray(preds), np.asarray(tgts)


def summarize(preds: np.ndarray, tgts: np.ndarray) -> dict:
    err = preds - tgts                                                  # (N, 8)
    per_dim_mse = (err ** 2).mean(axis=0)
    per_dim_rms = np.sqrt(per_dim_mse)
    per_dim_max_abs = np.abs(err).max(axis=0)
    per_step_l2 = np.linalg.norm(err, axis=1)
    return dict(
        n_steps=int(preds.shape[0]),
        per_dim_mse=per_dim_mse.tolist(),
        per_dim_rms_rad=per_dim_rms.tolist(),
        per_dim_max_abs=per_dim_max_abs.tolist(),
        mean_step_l2_rad=float(per_step_l2.mean()),
        max_step_l2_rad=float(per_step_l2.max()),
        worst_step_idx=int(np.argmax(per_step_l2)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', default='/iris/u/mikulrai/checkpoints/RoboFactory')
    ap.add_argument('--ckpt-suffix', default='workspace_overfit1')
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=2000)
    ap.add_argument('--task', default='TwoRobotsStackCube-rf')
    ap.add_argument('--zarr-root', default='/iris/u/mikulrai/datasets/multi_robot/RoboFactory/zarr_data')
    ap.add_argument('--ep-idx', type=int, default=13)
    ap.add_argument('--n-obs-steps', type=int, default=3)
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.add_argument('--out-json', default='/iris/u/mikulrai/runs/diagnostics/overfit_h3_tierB/summary.json')
    args = ap.parse_args()

    out = dict(args=vars(args), arms={})
    for arm in (0, 1):
        ckpt_dir = f'{args.ckpt_root}/{args.task}_agent{arm}_{args.ckpt_suffix}_{args.data_num}'
        ckpt = f'{ckpt_dir}/{args.checkpoint_num}.ckpt'
        zarr_path = f'{args.zarr_root}/{args.task}_workspace_overfit1_agent{arm}_{args.data_num}.zarr'
        print(f'\n=== arm {arm} ===')
        print(f'ckpt: {ckpt}')
        print(f'zarr: {zarr_path}')

        zr = zarr.open(zarr_path, mode='r')
        s, e = episode_slice(zr, args.ep_idx)
        print(f'episode {args.ep_idx}: [{s}, {e}), length={e-s}')

        arm_results = {}
        for use_ema in (True, False):
            tag = 'ema' if use_ema else 'raw'
            print(f'-- arm {arm} {tag} --')
            policy, cfg = load_policy(ckpt, use_ema=use_ema)
            preds, tgts = teacher_force(
                policy, zr, s, e,
                n_obs_steps=args.n_obs_steps,
                include_global=args.include_global,
            )
            stats = summarize(preds, tgts)
            arm_results[tag] = stats
            print(f'   mean step L2 = {stats["mean_step_l2_rad"]:.4f} rad   '
                  f'max = {stats["max_step_l2_rad"]:.4f}   '
                  f'worst t = {stats["worst_step_idx"]}')
            print(f'   per-dim RMS = {[round(x, 4) for x in stats["per_dim_rms_rad"]]}')
            del policy
            torch.cuda.empty_cache()
        out['arms'][arm] = arm_results

    # Verdict
    ema_l2 = [out['arms'][a]['ema']['mean_step_l2_rad'] for a in (0, 1)]
    raw_l2 = [out['arms'][a]['raw']['mean_step_l2_rad'] for a in (0, 1)]
    ema_pass = max(ema_l2) < 0.05
    raw_pass = max(raw_l2) < 0.05
    out['verdict'] = dict(
        ema_mean_step_l2=ema_l2,
        raw_mean_step_l2=raw_l2,
        ema_memorized=ema_pass,
        raw_memorized=raw_pass,
        decision=(
            'memorization_confirmed_ema' if ema_pass else
            'memorization_confirmed_raw_only' if raw_pass else
            'memorization_failed'
        ),
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n=== VERDICT ===')
    print(json.dumps(out['verdict'], indent=2))
    print(f'\nFull summary: {args.out_json}')


if __name__ == '__main__':
    main()
