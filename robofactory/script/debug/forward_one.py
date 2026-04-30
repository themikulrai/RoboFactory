"""T2/T3/T5 — feed one observation through the loaded policy and inspect every layer.

Two obs sources:
  --obs-source train  → use training h5 frames 0/1/2 + agent_pos = action[0/1/2]
                        (what training-time conditioning looked like)
  --obs-source env    → env.reset(seed=k); use 3 copies of frame 0 + sensed qpos
                        (what eval-time conditioning looks like)

Prints:
  - raw image stats (preprocessing) — T2
  - normalized image stats (post LinearNormalizer)            — T3
  - normalized agent_pos (post LinearNormalizer)              — T3
  - predicted action raw (pre-unnormalize, must be ~[-1, 1])  — T5
  - predicted action denormalized (must be in training range) — T5
  - if --obs-source train: compares to ground-truth action[3..(3+horizon)]

Usage:
  python script/debug/forward_one.py --ckpt <ckpt> --obs-source train \
    --h5 /iris/u/mikulrai/data/RoboFactory/hf_download/PickMeat/PickMeat.h5 --traj 0
  python script/debug/forward_one.py --ckpt <ckpt> --obs-source env \
    --config configs/table/pick_meat.yaml --seed 0
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import numpy as np
import torch
import dill
import h5py
import yaml
import gymnasium as gym
from collections import deque

from robofactory.tasks import *  # register envs
import hydra
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


def _load_policy(ckpt_path, device='cuda:0'):
    payload = torch.load(ckpt_path, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy, cfg


def _stack_obs(head_cam_chw_list, agent_pos_list):
    """Stack 3 obs into the input shape expected by predict_action: (B=1, To=3, ...)."""
    head = np.stack(head_cam_chw_list, axis=0)[None]      # (1, 3, 3, H, W)
    pos = np.stack(agent_pos_list, axis=0)[None]          # (1, 3, 8)
    return {'head_cam': head.astype(np.float32),
            'agent_pos': pos.astype(np.float32)}


def _from_train(h5_path, traj_id, n_obs=3):
    """agent_pos = action (commanded) — training contract."""
    with h5py.File(h5_path, 'r') as f:
        traj = f[f'traj_{traj_id}']
        rgb_all = traj['obs/sensor_data/head_camera/rgb'][:n_obs]   # (3, 240, 320, 3) uint8
        actions = traj['actions'][:]                                # (T-1, 8) float
    head_chw = [np.moveaxis(r, -1, 0).astype(np.float32) / 255.0 for r in rgb_all]
    agent_pos = [actions[i] for i in range(n_obs)]
    gt_actions = actions  # ground truth full sequence
    return _stack_obs(head_chw, agent_pos), gt_actions


def _from_env(config, seed, n_obs=3):
    """3 copies of frame 0 + sensed qpos + planner gripper=OPEN — eval contract."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'
    env = gym.make(env_id, config=config, obs_mode='rgb', control_mode='pd_joint_pos',
                   render_mode='sensors', num_envs=1, sim_backend='cpu', enable_shadow=False,
                   sensor_configs=dict(shader_pack='default'))
    obs, _ = env.reset(seed=seed)
    rgb = obs['sensor_data']['head_camera']['rgb']
    rgb_np = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
    if rgb_np.ndim == 4:
        rgb_np = rgb_np[0]
    head_chw = np.moveaxis(rgb_np, -1, 0).astype(np.float32) / 255.0
    qpos = obs['agent']['qpos']
    qpos_np = qpos.cpu().numpy() if hasattr(qpos, 'cpu') else np.asarray(qpos)
    qpos_np = qpos_np.squeeze(0)[:-2]   # 7 arm
    agent_pos = np.append(qpos_np, 1.0)  # OPEN gripper
    env.close()
    return _stack_obs([head_chw]*n_obs, [agent_pos]*n_obs), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--obs-source', choices=['train', 'env'], required=True)
    ap.add_argument('--h5', default=None)
    ap.add_argument('--traj', type=int, default=0)
    ap.add_argument('--config', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    policy, cfg = _load_policy(args.ckpt, device=device)
    print(f"=== ckpt: {args.ckpt}  obs-source: {args.obs_source} ===")
    print(f"horizon={cfg.policy.horizon}  n_obs_steps={cfg.policy.n_obs_steps}  n_action_steps={cfg.policy.n_action_steps}")

    if args.obs_source == 'train':
        assert args.h5, 'need --h5 for obs-source=train'
        obs_dict, gt = _from_train(args.h5, args.traj, n_obs=cfg.policy.n_obs_steps)
    else:
        assert args.config, 'need --config for obs-source=env'
        obs_dict, gt = _from_env(args.config, args.seed, n_obs=cfg.policy.n_obs_steps)

    # Move to device
    obs_t = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device))

    # T2 — raw stats
    print("\n--- T2 raw obs stats ---")
    for k, v in obs_dict.items():
        v_arr = np.asarray(v)
        print(f"  {k}: shape={v_arr.shape} dtype={v_arr.dtype} min={v_arr.min():.4f} max={v_arr.max():.4f} mean={v_arr.mean():.4f}")

    # T3 — post-normalize
    print("\n--- T3 post-LinearNormalizer ---")
    with torch.no_grad():
        nobs = policy.normalizer.normalize(obs_t)
    for k, v in nobs.items():
        a = v.detach().cpu().numpy()
        print(f"  norm({k}): shape={a.shape} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} std={a.std():.4f}")

    # T5 — predict
    print("\n--- T5 predict_action ---")
    with torch.no_grad():
        result = policy.predict_action(obs_t)
    a = result['action'].detach().cpu().numpy()       # denormalized (B, n_action_steps, A)
    a_pred = result.get('action_pred')
    if a_pred is not None:
        a_pred = a_pred.detach().cpu().numpy()        # full horizon, denormalized (B, horizon, A)
        print(f"  action_pred (full horizon, denorm): shape={a_pred.shape}")
        for d in range(a_pred.shape[-1]):
            col = a_pred[0, :, d]
            print(f"    dim {d}: min={col.min():+.4f} max={col.max():+.4f} mean={col.mean():+.4f}  step0={a_pred[0,0,d]:+.4f}")
    print(f"  action (executed slice, denorm): shape={a.shape}  step0={a[0,0].round(4).tolist()}")

    # Also dump pre-unnormalize raw action
    # Re-run sampling with unnormalize=False if API supports; fallback: normalize the prediction
    raw = policy.normalizer['action'].normalize(result['action_pred']).detach().cpu().numpy() if a_pred is not None else None
    if raw is not None:
        print(f"  raw_action (pre-unnormalize, should be ~[-1,1]): min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}")

    # Compare to ground truth if train source
    if gt is not None and a_pred is not None:
        H = a_pred.shape[1]
        # Training contract: prediction at obs t..t+H-1, action[t-1+1..t-1+H] = action[t..t+H-1]
        # We fed obs from indices 0..n_obs-1 so prediction starts at index n_obs-1+1? actually let's compare to
        # action[n_obs_steps-1 : n_obs_steps-1+H] (the target the policy was trained to produce).
        start = cfg.policy.n_obs_steps - 1
        target = gt[start: start + H]
        print(f"\n--- ground truth comparison (target = train action[{start}:{start+H}]) ---")
        print(f"  per-dim per-step abs err of predicted full-horizon vs train action:")
        err = np.abs(a_pred[0, :target.shape[0]] - target)
        for d in range(target.shape[-1]):
            print(f"    dim {d}: mean_err={err[:,d].mean():.4f}  max_err={err[:,d].max():.4f}")
        print(f"  overall: mean={err.mean():.4f}  max={err.max():.4f}")
        # Print step 0 ground truth vs prediction
        print(f"  step0 pred  : {a_pred[0, 0].round(4).tolist()}")
        print(f"  step0 target: {target[0].round(4).tolist()}")


if __name__ == '__main__':
    main()
