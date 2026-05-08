"""A3: joint (centralised) DP encoder probe.

Loads a SINGLE joint DP ckpt that consumes obs from all arms and emits a joint
action chunk.  Per seed, dumps the global encoder feature once + the joint
action chunk (which we slice per-arm for downstream analysis).
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import os
import numpy as np
import torch
import yaml
import gymnasium as gym

from robofactory.tasks import *  # registers env ids
import robofactory.agents  # noqa
from diffusion_policy.common.pytorch_util import dict_apply

from eval_joint_dp import load_policy


CAM_KEY_TPL = {
    'workspace': 'head_camera_agent{i}',
    'wristcam': 'hand_camera_{i}',
}


def cube_xyz(env, name):
    for a in env.unwrapped.scene.actors.values():
        if a.name.lower() == name.lower():
            p = a.pose.p
            if hasattr(p, 'cpu'):
                p = p.cpu().numpy()
            return np.asarray(p).flatten()[:3]
    return None


def _pull_rgb(raw_obs, key, resize):
    rgb = raw_obs['sensor_data'][key]['rgb']
    if hasattr(rgb, 'squeeze'):
        rgb = rgb.squeeze(0)
    rgb_np = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
    import cv2 as _cv2
    if rgb_np.shape[0] != resize or rgb_np.shape[1] != resize:
        rgb_np = _cv2.resize(rgb_np, (resize, resize), interpolation=_cv2.INTER_AREA)
    return np.moveaxis(rgb_np, -1, 0).astype(np.float32) / 255.0


def _initial_agent_pos(raw_obs, agent_id, agent_prefix):
    qpos = raw_obs['agent'][f'{agent_prefix}-{agent_id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
    return np.append(qpos, 1.0).astype(np.float32)


@torch.no_grad()
def encoder_features(policy, obs_dict_np):
    device = next(policy.parameters()).device
    To = policy.n_obs_steps
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_in = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in stacked.items()}
    nobs = policy.normalizer.normalize(obs_in)
    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    feats = policy.obs_encoder(this_nobs)
    return feats.cpu().numpy()


@torch.no_grad()
def predict_action_np(policy, obs_dict_np):
    device = next(policy.parameters()).device
    To = policy.n_obs_steps
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_in = {k: torch.from_numpy(v).to(device).unsqueeze(0) for k, v in stacked.items()}
    out = policy.predict_action(obs_in)
    return out['action'].squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt-path', required=True,
                    help='Path to joint DP ckpt (relative to robofactory CWD).')
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--n-agents', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    policy = load_policy(args.ckpt_path)
    print(f"Loaded joint policy from {args.ckpt_path}", flush=True)

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
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    n_seeds = len(args.seeds)
    cube_names = ['cubeA', 'cubeB', 'cubeC']
    cubes = np.full((n_seeds, 3, 3), np.nan, dtype=np.float32)
    feats_buf = None
    actions_buf = None
    qpos_buf = None
    images_buf = None

    tpl = CAM_KEY_TPL[args.obs_cam_family]
    for si, seed in enumerate(args.seeds):
        raw_obs, _ = env.reset(seed=int(seed))
        for ci, cn in enumerate(cube_names):
            p = cube_xyz(env, cn)
            if p is not None:
                cubes[si, ci] = p

        # Build joint obs dict
        agent_pos = np.concatenate(
            [_initial_agent_pos(raw_obs, i, agent_prefix) for i in range(args.n_agents)]
        ).astype(np.float32)
        obs_d = {'agent_pos': agent_pos}
        for i in range(args.n_agents):
            obs_d[f'head_camera_{i}'] = _pull_rgb(raw_obs, tpl.format(i=i), args.resize)
        if args.include_global:
            obs_d['head_camera_global'] = _pull_rgb(raw_obs, 'head_camera_global', args.resize)

        f = encoder_features(policy, obs_d)
        a = predict_action_np(policy, obs_d)
        if feats_buf is None:
            feat_dim = f.shape[-1]
            Ta, Da = a.shape
            qpos_dim = agent_pos.shape[0]
            feats_buf = np.zeros((n_seeds, feat_dim), dtype=np.float32)
            actions_buf = np.zeros((n_seeds, Ta, Da), dtype=np.float32)
            qpos_buf = np.zeros((n_seeds, qpos_dim), dtype=np.float32)
            # Save head_camera_0 thumbnail for visual sanity
            images_buf = np.zeros((n_seeds, 3, 60, 80), dtype=np.float32)
            print(f"  feat_dim={feat_dim}  action_chunk=({Ta},{Da})  qpos_dim={qpos_dim}", flush=True)

        feats_buf[si] = f[0]
        actions_buf[si] = a
        qpos_buf[si] = agent_pos
        import cv2 as _cv2
        img_hwc = np.moveaxis(obs_d['head_camera_0'], 0, -1)
        img_small = _cv2.resize(img_hwc, (80, 60), interpolation=_cv2.INTER_AREA)
        images_buf[si] = np.moveaxis(img_small, -1, 0)

        if (si + 1) % 4 == 0 or si == n_seeds - 1:
            print(f"  seed-idx {si+1}/{n_seeds} done", flush=True)
    env.close()

    # Slice per-arm action chunk: action_dim split equally across arms
    n_agents = args.n_agents
    Da_per = actions_buf.shape[-1] // n_agents
    actions_per_arm = actions_buf.reshape(n_seeds, actions_buf.shape[1], n_agents, Da_per)

    np.savez(
        args.out,
        seeds=np.array(args.seeds, dtype=np.int64),
        features=feats_buf,            # (N, F)  -- single shared joint feature
        actions=actions_buf,            # (N, Ta, Da_total)
        actions_per_arm=actions_per_arm,  # (N, Ta, n_agents, Da_per)
        qpos=qpos_buf,                  # (N, qpos_total)
        images=images_buf,              # (N, 3, 60, 80) head_camera_0 thumb
        cubes=cubes,                    # (N, 3, 3)
        ckpt_path=str(args.ckpt_path),
        config=args.config,
    )
    print(f"saved {args.out}", flush=True)

    # Diagnostics
    F = feats_buf
    cs = (F @ F.T) / (np.linalg.norm(F, axis=1, keepdims=True)
                      @ np.linalg.norm(F, axis=1, keepdims=True).T + 1e-9)
    off = cs[np.triu_indices(n_seeds, k=1)]
    var_per_dim = F.var(axis=0).mean()
    print(f"  joint feat cos μ={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}  var/dim={var_per_dim:.3e}",
          flush=True)


if __name__ == '__main__':
    main()
