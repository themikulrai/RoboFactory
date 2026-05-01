"""A4: in-distribution qpos-swap probe.

For each (seed si, arm aid):
  - Build obs_d normally for seed si.
  - Pick a different seed sj (random with fixed RNG seed for reproducibility).
  - Replace obs_d['agent_pos'] with the qpos that arm aid had at seed sj.
  - Re-run encoder + predict_action.

Saves both the standard (no-swap) buffers and the swap buffers in one .npz per arm.
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

from eval_multi_dp import DP, get_model_input


def cube_xyz(env, name):
    for a in env.unwrapped.scene.actors.values():
        if a.name.lower() == name.lower():
            p = a.pose.p
            if hasattr(p, 'cpu'):
                p = p.cpu().numpy()
            return np.asarray(p).flatten()[:3]
    return None


@torch.no_grad()
def encoder_features(policy, obs_dict_np):
    device = policy.device
    To = policy.n_obs_steps
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = dict_apply(stacked, lambda x: torch.from_numpy(x).to(device=device))
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    nobs = policy.normalizer.normalize(obs_dict_b)
    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    feats = policy.obs_encoder(this_nobs)
    return feats.cpu().numpy()


@torch.no_grad()
def predict_action_np(policy, obs_dict_np):
    device = policy.device
    To = policy.n_obs_steps
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = dict_apply(stacked, lambda x: torch.from_numpy(x).to(device=device))
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    out = policy.predict_action(obs_dict_b)
    return out['action'].squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=300)
    ap.add_argument('--ckpt-suffix', default="")
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=240)
    ap.add_argument('--img-width', type=int, default=320)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--rng-seed', type=int, default=12345)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    agent_num = len(env.unwrapped.agent.agents)
    print(f"env_id={env_id} agent_prefix={agent_prefix} agent_num={agent_num}", flush=True)

    dp_models = [DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix)
                 for i in range(agent_num)]
    print(f"Loaded {agent_num} ckpts.", flush=True)

    n_seeds = len(args.seeds)
    OPEN = 0.04
    cube_names = ['cubeA', 'cubeB', 'cubeC']

    cubes = np.full((n_seeds, 3, 3), np.nan, dtype=np.float32)

    # First pass: collect obs_d and qpos for each (seed, arm) without running the policy
    obs_per_seed_arm = {}
    qpos_per_seed_arm = {}
    for si, seed in enumerate(args.seeds):
        raw_obs, _ = env.reset(seed=int(seed))
        for ci, cn in enumerate(cube_names):
            p = cube_xyz(env, cn)
            if p is not None:
                cubes[si, ci] = p
        for aid in range(agent_num):
            qpos = raw_obs['agent'][f'{agent_prefix}-{aid}']['qpos'].squeeze(0)[:-2].cpu().numpy()
            qpos_with_grip = np.append(qpos, OPEN)
            obs_d = get_model_input(
                raw_obs, qpos_with_grip, aid,
                include_global=args.include_global,
                cam_family=args.obs_cam_family,
                img_h=args.img_height, img_w=args.img_width,
            )
            obs_per_seed_arm[(si, aid)] = obs_d
            qpos_per_seed_arm[(si, aid)] = qpos_with_grip
    env.close()

    # Pick swap source j for each i with fixed RNG (j != i)
    rng = np.random.default_rng(args.rng_seed)
    swap_sj = np.zeros(n_seeds, dtype=np.int64)
    for si in range(n_seeds):
        choices = [k for k in range(n_seeds) if k != si]
        swap_sj[si] = rng.choice(choices)

    # Allocate buffers
    feat_dim = None
    for si in range(n_seeds):
        for aid in range(agent_num):
            obs_d = obs_per_seed_arm[(si, aid)]
            f = encoder_features(dp_models[aid].policy, obs_d)
            a = predict_action_np(dp_models[aid].policy, obs_d)
            if feat_dim is None:
                feat_dim = f.shape[-1]
                Ta, Da = a.shape
                qpos_dim = qpos_per_seed_arm[(0, 0)].shape[0]
                feats_buf = np.zeros((n_seeds, agent_num, feat_dim), dtype=np.float32)
                actions_buf = np.zeros((n_seeds, agent_num, Ta, Da), dtype=np.float32)
                feats_swap = np.zeros((n_seeds, agent_num, feat_dim), dtype=np.float32)
                actions_swap = np.zeros((n_seeds, agent_num, Ta, Da), dtype=np.float32)
                qpos_buf = np.zeros((n_seeds, agent_num, qpos_dim), dtype=np.float32)
                qpos_swap_buf = np.zeros((n_seeds, agent_num, qpos_dim), dtype=np.float32)
                images_buf = np.zeros((n_seeds, agent_num, 3, 60, 80), dtype=np.float32)
            feats_buf[si, aid] = f[0]
            actions_buf[si, aid] = a
            qpos_buf[si, aid] = qpos_per_seed_arm[(si, aid)]
            import cv2 as _cv2
            img_hwc = np.moveaxis(obs_d['head_cam'], 0, -1)
            img_small = _cv2.resize(img_hwc, (80, 60), interpolation=_cv2.INTER_AREA)
            images_buf[si, aid] = np.moveaxis(img_small, -1, 0)

            # Swap variant: same obs image, but agent_pos from a different seed (same arm)
            sj = int(swap_sj[si])
            obs_d_swap = dict(obs_d)
            obs_d_swap['agent_pos'] = qpos_per_seed_arm[(sj, aid)]
            f_sw = encoder_features(dp_models[aid].policy, obs_d_swap)
            a_sw = predict_action_np(dp_models[aid].policy, obs_d_swap)
            feats_swap[si, aid] = f_sw[0]
            actions_swap[si, aid] = a_sw
            qpos_swap_buf[si, aid] = qpos_per_seed_arm[(sj, aid)]

        if (si + 1) % 4 == 0 or si == n_seeds - 1:
            print(f"  seed-idx {si+1}/{n_seeds} done", flush=True)

    out_path = os.path.join(args.out_dir, 'probe_qpos_swap.npz')
    np.savez(
        out_path,
        seeds=np.array(args.seeds, dtype=np.int64),
        swap_sj=swap_sj,
        features=feats_buf,
        features_swap=feats_swap,
        actions=actions_buf,
        actions_swap=actions_swap,
        qpos=qpos_buf,
        qpos_swap=qpos_swap_buf,
        images=images_buf,
        cubes=cubes,
        ckpt_paths=np.array([m.ckpt_path for m in dp_models]),
        config=args.config,
        rng_seed=args.rng_seed,
    )
    # Per-arm view too
    for aid in range(agent_num):
        per_arm = os.path.join(args.out_dir, f'probe_qpos_swap_arm{aid}.npz')
        np.savez(
            per_arm,
            seeds=np.array(args.seeds, dtype=np.int64),
            swap_sj=swap_sj,
            features=feats_buf[:, aid],
            features_swap=feats_swap[:, aid],
            actions=actions_buf[:, aid],
            actions_swap=actions_swap[:, aid],
            qpos=qpos_buf[:, aid],
            qpos_swap=qpos_swap_buf[:, aid],
            images=images_buf[:, aid],
            cubes=cubes,
            ckpt_path=str(dp_models[aid].ckpt_path),
        )
        print(f"saved {per_arm}", flush=True)
    print(f"saved {out_path}", flush=True)

    # Diagnostics
    for aid in range(agent_num):
        F = feats_buf[:, aid]
        Fsw = feats_swap[:, aid]
        n = np.linalg.norm(F, axis=1) + 1e-9
        nsw = np.linalg.norm(Fsw, axis=1) + 1e-9
        # cos sim between feature(seed_i) and feature_swap(seed_i)
        per_seed_cos = (F * Fsw).sum(axis=1) / (n * nsw)
        # Pairwise cos sim within swap features
        cs_sw = (Fsw @ Fsw.T) / (np.linalg.norm(Fsw, axis=1, keepdims=True)
                                 @ np.linalg.norm(Fsw, axis=1, keepdims=True).T + 1e-9)
        off_sw = cs_sw[np.triu_indices(n_seeds, k=1)]
        # Pairwise cos sim within original features
        cs = (F @ F.T) / (np.linalg.norm(F, axis=1, keepdims=True)
                          @ np.linalg.norm(F, axis=1, keepdims=True).T + 1e-9)
        off = cs[np.triu_indices(n_seeds, k=1)]
        # Action delta between original and swap
        A = actions_buf[:, aid].reshape(n_seeds, -1)
        Asw = actions_swap[:, aid].reshape(n_seeds, -1)
        a_delta = np.linalg.norm(A - Asw, axis=-1).mean()
        print(f"  arm{aid}: orig cos μ={off.mean():.4f}  swap cos μ={off_sw.mean():.4f}  "
              f"per-seed orig-vs-swap cos={per_seed_cos.mean():.4f}  Δaction L2={a_delta:.4f}",
              flush=True)


if __name__ == '__main__':
    main()
