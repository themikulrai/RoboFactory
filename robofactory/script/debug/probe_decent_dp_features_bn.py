"""A0: BN-mode x epoch sweep probe.

Identical to probe_decent_dp_features.py but:
- accepts multiple --checkpoint-num values (sweeps epochs)
- forwards encoder in BOTH eval-mode AND train-mode (for BN running-stats vs. weight collapse discrimination)
- writes per-(epoch, mode) feature buffers to a single .npz per arm and a global summary.txt

Skips epochs whose ckpts do not exist on disk.
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
def encoder_features(policy, obs_dict_np, train_mode: bool = False):
    device = policy.device
    To = policy.n_obs_steps
    if train_mode:
        policy.obs_encoder.train()
    else:
        policy.obs_encoder.eval()
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = dict_apply(stacked, lambda x: torch.from_numpy(x).to(device=device))
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    nobs = policy.normalizer.normalize(obs_dict_b)
    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    feats = policy.obs_encoder(this_nobs)
    return feats.cpu().numpy()


def ckpt_exists(env_id, data_num, agent_id, ckpt_suffix, epoch):
    if ckpt_suffix:
        ckpt_dir = f'checkpoints/{env_id}_agent{agent_id}_{ckpt_suffix}_{data_num}'
    else:
        ckpt_dir = f'checkpoints/{env_id}_Agent{agent_id}_{data_num}'
    return os.path.exists(f'{ckpt_dir}/{epoch}.ckpt')


def pairwise_cos(F):
    n = np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    return (F @ F.T) / (n @ n.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-nums', type=int, nargs='+', required=True,
                    help='List of epochs to sweep.')
    ap.add_argument('--ckpt-suffix', default="")
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=240)
    ap.add_argument('--img-width', type=int, default=320)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
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

    # Filter checkpoint epochs to existing ones (per arm 0; assume same for all arms)
    valid_epochs = [e for e in args.checkpoint_nums
                    if ckpt_exists(env_id, args.data_num, 0, args.ckpt_suffix, e)]
    missing_epochs = [e for e in args.checkpoint_nums if e not in valid_epochs]
    if missing_epochs:
        print(f"Missing ckpts (skipping): {missing_epochs}", flush=True)
    print(f"Sweeping epochs: {valid_epochs}", flush=True)

    n_seeds = len(args.seeds)
    cube_names = ['cubeA', 'cubeB', 'cubeC']

    summary_lines = []
    summary_lines.append(f"# A0 BN-mode x epoch sweep  config={args.config}  ckpt_suffix={args.ckpt_suffix}")
    summary_lines.append(f"# n_seeds={n_seeds}  agents={agent_num}  epochs={valid_epochs}")
    summary_lines.append(f"{'epoch':>6} {'mode':>6} {'arm':>3}  {'cos μ':>8} {'min':>8} {'max':>8}  {'var/F':>10}")

    # Pre-collect cubes + obs_d once across seeds (independent of ckpt)
    cubes = np.full((n_seeds, 3, 3), np.nan, dtype=np.float32)
    OPEN = 0.04
    obs_per_seed_arm = {}  # (si, aid) -> obs_d dict
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

    # Per-arm output: dict of arrays {epoch}_{mode} -> features
    for aid in range(agent_num):
        out_dict = {
            'seeds': np.array(args.seeds, dtype=np.int64),
            'cubes': cubes,
            'epochs': np.array(valid_epochs, dtype=np.int64),
        }
        # We'll store features as (n_epochs, n_seeds, feat_dim) for each mode
        feats_eval = None
        feats_train = None
        for ei, epoch in enumerate(valid_epochs):
            print(f"\n[arm{aid} epoch{epoch}] loading ckpt...", flush=True)
            try:
                dp = DP(env_id, epoch, args.data_num, id=aid, ckpt_suffix=args.ckpt_suffix)
            except Exception as e:
                print(f"  FAILED to load: {e}", flush=True)
                continue
            policy = dp.policy

            for mode_name, train_mode in [('eval', False), ('train', True)]:
                feats_seeds = []
                for si in range(n_seeds):
                    obs_d = obs_per_seed_arm[(si, aid)]
                    f = encoder_features(policy, obs_d, train_mode=train_mode)
                    feats_seeds.append(f[0])
                feats_seeds = np.stack(feats_seeds, axis=0)  # (n_seeds, feat_dim)
                if mode_name == 'eval':
                    if feats_eval is None:
                        feats_eval = np.zeros((len(valid_epochs), n_seeds, feats_seeds.shape[-1]), dtype=np.float32)
                    feats_eval[ei] = feats_seeds
                else:
                    if feats_train is None:
                        feats_train = np.zeros((len(valid_epochs), n_seeds, feats_seeds.shape[-1]), dtype=np.float32)
                    feats_train[ei] = feats_seeds

                cs = pairwise_cos(feats_seeds)
                cs_off = cs[np.triu_indices(n_seeds, k=1)]
                var_per_dim = feats_seeds.var(axis=0).mean()
                line = (f"{epoch:>6} {mode_name:>6} {aid:>3}  "
                        f"{cs_off.mean():>8.4f} {cs_off.min():>8.4f} {cs_off.max():>8.4f}  "
                        f"{var_per_dim:>10.4e}")
                summary_lines.append(line)
                print(line, flush=True)

            # Free
            del dp, policy
            torch.cuda.empty_cache()

        out_dict['features_eval'] = feats_eval     # (n_epochs, n_seeds, feat_dim)
        out_dict['features_train'] = feats_train   # (n_epochs, n_seeds, feat_dim)
        out_path = os.path.join(args.out_dir, f'probe_bn_arm{aid}.npz')
        np.savez(out_path, **out_dict)
        print(f"\nsaved {out_path}", flush=True)

    summary_path = os.path.join(args.out_dir, 'bn_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nsaved {summary_path}", flush=True)


if __name__ == '__main__':
    main()
