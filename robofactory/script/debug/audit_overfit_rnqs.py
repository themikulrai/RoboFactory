"""Tier R+N+Q+S — combined audit for the overfit ckpts.

R: print normalizer params (action + agent_pos) for both EMA and raw
N: env.reset(seed=17); predict_action; compare chunk[0] vs recorded zarr action[0]
Q: same as N but with raw (non-EMA) weights
S: 10x predict_action at same obs; measure std across DDPM samples

Runs for both ckpts:
  - state=action: workspace_overfit1 (the original buggy ckpt)
  - state=qpos:   workspace_overfit1qpos (the converter-fix ckpt)

Both arms (agent0, agent1).
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os
from collections import deque

import numpy as np
import torch
import dill
import hydra
import yaml
import zarr
import gymnasium as gym

from robofactory.tasks import *  # noqa
import robofactory.agents  # noqa
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

from eval_multi_dp import get_model_input
from diffusion_policy.common.pytorch_util import dict_apply


def _qpos_agent_pos(raw_obs, arm_id, agent_prefix):
    qp = raw_obs['agent'][f'{agent_prefix}-{arm_id}']['qpos'].squeeze(0).cpu().numpy()
    return np.concatenate([qp[:7], qp[7:8]], axis=0).astype(np.float32)


def _action_agent_pos(raw_obs, arm_id, agent_prefix, gripper_value=0.04):
    """Default eval proprio: qpos[:7] + gripper_value. Matches eval_multi_dp.py:248."""
    qp = raw_obs['agent'][f'{agent_prefix}-{arm_id}']['qpos'].squeeze(0).cpu().numpy()
    return np.concatenate([qp[:7], np.array([gripper_value], dtype=np.float32)]).astype(np.float32)


def load_policy(ckpt_path, use_ema=True, device='cuda:0'):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg, output_dir=None)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if (use_ema and cfg.training.use_ema) else ws.model
    policy.to(device).eval()
    return policy, cfg


def build_obs_dict(single_obs, n_obs_steps=3, device='cuda:0'):
    """Replicate the single-step obs n_obs_steps times to match runner behavior at t=0."""
    out = {}
    for k, v in single_obs.items():
        if k == 'agent_pos':
            arr = np.stack([v] * n_obs_steps, axis=0)  # (n_obs, D)
        else:
            arr = np.stack([v] * n_obs_steps, axis=0)  # (n_obs, C, H, W)
        out[k] = torch.from_numpy(arr).unsqueeze(0).to(device=device)
    return out


def normalizer_summary(policy):
    """Pull min/max/mean/scale per key from policy.normalizer."""
    nrm = policy.normalizer
    summary = {}
    for key in ['action', 'agent_pos']:
        if key not in nrm.params_dict:
            continue
        p = nrm.params_dict[key]
        d = {}
        for k in ['input_stats', 'scale', 'offset']:
            if k == 'input_stats' and k in p:
                d[k] = {kk: vv.detach().cpu().numpy().tolist() for kk, vv in p[k].items()}
            elif k in p:
                d[k] = p[k].detach().cpu().numpy().tolist()
        summary[key] = d
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/table/two_robots_stack_cube.yaml')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--n-samples', type=int, default=10)
    ap.add_argument('--robot-uids', default='panda_wristcam_multi,panda_wristcam_multi')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    with open(args.config) as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    env_kwargs = dict(
        config=args.config, obs_mode='rgb', control_mode='pd_joint_pos',
        render_mode='sensors',
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=1, sim_backend='gpu', enable_shadow=False,
    )
    if args.robot_uids:
        env_kwargs['robot_uids'] = tuple(args.robot_uids.split(','))
    env = gym.make(env_id, **env_kwargs)
    raw_obs, _ = env.reset(seed=int(args.seed))
    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'

    # Recorded action[0] from each ckpt's zarr
    def load_recorded_action0(zarr_path, ep_idx=13):
        r = zarr.open(zarr_path, mode='r')
        ends = np.asarray(r['meta/episode_ends'])
        start = int(0 if ep_idx == 0 else ends[ep_idx - 1])
        return np.asarray(r['data/action'][start]).tolist(), start

    out = {'env_id': env_id, 'seed': args.seed, 'agent_prefix': agent_prefix}

    ckpt_specs = [
        {
            'name': 'state=action',
            'ckpt_suffix': 'workspace_overfit1',
            'zarr_tpl': '/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_overfit1_agent{i}_150.zarr',
            'agent_pos_fn': _action_agent_pos,
        },
        {
            'name': 'state=qpos',
            'ckpt_suffix': 'workspace_overfit1qpos',
            'zarr_tpl': '/iris/u/mikulrai/data/RoboFactory/zarr_data/TwoRobotsStackCube-rf_workspace_overfit1qpos_agent{i}_150.zarr',
            'agent_pos_fn': _qpos_agent_pos,
        },
    ]

    for spec in ckpt_specs:
        spec_out = {'agents': {}}
        for arm_id in (0, 1):
            ckpt = f'checkpoints/{env_id}_{spec["ckpt_suffix"]}_agent{arm_id}_150/2000.ckpt'
            zpath = spec['zarr_tpl'].format(i=arm_id)
            recorded, ep_start = load_recorded_action0(zpath)
            agent_pos = spec['agent_pos_fn'](raw_obs, arm_id, agent_prefix)
            single = get_model_input(raw_obs, agent_pos, arm_id,
                                      include_global=True, cam_family='workspace',
                                      img_h=224, img_w=224)
            arm_out = {'ckpt': ckpt, 'zarr': zpath, 'ep_start': ep_start,
                       'recorded_action0': recorded, 'agent_pos_used': agent_pos.tolist(),
                       'modes': {}}
            for mode in ('ema', 'raw'):
                policy, cfg = load_policy(ckpt, use_ema=(mode == 'ema'))
                obs_dict = build_obs_dict(single, n_obs_steps=policy.n_obs_steps)
                # S: 10x sampling, measure std
                preds = []
                with torch.no_grad():
                    for _ in range(args.n_samples):
                        res = policy.predict_action(obs_dict)
                        # action shape (B, n_action_steps, D) -> take first action only
                        a0 = res['action'][0, 0].detach().cpu().numpy()
                        preds.append(a0)
                preds = np.stack(preds, axis=0)  # (S, D)
                mean_a0 = preds.mean(axis=0)
                std_a0 = preds.std(axis=0)
                err = np.array(recorded) - mean_a0
                arm_out['modes'][mode] = {
                    'mean_pred_a0': mean_a0.tolist(),
                    'std_pred_a0': std_a0.tolist(),
                    'recorded_a0': recorded,
                    'err': err.tolist(),
                    'err_arm_l2': float(np.linalg.norm(err[:7])),
                    'err_gripper': float(err[7]),
                    'arm_max_std': float(std_a0[:7].max()),
                    'gripper_std': float(std_a0[7]),
                    'normalizer_summary': normalizer_summary(policy),
                }
                del policy
                torch.cuda.empty_cache()
            spec_out['agents'][f'arm{arm_id}'] = arm_out
        out[spec['name']] = spec_out

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n=== wrote {args.out} ===')
    env.close()


if __name__ == '__main__':
    main()
