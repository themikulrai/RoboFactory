"""Probe whether D1 TSC decent DP encoder produces distinct features for different cube
positions, and whether the policy's predicted action chunk varies with the scene.

Usage (run from /iris/u/mikulrai/projects/RoboFactory/robofactory):
    python script/debug/probe_decent_dp_features.py \
        --config=configs/table/three_robots_stack_cube.yaml \
        --data-num=150 --checkpoint-num=300 --ckpt-suffix="" \
        --obs-cam-family=workspace --no-include-global \
        --img-height=240 --img-width=320 \
        --seeds 10000 10001 10002 10003 10004 10005 10006 10007 \
        --out /iris/u/mikulrai/logs/phase2_debug/probe_d1_decent_features.npz

Or with explicit ckpt paths (one per agent, repeat the flag) — used by the
training-loop auto-capture hook (workflow improvement #4):
    python script/debug/probe_decent_dp_features.py \
        --config=... --img-height=240 --img-width=320 \
        --ckpt-path /path/to/agent0_ckpt.ckpt \
        --ckpt-path /path/to/agent1_ckpt.ckpt \
        --seeds ... --out runs/calibration/<run_id>.npz

Saves a .npz with per-seed, per-agent: encoder feature vector, predicted action chunk,
head_cam thumbnail, and cubeA/B/C xyz.
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import numpy as np
import torch
import yaml
import gymnasium as gym

from robofactory.tasks import *  # registers env ids
import robofactory.agents  # noqa
from diffusion_policy.common.pytorch_util import dict_apply

# Reuse helpers from eval_multi_dp
from eval_multi_dp import DP, get_model_input, get_policy


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
    """Replicate dp_runner.get_action preprocessing up to obs_encoder, returns (To, feat_dim) numpy."""
    device = policy.device
    To = policy.n_obs_steps
    # Stack To copies of the same obs (matches DPRunner.stack_last_n_obs at reset time)
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = dict_apply(stacked, lambda x: torch.from_numpy(x).to(device=device))
    # Add batch dim → (B=1, To, ...)
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    nobs = policy.normalizer.normalize(obs_dict_b)
    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    feats = policy.obs_encoder(this_nobs)  # (To, feat_dim)
    return feats.cpu().numpy()


@torch.no_grad()
def predict_action_np(policy, obs_dict_np):
    """Full predict_action; returns action chunk (n_action_steps, action_dim) numpy."""
    device = policy.device
    To = policy.n_obs_steps
    stacked = {k: np.stack([v] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = dict_apply(stacked, lambda x: torch.from_numpy(x).to(device=device))
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    out = policy.predict_action(obs_dict_b)
    return out['action'].squeeze(0).cpu().numpy()


def probe_collapse_features(
    config_path: str,
    output_npz: str,
    seeds,
    *,
    ckpt_paths=None,
    data_num: int = 150,
    checkpoint_num: int = 300,
    ckpt_suffix: str = "",
    obs_cam_family: str = "workspace",
    include_global: bool = True,
    img_height: int = 224,
    img_width: int = 224,
    device: str = "cuda:0",
):
    """Programmatic entry-point for the encoder-collapse calibration probe.

    Two ckpt resolution modes:
    - ckpt_paths is None  → resolve via DP(env_id, checkpoint_num, data_num,
      id=i, ckpt_suffix=...) for i in [0..agent_num). Mirrors the original CLI.
    - ckpt_paths is a list → load each ckpt directly via get_policy(...).
      Length must equal env's agent_num.

    Saves the same NPZ schema as the original probe so downstream hash-pinning
    in `utils/test_calibration_integrity.py` and analysis tooling continue to
    work. NPZ keys: seeds, features, actions, images, qpos, cubes, ckpt_paths,
    config, obs_cam_family, include_global, img_height, img_width.

    Returns the absolute path to the saved NPZ.
    """
    seeds = list(seeds)

    with open(config_path, 'r') as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    env = gym.make(
        env_id,
        config=config_path,
        obs_mode='rgb',
        control_mode='pd_joint_pos',
        render_mode='sensors',
        num_envs=1,
        sim_backend='cpu',
        enable_shadow=False,
        sensor_configs=dict(shader_pack='default'),
    )
    raw_obs, _ = env.reset(seed=seeds[0])
    # Single-agent envs (PickMeat) expose env.unwrapped.agent as the Panda directly;
    # multi-agent envs expose a MultiAgent wrapper with .agents list.
    _agent = env.unwrapped.agent
    if hasattr(_agent, 'agents'):
        agent_prefix = _agent.agents[0].uid
        agent_num = len(_agent.agents)
    else:
        agent_prefix = getattr(_agent, 'uid', 'panda')
        agent_num = 1
    print(f"env_id={env_id} agent_prefix={agent_prefix} agent_num={agent_num}", flush=True)

    if ckpt_paths is None:
        dp_models = [
            DP(env_id, checkpoint_num, data_num, id=i, ckpt_suffix=ckpt_suffix)
            for i in range(agent_num)
        ]
        resolved_paths = [m.ckpt_path for m in dp_models]
        policies = [m.policy for m in dp_models]
    else:
        ckpt_paths = list(ckpt_paths)
        if len(ckpt_paths) == 1 and agent_num > 1:
            # auto-broadcast: same ckpt for every arm (single-policy multi-arm scenes
            # we don't currently use, but keeps the hook simple to wire in)
            ckpt_paths = ckpt_paths * agent_num
        if len(ckpt_paths) != agent_num:
            raise ValueError(
                f"ckpt_paths length {len(ckpt_paths)} != agent_num {agent_num} "
                f"for env_id={env_id}. Provide one ckpt per agent."
            )
        resolved_paths = ckpt_paths
        policies = [get_policy(p, None, device) for p in ckpt_paths]
    print(f"Loaded {agent_num} ckpts: {resolved_paths}", flush=True)

    n_seeds = len(seeds)
    feat_dim = None
    feats_buf = None
    actions_buf = None
    images_buf = None
    cubes = np.full((n_seeds, 3, 3), np.nan, dtype=np.float32)  # (N, {A,B,C}, xyz)
    qpos_buf = None
    OPEN = 0.04
    cube_names = ['cubeA', 'cubeB', 'cubeC']

    for si, seed in enumerate(seeds):
        raw_obs, _ = env.reset(seed=int(seed))
        for ci, cn in enumerate(cube_names):
            p = cube_xyz(env, cn)
            if p is not None:
                cubes[si, ci] = p

        for aid in range(agent_num):
            # Single-agent envs expose raw_obs['agent'] as flat {'qpos','qvel'};
            # multi-agent envs nest per-arm under '{prefix}-{aid}'.
            _ag = raw_obs['agent']
            _key = f'{agent_prefix}-{aid}'
            _qsrc = _ag[_key]['qpos'] if _key in _ag else _ag['qpos']
            qpos = _qsrc.squeeze(0)[:-2].cpu().numpy()
            qpos_with_grip = np.append(qpos, OPEN)
            obs_d = get_model_input(
                raw_obs, qpos_with_grip, aid,
                include_global=include_global,
                cam_family=obs_cam_family,
                img_h=img_height, img_w=img_width,
            )
            policy = policies[aid]
            feats = encoder_features(policy, obs_d)  # (To, feat_dim)
            action = predict_action_np(policy, obs_d)  # (Ta, Da)

            if feats_buf is None:
                feat_dim = feats.shape[-1]
                Ta, Da = action.shape
                qpos_dim = qpos_with_grip.shape[0]
                feats_buf = np.zeros((n_seeds, agent_num, feat_dim), dtype=np.float32)
                actions_buf = np.zeros((n_seeds, agent_num, Ta, Da), dtype=np.float32)
                # store small thumbnail of head_cam (60x80 uint8)
                images_buf = np.zeros((n_seeds, agent_num, 3, 60, 80), dtype=np.float32)
                qpos_buf = np.zeros((n_seeds, agent_num, qpos_dim), dtype=np.float32)
                print(f"feat_dim={feat_dim} action_chunk={action.shape}", flush=True)

            feats_buf[si, aid] = feats[0]  # all To timesteps identical at reset
            actions_buf[si, aid] = action
            qpos_buf[si, aid] = qpos_with_grip
            # downsample image for storage
            img = obs_d['head_cam']  # (3, H, W) float [0,1]
            import cv2 as _cv2
            img_hwc = np.moveaxis(img, 0, -1)
            img_small = _cv2.resize(img_hwc, (80, 60), interpolation=_cv2.INTER_AREA)
            images_buf[si, aid] = np.moveaxis(img_small, -1, 0)

        if (si + 1) % 4 == 0 or si == n_seeds - 1:
            print(f"  seed {seed} done ({si+1}/{n_seeds})", flush=True)

    env.close()

    np.savez(
        output_npz,
        seeds=np.array(seeds, dtype=np.int64),
        features=feats_buf,        # (N, A, F)
        actions=actions_buf,        # (N, A, Ta, Da)
        images=images_buf,          # (N, A, 3, 60, 80) float [0,1]
        qpos=qpos_buf,              # (N, A, qpos_dim)
        cubes=cubes,                # (N, 3, 3)
        ckpt_paths=np.array(resolved_paths),
        config=config_path,
        obs_cam_family=obs_cam_family,
        include_global=include_global,
        img_height=img_height,
        img_width=img_width,
    )
    print(f"\nSaved {output_npz}", flush=True)
    print(f"  features: {feats_buf.shape}  actions: {actions_buf.shape}", flush=True)
    print(f"  cube std (xy, m): A={np.nanstd(cubes[:,0,:2], axis=0).round(4)}  "
          f"B={np.nanstd(cubes[:,1,:2], axis=0).round(4)}  "
          f"C={np.nanstd(cubes[:,2,:2], axis=0).round(4)}", flush=True)
    # quick on-the-fly diagnostics
    for aid in range(agent_num):
        F = feats_buf[:, aid]                     # (N, F)
        cs = (F @ F.T) / (np.linalg.norm(F, axis=1, keepdims=True)
                          @ np.linalg.norm(F, axis=1, keepdims=True).T + 1e-9)
        off = cs[np.triu_indices(n_seeds, k=1)]
        A = actions_buf[:, aid].reshape(n_seeds, -1)  # (N, Ta*Da)
        a_pair = np.linalg.norm(A[:, None] - A[None, :], axis=-1)
        a_off = a_pair[np.triu_indices(n_seeds, k=1)]
        print(f"  arm{aid}: feat cos-sim mean={off.mean():.4f}  min={off.min():.4f}  max={off.max():.4f}  | "
              f"action L2 mean={a_off.mean():.4f}  min={a_off.min():.4f}  max={a_off.max():.4f}",
              flush=True)
    return output_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=300)
    ap.add_argument('--ckpt-suffix', default="")
    # Additive: caller may pass one or more explicit ckpt paths to skip the
    # implicit DP(env_id, checkpoint_num, data_num) filename derivation. Used
    # by the auto-capture-calibration training-loop hook (workflow #4) so the
    # hook can pass `--ckpt-path <path-trainer-just-saved>`. Repeat the flag
    # for multi-arm runs.
    ap.add_argument('--ckpt-path', dest='ckpt_paths', action='append', default=None,
                    help='explicit ckpt path; repeat for each agent')
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=224)
    ap.add_argument('--img-width', type=int, default=224)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    probe_collapse_features(
        config_path=args.config,
        output_npz=args.out,
        seeds=args.seeds,
        ckpt_paths=args.ckpt_paths,
        data_num=args.data_num,
        checkpoint_num=args.checkpoint_num,
        ckpt_suffix=args.ckpt_suffix,
        obs_cam_family=args.obs_cam_family,
        include_global=args.include_global,
        img_height=args.img_height,
        img_width=args.img_width,
    )


if __name__ == '__main__':
    main()
