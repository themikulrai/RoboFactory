"""T12: Check whether DDPM sampling is stochastic or deterministic.

Tests two things:
A) SAME seed, DIFFERENT torch seeds (5 runs) -> if outputs vary a lot,
   diffusion is stochastic (noise matters). If outputs are always the same,
   the model is effectively deterministic (or noise is washed out by conditioning).

B) DIFFERENT seeds (10011 success vs 10000 failure) with SAME torch seed ->
   if outputs are still identical, the conditioning truly has no effect.
   If outputs differ, the conditioning DOES have effect when noise is fixed.

Usage:
    python script/debug/check_diffusion_stochasticity.py
"""
import sys
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

import argparse
import numpy as np
import torch
import dill
import yaml
import gymnasium as gym
from collections import deque
from robofactory.tasks import *
import hydra
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

CKPT_PATH = '/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt'
CFG_PATH  = 'configs/table/pick_meat.yaml'

def load_policy(device='cuda:0'):
    payload = torch.load(CKPT_PATH, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy

def get_obs_from_env(seed, n_obs=3):
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'
    env = gym.make(env_id, config=CFG_PATH, obs_mode='rgb', control_mode='pd_joint_pos',
                   render_mode='sensors', num_envs=1, sim_backend='cpu', enable_shadow=False,
                   sensor_configs=dict(shader_pack='default'))
    obs, _ = env.reset(seed=seed)
    rgb = obs['sensor_data']['head_camera']['rgb']
    rgb_np = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
    if rgb_np.ndim == 4: rgb_np = rgb_np[0]
    head_chw = np.moveaxis(rgb_np, -1, 0).astype(np.float32) / 255.0
    qpos = obs['agent']['qpos']
    qpos_np = qpos.cpu().numpy() if hasattr(qpos, 'cpu') else np.asarray(qpos)
    qpos_np = qpos_np.squeeze(0)[:-2]
    agent_pos = np.append(qpos_np, 1.0)
    env.close()
    head = np.stack([head_chw]*n_obs, axis=0)[None]
    pos = np.stack([agent_pos]*n_obs, axis=0)[None]
    return {'head_cam': head.astype(np.float32), 'agent_pos': pos.astype(np.float32)}

def run_policy(policy, obs_dict, torch_seed=None, device='cuda:0'):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)
    obs_t = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device))
    with torch.no_grad():
        result = policy.predict_action(obs_t)
    return result['action'].cpu().numpy()[0]  # (n_exec, A)

print("Loading policy ...")
policy = load_policy()

print("\n=== TEST A: Same seed (10011, success), 5 different torch seeds ===")
print("Hypothesis: if outputs VARY -> DDPM is stochastic (noise matters)")
print("            if outputs are IDENTICAL -> DDPM is deterministic (or conditioning dominates)")
obs_11 = get_obs_from_env(10011)
results_a = []
for ts in [0, 42, 100, 999, 12345]:
    a = run_policy(policy, obs_11, torch_seed=ts)
    results_a.append(a)
    print(f"  torch_seed={ts:>6}: step0={a[0].round(4).tolist()}")

results_a = np.array(results_a)  # (5, 6, 8)
std_across_torch = results_a.std(axis=0)  # (6, 8)
print(f"\n  Std across torch seeds (mean per dim at step0): {std_across_torch[0].round(4).tolist()}")
print(f"  Max std at step0: {std_across_torch[0].max():.4f}")
if std_across_torch[0].max() > 0.01:
    print("  VERDICT: DDPM IS STOCHASTIC (noise matters significantly)")
else:
    print("  VERDICT: DDPM output is NEAR-DETERMINISTIC (same conditioning -> same output)")

print("\n=== TEST B: Different seeds (10011 success vs 10000 failure), same torch seed=42 ===")
print("Hypothesis: if outputs IDENTICAL -> policy has NO image conditioning effect")
print("            if outputs DIFFER    -> policy DOES condition on image (but maybe weakly)")
obs_00 = get_obs_from_env(10000)
obs_05 = get_obs_from_env(10005)

torch.manual_seed(42)
a_11 = run_policy(policy, obs_11, torch_seed=42)
a_00 = run_policy(policy, obs_00, torch_seed=42)
a_05 = run_policy(policy, obs_05, torch_seed=42)

print(f"\n  seed=10011 (success y=+0.021): step0={a_11[0].round(4).tolist()}")
print(f"  seed=10000 (failure y=-0.016): step0={a_00[0].round(4).tolist()}")
print(f"  seed=10005 (failure y=-0.044): step0={a_05[0].round(4).tolist()}")
diff_11_00 = np.abs(a_11 - a_00)
diff_11_05 = np.abs(a_11 - a_05)
print(f"\n  |pred(10011) - pred(10000)| mean={diff_11_00.mean():.6f}  max={diff_11_00.max():.6f}")
print(f"  |pred(10011) - pred(10005)| mean={diff_11_05.mean():.6f}  max={diff_11_05.max():.6f}")
if diff_11_00.max() < 1e-4:
    print("\n  VERDICT B: CONDITIONING HAS NO EFFECT (predictions identical regardless of image)")
elif diff_11_00.mean() < 0.01:
    print("\n  VERDICT B: CONDITIONING IS VERY WEAK (predictions nearly identical despite different images)")
else:
    print(f"\n  VERDICT B: CONDITIONING IS ACTIVE (different images -> different predictions, diff={diff_11_00.mean():.4f})")

print("\n=== TEST C: ResNet features for success vs failure seed ===")
# Feed same image to both seeds; if features differ, conditioning input is different
obs_t_11 = dict_apply(obs_11, lambda x: torch.from_numpy(x).to('cuda:0'))
obs_t_00 = dict_apply(obs_00, lambda x: torch.from_numpy(x).to('cuda:0'))

with torch.no_grad():
    # Normalize observations
    nobs_11 = policy.normalizer.normalize(obs_t_11)
    nobs_00 = policy.normalizer.normalize(obs_t_00)
    # Get image features through obs encoder
    # For ConditionalUnet1D policy, the obs encoder processes head_cam
    try:
        feat_11 = policy.obs_encoder(nobs_11)  # (B, D)
        feat_00 = policy.obs_encoder(nobs_00)
        diff_feat = (feat_11 - feat_00).abs()
        print(f"  ObsEncoder feature shape: {feat_11.shape}")
        print(f"  Feature |diff| mean={diff_feat.mean().item():.6f}  max={diff_feat.max().item():.6f}")
        rel_diff = diff_feat.mean().item() / (feat_11.abs().mean().item() + 1e-9)
        print(f"  Relative feature diff: {rel_diff:.4f} (>0.01 = meaningful conditioning signal)")
        if rel_diff < 0.001:
            print("  VERDICT C: IMAGE FEATURES ARE NEAR-IDENTICAL for different meat positions")
            print("    -> The image encoder does NOT discriminate meat position sufficiently")
        else:
            print(f"  VERDICT C: Image features differ by {rel_diff:.2%} -> conditioning signal IS present")
    except Exception as e:
        print(f"  Could not access obs_encoder directly: {e}")
        print("  Trying manual feature extraction ...")
        # Try accessing through model structure
        try:
            heads = {'head_cam': nobs_11['head_cam'], 'agent_pos': nobs_11['agent_pos']}
            feat_11 = policy.model.obs_encoder(heads)
            heads = {'head_cam': nobs_00['head_cam'], 'agent_pos': nobs_00['agent_pos']}
            feat_00 = policy.model.obs_encoder(heads)
            diff_feat = (feat_11 - feat_00).abs()
            print(f"  Feature diff mean={diff_feat.mean().item():.6f}  max={diff_feat.max().item():.6f}")
        except Exception as e2:
            print(f"  Also failed: {e2}")
