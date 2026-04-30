"""T17: Test whether the policy conditions on image at step N (after arm has moved).

At step 0: obs deque = [frame0, frame0, frame0] (degenerate padding)
At step N: obs deque = [frame_{N-2}, frame_{N-1}, frame_N] (real temporal context)

Hypothesis H1: if policy conditions differently at step N vs step 0,
the problem is the degenerate step-0 padding, NOT weak image conditioning.
Fix: warm up the obs deque with real env steps before starting DP.

Protocol:
1. Reset env for success/failure seeds
2. Execute a fixed "hold position" action for n_warmup steps to build a real obs deque
3. Query policy at step N with real temporal context
4. Compare predictions across success vs failure seed

Usage:
    python script/debug/test_step_n_conditioning.py
"""
import sys
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

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
N_OBS     = 3   # n_obs_steps from checkpoint

def load_policy(device='cuda:0'):
    payload = torch.load(CKPT_PATH, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy, cfg

def make_env(seed):
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'
    env = gym.make(env_id, config=CFG_PATH, obs_mode='rgb', control_mode='pd_joint_pos',
                   render_mode='sensors', num_envs=1, sim_backend='cpu', enable_shadow=False,
                   sensor_configs=dict(shader_pack='default'))
    obs, _ = env.reset(seed=seed)
    return env, obs

def obs_to_np(obs):
    rgb = obs['sensor_data']['head_camera']['rgb']
    rgb_np = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
    if rgb_np.ndim == 4: rgb_np = rgb_np[0]
    head_chw = np.moveaxis(rgb_np, -1, 0).astype(np.float32) / 255.0
    qpos = obs['agent']['qpos']
    qpos_np = qpos.cpu().numpy() if hasattr(qpos, 'cpu') else np.asarray(qpos)
    qpos_np = qpos_np.squeeze(0)[:-2]
    agent_pos = np.append(qpos_np, 1.0)
    return head_chw, agent_pos

def get_meat_pos(env):
    for a in env.unwrapped.scene.actors.values():
        if 'meat' in a.name.lower():
            p = a.pose.p
            if hasattr(p, 'cpu'): p = p.cpu().numpy()
            return np.asarray(p).flatten()[:3]
    return None

def build_obs_dict_from_deque(frame_deque, pos_deque):
    head = np.stack(list(frame_deque), axis=0)[None]   # (1, N_OBS, C, H, W)
    pos  = np.stack(list(pos_deque),  axis=0)[None]    # (1, N_OBS, D)
    return {'head_cam': head.astype(np.float32), 'agent_pos': pos.astype(np.float32)}

def run_policy(policy, obs_dict, torch_seed=42, device='cuda:0'):
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(torch_seed)
    obs_t = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device))
    with torch.no_grad():
        result = policy.predict_action(obs_t)
    return result['action'].cpu().numpy()[0]

print("Loading policy ...")
policy, cfg = load_policy()

# Seeds: success=10011 (meat_y≈+0.021), failure=10000 (meat_y≈-0.016), failure=10005 (meat_y≈-0.044)
TEST_SEEDS = [
    (10011, 'SUCCESS y≈+0.021'),
    (10000, 'FAILURE y≈-0.016'),
    (10005, 'FAILURE y≈-0.044'),
]

for n_warmup in [0, 2, 5, 10]:
    print(f"\n{'='*60}")
    print(f"WARMUP STEPS = {n_warmup}  (obs deque has {'degenerate pad' if n_warmup==0 else 'real temporal context'})")
    print(f"{'='*60}")

    preds = {}
    meat_positions = {}
    for seed, label in TEST_SEEDS:
        env, obs = make_env(seed)
        meat_pos = get_meat_pos(env)
        meat_positions[seed] = meat_pos

        # Build initial obs deque with padding (all copies of frame 0)
        frame0, pos0 = obs_to_np(obs)
        frame_deque = deque([frame0.copy() for _ in range(N_OBS)], maxlen=N_OBS)
        pos_deque   = deque([pos0.copy()   for _ in range(N_OBS)], maxlen=N_OBS)

        # Execute hold-position (repeat qpos) for n_warmup steps
        for step in range(n_warmup):
            # Hold: command current qpos (first 7 dims) + open gripper.
            # PM is single-robot (panda), so action is a plain (1, 8) array.
            qpos_cmd = np.concatenate([pos0[:7], [1.0]])[None]  # (1, 8)
            obs, _, term, trunc, _ = env.step(qpos_cmd)
            frame_new, pos_new = obs_to_np(obs)
            frame_deque.append(frame_new)
            pos_deque.append(pos_new)

        obs_dict = build_obs_dict_from_deque(frame_deque, pos_deque)
        pred = run_policy(policy, obs_dict, torch_seed=42)
        preds[seed] = pred
        env.close()

        if meat_pos is not None:
            print(f"  seed={seed:5d} ({label}): meat=({meat_pos[0]:.4f},{meat_pos[1]:.4f})  step0_pred={pred[0].round(4).tolist()}")
        else:
            print(f"  seed={seed:5d} ({label}): meat=None  step0_pred={pred[0].round(4).tolist()}")

    # Compare predictions
    a_11, a_00, a_05 = preds[10011], preds[10000], preds[10005]
    diff_11_00 = np.abs(a_11 - a_00)
    diff_11_05 = np.abs(a_11 - a_05)
    print(f"\n  |pred(10011) - pred(10000)| mean={diff_11_00.mean():.6f}  max={diff_11_00.max():.6f}")
    print(f"  |pred(10011) - pred(10005)| mean={diff_11_05.mean():.6f}  max={diff_11_05.max():.6f}")
    if diff_11_00.max() < 1e-4:
        print(f"  VERDICT(warmup={n_warmup}): STILL NO CONDITIONING EFFECT")
    elif diff_11_00.mean() < 0.01:
        print(f"  VERDICT(warmup={n_warmup}): VERY WEAK conditioning (mean diff < 0.01)")
    else:
        print(f"  VERDICT(warmup={n_warmup}): CONDITIONING IS ACTIVE (mean diff={diff_11_00.mean():.4f})")

print("\n=== SUMMARY ===")
print("If warmup=0 shows no conditioning but warmup>0 does -> H1 confirmed (degenerate padding)")
print("If all warmup levels show no conditioning -> H2 confirmed (weak image features)")
print("If warmup=0 already shows conditioning -> policy conditions from frame 0 (contradicts smoking gun)")
