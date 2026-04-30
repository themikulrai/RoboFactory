"""Encoder feature-discrimination probe: DINOv2 vs ResNet18 variants.

Loads head_camera frames from the eval env at 3 seeds (10011 success, 10000/10005 failure).
Measures mean relative feature diff between seeds for each encoder — the H2 metric.

Baseline reference:
  - random-init ResNet18 (current DP config)  → 0.03%  (policy is blind)
  - ImageNet ResNet18                          → 4.49%
  - R3M ResNet18                              → 6.37%

Run on a compute node (needs SAPIEN/Vulkan and internet for model download).

Usage:
    python script/debug/probe_encoder_dino.py [--device cuda:0] [--skip-env]
"""
import sys
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

import argparse
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as tvm
import yaml

import gymnasium as gym
from robofactory.tasks import *  # noqa: F401,F403 — registers gym envs

SEEDS = [10011, 10000, 10005]  # success, failure, failure
CFG_PATH = 'configs/table/pick_meat.yaml'


def get_rgb_from_env(seeds):
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    env_id = cfg['task_name'] + '-rf'
    env = gym.make(env_id, config=CFG_PATH, obs_mode='rgb', control_mode='pd_joint_pos',
                   render_mode='sensors', num_envs=1, sim_backend='cpu', enable_shadow=False,
                   sensor_configs=dict(shader_pack='default'))
    frames = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        rgb = obs['sensor_data']['head_camera']['rgb']
        rgb_np = rgb.cpu().numpy() if hasattr(rgb, 'cpu') else np.asarray(rgb)
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        frames.append(rgb_np.copy())  # (H, W, 3) uint8
    env.close()
    return frames


def probe_encoder(encoder, frames, device, name, input_size=224):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = torch.stack([transform(f) for f in frames]).to(device)  # (N, 3, H, W)
    encoder = encoder.to(device).eval()
    with torch.no_grad():
        features = encoder(tensors)  # (N, D)
    if isinstance(features, dict):
        features = features['x_norm_clstoken']  # timm DINOv2 style
    features = features.float()

    feat_norm = features.abs().mean().item()
    diffs = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            diff = (features[i] - features[j]).abs().mean().item()
            diffs.append(diff / (feat_norm + 1e-9))

    print(f"\n{'='*60}")
    print(f"Encoder: {name}")
    print(f"  Feature dim: {features.shape[1]},  ||f||_mean = {feat_norm:.4f}")
    labels = [f"seed{s}" for s in SEEDS]
    k = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            print(f"  {labels[i]} vs {labels[j]}: rel_diff = {diffs[k]:.4%}")
            k += 1
    print(f"  Mean rel_diff (all pairs): {np.mean(diffs):.4%}")
    return np.mean(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip-env', action='store_true',
                    help='Load frames from zarr instead of env (faster, no Vulkan needed)')
    args = ap.parse_args()

    device = args.device

    if args.skip_env:
        # Load first 3 episodes from zarr as a quick proxy
        import zarr
        z = zarr.open('/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/'
                      'PickMeat-rf_150.zarr', mode='r')
        ends = z['meta/episode_ends'][:]
        frames = []
        prev = 0
        for ep_end in ends[:3]:
            idx = prev  # first frame of each episode
            rgb = z['data/head_camera'][idx]  # (3, H, W) uint8
            frames.append(np.moveaxis(rgb, 0, -1))  # → (H, W, 3)
            prev = ep_end
        print(f"Loaded {len(frames)} zarr frames, shape {frames[0].shape}")
    else:
        print(f"Loading frames from eval env for seeds {SEEDS} ...")
        frames = get_rgb_from_env(SEEDS)
        print(f"Frame shape: {frames[0].shape}")

    results = {}

    # 1. Random-init ResNet18
    print("\n[1/4] Random-init ResNet18 ...")
    enc = tvm.resnet18(weights=None)
    enc.fc = torch.nn.Identity()
    results['resnet18_random'] = probe_encoder(enc, frames, device,
                                               "ResNet18 (random-init, current DP config)")

    # 2. ImageNet ResNet18
    print("\n[2/4] ImageNet ResNet18 ...")
    enc = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    enc.fc = torch.nn.Identity()
    results['resnet18_in1k'] = probe_encoder(enc, frames, device,
                                             "ResNet18 (ImageNet IMAGENET1K_V1)")

    import timm

    # 3. DINOv2 ViT-S/14 — CLS token (384-d) via timm (Python 3.9 compatible)
    print("\n[3/4] DINOv2 ViT-S/14 (timm) ...")
    dino_s = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0, img_size=224)
    results['dinov2_vits14'] = probe_encoder(dino_s, frames, device,
                                             "DINOv2 ViT-S/14 (CLS token, 384-d)")

    # 4. DINOv2 ViT-B/14 — CLS token (768-d) via timm
    print("\n[4/4] DINOv2 ViT-B/14 (timm) ...")
    dino_b = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0, img_size=224)
    results['dinov2_vitb14'] = probe_encoder(dino_b, frames, device,
                                             "DINOv2 ViT-B/14 (CLS token, 768-d)")

    print("\n" + "="*60)
    print("SUMMARY (mean pairwise relative feature diff across 3 seeds)")
    print("="*60)
    baseline = {'resnet18_random': 0.0003, 'resnet18_in1k': 0.0449, 'r3m': 0.0637}
    for k, v in results.items():
        print(f"  {k:<25} {v:.4%}")
    print("\n  Reference baselines:")
    for k, v in baseline.items():
        print(f"  {k:<25} {v:.4%}  (previous probe)")
    print("="*60)


if __name__ == "__main__":
    main()
