"""T0 — Checkpoint contract audit.

Usage: python script/debug/inspect_ckpt.py <ckpt_path>

Prints the training contract embedded in a Diffusion Policy checkpoint:
  - shape_meta (camera keys + image shapes consumed by the encoder)
  - n_obs_steps, horizon, n_action_steps
  - dataset / exp_name / training data path
  - per-dim normalizer offset/scale for action and agent_pos
  - whether image normalizer is identity vs [-1,1] mapping
"""
import sys
import argparse
import torch
import dill
import numpy as np
import yaml


def _to_dict(obj):
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return obj


def _summarize_normalizer(payload):
    """Return per-dim offset/scale arrays for each normalizer key, plus a
    'bytes' fingerprint so we can tell identity vs fitted at a glance."""
    sd = payload.get('state_dicts', {}) or {}
    model_sd = sd.get('model', sd.get('ema_model', None))
    if model_sd is None:
        return {}
    out = {}
    # LinearNormalizer stores per-key offset/scale under model.normalizer.params_dict.<key>.{offset,scale,input_stats.*}
    for k in list(model_sd.keys()):
        if 'normalizer.params_dict' in k and k.endswith('.scale'):
            base = k[: -len('.scale')]
            scale = model_sd[k]
            offset = model_sd.get(base + '.offset')
            key_name = base.split('normalizer.params_dict.')[-1]
            out[key_name] = {
                'scale': scale,
                'offset': offset,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = payload.get('cfg', None)

    print(f"=== ckpt: {args.ckpt} ===")
    print(f"top-level keys: {list(payload.keys())}")
    if cfg is None:
        print("!! no 'cfg' key in payload — cannot extract training contract")
        sys.exit(1)

    # Flatten the bits we care about
    try:
        from omegaconf import OmegaConf
        cfg_d = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_d = dict(cfg) if hasattr(cfg, 'keys') else {}

    print("\n--- exp / training ---")
    print(f"exp_name        = {cfg_d.get('exp_name')}")
    print(f"task.name       = {cfg_d.get('task', {}).get('name')}")
    print(f"dataset.zarr    = {cfg_d.get('task', {}).get('dataset', {}).get('zarr_path')}")
    print(f"max_train_eps   = {cfg_d.get('task', {}).get('dataset', {}).get('max_train_episodes')}")
    print(f"current_agent_id= {cfg_d.get('current_agent_id')}")
    print(f"training.num_epochs = {cfg_d.get('training', {}).get('num_epochs')}")
    print(f"training.seed   = {cfg_d.get('training', {}).get('seed')}")

    print("\n--- shape_meta (encoder contract) ---")
    sm = cfg_d.get('shape_meta') or cfg_d.get('task', {}).get('shape_meta', {})
    print(yaml.safe_dump(sm, sort_keys=False))

    print("--- horizons ---")
    print(f"n_obs_steps     = {cfg_d.get('n_obs_steps') or cfg_d.get('policy', {}).get('n_obs_steps')}")
    print(f"horizon         = {cfg_d.get('horizon') or cfg_d.get('policy', {}).get('horizon')}")
    print(f"n_action_steps  = {cfg_d.get('n_action_steps') or cfg_d.get('policy', {}).get('n_action_steps')}")

    print("\n--- normalizer ---")
    norms = _summarize_normalizer(payload)
    if not norms:
        print("!! no normalizer params found in state_dict — model uninitialized?")
    for k, v in norms.items():
        sc = v['scale'].numpy() if hasattr(v['scale'], 'numpy') else np.asarray(v['scale'])
        of = v['offset'].numpy() if hasattr(v['offset'], 'numpy') else np.asarray(v['offset'])
        is_identity = np.allclose(sc, 1.0) and np.allclose(of, 0.0)
        print(f"\n  key={k}  shape={sc.shape}  identity={is_identity}")
        if sc.size <= 32:
            print(f"    scale  per-dim: {sc.tolist()}")
            print(f"    offset per-dim: {of.tolist()}")
        else:
            print(f"    scale  min/mean/max = {sc.min():.4f} / {sc.mean():.4f} / {sc.max():.4f}")
            print(f"    offset min/mean/max = {of.min():.4f} / {of.mean():.4f} / {of.max():.4f}")
        # For 8-dim action/state, gripper is dim 7; should be near identity in raw [-1,1] data.
        if sc.size == 8:
            print(f"    gripper dim (7): scale={sc[7]:.4f} offset={of[7]:.4f}  "
                  f"({'identity' if (abs(sc[7]-1)<1e-3 and abs(of[7])<1e-3) else 'fitted'})")


if __name__ == '__main__':
    main()
