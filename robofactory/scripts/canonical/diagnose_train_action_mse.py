"""Compute train_action_mse on the actual training data for a DP ckpt.

Diagnoses whether the diffusion model can reproduce its training actions.
If MSE ~ 0 -> model converged, eval failure is pipeline/OOD-seed related.
If MSE ~ 0.1+ -> diffusion sampling can't recover trained actions -> deep bug.

Compare against PM baseline (train_action_mse = 1.4e-07 in working DP run).
"""
import os
import sys
import argparse
import torch
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf

# Allow imports from the diffusion_policy package
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')
os.chdir('/iris/u/mikulrai/projects/RoboFactory/robofactory')

from diffusion_policy.workspace.robotworkspace import RobotWorkspace


def diagnose(ckpt_path: str, n_batches: int = 8) -> dict:
    print(f"\n{'=' * 70}\nLoading: {ckpt_path}\n{'=' * 70}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    device = torch.device('cuda:0')
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    # load_payload already restored the trained normalizer; move *after* that.
    policy.to(device)
    policy.eval()

    # Recreate the training dataset (only for sampling training batches; do NOT overwrite normalizer)
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=False,
    )

    print(f"  train episodes: {int(dataset.train_mask.sum())}")
    print(f"  train sequences: {len(dataset)}")
    print(f"  action range: [{cfg.task.shape_meta.action.shape}]")

    all_mse = []
    per_dim_mse_list = []
    sample_pairs = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            batch = dataset.postprocess(batch, device)
            obs = batch['obs']
            gt = batch['action']  # (B, T, A)
            result = policy.predict_action(obs)
            pred = result['action_pred']  # (B, T, A)
            mse = torch.nn.functional.mse_loss(pred, gt).item()
            per_dim = torch.nn.functional.mse_loss(
                pred, gt, reduction='none'
            ).mean(dim=tuple(range(pred.ndim - 1))).cpu().numpy()
            all_mse.append(mse)
            per_dim_mse_list.append(per_dim)
            if sample_pairs is None:
                # save one sample for visual inspection (first batch, first sample, first timestep)
                sample_pairs = {
                    'pred_t0': pred[0, 0].cpu().numpy().tolist(),
                    'gt_t0': gt[0, 0].cpu().numpy().tolist(),
                }
            print(f"    batch {i}: mse={mse:.6f}")

    avg_mse = float(np.mean(all_mse))
    avg_per_dim = np.mean(per_dim_mse_list, axis=0).tolist()
    return {
        'ckpt': ckpt_path,
        'avg_train_action_mse': avg_mse,
        'per_dim_mse': avg_per_dim,
        'sample_pred_t0': sample_pairs['pred_t0'],
        'sample_gt_t0': sample_pairs['gt_t0'],
        'n_batches_evaluated': len(all_mse),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True)
    parser.add_argument('--n_batches', type=int, default=8)
    args = parser.parse_args()

    results = []
    for ckpt in args.ckpts:
        try:
            r = diagnose(ckpt, n_batches=args.n_batches)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'ckpt': ckpt, 'error': str(e)})

    print(f"\n{'#' * 70}\n# SUMMARY\n{'#' * 70}")
    for r in results:
        if 'error' in r:
            print(f"  {r['ckpt']}\n    ERROR: {r['error']}")
            continue
        print(f"\n  {r['ckpt']}")
        print(f"    avg_train_action_mse = {r['avg_train_action_mse']:.6f}")
        print(f"    per_dim_mse           = {[f'{x:.4f}' for x in r['per_dim_mse']]}")
        print(f"    sample pred[t=0]      = {[f'{x:.3f}' for x in r['sample_pred_t0']]}")
        print(f"    sample gt[t=0]        = {[f'{x:.3f}' for x in r['sample_gt_t0']]}")
    print(f"\nPM baseline (working): train_action_mse = 1.4e-07")
