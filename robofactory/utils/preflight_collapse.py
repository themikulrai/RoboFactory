"""Encoder-collapse probe (metric-only).

Why this exists
---------------
End-to-end finetuning of a vision encoder on highly templated demonstrations
can collapse the visual features (every input maps to the same feature vector)
while `val_action_mse` stays low because the policy has learned to ignore the
visual stream and route everything through proprioception.  The D1 + D2 TSC
decent ckpts are the canonical observation: 0% rollout SR yet val_action_mse
≈ 0.003 (see project_d1_tsc_encoder_collapse memory).

This module computes a small set of cheap diagnostic statistics that surface
this failure mode *before* a 60-seed eval is launched.  Per the plan, it is
**metric-only** — it never blocks training or eval.  It logs to wandb so a
human can compare values across runs and decide; once enough calibration
ckpts exist (≥4) we can revisit threshold-based hard gating.

Usage from an eval launcher::

    from utils.preflight_collapse import probe_collapse
    report = probe_collapse(policy, val_episodes)
    wandb_run.log_raw(report.to_wandb_payload())

Errors
------
- `DegenerateBaselineError`: every baseline action prediction is identical to
  the ground truth (mse_baseline ≈ 0). Means the eval set is trivial / leaked.
- `CollapseProbeError`: any non-finite values in the loss path (NaNs, infs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


class CollapseProbeError(RuntimeError):
    """Raised when the probe encounters non-finite values it can't divide past."""


class DegenerateBaselineError(RuntimeError):
    """Raised when mse_baseline ≈ 0 — no useful signal extractable."""


@dataclass
class CollapseReport:
    mse_baseline: float
    mse_zero_image: float
    mse_zero_proprio: float
    feature_rank: int
    per_channel_variance: np.ndarray  # shape (feat_dim,)
    n_episodes: int
    image_keys: Tuple[str, ...]
    proprio_keys: Tuple[str, ...]
    feature_dim: int

    @property
    def image_to_baseline_ratio(self) -> float:
        return float(self.mse_zero_image / max(self.mse_baseline, 1e-12))

    @property
    def proprio_to_baseline_ratio(self) -> float:
        return float(self.mse_zero_proprio / max(self.mse_baseline, 1e-12))

    def to_wandb_payload(self, prefix: str = "collapse") -> dict:
        return {
            f"{prefix}/mse_baseline": self.mse_baseline,
            f"{prefix}/mse_zero_image": self.mse_zero_image,
            f"{prefix}/mse_zero_proprio": self.mse_zero_proprio,
            f"{prefix}/image_to_baseline_ratio": self.image_to_baseline_ratio,
            f"{prefix}/proprio_to_baseline_ratio": self.proprio_to_baseline_ratio,
            f"{prefix}/feature_rank": self.feature_rank,
            f"{prefix}/feature_dim": self.feature_dim,
            f"{prefix}/feature_var_mean": float(self.per_channel_variance.mean()),
            f"{prefix}/feature_var_min": float(self.per_channel_variance.min()),
            f"{prefix}/feature_var_max": float(self.per_channel_variance.max()),
            f"{prefix}/n_episodes": self.n_episodes,
        }

    def summary(self) -> str:
        return (
            f"CollapseReport(n_episodes={self.n_episodes}, "
            f"mse_baseline={self.mse_baseline:.6g}, "
            f"image_ratio={self.image_to_baseline_ratio:.3g}, "
            f"proprio_ratio={self.proprio_to_baseline_ratio:.3g}, "
            f"feature_rank={self.feature_rank}/{self.feature_dim}, "
            f"feat_var_mean={float(self.per_channel_variance.mean()):.3g})"
        )


def _auto_image_keys(obs_dict: dict) -> list:
    """Heuristic: keys whose value has shape (..., C, H, W) with C == 3 are images."""
    out = []
    for k, v in obs_dict.items():
        a = np.asarray(v)
        if a.ndim >= 3 and a.shape[-3] == 3:
            out.append(k)
    return out


def _auto_proprio_keys(obs_dict: dict, image_keys: Sequence[str]) -> list:
    return [k for k in obs_dict.keys() if k not in image_keys]


def _zero_keys(obs_dict: dict, keys: Sequence[str]) -> dict:
    out = {}
    for k, v in obs_dict.items():
        a = np.asarray(v)
        out[k] = np.zeros_like(a) if k in keys else a
    return out


def _to_torch_dict(d: dict, device) -> dict:
    """Convert a dict of np.ndarray to torch tensors on `device`."""
    import torch
    return {
        k: torch.from_numpy(np.ascontiguousarray(v)).to(device=device)
        for k, v in d.items()
    }


def _predict_action(policy, obs_dict_np: dict) -> np.ndarray:
    """Run policy.predict_action; mirrors probe_decent_dp_features.predict_action_np."""
    import torch

    device = policy.device
    To = policy.n_obs_steps
    stacked = {k: np.stack([np.asarray(v)] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = _to_torch_dict(stacked, device)
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    with torch.no_grad():
        out = policy.predict_action(obs_dict_b)
    return out["action"].squeeze(0).cpu().numpy()


def _encoder_features(policy, obs_dict_np: dict) -> np.ndarray:
    """Forward through obs_encoder only; returns (To, feat_dim) numpy."""
    import torch

    device = policy.device
    To = policy.n_obs_steps
    stacked = {k: np.stack([np.asarray(v)] * To, axis=0) for k, v in obs_dict_np.items()}
    obs_dict = _to_torch_dict(stacked, device)
    obs_dict_b = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
    nobs = policy.normalizer.normalize(obs_dict_b)
    this_nobs = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
    with torch.no_grad():
        feats = policy.obs_encoder(this_nobs)
    return feats.cpu().numpy()


def _safe_mse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = (pred - target).astype(np.float64)
    if not np.all(np.isfinite(diff)):
        raise CollapseProbeError(
            "non-finite values encountered in (pred - target). "
            "Inputs likely contain NaN/inf — fix upstream rather than masking here."
        )
    return float(np.mean(diff * diff))


def probe_collapse(
    policy,
    val_episodes: Sequence[Tuple[dict, np.ndarray]],
    *,
    image_keys: Optional[Sequence[str]] = None,
    proprio_keys: Optional[Sequence[str]] = None,
    feature_episodes_cap: int = 16,
) -> CollapseReport:
    """Compute encoder-collapse diagnostics on a small set of validation episodes.

    Args:
        policy: loaded DP policy (must expose `.predict_action`, `.obs_encoder`,
            `.normalizer`, `.n_obs_steps`, `.device`).
        val_episodes: sequence of `(obs_dict_np, gt_action_chunk)` pairs.
            `gt_action_chunk` has shape (Ta, Da) and matches the chunk produced
            by the policy.  At least 1 episode is required.
        image_keys: keys to treat as image inputs (auto-detected when None).
        proprio_keys: keys to treat as proprio (auto-detected when None).
        feature_episodes_cap: stop collecting features once this many episodes
            have been seen (rank is meaningful with <= feat_dim samples).

    Returns:
        `CollapseReport`. Crashes with `DegenerateBaselineError` or
        `CollapseProbeError` rather than returning suspicious values.
    """
    if len(val_episodes) == 0:
        raise ValueError("probe_collapse needs at least one validation episode")

    first_obs, first_act = val_episodes[0]
    _image_keys = tuple(image_keys) if image_keys is not None else tuple(_auto_image_keys(first_obs))
    _proprio_keys = tuple(proprio_keys) if proprio_keys is not None else tuple(
        _auto_proprio_keys(first_obs, _image_keys)
    )

    if not _image_keys:
        raise ValueError(f"no image keys detected; obs_dict keys = {list(first_obs.keys())}")
    if not _proprio_keys:
        raise ValueError(f"no proprio keys detected; obs_dict keys = {list(first_obs.keys())}")

    sse_base = 0.0
    sse_zimg = 0.0
    sse_zpro = 0.0
    n_count = 0
    feats_acc = []
    feature_dim = None

    for ep_i, (obs_np, gt_action) in enumerate(val_episodes):
        gt_action = np.asarray(gt_action)
        if gt_action.ndim != 2:
            raise ValueError(
                f"episode {ep_i}: gt_action must be 2D (Ta, Da); got shape {gt_action.shape}"
            )

        pred_base = _predict_action(policy, obs_np)
        pred_zimg = _predict_action(policy, _zero_keys(obs_np, _image_keys))
        pred_zpro = _predict_action(policy, _zero_keys(obs_np, _proprio_keys))

        if pred_base.shape != gt_action.shape:
            raise ValueError(
                f"episode {ep_i}: predicted shape {pred_base.shape} != gt {gt_action.shape}"
            )

        sse_base += _safe_mse(pred_base, gt_action)
        sse_zimg += _safe_mse(pred_zimg, gt_action)
        sse_zpro += _safe_mse(pred_zpro, gt_action)
        n_count += 1

        if ep_i < feature_episodes_cap:
            feats = _encoder_features(policy, obs_np)  # (To, feat_dim)
            if feature_dim is None:
                feature_dim = int(feats.shape[-1])
            feats_acc.append(feats[0])  # all To rows are identical at episode start

    mse_baseline = sse_base / n_count
    mse_zero_image = sse_zimg / n_count
    mse_zero_proprio = sse_zpro / n_count

    if mse_baseline < 1e-12:
        raise DegenerateBaselineError(
            f"mse_baseline = {mse_baseline:.3e} — predictions match GT exactly. "
            "Eval set is trivial / leaked, or policy has memorised the val set."
        )

    F = np.stack(feats_acc, axis=0)  # (E, feat_dim)
    feature_rank = int(np.linalg.matrix_rank(F))
    per_channel_variance = F.var(axis=0)

    return CollapseReport(
        mse_baseline=mse_baseline,
        mse_zero_image=mse_zero_image,
        mse_zero_proprio=mse_zero_proprio,
        feature_rank=feature_rank,
        per_channel_variance=per_channel_variance,
        n_episodes=n_count,
        image_keys=_image_keys,
        proprio_keys=_proprio_keys,
        feature_dim=int(feature_dim if feature_dim is not None else 0),
    )
