"""Tests for utils.preflight_collapse.

Run from the robofactory env:
    python -m unittest robofactory.utils.test_preflight_collapse -v
or directly:
    python utils/test_preflight_collapse.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import preflight_collapse  # noqa: E402


class _FakePolicy:
    """Stub policy that mimics the small slice of the DP policy API the probe touches.

    `mode` controls behaviour:
      - "healthy": predicted action depends on both image and proprio (zeroing
        either degrades the prediction relative to GT).
      - "collapsed": predicted action depends only on proprio (zeroing image
        leaves prediction unchanged; zeroing proprio destroys it).
    `feature_func`: callable that produces (To, feat_dim) features for a given
        obs_dict — used to simulate rank-collapse.
    """

    def __init__(self, mode: str = "healthy", n_obs_steps: int = 2, feat_dim: int = 8):
        self.mode = mode
        self.n_obs_steps = n_obs_steps
        self._feat_dim = feat_dim

    @property
    def device(self):
        import torch
        return torch.device("cpu")

    @property
    def normalizer(self):
        return _IdentityNormalizer()

    def obs_encoder(self, batched_inputs: dict):
        import torch

        proprio = batched_inputs["agent_pos"]  # (To, P)
        image = batched_inputs["head_cam"]  # (To, 3, H, W)
        if self.mode == "healthy":
            img_summary = image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)  # (To, 1)
            feats = torch.cat([proprio, img_summary.expand(-1, self._feat_dim - proprio.shape[-1])], dim=-1)
        elif self.mode == "collapsed":
            feats = proprio.repeat(1, self._feat_dim // proprio.shape[-1] + 1)[:, : self._feat_dim]
        elif self.mode in ("rank_collapsed", "constant"):
            feats = torch.ones(proprio.shape[0], self._feat_dim, device=proprio.device)
        elif self.mode == "nan":
            feats = torch.zeros(proprio.shape[0], self._feat_dim, device=proprio.device)
        else:
            raise ValueError(self.mode)
        return feats

    def predict_action(self, obs_dict_b: dict):
        import torch
        proprio = obs_dict_b["agent_pos"]  # (B, To, P)
        image = obs_dict_b["head_cam"]  # (B, To, 3, H, W)
        B = proprio.shape[0]
        Ta = 4
        Da = 7
        if self.mode == "healthy":
            # action chunk depends on both proprio (last step) and image mean
            base = proprio[:, -1, :Da] if proprio.shape[-1] >= Da else proprio[:, -1].repeat(1, (Da // proprio.shape[-1]) + 1)[:, :Da]
            img_mean = image.mean(dim=(2, 3, 4))[:, -1:].expand(-1, Da)
            action = base.unsqueeze(1).expand(-1, Ta, -1) + 0.1 * img_mean.unsqueeze(1).expand(-1, Ta, -1)
        elif self.mode == "collapsed":
            # action depends only on proprio
            base = proprio[:, -1, :Da] if proprio.shape[-1] >= Da else proprio[:, -1].repeat(1, (Da // proprio.shape[-1]) + 1)[:, :Da]
            action = base.unsqueeze(1).expand(-1, Ta, -1)
        elif self.mode == "constant":
            action = torch.zeros(B, Ta, Da)
        elif self.mode == "nan":
            action = torch.full((B, Ta, Da), float("nan"))
        else:
            raise ValueError(self.mode)
        return {"action": action}


class _IdentityNormalizer:
    def normalize(self, d):
        return d


def _make_obs(seed: int = 0, image_value: float = 0.5):
    rng = np.random.default_rng(seed)
    return {
        "agent_pos": rng.standard_normal(8).astype(np.float32),
        "head_cam": (np.ones((3, 32, 32), dtype=np.float32) * image_value),
    }


def _gt_action_for(policy: _FakePolicy, obs: dict) -> np.ndarray:
    """Return the ground truth that the *healthy* policy would output for obs.

    By using this as GT, mse_baseline ≈ 0 only for `mode='healthy'`.  For
    `mode='collapsed'` against this same GT, image-zeroing leaves prediction
    unchanged (so mse_zero_image ≈ mse_baseline) while proprio-zeroing destroys
    it.
    """
    healthy = _FakePolicy(mode="healthy")
    return preflight_collapse._predict_action(healthy, obs)


class TestProbeCollapseHealthy(unittest.TestCase):
    def test_healthy_policy_zero_image_increases_mse(self):
        policy = _FakePolicy(mode="healthy")
        episodes = []
        for i in range(4):
            obs = _make_obs(i, image_value=0.5 + 0.1 * i)
            gt = _gt_action_for(policy, obs) + 0.01  # perturb GT so baseline > 0
            episodes.append((obs, gt))
        report = preflight_collapse.probe_collapse(policy, episodes)
        # zeroing image should hurt for the healthy policy → ratio > 1
        self.assertGreater(report.image_to_baseline_ratio, 1.0)
        # zeroing proprio should also hurt, often more severely
        self.assertGreater(report.proprio_to_baseline_ratio, 1.0)


class TestProbeCollapseCollapsed(unittest.TestCase):
    def test_collapsed_policy_image_ratio_near_one(self):
        """A policy that ignores image: zeroing image does not change predictions,
        so mse_zero_image ≈ mse_baseline (ratio ≈ 1)."""
        policy = _FakePolicy(mode="collapsed")
        episodes = []
        for i in range(4):
            obs = _make_obs(i, image_value=0.5 + 0.1 * i)
            # GT is what a *healthy* policy would output (different from collapsed prediction)
            gt = _gt_action_for(_FakePolicy(mode="healthy"), obs)
            episodes.append((obs, gt))
        report = preflight_collapse.probe_collapse(policy, episodes)

        # image ratio close to 1 (zeroing image doesn't change prediction)
        self.assertAlmostEqual(report.image_to_baseline_ratio, 1.0, delta=0.5)
        # proprio ratio much higher (zeroing proprio breaks everything)
        self.assertGreater(report.proprio_to_baseline_ratio, report.image_to_baseline_ratio)


class TestProbeCollapseRankCollapse(unittest.TestCase):
    def test_rank_collapsed_features_yield_rank_one(self):
        policy = _FakePolicy(mode="rank_collapsed")
        # Override predict_action so baseline ≠ 0 (rank-collapsed mode has no
        # predict_action implementation by default)
        episodes = []
        for i in range(8):
            obs = _make_obs(i, image_value=0.5 + 0.1 * i)
            gt = np.full((4, 7), 0.1, dtype=np.float32)
            episodes.append((obs, gt))

        # rebind predict_action to a deterministic non-zero output
        def fake_predict(obs_dict_b):
            import torch
            B = next(iter(obs_dict_b.values())).shape[0]
            return {"action": torch.full((B, 4, 7), 0.5)}

        policy.predict_action = fake_predict

        report = preflight_collapse.probe_collapse(policy, episodes)
        self.assertEqual(report.feature_rank, 1)
        self.assertLess(float(report.per_channel_variance.max()), 1e-6)


class TestErrorPaths(unittest.TestCase):
    def test_nan_in_prediction_raises(self):
        policy = _FakePolicy(mode="nan")
        episodes = [(_make_obs(0), np.zeros((4, 7), dtype=np.float32))]
        with self.assertRaises(preflight_collapse.CollapseProbeError):
            preflight_collapse.probe_collapse(policy, episodes)

    def test_zero_baseline_raises_degenerate(self):
        # Constant-zero policy + zero GT → baseline is exactly 0 → degenerate
        policy = _FakePolicy(mode="constant")
        episodes = [(_make_obs(i), np.zeros((4, 7), dtype=np.float32)) for i in range(3)]
        with self.assertRaises(preflight_collapse.DegenerateBaselineError):
            preflight_collapse.probe_collapse(policy, episodes)

    def test_no_image_keys_raises(self):
        policy = _FakePolicy(mode="healthy")
        # only-proprio obs_dict
        episodes = [({"agent_pos": np.zeros(8, dtype=np.float32)}, np.zeros((4, 7), dtype=np.float32))]
        with self.assertRaises(ValueError):
            preflight_collapse.probe_collapse(policy, episodes)

    def test_empty_episodes_raises(self):
        policy = _FakePolicy(mode="healthy")
        with self.assertRaises(ValueError):
            preflight_collapse.probe_collapse(policy, [])

    def test_shape_mismatch_raises(self):
        policy = _FakePolicy(mode="healthy")
        bad_gt = np.zeros((1, 7), dtype=np.float32)  # wrong Ta dim
        episodes = [(_make_obs(0), bad_gt)]
        with self.assertRaises(ValueError):
            preflight_collapse.probe_collapse(policy, episodes)


class TestReportPayload(unittest.TestCase):
    def test_to_wandb_payload_has_required_keys(self):
        report = preflight_collapse.CollapseReport(
            mse_baseline=0.01,
            mse_zero_image=0.02,
            mse_zero_proprio=0.5,
            feature_rank=4,
            per_channel_variance=np.array([0.1, 0.2, 0.3]),
            n_episodes=8,
            image_keys=("head_cam",),
            proprio_keys=("agent_pos",),
            feature_dim=64,
        )
        payload = report.to_wandb_payload()
        self.assertIn("collapse/mse_baseline", payload)
        self.assertIn("collapse/image_to_baseline_ratio", payload)
        self.assertIn("collapse/feature_rank", payload)
        self.assertEqual(payload["collapse/mse_baseline"], 0.01)
        self.assertAlmostEqual(payload["collapse/image_to_baseline_ratio"], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
