"""Tests for the loaded-policy variants added in S1 wiring.

These cover:
  - probe_collapse_with_loaded_policy (DP path): builds episodes from a
    synthetic .npz and runs the existing probe_collapse against a fake DP
    policy.
  - probe_collapse_pi05_loaded_policy (opaque server path): a fake
    .infer()-style policy returning a fixed action chunk; verifies the
    report fields are populated and finite.

Run:
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \\
        /iris/u/mikulrai/projects/RoboFactory/robofactory/utils/test_preflight_collapse_loaded.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import preflight_collapse  # noqa: E402
from utils.test_preflight_collapse import _FakePolicy  # noqa: E402


def _make_synthetic_npz(path: str, n: int = 6, To: int = 1, H: int = 32, W: int = 32,
                        P: int = 8, Ta: int = 4, Da: int = 7, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        seeds=np.arange(n, dtype=np.int64),
        images=rng.uniform(0, 1, size=(n, To, 3, H, W)).astype(np.float32),
        qpos=rng.standard_normal((n, To, P)).astype(np.float32),
        # actions are a chunk per To (we use the first To slot — Ta=4, Da=7)
        actions=rng.standard_normal((n, To, Ta, Da)).astype(np.float32) * 0.1,
        features=np.zeros((n, To, 16), dtype=np.float32),
    )


class _FakePi05Policy:
    """Minimal pi0.5-server stand-in: .infer(obs_dict) -> {'actions': np.ndarray}.

    `mode='healthy'` makes the action depend on both image and state (so
    zeroing either changes prediction). `mode='collapsed'` makes it depend
    only on state (zeroing image leaves output unchanged).
    """

    def __init__(self, mode: str = "healthy", action_horizon: int = 6,
                 active_dim: int = 8, image_slots=("base_0_rgb_raw",),
                 proprio_key: str = "state"):
        self.mode = mode
        self.H = action_horizon
        self.D = active_dim
        self.image_slots = image_slots
        self.proprio_key = proprio_key

    def infer(self, obs_dict: dict) -> dict:
        state = np.asarray(obs_dict[self.proprio_key], dtype=np.float32).reshape(-1)
        # keep length consistent
        s_pad = np.resize(state, self.D).astype(np.float32)
        if self.mode == "healthy":
            img_means = []
            for slot in self.image_slots:
                if slot in obs_dict:
                    img_means.append(float(np.asarray(obs_dict[slot]).mean()))
            img_signal = float(np.mean(img_means)) if img_means else 0.0
            base = np.tile(s_pad, (self.H, 1)) + 0.5 * img_signal
        elif self.mode == "collapsed":
            base = np.tile(s_pad, (self.H, 1))
        else:
            raise ValueError(self.mode)
        return {"actions": base.astype(np.float32)}


class TestProbeCollapseWithLoadedPolicyDP(unittest.TestCase):
    def test_dp_loaded_policy_runs_and_populates_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = str(Path(tmp) / "synthetic.npz")
            _make_synthetic_npz(npz_path, n=6, Ta=4, Da=7)

            policy = _FakePolicy(mode="healthy", n_obs_steps=2, feat_dim=8)
            report = preflight_collapse.probe_collapse_with_loaded_policy(
                policy, npz_path, max_episodes=6
            )

            self.assertIsInstance(report, preflight_collapse.CollapseReport)
            self.assertEqual(report.n_episodes, 6)
            self.assertTrue(np.isfinite(report.mse_baseline))
            self.assertTrue(np.isfinite(report.mse_zero_image))
            self.assertTrue(np.isfinite(report.mse_zero_proprio))
            self.assertGreater(report.mse_baseline, 0.0)
            self.assertGreaterEqual(report.feature_rank, 1)
            self.assertGreater(report.feature_dim, 0)

    def test_dp_collapsed_policy_image_ratio_near_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = str(Path(tmp) / "synthetic.npz")
            _make_synthetic_npz(npz_path, n=6, Ta=4, Da=7)
            policy = _FakePolicy(mode="collapsed", n_obs_steps=2, feat_dim=8)
            report = preflight_collapse.probe_collapse_with_loaded_policy(
                policy, npz_path, max_episodes=6
            )
            # Collapsed policy ignores image -> zero-image prediction equals baseline
            self.assertAlmostEqual(report.image_to_baseline_ratio, 1.0, delta=0.5)


class TestProbeCollapsePi05LoadedPolicy(unittest.TestCase):
    def _build_obs_dict_factory(self, prompt: str = "stack the cubes",
                                image_slots=("base_0_rgb_raw",)):
        def _build(img_chw: np.ndarray, qpos: np.ndarray) -> dict:
            # convert CHW -> HWC uint8 (server convention)
            hwc = np.moveaxis(img_chw, 0, -1)
            hwc_u8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
            out = {"prompt": prompt, "state": qpos.astype(np.float32)}
            for slot in image_slots:
                out[slot] = hwc_u8
            return out
        return _build

    def _build_npz_with_gt_from_policy(self, path: str, policy: "_FakePi05Policy",
                                       build_fn, n: int = 4, To: int = 1,
                                       H: int = 32, W: int = 32, P: int = 8) -> None:
        """Build calibration npz where actions == policy(image, qpos) plus a
        small perturbation. This ensures MSE_baseline > 0 but image-zeroing
        moves prediction further from GT."""
        rng = np.random.default_rng(7)
        images = rng.uniform(0, 1, size=(n, To, 3, H, W)).astype(np.float32)
        qpos = rng.standard_normal((n, To, P)).astype(np.float32)
        Ta = policy.H
        Da = policy.D
        actions = np.zeros((n, To, Ta, Da), dtype=np.float32)
        for i in range(n):
            obs = build_fn(images[i, 0], qpos[i, 0])
            actions[i, 0] = np.asarray(policy.infer(obs)["actions"])[:Ta, :Da]
        # add perturbation so baseline isn't degenerate (MSE > 0)
        actions += rng.standard_normal(actions.shape).astype(np.float32) * 0.02
        np.savez(path, seeds=np.arange(n, dtype=np.int64), images=images,
                 qpos=qpos, actions=actions,
                 features=np.zeros((n, To, 16), dtype=np.float32))

    def test_pi05_healthy_policy_image_ratio_above_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = str(Path(tmp) / "synthetic.npz")
            image_slots = ("base_0_rgb_raw",)
            build = self._build_obs_dict_factory(image_slots=image_slots)
            policy = _FakePi05Policy(mode="healthy", action_horizon=6,
                                     active_dim=8, image_slots=image_slots)
            self._build_npz_with_gt_from_policy(npz_path, policy, build, n=4)

            report = preflight_collapse.probe_collapse_pi05_loaded_policy(
                policy, npz_path, build_obs_dict=build,
                image_slots=image_slots, proprio_key="state", max_episodes=4
            )
            self.assertEqual(report.n_episodes, 4)
            self.assertTrue(np.isfinite(report.mse_baseline))
            self.assertTrue(np.isfinite(report.mse_zero_image))
            self.assertTrue(np.isfinite(report.mse_zero_proprio))
            # opaque path: feature_rank/var unsupported -> zeroed but valid
            self.assertEqual(report.feature_rank, 0)
            self.assertEqual(report.feature_dim, 0)
            self.assertEqual(report.per_channel_variance.size, 0)
            # image ratio should be above 1: healthy mode uses image, so
            # zeroing it pushes prediction away from GT
            self.assertGreater(report.image_to_baseline_ratio, 1.0)

    def test_pi05_payload_skips_feature_var_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = str(Path(tmp) / "synthetic.npz")
            _make_synthetic_npz(npz_path, n=3, To=1, Ta=6, Da=8)
            image_slots = ("base_0_rgb_raw",)
            build = self._build_obs_dict_factory(image_slots=image_slots)
            policy = _FakePi05Policy(mode="healthy", action_horizon=6,
                                     active_dim=8, image_slots=image_slots)
            report = preflight_collapse.probe_collapse_pi05_loaded_policy(
                policy, npz_path, build_obs_dict=build,
                image_slots=image_slots, proprio_key="state", max_episodes=3
            )
            payload = report.to_wandb_payload(prefix="collapse")
            # core keys present
            self.assertIn("collapse/mse_baseline", payload)
            self.assertIn("collapse/feature_rank", payload)
            # variance keys SKIPPED (empty per_channel_variance)
            self.assertNotIn("collapse/feature_var_mean", payload)
            self.assertNotIn("collapse/feature_var_max", payload)


if __name__ == "__main__":
    unittest.main()
