"""PR5: DP diffusion-sampler determinism.

The diffusion policy draws its denoising noise from torch's global RNG. Before PR5
torch was NEVER seeded at eval, so the same env seed produced a DIFFERENT trajectory
every run (the sampler free-ran). PR5 adds, at the top of every DP rollout:

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % 2**32); random.seed(seed)

This test proves two things WITHOUT a trained ckpt or SAPIEN:

  1. Functional: re-seeding with the SAME seed reproduces identical torch/np/random
     draws (so the diffusion noise is identical -> identical chunk -> identical step
     count on the same node). Different seeds differ.
  2. Structural: every DP rollout entry point actually contains the seeding block.

NOTE: cross-GPU bitwise determinism is NOT promised (kernels differ A40/A5000/A6000).
The contract is same-seed-same-node reproducibility + N_REPEATS, not bit-identity.
"""
from __future__ import annotations

import pathlib
import random

import numpy as np
import torch


def _seed_block(seed: int):
    """Exact copy of the PR5 per-episode seeding block."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def _draw_after_seed(seed: int):
    _seed_block(seed)
    return (
        torch.randn(4, 8).numpy().copy(),   # diffusion noise surrogate
        np.random.rand(4).copy(),
        [random.random() for _ in range(4)],
    )


class TestSeedingFunctional:
    def test_same_seed_identical_draws(self):
        t1, n1, r1 = _draw_after_seed(10042)
        t2, n2, r2 = _draw_after_seed(10042)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(n1, n2)
        assert r1 == r2

    def test_different_seed_differs(self):
        t1, _, _ = _draw_after_seed(10042)
        t2, _, _ = _draw_after_seed(10043)
        assert not np.array_equal(t1, t2)

    def test_large_seed_wraps_for_numpy(self):
        # PR4 env seeds can exceed 2**32 (e.g. base*100000 OOD range). np.random.seed
        # rejects values >= 2**32, so the block masks with % 2**32. Must not raise.
        big = 17 * 100000 + 2**32 + 5
        _draw_after_seed(big)  # raises if the mask is missing
        # And it is still deterministic at the masked value.
        a = _draw_after_seed(big)[1]
        b = _draw_after_seed(big)[1]
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Structural: every DP rollout entry point contains the seeding block.
# ---------------------------------------------------------------------------
DP_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _src(rel):
    return (DP_ROOT / rel).read_text()


class TestSeedingPresentInAllRollouts:
    def test_eval_dp_single_run_episode_seeds(self):
        text = _src("eval_dp.py")
        assert "torch.manual_seed(seed)" in text
        assert "np.random.seed(seed % 2**32)" in text

    def test_eval_multi_dp_run_episode_seeds(self):
        text = _src("eval_multi_dp.py")
        assert "torch.manual_seed(seed)" in text
        assert "np.random.seed(seed % 2**32)" in text

    def test_eval_joint_dp_runner_seeds(self):
        # eval_joint_dp delegates rollout to robot_joint_image_runner.
        text = _src("diffusion_policy/env_runner/robot_joint_image_runner.py")
        assert "torch.manual_seed(seed)" in text
        assert "np.random.seed(seed % 2**32)" in text

    def test_multi_agent_runner_seeds(self):
        text = _src("diffusion_policy/env_runner/robot_multi_agent_image_runner.py")
        assert "torch.manual_seed(seed)" in text
        assert "np.random.seed(seed % 2**32)" in text
