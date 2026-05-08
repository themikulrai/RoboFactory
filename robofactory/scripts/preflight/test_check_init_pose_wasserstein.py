"""Tests for scripts/preflight/check_init_pose_wasserstein.py.

These mock the env factory + the train-data loader so nothing imports
SAPIEN.  The login node renders SAPIEN scenes ~32% darker than compute
nodes (per CLAUDE.md), and besides we don't want to spawn a sim env
just to unit-test a stats helper.

Run from repo root::

    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \\
        -m pytest robofactory/scripts/preflight/test_check_init_pose_wasserstein.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the script importable regardless of pytest cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))  # repo root

from robofactory.scripts.preflight import check_init_pose_wasserstein as cw  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: synthetic train + eval init-pose distributions
# ---------------------------------------------------------------------------


def _make_env_factory(eval_states: np.ndarray):
    """Return an env_factory whose env.reset(seed=k) yields eval_states[k - SEED0]."""

    class _FakeEnv:
        def __init__(self, states):
            self._states = states
            self._call = 0

        def reset(self, seed=None):
            # Index by call order; tests pass a contiguous block of seeds so
            # this is deterministic.
            obs = {"agent": {"qpos": self._states[self._call]}}
            self._call += 1
            return obs, {}

        def close(self):
            pass

    def factory(_eval_config_path):
        return _FakeEnv(eval_states)
    return factory


def _patch_train_loader(monkeypatch, train_states: np.ndarray):
    """Monkeypatch ``load_train_init_poses`` to return canned states."""
    def _fake(path, n_samples):
        return train_states[:n_samples].astype(np.float64)
    monkeypatch.setattr(cw, "load_train_init_poses", _fake)


def _run(
    monkeypatch,
    train_states: np.ndarray,
    eval_states: np.ndarray,
    *,
    n_pos_dims: int = 3,
    pos_tolerance_cm: float = 0.5,
    rot_tolerance_deg: float = 0.5,
    bootstrap_iters: int = 200,
    n_samples: int | None = None,
):
    if n_samples is None:
        n_samples = train_states.shape[0]
    _patch_train_loader(monkeypatch, train_states)
    return cw.run_check(
        Path("/dev/null/train.zarr"),
        Path("/dev/null/eval.yaml"),
        n_samples=n_samples,
        bootstrap_iters=bootstrap_iters,
        pos_tolerance_cm=pos_tolerance_cm,
        rot_tolerance_deg=rot_tolerance_deg,
        n_pos_dims=n_pos_dims,
        seed_start=100,
        rng_seed=0,
        env_factory=_make_env_factory(eval_states),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestIdenticalDistributions:
    """train == eval -> per-dim W1 ~ 0 -> all pass."""

    def test_zero_pose_passes(self, monkeypatch):
        # 7-D qpos (rot, all radians) + 3-D xyz (m) = 10-D state.
        rng = np.random.default_rng(42)
        states = rng.normal(loc=0.0, scale=0.01, size=(50, 10))
        report = _run(monkeypatch, states.copy(), states.copy())
        assert report["passed_overall"] is True
        assert report["n_fail"] == 0
        # W1 of a sample with itself is exactly zero.
        for d in report["per_dim"]:
            assert d["wasserstein_human"] == pytest.approx(0.0, abs=1e-12)

    def test_same_distribution_passes_within_bootstrap_noise(self, monkeypatch):
        rng = np.random.default_rng(123)
        train = rng.normal(loc=0.0, scale=0.001, size=(50, 7))  # 7 rot dims
        evalu = rng.normal(loc=0.0, scale=0.001, size=(50, 7))
        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=0,
            rot_tolerance_deg=0.5,
        )
        # 0.001 rad ~ 0.057 deg, well under 0.5 deg even after sampling noise.
        assert report["passed_overall"] is True


class TestPositionShift:
    """A 2-cm shift on one xyz dim must trip the 0.5-cm tolerance."""

    def test_2cm_shift_one_dim_fails_only_that_dim(self, monkeypatch):
        rng = np.random.default_rng(7)
        # 7 rot dims (small noise) + 3 pos dims (small noise).
        train = np.zeros((50, 10))
        train[:, :7] = rng.normal(0.0, 0.0001, size=(50, 7))
        train[:, 7:] = rng.normal(0.0, 0.0001, size=(50, 3))

        evalu = train.copy()
        evalu[:, 8] += 0.02  # shift y (middle pos dim) by 2 cm

        report = _run(monkeypatch, train, evalu, n_pos_dims=3)
        assert report["passed_overall"] is False
        # Per-dim isolation: only dim 8 fails.
        failed = [d for d in report["per_dim"] if not d["passed"]]
        assert len(failed) == 1
        assert failed[0]["dim"] == 8
        assert failed[0]["kind"] == "pos"
        # Wasserstein in cm is ~2.0 (the deterministic shift), well above tol.
        assert failed[0]["wasserstein_human"] == pytest.approx(2.0, rel=0.05)
        assert failed[0]["wasserstein_human"] > failed[0]["tolerance_human"]


class TestRotationShift:
    """A 1-deg shift on one rot dim must trip the 0.5-deg tolerance."""

    def test_1deg_shift_one_dim_fails_only_that_dim(self, monkeypatch):
        rng = np.random.default_rng(99)
        train = np.zeros((50, 10))
        train[:, :7] = rng.normal(0.0, 1e-5, size=(50, 7))
        train[:, 7:] = rng.normal(0.0, 1e-5, size=(50, 3))

        evalu = train.copy()
        evalu[:, 3] += math.radians(1.0)  # shift rot dim 3 by 1 deg

        report = _run(monkeypatch, train, evalu, n_pos_dims=3)
        assert report["passed_overall"] is False
        failed = [d for d in report["per_dim"] if not d["passed"]]
        assert len(failed) == 1
        assert failed[0]["dim"] == 3
        assert failed[0]["kind"] == "rot"
        assert failed[0]["wasserstein_human"] == pytest.approx(1.0, rel=0.05)


class TestStdMismatch:
    """Same mean but different spread: W1 must be > 0 (a fact about the
    1-D Wasserstein distance between two zero-mean distributions of
    different scale)."""

    def test_different_std_yields_positive_wasserstein(self, monkeypatch):
        rng = np.random.default_rng(2024)
        train = rng.normal(0.0, 0.001, size=(200, 1))   # narrow, in radians
        evalu = rng.normal(0.0, 0.05, size=(200, 1))    # wide
        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=0,
            bootstrap_iters=100,
            n_samples=200,
            rot_tolerance_deg=0.5,
        )
        d = report["per_dim"][0]
        assert d["wasserstein_human"] > 0.0
        # 0.05 rad ~ 2.86 deg => W1 dominated by the wide tail; must fail.
        assert d["passed"] is False


class TestPerDimIsolation:
    """7-D pose, mismatched only on dim 4 -> exactly that dim flagged."""

    def test_only_mismatched_dim_flagged(self, monkeypatch):
        rng = np.random.default_rng(31337)
        train = rng.normal(0.0, 1e-5, size=(50, 7))
        evalu = train.copy()
        evalu[:, 4] += math.radians(2.0)  # 2 deg, well above 0.5 deg tol

        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=0,
            rot_tolerance_deg=0.5,
        )
        passed_dims = [d["dim"] for d in report["per_dim"] if d["passed"]]
        failed_dims = [d["dim"] for d in report["per_dim"] if not d["passed"]]
        assert failed_dims == [4]
        assert sorted(passed_dims) == [0, 1, 2, 3, 5, 6]


class TestUnitConversion:
    """Sanity-check the m->cm and rad->deg scaling."""

    def test_pos_dim_native_to_cm(self, monkeypatch):
        train = np.zeros((50, 4))
        evalu = np.zeros((50, 4))
        evalu[:, -1] += 0.005  # 5 mm = 0.5 cm exactly
        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=1,
            pos_tolerance_cm=0.5,
        )
        d = report["per_dim"][-1]
        assert d["kind"] == "pos"
        assert d["wasserstein_native"] == pytest.approx(0.005, rel=1e-6)
        assert d["wasserstein_human"] == pytest.approx(0.5, rel=1e-6)

    def test_rot_dim_native_to_deg(self, monkeypatch):
        train = np.zeros((50, 1))
        evalu = np.zeros((50, 1)) + math.radians(0.25)  # well under 0.5 deg
        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=0,
            rot_tolerance_deg=0.5,
        )
        d = report["per_dim"][0]
        assert d["kind"] == "rot"
        assert d["wasserstein_human"] == pytest.approx(0.25, rel=1e-4)
        assert d["passed"] is True


class TestBootstrapCI:
    """CI bounds bracket the point estimate (a non-trivial sanity check
    that the bootstrap actually runs)."""

    def test_ci_brackets_point_estimate(self, monkeypatch):
        rng = np.random.default_rng(0)
        train = rng.normal(0.0, 0.01, size=(50, 3))
        evalu = rng.normal(0.005, 0.01, size=(50, 3))
        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=0,
            bootstrap_iters=300,
            rot_tolerance_deg=999.0,  # don't care about pass/fail here
        )
        for d in report["per_dim"]:
            assert d["ci_low_human"] <= d["wasserstein_human"] + 1e-9
            assert d["ci_high_human"] >= d["wasserstein_human"] - 1e-9


class TestReportShape:
    """The JSON report has the keys the SLURM launcher will key on."""

    def test_report_has_expected_keys(self, monkeypatch):
        train = np.zeros((10, 4))
        evalu = np.zeros((10, 4))
        report = _run(
            monkeypatch, train, evalu,
            n_pos_dims=1,
            n_samples=10,
            bootstrap_iters=50,
        )
        for k in ("train_data_path", "eval_config", "n_train_samples",
                  "n_eval_samples", "state_dim", "n_pos_dims",
                  "pos_tolerance_cm", "rot_tolerance_deg",
                  "bootstrap_iters", "seed_start",
                  "per_dim", "n_fail", "passed_overall"):
            assert k in report, f"missing key: {k}"
        # Per-dim entries are JSON-serialisable.
        json.dumps(report)


class TestDimMismatchRaises:
    """Train state dim != eval state dim -> ValueError, not a silent bug."""

    def test_mismatched_state_dim_raises(self, monkeypatch):
        train = np.zeros((10, 7))
        evalu = np.zeros((10, 8))
        with pytest.raises(ValueError, match="state-dim mismatch"):
            _run(monkeypatch, train, evalu, n_pos_dims=0, n_samples=10,
                 bootstrap_iters=10)


class TestProprioExtractor:
    """The default extractor handles both qpos and qpos_<i> conventions."""

    def test_single_arm_qpos(self):
        obs = {"agent": {"qpos": np.array([0.1, 0.2, 0.3])}}
        out = cw._default_proprio_extractor(obs)
        np.testing.assert_array_equal(out, np.array([0.1, 0.2, 0.3]))

    def test_multi_arm_qpos_concatenates(self):
        obs = {"agent": {
            "qpos_1": np.array([2.0, 2.1]),
            "qpos_0": np.array([1.0, 1.1]),
        }}
        out = cw._default_proprio_extractor(obs)
        # sorted by name: qpos_0 before qpos_1
        np.testing.assert_array_equal(out, np.array([1.0, 1.1, 2.0, 2.1]))

    def test_missing_agent_raises(self):
        with pytest.raises(KeyError, match="agent"):
            cw._default_proprio_extractor({"observation": {}})

    def test_missing_qpos_raises(self):
        with pytest.raises(KeyError, match="qpos"):
            cw._default_proprio_extractor({"agent": {"foo": 1}})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
