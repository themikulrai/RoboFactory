"""Tests for scripts/preflight/check_overfit_replay_sanity.py.

These mock the env factory, the policy loader, and the dataset loader so
nothing imports SAPIEN, torch, zarr, or the HF ``datasets`` library.
The login node renders SAPIEN scenes ~32% darker than compute nodes
(per CLAUDE.md), and we don't want to spawn a sim env or load a torch
checkpoint just to unit-test the replay/MSE math.

Run from repo root::

    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \\
        -m pytest robofactory/scripts/preflight/test_check_overfit_replay_sanity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Make the script importable regardless of pytest cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))  # repo root

from robofactory.scripts.preflight import check_overfit_replay_sanity as cor  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal env fake matching the contract used by ``replay_episode``.

    ``reset(init_pose=...) -> (obs, info)``.  ``step(action) ->
    (obs, reward, term, trunc, info)``.

    Each step returns the next recorded observation if available, so
    that "the policy is fed the env-rendered obs" path is deterministic
    in tests.  The real env renders pixels; here we just hand back the
    canned per-step state vector — that's plenty to test the wiring.
    """

    def __init__(self, observations: np.ndarray):
        self._obs = observations
        self._t = 0
        self.closed = False
        self.reset_init_pose: Optional[np.ndarray] = None

    def reset(self, init_pose=None, seed=None):
        self.reset_init_pose = (
            None if init_pose is None else np.asarray(init_pose).copy()
        )
        self._t = 0
        return self._obs[0], {}

    def step(self, action):
        self._t += 1
        idx = min(self._t, len(self._obs) - 1)
        return self._obs[idx], 0.0, False, False, {}

    def close(self):
        self.closed = True


def _make_env_factory(observations: np.ndarray):
    def factory(_scene_config, _episode_idx):
        return _FakeEnv(observations)
    return factory


def _make_policy_loader(policy_fn):
    """Wrap a plain callable into a ``policy_loader`` seam."""
    def loader(_ckpt_path):
        return policy_fn
    return loader


def _make_dataset_loader(init_pose, observations, actions):
    def loader(_path, _episode_idx):
        return init_pose, observations, actions
    return loader


def _run(
    policy_fn,
    *,
    init_pose: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    mode: str = "full",
    max_steps: int = 50,
    mse_tolerance: float = 0.01,
    episode_idx: int = 0,
):
    return cor.run_check(
        ckpt_path=Path("/dev/null/ckpt.pt"),
        dataset_path=Path("/dev/null/data.zarr"),
        scene_config=Path("/dev/null/scene.yaml"),
        episode_idx=episode_idx,
        max_steps=max_steps,
        mse_tolerance=mse_tolerance,
        mode=mode,
        env_factory=_make_env_factory(observations),
        policy_loader=_make_policy_loader(policy_fn),
        dataset_loader=_make_dataset_loader(init_pose, observations, actions),
    )


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


def _toy_episode(T: int = 10, A: int = 7, K: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    init_pose = rng.normal(size=(A,))
    observations = rng.normal(size=(T, A))   # treat obs as same shape as state
    actions = rng.normal(size=(T, A))
    return init_pose, observations, actions, K


# ---------------------------------------------------------------------------
# 1) Identity policy -> MSE == 0
# ---------------------------------------------------------------------------


class TestIdentityPolicy:
    def test_zero_mse_passes(self):
        init_pose, obs, actions, K = _toy_episode(T=8, A=7, K=4, seed=1)
        # Identity policy: at step t, return actions[t : t+K] (clipped).
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            return actions[t : t + k]

        report = _run(policy, init_pose=init_pose, observations=obs, actions=actions)
        assert report["passed"] is True
        assert report["chunk0_passed"] is True
        assert report["full_chunk_passed"] is True
        assert report["chunk0_mse_mean"] == pytest.approx(0.0, abs=1e-12)
        assert report["full_chunk_mse_mean"] == pytest.approx(0.0, abs=1e-12)
        for v in report["chunk0_mse_per_step"]:
            assert v == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2) Small noise under tolerance
# ---------------------------------------------------------------------------


class TestSmallNoiseUnderTolerance:
    def test_005_noise_passes_at_tol_001(self):
        init_pose, obs, actions, K = _toy_episode(T=8, A=7, K=4, seed=2)
        rng = np.random.default_rng(99)
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            # noise std = 0.005 => MSE expectation ~ 0.005^2 = 2.5e-5 << 1e-2.
            return actions[t : t + k] + rng.normal(0.0, 0.005, size=(k, actions.shape[1]))

        report = _run(policy, init_pose=init_pose, observations=obs, actions=actions)
        assert report["chunk0_mse_mean"] < 0.01
        assert report["full_chunk_mse_mean"] < 0.01
        assert report["passed"] is True


# ---------------------------------------------------------------------------
# 3) Larger noise over tolerance
# ---------------------------------------------------------------------------


class TestLargerNoiseOverTolerance:
    def test_05_noise_fails(self):
        init_pose, obs, actions, K = _toy_episode(T=8, A=7, K=4, seed=3)
        rng = np.random.default_rng(7)
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            # noise std = 0.5 => MSE expectation ~ 0.25 >> 1e-2.
            return actions[t : t + k] + rng.normal(0.0, 0.5, size=(k, actions.shape[1]))

        report = _run(policy, init_pose=init_pose, observations=obs, actions=actions)
        assert report["chunk0_mse_mean"] > 0.01
        assert report["full_chunk_mse_mean"] > 0.01
        assert report["passed"] is False
        assert report["chunk0_passed"] is False
        assert report["full_chunk_passed"] is False


# ---------------------------------------------------------------------------
# 4) chunk[0] passes but full chunk fails -> diagnostic
# ---------------------------------------------------------------------------


class TestChunk0PassesFullChunkFails:
    def test_first_action_correct_rest_garbage(self):
        init_pose, obs, actions, K = _toy_episode(T=8, A=7, K=5, seed=4)
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            out = np.tile(actions[t], (k, 1)).copy()  # row 0 correct
            if k > 1:
                # rows 1..k-1 deliberately wrong by ~0.5 in every dim.
                out[1:] = out[1:] + 0.5
            return out

        report = _run(policy, init_pose=init_pose, observations=obs, actions=actions)
        assert report["chunk0_passed"] is True
        assert report["chunk0_mse_mean"] == pytest.approx(0.0, abs=1e-12)
        assert report["full_chunk_passed"] is False
        assert report["full_chunk_mse_mean"] > 0.01
        assert report["passed"] is False


# ---------------------------------------------------------------------------
# 5) full chunk passes but chunk[0] fails (don't conflate them)
# ---------------------------------------------------------------------------


class TestFullChunkPassesChunk0Fails:
    def test_first_action_wrong_rest_correct_long_chunk(self):
        # K large so the lone-bad first slot averages out below tolerance
        # in the full-chunk metric while still tripping the chunk[0]
        # metric.  Numerically (see file header in script):
        # chunk0 = err^2 / A;  full = err^2 / (K * A).
        # err = 0.5, A = 7, K = 20  -> chunk0 ~ 0.0357 (> 0.01),
        #                              full   ~ 0.00179 (< 0.01).
        T, A, K = 30, 7, 20
        rng = np.random.default_rng(5)
        init_pose = rng.normal(size=(A,))
        observations = rng.normal(size=(T, A))
        actions = rng.normal(size=(T, A))
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            out = actions[t : t + k].copy()
            # Perturb only the very first slot, only on dim 0, by 0.5.
            perturbation = np.zeros_like(out[0])
            perturbation[0] = 0.5
            out[0] = out[0] + perturbation
            return out

        report = cor.run_check(
            ckpt_path=Path("/dev/null/ckpt.pt"),
            dataset_path=Path("/dev/null/data.zarr"),
            scene_config=Path("/dev/null/scene.yaml"),
            episode_idx=0,
            max_steps=T,           # replay full episode so K-clipping stays K most of the time
            mse_tolerance=0.01,
            mode="full",
            env_factory=_make_env_factory(observations),
            policy_loader=_make_policy_loader(policy),
            dataset_loader=_make_dataset_loader(init_pose, observations, actions),
        )
        # chunk0 sees the lone bad slot every step; full averages it over the chunk.
        # Restrict the comparison to the steps where K-clipping doesn't kick in
        # (t <= T - K) so the analytic prediction holds; pytest's tolerance
        # on the means is loose enough either way.
        assert report["chunk0_mse_mean"] > 0.01, report["chunk0_mse_mean"]
        # The full mean might be marginal due to end-of-episode K-clipping (the
        # tail steps see fewer "good" rows divided over fewer slots).  Test at
        # the per-step level for the steady-state regime instead:
        steady = [v for i, v in enumerate(report["full_chunk_mse_per_step"])
                  if i <= T - K]
        assert max(steady) < 0.01
        # And chunk0_passed is unambiguously False.
        assert report["chunk0_passed"] is False


# ---------------------------------------------------------------------------
# 6) mode='first-only' -> full_chunk_* fields are None
# ---------------------------------------------------------------------------


class TestFirstOnlyMode:
    def test_first_only_skips_full_chunk(self):
        init_pose, obs, actions, K = _toy_episode(T=6, A=7, K=4, seed=6)
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            return actions[t]  # 1-D output -> wrapped as (1, A)

        report = _run(policy, init_pose=init_pose, observations=obs, actions=actions,
                      mode="first-only")
        assert report["full_chunk_mse_per_step"] is None
        assert report["full_chunk_mse_mean"] is None
        # Overall verdict reduces to chunk0 verdict.
        assert report["full_chunk_passed"] is True
        assert report["chunk0_passed"] is True
        assert report["passed"] is True


# ---------------------------------------------------------------------------
# 7) Episode shorter than max_steps -> n_steps clipped
# ---------------------------------------------------------------------------


class TestEpisodeShorterThanMaxSteps:
    def test_n_steps_clipped(self):
        # Episode of length 3, max_steps = 50 -> n_steps_replayed == 3.
        T, A, K = 3, 7, 4
        rng = np.random.default_rng(11)
        init_pose = rng.normal(size=(A,))
        observations = rng.normal(size=(T, A))
        actions = rng.normal(size=(T, A))
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            return actions[t : t + k]

        report = _run(policy, init_pose=init_pose, observations=observations,
                      actions=actions, max_steps=50)
        assert report["n_steps_replayed"] == 3
        assert len(report["chunk0_mse_per_step"]) == 3
        assert len(report["full_chunk_mse_per_step"]) == 3


# ---------------------------------------------------------------------------
# 8) Single-step episode
# ---------------------------------------------------------------------------


class TestSingleStepEpisode:
    def test_T1_handled(self):
        T, A, K = 1, 5, 8
        init_pose = np.zeros(A)
        observations = np.zeros((T, A))
        actions = np.zeros((T, A))

        def policy(_obs):
            # K=8 chunk but only 1 step left; replay should clip to 1 row.
            return np.zeros((K, A))

        report = _run(policy, init_pose=init_pose, observations=observations,
                      actions=actions, max_steps=50)
        assert report["n_steps_replayed"] == 1
        assert report["chunk0_mse_mean"] == pytest.approx(0.0)
        assert report["full_chunk_mse_mean"] == pytest.approx(0.0)
        assert report["passed"] is True


# ---------------------------------------------------------------------------
# 9) Action chunk extending past episode end -> clip
# ---------------------------------------------------------------------------


class TestChunkPastEpisodeEnd:
    def test_no_index_error_on_tail(self):
        T, A, K = 5, 4, 10  # K >> T-t for the tail steps
        rng = np.random.default_rng(42)
        init_pose = rng.normal(size=(A,))
        observations = rng.normal(size=(T, A))
        actions = rng.normal(size=(T, A))

        def policy(_obs):
            return np.zeros((K, A))  # always emit a 10-row chunk

        report = _run(policy, init_pose=init_pose, observations=observations,
                      actions=actions, max_steps=50, mse_tolerance=999.0)
        assert report["n_steps_replayed"] == 5
        # Each per-step full MSE is finite (no out-of-bounds slicing).
        for v in report["full_chunk_mse_per_step"]:
            assert np.isfinite(v)


# ---------------------------------------------------------------------------
# 10) Mismatched obs / action shapes -> ValueError with clear message
# ---------------------------------------------------------------------------


class TestObsActionShapeMismatch:
    def test_mismatch_raises(self):
        init_pose = np.zeros(7)
        observations = np.zeros((5, 7))
        actions = np.zeros((4, 7))  # length mismatch

        def policy(_obs):
            return np.zeros((1, 7))

        with pytest.raises(ValueError, match="length mismatch"):
            _run(policy, init_pose=init_pose, observations=observations,
                 actions=actions)

    def test_action_dim_mismatch_raises(self):
        init_pose = np.zeros(7)
        observations = np.zeros((4, 7))
        actions = np.zeros((4, 7))

        def policy(_obs):
            return np.zeros((1, 6))  # wrong action dim

        with pytest.raises(ValueError, match="action dim"):
            _run(policy, init_pose=init_pose, observations=observations,
                 actions=actions)


# ---------------------------------------------------------------------------
# 11) Format auto-detect: *.zarr -> zarr loader, else -> HF
# ---------------------------------------------------------------------------


class TestFormatAutoDetect:
    def test_zarr_suffix_dispatches_to_zarr_loader(self, monkeypatch, tmp_path):
        called = {"zarr": False, "hf": False}

        def fake_zarr(path, idx):
            called["zarr"] = True
            return np.zeros(3), np.zeros((1, 3)), np.zeros((1, 3))

        def fake_hf(path, idx):
            called["hf"] = True
            return np.zeros(3), np.zeros((1, 3)), np.zeros((1, 3))

        monkeypatch.setattr(cor, "_load_zarr_episode", fake_zarr)
        monkeypatch.setattr(cor, "_load_lerobot_episode", fake_hf)

        cor.load_episode(Path("/tmp/something.zarr"), 0)
        assert called["zarr"] is True
        assert called["hf"] is False

    def test_non_zarr_suffix_dispatches_to_hf_loader(self, monkeypatch, tmp_path):
        called = {"zarr": False, "hf": False}

        def fake_zarr(path, idx):
            called["zarr"] = True
            return np.zeros(3), np.zeros((1, 3)), np.zeros((1, 3))

        def fake_hf(path, idx):
            called["hf"] = True
            return np.zeros(3), np.zeros((1, 3)), np.zeros((1, 3))

        monkeypatch.setattr(cor, "_load_zarr_episode", fake_zarr)
        monkeypatch.setattr(cor, "_load_lerobot_episode", fake_hf)

        # An ordinary file path with no zarr suffix and no zarr-shaped
        # directory layout falls through to the HF loader.
        cor.load_episode(tmp_path / "lerobot_dataset", 0)
        assert called["hf"] is True
        assert called["zarr"] is False


# ---------------------------------------------------------------------------
# 12) JSON serialization round-trip of the report
# ---------------------------------------------------------------------------


class TestJSONRoundTrip:
    def test_report_is_json_serialisable(self):
        init_pose, obs, actions, K = _toy_episode(T=4, A=3, K=2, seed=12)
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            return actions[t : t + k]

        report = _run(policy, init_pose=init_pose, observations=obs, actions=actions)
        s = json.dumps(report)
        round_trip = json.loads(s)
        # Top-level keys the SLURM launcher will key on.
        for k in (
            "ckpt_path", "dataset_path", "episode_idx", "n_steps_replayed",
            "chunk0_mse_per_step", "chunk0_mse_mean",
            "full_chunk_mse_per_step", "full_chunk_mse_mean",
            "mse_tolerance", "chunk0_passed", "full_chunk_passed", "passed",
        ):
            assert k in round_trip, f"missing key: {k}"
        # And via the dataclass directly:
        rep = cor.OverfitReplayReport(**{**round_trip})
        assert rep.passed == report["passed"]


# ---------------------------------------------------------------------------
# 13) NaN in policy predictions -> raise
# ---------------------------------------------------------------------------


class TestNaNPolicyPrediction:
    def test_nan_action_raises(self):
        init_pose, obs, actions, K = _toy_episode(T=3, A=4, K=2, seed=13)

        def policy(_obs):
            out = np.zeros((K, actions.shape[1]))
            out[0, 0] = np.nan
            return out

        with pytest.raises(ValueError, match="non-finite"):
            _run(policy, init_pose=init_pose, observations=obs, actions=actions)

    def test_inf_action_raises(self):
        init_pose, obs, actions, K = _toy_episode(T=3, A=4, K=2, seed=131)

        def policy(_obs):
            out = np.zeros((K, actions.shape[1]))
            out[0, 0] = np.inf
            return out

        with pytest.raises(ValueError, match="non-finite"):
            _run(policy, init_pose=init_pose, observations=obs, actions=actions)


# ---------------------------------------------------------------------------
# 14) Multi-arm action vector (T, 21) handled flatly via mean()
# ---------------------------------------------------------------------------


class TestMultiArmActions:
    def test_21d_action_no_special_case(self):
        # Three-arm setup: 7 + 7 + 7 = 21-D action vector.
        T, A, K = 6, 21, 4
        rng = np.random.default_rng(14)
        init_pose = rng.normal(size=(A,))
        observations = rng.normal(size=(T, A))
        actions = rng.normal(size=(T, A))
        state = {"t": 0}

        def policy(_obs):
            t = state["t"]
            state["t"] += 1
            k = min(K, len(actions) - t)
            return actions[t : t + k]

        report = _run(policy, init_pose=init_pose, observations=observations,
                      actions=actions)
        assert report["passed"] is True
        assert report["chunk0_mse_mean"] == pytest.approx(0.0, abs=1e-12)
        assert report["full_chunk_mse_mean"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Bonus: env wiring sanity (init_pose actually flows into env.reset)
# ---------------------------------------------------------------------------


class TestEnvResetReceivesInitPose:
    def test_init_pose_is_forwarded(self):
        init_pose, obs, actions, K = _toy_episode(T=3, A=5, K=2, seed=21)
        captured = {"env": None}

        def factory(_scene_config, _episode_idx):
            env = _FakeEnv(obs)
            captured["env"] = env
            return env

        def policy(_obs):
            return np.zeros((K, actions.shape[1]))

        cor.run_check(
            ckpt_path=Path("/dev/null/ckpt.pt"),
            dataset_path=Path("/dev/null/data.zarr"),
            scene_config=Path("/dev/null/scene.yaml"),
            mode="first-only",
            mse_tolerance=999.0,
            env_factory=factory,
            policy_loader=_make_policy_loader(policy),
            dataset_loader=_make_dataset_loader(init_pose, obs, actions),
        )
        env = captured["env"]
        assert env is not None
        assert env.reset_init_pose is not None
        np.testing.assert_array_equal(env.reset_init_pose, init_pose)
        assert env.closed is True  # run_check closes the env in finally.


# ---------------------------------------------------------------------------
# Bonus: mode validation
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_invalid_mode_raises(self):
        init_pose, obs, actions, K = _toy_episode(T=3, A=4, K=2, seed=22)

        def policy(_obs):
            return np.zeros((K, actions.shape[1]))

        with pytest.raises(ValueError, match="mode"):
            _run(policy, init_pose=init_pose, observations=obs, actions=actions,
                 mode="bogus")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
