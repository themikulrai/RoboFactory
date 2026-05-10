"""Tests for robofactory.evaluation.policy_loader.

These mock the openpi WebsocketClientPolicy so the test suite does NOT
require an openpi server and does not even require the openpi_client
package to be importable.

Run from repo root::

    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \\
        -m pytest robofactory/evaluation/test_policy_loader.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable regardless of pytest cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # repo root

from robofactory.evaluation import policy_loader as pl  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeWebsocketClient:
    """Mimics openpi_client.websocket_client_policy.WebsocketClientPolicy."""

    last_construction: dict = {}

    def __init__(self, host: str = "0.0.0.0", port=None, api_key=None):
        type(self).last_construction = dict(host=host, port=port, api_key=api_key)
        self.host = host
        self.port = port
        self.api_key = api_key
        self.calls: list = []

    def infer(self, obs):
        self.calls.append(obs)
        # Pretend the server gave us a chunk of (H=4, A=8).
        return {"actions": np.zeros((4, 8), dtype=np.float32)}


# ---------------------------------------------------------------------------
# 1) Dispatch: DP file extensions go to the DP backend (NotImplementedError)
# ---------------------------------------------------------------------------


class TestLoadPolicyDispatchDP:
    @pytest.mark.parametrize(
        "ckpt_name",
        ["latest.ckpt", "300.ckpt", "epoch_10.pt", "weights.bin"],
    )
    def test_dp_extensions_route_to_dp_loader(self, ckpt_name):
        with pytest.raises(NotImplementedError) as exc_info:
            pl.load_policy_for_eval(f"/tmp/some/dir/{ckpt_name}")
        msg = str(exc_info.value)
        # The error message must clearly call out the DP coupling so the
        # operator knows why it failed and what to do instead.
        assert "Diffusion-Policy" in msg or "DP" in msg
        assert "DPRunner" in msg or "RobotWorkspace" in msg or "BCRL" in msg


# ---------------------------------------------------------------------------
# 2) Dispatch: Pi0.5 directories route to the Pi0.5 backend (mocked)
# ---------------------------------------------------------------------------


class TestLoadPolicyDispatchPi05:
    def test_params_dir_with_explicit_port(self, tmp_path):
        # Realistic openpi-serve layout: the final component is `params`
        # but it's a directory inside a model dir.
        model_dir = tmp_path / "pi05_model"
        params_dir = model_dir / "params"
        params_dir.mkdir(parents=True)

        # Pass tmp_path/pi05_model (parent) — has a params/ child.
        policy = pl.load_policy_for_eval(
            model_dir,
            port=8000,
            client_factory=_FakeWebsocketClient,
        )
        assert callable(policy)
        assert policy.backend == "pi05"  # type: ignore[attr-defined]
        assert policy.port == 8000       # type: ignore[attr-defined]
        # Construction args got through to the client factory.
        assert _FakeWebsocketClient.last_construction["port"] == 8000
        assert _FakeWebsocketClient.last_construction["host"] == "0.0.0.0"

    def test_params_terminal_path_routes_to_pi05(self):
        # Even a non-existent path with terminal component "params" works
        # for dispatch — useful for unit tests without filesystem state.
        policy = pl.load_policy_for_eval(
            "/iris/u/mikulrai/runs/pi05/some_run/params",
            port=9999,
            client_factory=_FakeWebsocketClient,
        )
        assert policy.backend == "pi05"  # type: ignore[attr-defined]
        assert policy.port == 9999       # type: ignore[attr-defined]

    def test_params_in_path_parts_routes_to_pi05(self):
        policy = pl.load_policy_for_eval(
            "/iris/u/mikulrai/runs/pi05/run42/params/weights",
            port=4242,
            client_factory=_FakeWebsocketClient,
        )
        assert policy.backend == "pi05"  # type: ignore[attr-defined]

    def test_pi05_call_returns_chunk_array(self):
        policy = pl.load_policy_for_eval(
            "/some/pi05/params",
            port=8000,
            client_factory=_FakeWebsocketClient,
        )
        chunk = policy({"obs": "anything"})
        assert isinstance(chunk, np.ndarray)
        assert chunk.shape == (4, 8)

    def test_pi05_active_dim_slices_chunk(self):
        policy = pl.load_policy_for_eval(
            "/some/pi05/params",
            port=8000,
            active_dim=5,
            client_factory=_FakeWebsocketClient,
        )
        chunk = policy({"obs": "anything"})
        assert chunk.shape == (4, 5)

    def test_pi05_missing_port_raises(self):
        with pytest.raises(ValueError, match="port"):
            pl.load_policy_for_eval(
                "/some/pi05/params",
                client_factory=_FakeWebsocketClient,
            )

    def test_pi05_custom_host(self):
        policy = pl.load_policy_for_eval(
            "/some/pi05/params",
            host="iris-gpu-9.stanford.edu",
            port=12345,
            client_factory=_FakeWebsocketClient,
        )
        assert policy.host == "iris-gpu-9.stanford.edu"  # type: ignore[attr-defined]
        assert _FakeWebsocketClient.last_construction["host"] == "iris-gpu-9.stanford.edu"
        assert _FakeWebsocketClient.last_construction["port"] == 12345


# ---------------------------------------------------------------------------
# 3) Unknown formats raise a helpful ValueError
# ---------------------------------------------------------------------------


class TestLoadPolicyUnknownFormat:
    @pytest.mark.parametrize(
        "ckpt",
        [
            "/tmp/garbage.txt",
            "/tmp/no_extension",
            "/tmp/some_dir/",          # directory with no params/ child
            "/tmp/foo.json",
            "/tmp/random.npz",
        ],
    )
    def test_unknown_format_raises_valueerror(self, ckpt, tmp_path):
        # If `some_dir/` is requested, give it a real (empty) directory so
        # the looks_like_pi05 path-shape check actually runs.
        if ckpt == "/tmp/some_dir/":
            target = tmp_path / "empty_dir"
            target.mkdir()
            ckpt = str(target)
        with pytest.raises(ValueError) as exc_info:
            pl.load_policy_for_eval(ckpt)
        msg = str(exc_info.value)
        assert "cannot infer backend" in msg
        assert "Diffusion-Policy" in msg or ".ckpt" in msg
        assert "params" in msg


# ---------------------------------------------------------------------------
# 4) Module is importable without torch/openpi at import time
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_no_torch_at_import_time(self):
        # We can't unimport torch in-process, but we can verify that
        # the module's symbol table doesn't reference torch at the top
        # level — the only torch use should be inside the (currently
        # NotImplementedError) DP loader's eventual body.
        import robofactory.evaluation.policy_loader as mod
        # numpy is fine; torch / openpi are not.
        assert not hasattr(mod, "torch")
        assert not hasattr(mod, "openpi_client")
        assert not hasattr(mod, "WebsocketClientPolicy")


# ---------------------------------------------------------------------------
# 5) Integration with the heavy-preflight script's _default_policy_loader
# ---------------------------------------------------------------------------


class TestDefaultPolicyLoaderIntegration:
    """The script's ``_default_policy_loader`` calls
    ``load_policy_for_eval(ckpt_path)`` with no extra kwargs.  For DP
    checkpoints that triggers NotImplementedError; for Pi0.5 it would
    error out on missing ``port``.  This documents the *current*
    contract — when DP gets implemented, this test should be updated to
    exercise the round-trip dispatch.
    """

    def test_default_loader_propagates_dp_not_implemented(self):
        # The script's _default_policy_loader is a thin wrapper; we
        # verify it forwards the path and the NotImplementedError
        # bubbles up through the dispatcher.
        from robofactory.scripts.preflight import (  # noqa: E402
            check_overfit_replay_sanity as cor,
        )
        with pytest.raises(NotImplementedError):
            cor._default_policy_loader(Path("/tmp/fake.ckpt"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
