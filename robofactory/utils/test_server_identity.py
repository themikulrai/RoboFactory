"""GATE 3 negative test for the PR2 client-side server-identity handshake.

CPU-only: the handshake is a pure metadata comparison performed BEFORE any
inference, so no GPU / SAPIEN / trained checkpoint is needed.

(b) Pure-function unit tests of ``check_server_identity`` / ``assert_server_identity_or_exit``
    (fast; the primary gate). Proves: config mismatch -> non-zero exit (3) + one-line
    diagnosis; action_dim mismatch -> same; match -> proceeds (returns ok, no exit);
    missing metadata (legacy server) -> un-verifiable -> ok.

(a) A light integration test: spin up a minimal websocket server that sends a packed
    metadata dict (mimicking openpi serve_policy.py) on an ephemeral free port, connect
    with the real ``WebsocketClientPolicy``, fetch ``get_server_metadata()``, and assert
    the check fails for a mismatched --expect-config and passes for the matching one.
    Marked ``slow`` (network/asyncio) but still CPU-only and fast in practice.
"""
from __future__ import annotations

import asyncio
import socket
import threading

import pytest

from robofactory.utils.server_identity import (
    IDENTITY_MISMATCH_EXIT,
    assert_server_identity_or_exit,
    check_server_identity,
)

ARM0_META = {
    "config_name": "pi05_robofactory_lb_wc_decent_arm0",
    "action_dim": 32,
    "real_action_dim": 8,
    "ckpt_dir": "/ckpts/arm0/19999",
    "ckpt_step": 19999,
    "norm_stats_md5": "deadbeef",
    "wandb_id": "abc123",
    "openpi_zero_state": None,
    "openpi_mask_extra_rgb": None,
}


# --------------------------------------------------------------------------- (b) pure fn
def test_config_mismatch_returns_diagnosis():
    ok, diag = check_server_identity(
        ARM0_META, expect_config="pi05_robofactory_lb_wc_decent_arm1", label="arm1"
    )
    assert ok is False
    assert diag is not None
    # one line
    assert "\n" not in diag.strip()
    assert "SERVER IDENTITY MISMATCH" in diag
    assert "pi05_robofactory_lb_wc_decent_arm0" in diag  # got
    assert "pi05_robofactory_lb_wc_decent_arm1" in diag  # want
    assert "refusing to run" in diag


def test_config_match_ok():
    ok, diag = check_server_identity(
        ARM0_META, expect_config="pi05_robofactory_lb_wc_decent_arm0"
    )
    assert ok is True
    assert diag is None


def test_action_dim_mismatch_returns_diagnosis():
    # match config but wrong arm count -> expect per-arm dim 8 but served 32 (cent model)
    ok, diag = check_server_identity(
        ARM0_META,
        expect_config="pi05_robofactory_lb_wc_decent_arm0",
        expect_action_dim=8,
        label="arm0",
    )
    assert ok is False
    assert "action_dim=32" in diag
    assert "expects 8" in diag


def test_action_dim_match_ok():
    ok, diag = check_server_identity(
        ARM0_META,
        expect_config="pi05_robofactory_lb_wc_decent_arm0",
        expect_action_dim=32,
    )
    assert ok is True
    assert diag is None


def test_missing_metadata_is_unverifiable_ok():
    # Legacy server (pre-PR2) sends {} -> cannot verify -> proceed.
    ok, diag = check_server_identity({}, expect_config="anything", expect_action_dim=8)
    assert ok is True
    assert diag is None
    ok, diag = check_server_identity(None, expect_config="anything")
    assert ok is True


def test_no_expectation_is_ok():
    ok, diag = check_server_identity(ARM0_META, expect_config=None, expect_action_dim=None)
    assert ok is True


def test_assert_exits_3_with_one_line_diagnosis(capsys):
    with pytest.raises(SystemExit) as ei:
        assert_server_identity_or_exit(
            ARM0_META, expect_config="pi05_robofactory_lb_wc_decent_arm1", label="port 8123"
        )
    assert ei.value.code == IDENTITY_MISMATCH_EXIT == 3
    err = capsys.readouterr().err
    printed = err.strip()
    assert printed.count("\n") == 0  # exactly one line
    assert "SERVER IDENTITY MISMATCH" in printed
    assert "port 8123" in printed
    assert "refusing to run" in printed


def test_assert_match_is_noop(capsys):
    # Must NOT raise; nothing printed.
    assert_server_identity_or_exit(
        ARM0_META, expect_config="pi05_robofactory_lb_wc_decent_arm0", expect_action_dim=32
    )
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- (a) integration
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _StubMetadataServer:
    """Minimal websocket server that, on connect, sends one packed metadata dict
    (exactly the protocol openpi's WebsocketPolicyServer uses) then idles. No policy,
    no GPU. Runs its asyncio loop in a background thread."""

    def __init__(self, metadata: dict, port: int):
        self.metadata = metadata
        self.port = port
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        import websockets.asyncio.server as _server
        from openpi_client import msgpack_numpy

        packer = msgpack_numpy.Packer()

        async def handler(ws):
            await ws.send(packer.pack(self.metadata))
            try:
                async for _ in ws:  # idle; never infer
                    pass
            except Exception:
                pass

        self._stop = asyncio.Event()
        async with await _server.serve(handler, "127.0.0.1", self.port, compression=None):
            self._ready.set()
            await self._stop.wait()  # idle until stop() (clean shutdown, no leaked loop)

    def start(self):
        self._thread.start()
        assert self._ready.wait(timeout=10.0), "stub server never came up"

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5.0)


@pytest.mark.slow
def test_integration_real_client_mismatch_then_match():
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    port = _free_port()
    srv = _StubMetadataServer(ARM0_META, port)
    srv.start()
    try:
        client = WebsocketClientPolicy(host="127.0.0.1", port=port)
        meta = client.get_server_metadata()
        assert meta["config_name"] == "pi05_robofactory_lb_wc_decent_arm0"

        # Mismatch: client expects arm1 -> hard-fail (exit 3) + one-line diagnosis.
        with pytest.raises(SystemExit) as ei:
            assert_server_identity_or_exit(
                meta, expect_config="pi05_robofactory_lb_wc_decent_arm1",
                expect_action_dim=8, label=f"arm1 (port {port})",
            )
        assert ei.value.code == 3

        # Match: proceeds (action_dim 32 matches served).
        assert_server_identity_or_exit(
            meta, expect_config="pi05_robofactory_lb_wc_decent_arm0",
            expect_action_dim=32, label=f"arm0 (port {port})",
        )
    finally:
        srv.stop()
