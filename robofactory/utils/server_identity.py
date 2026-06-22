"""Client-side pi0.5 server-identity handshake (solution_E6.md §PR2 steps 2-3).

openpi's ``scripts/serve_policy.py`` sends a provenance/identity metadata dict at
websocket connect (``config_name``, ``ckpt_dir``, ``ckpt_step``, ``norm_stats_md5``,
``wandb_id``, ``action_dim``, ``real_action_dim``, ``openpi_zero_state``,
``openpi_mask_extra_rgb``); the client exposes it via
``WebsocketClientPolicy.get_server_metadata()``. Nothing checked it, so a server
launched with the wrong checkpoint behind the right port produced 60 silent
IndexError episodes instead of a single up-front failure.

This module is the pure, GPU-free comparison the eval clients run BEFORE episode 1:
``check_server_identity`` returns a one-line diagnosis on mismatch; the eval drivers
turn that into a hard ``sys.exit(3)``. Kept here (not inline in the drivers) so it can
be unit-tested with mocked metadata on CPU.
"""
from __future__ import annotations

import sys
from typing import Mapping, Optional, Tuple

# Hard-fail exit code for a server-identity mismatch (distinct from generic 1/2 so
# launchers/CI can recognise "wrong policy behind the port" specifically).
IDENTITY_MISMATCH_EXIT = 3


def check_server_identity(
    metadata: Optional[Mapping],
    expect_config: Optional[str],
    expect_action_dim: Optional[int] = None,
    *,
    label: str = "server",
) -> Tuple[bool, Optional[str]]:
    """Compare a server's identity metadata against what the client expects.

    Args:
        metadata: the dict returned by ``WebsocketClientPolicy.get_server_metadata()``.
            May be ``None``/empty if connecting to a legacy server that predates the
            PR2 handshake — that is treated as "cannot verify" and is OK (we never
            block on a missing field; the goal is to catch a *populated, wrong* value).
        expect_config: the config name this client was told to expect (e.g.
            ``pi05_robofactory_lb_wc_decent_arm0``). ``None``/empty disables the
            config check.
        expect_action_dim: the model action dim this client expects (per-arm dim for
            a decent arm, or the centralised total). ``None`` disables the dim check.
        label: a short label for the one-line diagnosis (e.g. ``port 8123`` or
            ``arm0``) so multi-server clients name the offending connection.

    Returns:
        ``(ok, diagnosis)``. ``ok`` is True when nothing populated mismatches the
        expectation. On mismatch ``ok`` is False and ``diagnosis`` is a single line
        suitable for printing before refusing to run.
    """
    meta = dict(metadata or {})

    # --- config_name check ---
    if expect_config:
        got = meta.get("config_name")
        if got is not None and got != expect_config:
            return False, (
                f"SERVER IDENTITY MISMATCH: {label} served config={got!r} but "
                f"--expect-config={expect_config!r}; refusing to run"
            )

    # --- action_dim check ---
    if expect_action_dim is not None:
        got_dim = meta.get("action_dim")
        if got_dim is not None and int(got_dim) != int(expect_action_dim):
            return False, (
                f"SERVER IDENTITY MISMATCH: {label} served action_dim={got_dim} but "
                f"client expects {int(expect_action_dim)} "
                f"(config={meta.get('config_name')!r}); refusing to run"
            )

    return True, None


def assert_server_identity_or_exit(
    metadata: Optional[Mapping],
    expect_config: Optional[str],
    expect_action_dim: Optional[int] = None,
    *,
    label: str = "server",
) -> None:
    """Hard-fail wrapper: print the one-line diagnosis and ``sys.exit(3)`` on mismatch.

    Called by the eval drivers after connect, before episode 1. A clean match is a
    no-op. Missing metadata (legacy server) is treated as un-verifiable -> proceeds.
    """
    ok, diagnosis = check_server_identity(
        metadata, expect_config, expect_action_dim, label=label
    )
    if not ok:
        print(diagnosis, file=sys.stderr, flush=True)
        sys.exit(IDENTITY_MISMATCH_EXIT)


__all__ = [
    "IDENTITY_MISMATCH_EXIT",
    "check_server_identity",
    "assert_server_identity_or_exit",
]
