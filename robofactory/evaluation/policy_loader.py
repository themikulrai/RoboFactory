"""Production policy loader for eval-side preflight checks.

This module is the production wiring referenced by
``robofactory/scripts/preflight/check_overfit_replay_sanity.py``'s
``_default_policy_loader``.  It returns a callable that the heavy
preflight uses as ``policy(obs) -> action_chunk: np.ndarray (K, A)``.

Design notes
------------

The overfit-replay-sanity check must work with both supported eval
drivers in the repo:

* **Diffusion Policy (DP)** — ``policy/Diffusion-Policy/eval_dp.py``.
  Loads via ``torch.load(ckpt_path, pickle_module=dill)`` + a hydra
  ``RobotWorkspace`` rebuilt from ``payload['cfg']`` + a stateful
  ``DPRunner`` that maintains an obs deque and emits action chunks.
  The runner is *deeply* coupled to the train-side dataloader's
  normalisation stats (``policy.normalizer``) and to the ``DP`` wrapper
  class' frame-stacking behaviour.  Reproducing all of that here
  faithfully is the same amount of work as just calling the eval
  driver, so the DP path is **NotImplementedError** until someone
  factors a reusable in-process driver out of ``eval_dp.py``.

* **Pi0.5 (openpi)** — ``policy/openpi_pi05/eval_pi05.py``.  Loads via
  a websocket *client* talking to a separate openpi server process.
  No torch on the eval side.  The contract is a single ``policy.infer
  (obs) -> {"actions": (H, A)}`` call per chunk.  We honour that by
  wrapping a ``WebsocketClientPolicy`` so the returned callable does
  ``np.asarray(client.infer(obs)["actions"])``.  Caller is responsible
  for having a server running on the configured ``host:port``.

Dispatch is by checkpoint *path shape* (NOT file content), so we never
need torch/openpi at module load:

* file ending in ``.ckpt`` (or ``.pt`` / ``.bin``) → DP
* a directory containing a ``params/`` subdir, OR a path whose final
  component is ``params`` → Pi0.5 (the openpi serve directory layout)

Anything else raises ``ValueError`` with a helpful message.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


PolicyFn = Callable[[Any], np.ndarray]


_DP_SUFFIXES = {".ckpt", ".pt", ".bin"}


def _looks_like_pi05_dir(path: Path) -> bool:
    """A Pi0.5 / openpi checkpoint is a directory with a ``params/`` child,
    or a path whose final component is ``params`` (the openpi-serve
    convention).  We don't read any files — just look at the layout.
    """
    if path.is_dir():
        if (path / "params").is_dir():
            return True
        if path.name == "params":
            return True
    # Allow non-existent paths whose name component looks Pi0.5-shaped,
    # so unit tests can exercise dispatch without filesystem state.
    if path.name == "params":
        return True
    parts = path.parts
    if "params" in parts:
        return True
    return False


def _looks_like_dp_ckpt(path: Path) -> bool:
    return path.suffix.lower() in _DP_SUFFIXES


# ---------------------------------------------------------------------------
# Pi0.5 backend
# ---------------------------------------------------------------------------


def _load_pi05_policy(
    ckpt_path: Path,
    *,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    active_dim: Optional[int] = None,
    prompt: Optional[str] = None,
    client_factory: Optional[Callable[..., Any]] = None,
) -> PolicyFn:
    """Wrap a ``WebsocketClientPolicy`` into the heavy-preflight policy
    contract.

    The caller MUST have an openpi server running and reachable at
    ``host:port`` — this loader does not start one.  See
    ``policy/openpi_pi05/eval_pi05.py`` for the serve-side recipe.

    ``ckpt_path`` is informational here (the server already loaded the
    weights); we keep it in the signature so the dispatch contract is
    uniform.

    Returns a callable ``policy(obs) -> action_chunk (K, A)``.

    ``active_dim`` (optional): if set, slice the returned chunk to
    ``[:, :active_dim]``; matches the ``eval_pi05.py`` slicing.
    """
    if port is None:
        raise ValueError(
            "load_policy_for_eval(...) for Pi0.5 requires an explicit `port=`; "
            "the openpi server runs in a separate process and we won't guess."
        )

    if client_factory is None:
        # Local import — keeps this module importable in environments
        # that don't have openpi_client (e.g. on the login node).
        from openpi_client.websocket_client_policy import (  # type: ignore
            WebsocketClientPolicy,
        )
        client_factory = WebsocketClientPolicy

    client = client_factory(host=host, port=port)

    def policy(obs: Any) -> np.ndarray:
        # The script lives upstream of any per-driver obs preprocessing —
        # the whole point of the overfit-replay check is that THE POLICY
        # owns its obs preprocessing.  For Pi0.5 that means the caller
        # has already shaped ``obs`` into the dict the openpi server
        # expects.  We just hand it through and unwrap the chunk.
        result = client.infer(obs)
        chunk = np.asarray(result["actions"])
        if active_dim is not None:
            chunk = chunk[:, :active_dim]
        return chunk

    # Stash provenance / inspection hooks for tests + debugging.
    policy.client = client          # type: ignore[attr-defined]
    policy.host = host              # type: ignore[attr-defined]
    policy.port = port              # type: ignore[attr-defined]
    policy.ckpt_path = str(ckpt_path)  # type: ignore[attr-defined]
    policy.backend = "pi05"         # type: ignore[attr-defined]
    return policy


# ---------------------------------------------------------------------------
# DP backend (stub — see module docstring for why)
# ---------------------------------------------------------------------------


def _load_dp_policy(ckpt_path: Path, **kwargs: Any) -> PolicyFn:
    raise NotImplementedError(
        "Diffusion-Policy (DP) load_policy_for_eval is not yet implemented. "
        "DP loading is tightly coupled to the BCRLImageWorkspace / "
        "RobotWorkspace + DPRunner machinery in "
        "policy/Diffusion-Policy/eval_dp.py: the runner owns frame-stacking, "
        "the action-chunk cache, and the normalizer stats. Reproducing all "
        "of that faithfully is equivalent to importing the eval driver "
        "in-process. Until a reusable in-process DP driver is factored out, "
        "use the Pi0.5 backend for the overfit-replay heavy preflight, or "
        "drive DP eval through eval_dp.py directly. "
        f"(requested ckpt: {ckpt_path})"
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def load_policy_for_eval(
    ckpt_path: str | Path,
    **kwargs: Any,
) -> PolicyFn:
    """Load a policy for the eval-side overfit-replay sanity check.

    Parameters
    ----------
    ckpt_path
        Path to the checkpoint.  Dispatch is path-shape based:

        * ``*.ckpt`` / ``*.pt`` / ``*.bin``  → Diffusion Policy backend
          (currently ``NotImplementedError`` — see module docstring).
        * directory with ``params/`` subdir, or a path whose final
          component is ``params``, or any path containing a ``params``
          part → Pi0.5 backend.
        * anything else → ``ValueError``.

    **kwargs
        Forwarded to the per-backend loader.  For Pi0.5: ``host``,
        ``port`` (required), ``active_dim``, ``prompt``,
        ``client_factory`` (test seam).

    Returns
    -------
    policy : Callable[[obs], np.ndarray]
        Callable returning an action chunk of shape ``(K, A)`` (or
        ``(A,)`` — the caller's ``_ensure_chunked`` handles both).
    """
    path = Path(ckpt_path)

    if _looks_like_dp_ckpt(path):
        return _load_dp_policy(path, **kwargs)

    if _looks_like_pi05_dir(path):
        return _load_pi05_policy(path, **kwargs)

    raise ValueError(
        f"load_policy_for_eval: cannot infer backend from ckpt_path={path!s}. "
        f"Expected either a Diffusion-Policy checkpoint file ending in one of "
        f"{sorted(_DP_SUFFIXES)}, or a Pi0.5 / openpi directory containing a "
        f"'params/' subdir (or whose final component is 'params'). "
        f"If your layout is unusual, pass an explicit backend by calling "
        f"_load_dp_policy / _load_pi05_policy directly."
    )
