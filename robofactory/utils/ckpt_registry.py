"""Checkpoint-registry drift guard (Week-2 lane L3, decision-table row #8).

THE BUG THIS PREVENTS
    One openpi ``config_name`` resolves to MULTIPLE training-run subdirs on disk.
    An eval that resolves "the" checkpoint by config_name silently grabs whichever
    run a launcher pointed at. That produced the bogus "LB ws 100% -> 20%" reading:
    E10 resolved ``pi05_robofactory_lb_ws_cent`` to the *worse* of two training
    runs (05-16-fp/19999) and reported a fake regression.

WHAT THIS MODULE DOES
    Loads the pinned registry (``scripts/canonical/eval/ckpt_registry.yaml``) and,
    given the checkpoint an eval ACTUALLY resolved for a config_name -- either from
    the manifest (``cfg.ckpt.dir`` / ``cfg.server.arm_dirs``) or, better, from the
    PR2 server-identity metadata the served policy announces over the websocket
    (``config_name`` / ``ckpt_dir`` / ``ckpt_step`` / ``wandb_id`` / ``norm_stats_md5``)
    -- compares resolved-vs-registered and:

      * MATCH                  -> ok, no-op.
      * DRIFT (dir/step diff)  -> hard-fail (or loud warn), recorded for the result JSON.
      * UNVERIFIED/AMBIGUOUS/MISSING config -> hard-fail by default (``guard_action: fail``)
                                  so nobody silently auto-evals an un-pinned run.
      * config NOT in registry -> "unregistered": warn, do not block (PM/2SC/LP are
                                  intentionally unscoped for L3).

    The comparison key is ``ckpt_dir`` + ``step``. ``norm_stats_md5`` is a DATA hash
    (ws/wc arms of one task share it; the two lb_ws_cent runs share it) so it is a
    secondary cross-check only, never the discriminator between sibling runs.

DESIGN MIRRORS ``server_identity.py``
    Pure, GPU-free, returns ``(ok, diagnosis, record)``; an ``assert_*_or_exit``
    wrapper does ``sys.exit(4)`` on drift (distinct from server_identity's 3 so
    launchers/CI can tell "wrong run for the right config" from "wrong config").
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

# Hard-fail exit code for a checkpoint-registry drift (distinct from
# server_identity's IDENTITY_MISMATCH_EXIT=3).
REGISTRY_DRIFT_EXIT = 4

# Default registry location (next to the eval manifest).
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "canonical" / "eval" / "ckpt_registry.yaml"
)


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    config_name: str
    canonical_ckpt_dir: Optional[str]
    step: Optional[int]
    wandb_id: Optional[str]
    git_sha: Optional[str]
    norm_stats_md5: Optional[str]
    trust: str
    guard_action: str  # "fail" or "warn"; how to treat a non-pinned/UNVERIFIED entry

    @property
    def is_pinned(self) -> bool:
        """True when this entry pins a concrete canonical dir to compare against."""
        return self.canonical_ckpt_dir is not None


# Trust tags whose canonical ckpt is deliberately NOT pinned. Evaluating them is a
# provenance hazard, so the default guard_action for these is "fail".
_UNPINNED_TRUST = {"UNVERIFIED", "AMBIGUOUS", "MISSING"}


class CkptRegistry:
    """In-memory view of ``ckpt_registry.yaml``."""

    def __init__(self, entries: Mapping[str, RegistryEntry], ckpt_root: Optional[str] = None):
        self._entries = dict(entries)
        self.ckpt_root = ckpt_root

    @classmethod
    def load(cls, path: Path | str = DEFAULT_REGISTRY_PATH) -> "CkptRegistry":
        path = Path(path)
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        ckpt_root = data.get("ckpt_root")
        raw = data.get("registry", {}) or {}
        entries: dict[str, RegistryEntry] = {}
        for name, row in raw.items():
            row = row or {}
            trust = str(row.get("trust", "UNVERIFIED")).upper()
            # guard_action precedence: explicit field > derived from trust.
            default_action = "fail" if trust in _UNPINNED_TRUST else "warn"
            entries[name] = RegistryEntry(
                config_name=name,
                canonical_ckpt_dir=row.get("canonical_ckpt_dir"),
                step=row.get("step"),
                wandb_id=row.get("wandb_id"),
                git_sha=row.get("git_sha"),
                norm_stats_md5=row.get("norm_stats_md5"),
                trust=trust,
                guard_action=str(row.get("guard_action", default_action)).lower(),
            )
        return cls(entries, ckpt_root=ckpt_root)

    def get(self, config_name: Optional[str]) -> Optional[RegistryEntry]:
        if not config_name:
            return None
        return self._entries.get(config_name)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, config_name: str) -> bool:
        return config_name in self._entries


def _norm_dir(path: Optional[str]) -> Optional[str]:
    """Normalise a ckpt dir for comparison: collapse `.`/`//`, strip trailing slash.

    We do NOT resolve symlinks (the whole ckpt tree is symlinked outside the repo;
    resolving would make every pin compare against a volatile real path). Pure
    string normalisation is enough to catch a wrong-run/wrong-step path.
    """
    if path is None:
        return None
    return str(Path(path)).rstrip("/")


def check_registry_drift(
    registry: CkptRegistry,
    config_name: Optional[str],
    resolved_ckpt_dir: Optional[str],
    resolved_step: Optional[int] = None,
    *,
    resolved_norm_stats_md5: Optional[str] = None,
    label: str = "ckpt",
) -> Tuple[bool, Optional[str], dict]:
    """Compare the checkpoint an eval resolved against the pinned canonical one.

    Args:
        registry: a loaded ``CkptRegistry``.
        config_name: the openpi config name the eval is serving (e.g.
            ``pi05_robofactory_lb_ws_cent``). Typically taken from the PR2 server
            metadata (``meta['config_name']``) or the manifest ``train_config``.
        resolved_ckpt_dir: the checkpoint DIR actually resolved — from the PR2
            metadata ``meta['ckpt_dir']`` (preferred: it is what the server really
            loaded) or the manifest ``cfg.ckpt.dir`` / per-arm ``arm_dirs[i]``.
        resolved_step: optional resolved step (PR2 ``meta['ckpt_step']``). When the
            resolved dir already ends in the step (``.../19999``), passing it is
            redundant but harmless; we compare the dir's trailing step too.
        resolved_norm_stats_md5: optional; cross-checked only, never the sole basis
            for a mismatch (it is a data hash, shared across sibling runs).
        label: short label for the diagnosis line (e.g. ``arm0`` / ``port 8123``).

    Returns:
        ``(ok, diagnosis, record)``.
          * ``ok`` is True when the resolved ckpt matches the pinned canonical, OR
            the config is unregistered (un-scoped) and thus un-checkable.
          * ``ok`` is False on DRIFT, or on an UNVERIFIED/AMBIGUOUS/MISSING config
            whose ``guard_action`` is ``fail``.
          * ``diagnosis`` is a one-line string on mismatch, else None.
          * ``record`` is always returned — a dict to embed in the eval result JSON
            so the provenance check is auditable even when it passes.
    """
    record: dict[str, Any] = {
        "config_name": config_name,
        "resolved_ckpt_dir": resolved_ckpt_dir,
        "resolved_step": resolved_step,
        "resolved_norm_stats_md5": resolved_norm_stats_md5,
        "registry_status": None,
        "registered_ckpt_dir": None,
        "registered_step": None,
        "registered_wandb_id": None,
        "trust": None,
        "drift": False,
        "checked": False,
    }

    entry = registry.get(config_name)

    # --- config not in registry: unscoped (PM/2SC/LP). Warn, do not block. -----
    if entry is None:
        record["registry_status"] = "unregistered"
        return True, None, record

    record["trust"] = entry.trust
    record["registered_ckpt_dir"] = entry.canonical_ckpt_dir
    record["registered_step"] = entry.step
    record["registered_wandb_id"] = entry.wandb_id

    # --- config registered but NOT pinned (UNVERIFIED/AMBIGUOUS/MISSING) -------
    if not entry.is_pinned:
        record["registry_status"] = f"unpinned:{entry.trust}"
        record["checked"] = True
        if entry.guard_action == "fail":
            diag = (
                f"CKPT REGISTRY: {label} config={config_name!r} is "
                f"{entry.trust} (no canonical checkpoint pinned); "
                f"refusing to auto-eval an un-resolved run "
                f"(resolved={resolved_ckpt_dir!r})"
            )
            return False, diag, record
        # warn-mode: surface but proceed.
        diag = (
            f"CKPT REGISTRY WARN: {label} config={config_name!r} is {entry.trust}; "
            f"resolved={resolved_ckpt_dir!r} is unverified against a canonical pin"
        )
        return True, diag, record

    # --- config pinned: the real drift comparison ------------------------------
    record["registry_status"] = "pinned"
    record["checked"] = True

    canon_dir = _norm_dir(entry.canonical_ckpt_dir)
    got_dir = _norm_dir(resolved_ckpt_dir)

    if got_dir is None:
        # Pinned config but the eval gave us nothing to compare -> cannot verify.
        diag = (
            f"CKPT REGISTRY: {label} config={config_name!r} is pinned to "
            f"{entry.canonical_ckpt_dir!r} but the eval resolved no ckpt dir; "
            f"refusing to run unverified"
        )
        record["drift"] = True
        return False, diag, record

    # Primary check: resolved dir must equal the pinned canonical dir.
    if got_dir != canon_dir:
        record["drift"] = True
        diag = (
            f"CKPT REGISTRY DRIFT: {label} config={config_name!r} resolved "
            f"ckpt={resolved_ckpt_dir!r} but registry pins "
            f"{entry.canonical_ckpt_dir!r} (wandb {entry.wandb_id}); "
            f"refusing to run (this is the wrong-checkpoint-for-config disaster)"
        )
        return False, diag, record

    # Secondary check: step, when both are known. The pinned dir already ends in
    # the canonical step, so a resolved_step that disagrees is a real drift.
    if resolved_step is not None and entry.step is not None and int(resolved_step) != int(entry.step):
        record["drift"] = True
        diag = (
            f"CKPT REGISTRY DRIFT: {label} config={config_name!r} resolved "
            f"step={resolved_step} but registry pins step={entry.step} "
            f"(dir {entry.canonical_ckpt_dir!r}); refusing to run"
        )
        return False, diag, record

    # Tertiary cross-check: norm_stats_md5 mismatch is a WARN (data hash, not a
    # per-run discriminator) — it never flips ok to False on its own, but a
    # mismatch on a pinned dir is worth recording.
    if (
        resolved_norm_stats_md5 is not None
        and entry.norm_stats_md5 is not None
        and resolved_norm_stats_md5 != entry.norm_stats_md5
    ):
        record["norm_stats_md5_warn"] = (
            f"norm_stats_md5 {resolved_norm_stats_md5} != registered "
            f"{entry.norm_stats_md5} on an otherwise-matching pinned dir"
        )

    return True, None, record


def assert_registry_or_exit(
    registry: CkptRegistry,
    config_name: Optional[str],
    resolved_ckpt_dir: Optional[str],
    resolved_step: Optional[int] = None,
    *,
    resolved_norm_stats_md5: Optional[str] = None,
    label: str = "ckpt",
) -> dict:
    """Hard-fail wrapper: print the one-line diagnosis and ``sys.exit(4)`` on drift.

    Returns the provenance ``record`` (for embedding in the result JSON) when it
    does NOT exit. In warn-mode it prints the warning to stderr and returns the
    record. A clean match prints nothing.
    """
    ok, diagnosis, record = check_registry_drift(
        registry,
        config_name,
        resolved_ckpt_dir,
        resolved_step,
        resolved_norm_stats_md5=resolved_norm_stats_md5,
        label=label,
    )
    if not ok:
        print(diagnosis, file=sys.stderr, flush=True)
        sys.exit(REGISTRY_DRIFT_EXIT)
    if diagnosis:  # warn-mode message
        print(diagnosis, file=sys.stderr, flush=True)
    return record


def check_from_server_metadata(
    registry: CkptRegistry,
    metadata: Optional[Mapping],
    *,
    label: str = "server",
) -> Tuple[bool, Optional[str], dict]:
    """Convenience: run the drift check straight off the PR2 server-identity dict.

    ``metadata`` is what ``WebsocketClientPolicy.get_server_metadata()`` returns
    (see ``serve_policy.py::build_identity_metadata``). Missing metadata (legacy
    server) -> unverifiable -> ok (mirrors ``server_identity`` semantics).
    """
    meta = dict(metadata or {})
    if not meta:
        return True, None, {"registry_status": "no_server_metadata", "checked": False}
    return check_registry_drift(
        registry,
        meta.get("config_name"),
        meta.get("ckpt_dir"),
        meta.get("ckpt_step") if isinstance(meta.get("ckpt_step"), int) else None,
        resolved_norm_stats_md5=meta.get("norm_stats_md5"),
        label=label,
    )


__all__ = [
    "REGISTRY_DRIFT_EXIT",
    "DEFAULT_REGISTRY_PATH",
    "RegistryEntry",
    "CkptRegistry",
    "check_registry_drift",
    "assert_registry_or_exit",
    "check_from_server_metadata",
]
