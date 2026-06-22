"""Unit tests for the checkpoint-registry drift guard (Week-2 lane L3).

CPU-only, no GPU/SAPIEN/checkpoint needed: the guard is a pure metadata
comparison. Tests cover:

  * the EXACT disaster (lb_ws_cent: config registered UNVERIFIED -> hard-fail
    so E10 can never silently re-grab the 05-16-fp/19999 run);
  * DRIFT: a pinned config resolved to the WRONG run dir -> hard-fail (4);
  * DRIFT: right dir, wrong step -> hard-fail;
  * MATCH on the pinned canonical -> ok + a provenance record;
  * unregistered config (PM/2SC/LP, out of L3 scope) -> ok, never blocks;
  * the PR2-server-metadata convenience path;
  * norm_stats_md5 mismatch on a matching dir -> WARN only (never flips ok);
  * the real on-disk ckpt_registry.yaml loads and pins what we expect.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from robofactory.utils.ckpt_registry import (
    REGISTRY_DRIFT_EXIT,
    DEFAULT_REGISTRY_PATH,
    CkptRegistry,
    RegistryEntry,
    assert_registry_or_exit,
    check_from_server_metadata,
    check_registry_drift,
)


# --------------------------------------------------------------------------- fixtures
def _registry(entries: dict[str, dict]) -> CkptRegistry:
    """Build an in-memory registry from a compact dict spec."""
    built = {}
    for name, row in entries.items():
        trust = row.get("trust", "CITE")
        default_action = "fail" if trust in {"UNVERIFIED", "AMBIGUOUS", "MISSING"} else "warn"
        built[name] = RegistryEntry(
            config_name=name,
            canonical_ckpt_dir=row.get("dir"),
            step=row.get("step"),
            wandb_id=row.get("wandb_id"),
            git_sha=row.get("git_sha"),
            norm_stats_md5=row.get("md5"),
            trust=trust,
            guard_action=row.get("guard_action", default_action),
        )
    return CkptRegistry(built)


CANON_DIR = "/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm0/lb-wc-dec-a0-bs16-20k-2026-05-16-fp/19999"
WRONG_RUN_DIR = "/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_wc_decent_arm0/lb-wc-decent-a0-bs16-20k-2026-05-13/19999"


@pytest.fixture
def reg() -> CkptRegistry:
    return _registry(
        {
            "pi05_robofactory_lb_wc_decent_arm0": {
                "dir": CANON_DIR, "step": 19999, "wandb_id": "0htz9zrs",
                "md5": "8eceaaae7a26b19e3417a53bc5f2103e", "trust": "CITE",
            },
            "pi05_robofactory_lb_ws_cent": {
                "dir": None, "step": None, "trust": "UNVERIFIED",
            },
            "pi05_robofactory_tsc_wc_decent_arm0": {
                "dir": None, "step": None, "trust": "AMBIGUOUS",
            },
            "pi05_robofactory_decent_lora_finetune_arm0": {
                "dir": None, "step": None, "trust": "MISSING",
            },
        }
    )


# --------------------------------------------------------------------------- match
def test_match_pinned_dir_ok(reg):
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_lb_wc_decent_arm0", CANON_DIR, 19999, label="arm0"
    )
    assert ok is True
    assert diag is None
    assert record["registry_status"] == "pinned"
    assert record["drift"] is False
    assert record["checked"] is True
    assert record["registered_wandb_id"] == "0htz9zrs"


def test_match_dir_without_explicit_step_ok(reg):
    # resolved_step omitted; dir match alone is sufficient.
    ok, diag, _ = check_registry_drift(reg, "pi05_robofactory_lb_wc_decent_arm0", CANON_DIR)
    assert ok is True and diag is None


def test_match_trailing_slash_normalised(reg):
    ok, diag, _ = check_registry_drift(
        reg, "pi05_robofactory_lb_wc_decent_arm0", CANON_DIR + "/", 19999
    )
    assert ok is True and diag is None


# --------------------------------------------------------------------------- drift
def test_wrong_run_dir_is_drift(reg):
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_lb_wc_decent_arm0", WRONG_RUN_DIR, 19999, label="arm0"
    )
    assert ok is False
    assert record["drift"] is True
    assert "CKPT REGISTRY DRIFT" in diag
    assert "2026-05-13" in diag        # the wrong run
    assert "2026-05-16-fp" in diag     # the pinned canonical
    assert "\n" not in diag.strip()    # one line


def test_right_dir_wrong_step_is_drift(reg):
    # Same run, but step 10000 instead of pinned 19999.
    bad = CANON_DIR.replace("/19999", "/10000")
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_lb_wc_decent_arm0", bad, 10000, label="arm0"
    )
    assert ok is False
    assert record["drift"] is True
    assert "DRIFT" in diag


def test_pinned_but_no_resolved_dir_refuses(reg):
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_lb_wc_decent_arm0", None, label="arm0"
    )
    assert ok is False
    assert record["drift"] is True
    assert "resolved no ckpt dir" in diag


# --------------------------------------------------------------------------- the disaster
def test_lb_ws_cent_unverified_hard_fails(reg):
    """THE provenance disaster: lb_ws_cent has no canonical pin -> any auto-eval
    that resolves a run (even the 05-16-fp/19999 E10 picked) must hard-fail."""
    e10_picked = "/iris/u/mikulrai/checkpoints/openpi/pi05_robofactory_lb_ws_cent/lb-ws-cent-bs16-20k-2026-05-16-fp/19999"
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_lb_ws_cent", e10_picked, 19999, label="lb_ws_cent"
    )
    assert ok is False
    assert record["trust"] == "UNVERIFIED"
    assert record["registry_status"] == "unpinned:UNVERIFIED"
    assert "UNVERIFIED" in diag
    assert "refusing to auto-eval" in diag


def test_ambiguous_and_missing_hard_fail(reg):
    for cfg in (
        "pi05_robofactory_tsc_wc_decent_arm0",   # AMBIGUOUS
        "pi05_robofactory_decent_lora_finetune_arm0",  # MISSING
    ):
        ok, diag, record = check_registry_drift(reg, cfg, "/some/resolved/dir/19999")
        assert ok is False, cfg
        assert record["registry_status"].startswith("unpinned:")
        assert "refusing to auto-eval" in diag


def test_unpinned_warn_mode_proceeds():
    # An unpinned entry explicitly set to guard_action=warn proceeds with a message.
    reg = _registry({"cfg": {"dir": None, "trust": "AMBIGUOUS", "guard_action": "warn"}})
    ok, diag, record = check_registry_drift(reg, "cfg", "/x/19999")
    assert ok is True
    assert "CKPT REGISTRY WARN" in diag
    assert record["checked"] is True


# --------------------------------------------------------------------------- unregistered
def test_unregistered_config_passes(reg):
    # PM is out of L3 scope -> unregistered -> never blocks.
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_pm_lora_finetune", "/whatever/19999"
    )
    assert ok is True
    assert diag is None
    assert record["registry_status"] == "unregistered"
    assert record["checked"] is False


def test_none_config_passes(reg):
    ok, diag, record = check_registry_drift(reg, None, "/x")
    assert ok is True
    assert record["registry_status"] == "unregistered"


# --------------------------------------------------------------------------- md5 cross-check
def test_md5_mismatch_on_matching_dir_is_warn_only(reg):
    ok, diag, record = check_registry_drift(
        reg, "pi05_robofactory_lb_wc_decent_arm0", CANON_DIR, 19999,
        resolved_norm_stats_md5="deadbeef",
    )
    assert ok is True               # md5 never flips ok by itself
    assert diag is None
    assert "norm_stats_md5_warn" in record


# --------------------------------------------------------------------------- assert_*_or_exit
def test_assert_exits_4_on_drift(reg, capsys):
    with pytest.raises(SystemExit) as ei:
        assert_registry_or_exit(
            reg, "pi05_robofactory_lb_wc_decent_arm0", WRONG_RUN_DIR, 19999, label="arm0"
        )
    assert ei.value.code == REGISTRY_DRIFT_EXIT == 4
    err = capsys.readouterr().err.strip()
    assert err.count("\n") == 0
    assert "CKPT REGISTRY DRIFT" in err


def test_assert_match_is_noop(reg, capsys):
    record = assert_registry_or_exit(
        reg, "pi05_robofactory_lb_wc_decent_arm0", CANON_DIR, 19999
    )
    assert capsys.readouterr().err == ""
    assert record["drift"] is False


def test_assert_unverified_exits_4(reg):
    with pytest.raises(SystemExit) as ei:
        assert_registry_or_exit(reg, "pi05_robofactory_lb_ws_cent", "/x/19999")
    assert ei.value.code == 4


# --------------------------------------------------------------------------- PR2 metadata path
def test_check_from_server_metadata_match(reg):
    meta = {
        "config_name": "pi05_robofactory_lb_wc_decent_arm0",
        "ckpt_dir": CANON_DIR,
        "ckpt_step": 19999,
        "norm_stats_md5": "8eceaaae7a26b19e3417a53bc5f2103e",
        "wandb_id": "0htz9zrs",
    }
    ok, diag, record = check_from_server_metadata(reg, meta, label="port 8123")
    assert ok is True and diag is None
    assert record["resolved_step"] == 19999


def test_check_from_server_metadata_drift(reg):
    meta = {
        "config_name": "pi05_robofactory_lb_wc_decent_arm0",
        "ckpt_dir": WRONG_RUN_DIR,
        "ckpt_step": 19999,
    }
    ok, diag, _ = check_from_server_metadata(reg, meta)
    assert ok is False
    assert "DRIFT" in diag


def test_check_from_server_metadata_legacy_empty_is_ok(reg):
    # Pre-PR2 server sends {} -> unverifiable -> proceed (mirrors server_identity).
    ok, diag, record = check_from_server_metadata(reg, {})
    assert ok is True
    assert record["registry_status"] == "no_server_metadata"
    ok2, _, _ = check_from_server_metadata(reg, None)
    assert ok2 is True


# --------------------------------------------------------------------------- real file
def test_real_registry_loads_and_pins_expected():
    reg = CkptRegistry.load(DEFAULT_REGISTRY_PATH)
    assert len(reg) >= 18  # LB + TSC entries

    # lb_ws_cent is the disaster row: registered but UNVERIFIED + hard-fail.
    ws_cent = reg.get("pi05_robofactory_lb_ws_cent")
    assert ws_cent is not None
    assert ws_cent.trust == "UNVERIFIED"
    assert ws_cent.is_pinned is False
    assert ws_cent.guard_action == "fail"

    # lb_wc_decent_arm0 is a CITE pin to the 05-16-fp/19999 run.
    wc0 = reg.get("pi05_robofactory_lb_wc_decent_arm0")
    assert wc0.trust == "CITE"
    assert wc0.is_pinned is True
    assert wc0.step == 19999
    assert "2026-05-16-fp" in wc0.canonical_ckpt_dir
    assert wc0.wandb_id == "0htz9zrs"

    # TSC d2_wristcam decent is pinned to IMPORTANT_TSC_ARM*.
    tsc0 = reg.get("pi05_robofactory_decent_wristcam_lora_finetune_arm0")
    assert tsc0.is_pinned is True
    assert "IMPORTANT_TSC_ARM0" in tsc0.canonical_ckpt_dir

    # TSC d1 decent_lora is MISSING (empty dir on disk).
    d1 = reg.get("pi05_robofactory_decent_lora_finetune_arm0")
    assert d1.trust == "MISSING"
    assert d1.guard_action == "fail"
