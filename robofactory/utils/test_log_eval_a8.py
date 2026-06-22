"""Tests for the A8 loop-killer convention (WEEK1_EXECUTION §A8).

Two halves:
  1. ``lint_eval_launchers`` rule H — an inline 60-seed launcher that does NOT
     call ``scripts/log_eval.py`` at the end is a FAILURE; one that does call it
     passes; a thin ``exec .../submit_eval.sh`` dispatch wrapper is EXEMPT.
  2. ``scripts/log_eval.py`` — byte-compiles, parses both eval output formats
     (pi0.5 single-JSON and DP per-line JSONL), and a dry-run / mocked POST
     builds the correct cell payload (SR recomputed, validity propagated,
     metrics in {k,v,d} wire format, key read from env never hardcoded).
"""
from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from robofactory.utils.lint_eval_launchers import lint_one


# Locate scripts/log_eval.py relative to this test (robofactory/utils/ -> robofactory/scripts/).
_REPO = Path(__file__).resolve().parents[1]  # robofactory/
LOG_EVAL_PATH = _REPO / "scripts" / "log_eval.py"


def _load_log_eval():
    """Import scripts/log_eval.py as a module (it is not a package member)."""
    spec = importlib.util.spec_from_file_location("log_eval_a8", LOG_EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Rule H — lint enforcement of the auto-post
# ---------------------------------------------------------------------------
def _h_findings(rpt):
    return [f for f in rpt.findings if f.rule == "H"]


def test_rule_h_fires_when_log_eval_missing(tmp_path: Path):
    """An inline 60-seed launcher with NO log_eval.py call -> rule H failure."""
    f = tmp_path / "eval_demo_60seeds.sh"
    f.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_env_60.txt)\n"
        'python -u ./policy/openpi_pi05/eval_pi05.py --seeds "$SEEDS" '
        '--out-dir "$OUT_DIR"\n'
        'echo "Done at $(date)"\n'
    )
    rpt = lint_one(f)
    assert len(_h_findings(rpt)) == 1, f"expected one rule-H finding, got {rpt.findings}"


def test_rule_h_passes_when_log_eval_present(tmp_path: Path):
    """The same launcher with a trailing log_eval.py call -> no rule-H finding."""
    f = tmp_path / "eval_demo_60seeds.sh"
    f.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "SEEDS=$(paste -sd, /iris/u/mikulrai/runs/eval_seeds_env_60.txt)\n"
        'RESULT_FILE="$(ls -t $OUT_DIR/eval_*.json | head -1)"\n'
        'python -u ./policy/openpi_pi05/eval_pi05.py --seeds "$SEEDS" '
        '--out-dir "$OUT_DIR"\n'
        "python scripts/log_eval.py "
        '--jsonl "$RESULT_FILE" --job "${SLURM_JOB_ID:-manual}" '
        '--title "Eval demo" || true\n'
    )
    rpt = lint_one(f)
    assert _h_findings(rpt) == [], f"no rule-H finding expected, got {_h_findings(rpt)}"


def test_rule_h_exempts_exec_dispatch_wrapper(tmp_path: Path):
    """A thin `exec .../submit_eval.sh` wrapper never runs the eval inline, so
    rule H must NOT fire (the auto-post belongs in the dispatcher)."""
    f = tmp_path / "eval_thin_60seeds.sh"
    f.write_text(
        "#!/bin/bash\n"
        "# Thin wrapper -> manifest launcher (see eval/manifest.yaml).\n"
        'exec "$(dirname "$0")/eval/submit_eval.sh" some_launcher_id "$@"\n'
    )
    rpt = lint_one(f)
    assert _h_findings(rpt) == [], f"exec-wrapper must be exempt, got {_h_findings(rpt)}"


def test_rule_h_only_scopes_60seeds_named(tmp_path: Path):
    """Rule H keys on the `60seeds` name token; a non-60seeds smoke script that
    runs a driver inline is out of scope."""
    f = tmp_path / "smoke_5seeds.sh"
    f.write_text(
        "#!/bin/bash\n"
        "python -u ./policy/openpi_pi05/eval_pi05.py --out-dir x\n"
    )
    rpt = lint_one(f)
    assert _h_findings(rpt) == [], "non-60seeds script is out of rule-H scope"


# ---------------------------------------------------------------------------
# log_eval.py — compile + parse + payload
# ---------------------------------------------------------------------------
def test_log_eval_byte_compiles():
    py_compile.compile(str(LOG_EVAL_PATH), doraise=True)


PI05_JSON = {
    "args": {"task": "LiftBarrier-rf", "run_id": "9999001"},
    "validity": {"valid": True, "invalid_reason": None},
    "results": [
        {"seed": 100, "success": True, "steps": 120},
        {"seed": 101, "success": False, "steps": 400},
        {"seed": 102, "success": True, "steps": 88},
    ],
}

DP_JSONL_LINES = [
    {"kind": "manifest", "env_id": "LiftBarrier-rf", "checkpoint_num": 300},
    {"kind": "episode", "seed": 100, "success": False, "steps": 150},
    {"kind": "episode", "seed": 101, "success": True, "steps": 95},
    {"kind": "episode", "seed": 102, "success": False, "steps": 150},
    {"kind": "episode", "seed": 103, "success": False, "steps": 150},
    {"kind": "summary", "n_total": 4, "n_success": 1, "success_rate": 0.25},
    {"kind": "validity", "valid": False,
     "invalid_reason": "12% episodes steps==-1 (port-collision signature)"},
]


def _write_pi05(tmp_path: Path) -> Path:
    p = tmp_path / "eval_LiftBarrier-rf_123.json"
    p.write_text(json.dumps(PI05_JSON))
    return p


def _write_dp(tmp_path: Path) -> Path:
    p = tmp_path / "eval_dp.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in DP_JSONL_LINES) + "\n")
    return p


def test_parse_pi05_json(tmp_path: Path):
    mod = _load_log_eval()
    s = mod.parse_result_file(_write_pi05(tmp_path))
    assert s["fmt"] == "pi0.5-json"
    assert s["n"] == 3 and s["n_success"] == 2
    assert abs(s["sr"] - 2 / 3) < 1e-9
    assert s["valid"] is True


def test_parse_dp_jsonl(tmp_path: Path):
    mod = _load_log_eval()
    s = mod.parse_result_file(_write_dp(tmp_path))
    assert s["fmt"] == "dp-jsonl"
    assert s["n"] == 4 and s["n_success"] == 1
    assert abs(s["sr"] - 0.25) < 1e-9
    assert s["valid"] is False
    assert "port-collision" in s["invalid_reason"]


def test_build_cell_payload_shape(tmp_path: Path):
    mod = _load_log_eval()
    s = mod.parse_result_file(_write_pi05(tmp_path))
    cell = mod.build_cell(
        title="Eval LB wc Pi0.5 [Cent]",
        job="9999001",
        jsonl_path=str(_write_pi05(tmp_path)),
        summary=s,
    )
    # title / kind / status
    assert cell["title"] == "Eval LB wc Pi0.5 [Cent]"
    assert cell["kind"] == "agent"
    assert cell["status"] == "done"  # valid -> done
    # metrics use the {k, v, d} wire format (NOT {label, value} -> 422)
    assert isinstance(cell["metrics"], list) and cell["metrics"]
    for m in cell["metrics"]:
        assert set(m.keys()) <= {"k", "v", "d"}
        assert "k" in m and "v" in m
    # job id + result path + SR all present in the body / conclusion
    assert "9999001" in cell["body"]
    assert "2/3" in cell["conclusion"]
    assert "66.7%" in cell["body"]


def test_build_cell_invalid_run_status_blocked(tmp_path: Path):
    mod = _load_log_eval()
    s = mod.parse_result_file(_write_dp(tmp_path))
    cell = mod.build_cell(title="Eval LB wc DP", job="9999002",
                          jsonl_path="x.jsonl", summary=s)
    assert cell["status"] == "blocked"  # invalid -> blocked
    assert "INVALID" in cell["conclusion"]


def test_resolve_key_from_env_priority(monkeypatch):
    mod = _load_log_eval()
    for v in mod.KEY_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("FIELD_NOTES_KEY_FILE", raising=False)
    assert mod.resolve_key(None) is None  # no key anywhere -> None (soft skip)
    monkeypatch.setenv("FIELD_NOTES_KEY", "abc123")
    assert mod.resolve_key(None) == "abc123"
    monkeypatch.setenv("X_FIELD_NOTES_KEY", "xyz789")  # higher priority
    assert mod.resolve_key(None) == "xyz789"


def test_resolve_key_never_hardcoded():
    """Guard: the key must never be baked into the source (only env/file)."""
    src = LOG_EVAL_PATH.read_text()
    # the historical hardcoded render_eval_cell key must not appear here
    assert "78fc1af37b213bff1c9bc2c9c3b3897dc8ad4b7e2ff4cd110ef9f0e87e9c7428" not in src


def test_dry_run_cli_builds_payload_no_http(tmp_path: Path, monkeypatch):
    """End-to-end --dry-run via the CLI: prints a valid cell payload, makes no
    HTTP call (so it is safe in CI / on a node with no network)."""
    p = _write_dp(tmp_path)
    # Ensure no key in env so a non-dry path would soft-skip — dry-run ignores it.
    for v in ("X_FIELD_NOTES_KEY", "FIELD_NOTES_KEY", "FIELD_NOTES_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    out = subprocess.run(
        [sys.executable, str(LOG_EVAL_PATH),
         "--jsonl", str(p), "--job", "777", "--title", "Eval DP smoke",
         "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["title"] == "Eval DP smoke"
    assert payload["status"] == "blocked"  # the DP fixture is invalid
    assert any(m["v"] == "777" for m in payload["metrics"])


def test_missing_file_soft_skips(tmp_path: Path):
    """A missing result file must NOT fail the eval job (exit 0, no post)."""
    out = subprocess.run(
        [sys.executable, str(LOG_EVAL_PATH),
         "--jsonl", str(tmp_path / "nope.json"), "--job", "1", "--title", "x"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0


def test_post_called_with_correct_payload(tmp_path: Path, monkeypatch):
    """Mock the HTTP layer: verify post_cell is reached with the right base /
    project / key / payload when a key + project are present in env."""
    mod = _load_log_eval()
    captured = {}

    def fake_post(base, project_id, key, cell):
        captured["base"] = base
        captured["project_id"] = project_id
        captured["key"] = key
        captured["cell"] = cell
        return "cell-id-abc"

    monkeypatch.setattr(mod, "post_cell", fake_post)
    monkeypatch.setenv("FIELD_NOTES_KEY", "secret-key")
    monkeypatch.setenv("FIELD_NOTES_PROJECT", "proj-123")
    monkeypatch.setenv("FIELD_NOTES_API_URL", "https://example.test")

    rc = mod.main([
        "--jsonl", str(_write_pi05(tmp_path)),
        "--job", "555", "--title", "Eval mock",
    ])
    assert rc == 0
    assert captured["base"] == "https://example.test"
    assert captured["project_id"] == "proj-123"
    assert captured["key"] == "secret-key"
    assert captured["cell"]["title"] == "Eval mock"
    assert any(m["v"] == "555" for m in captured["cell"]["metrics"])


def test_no_key_soft_skips_without_post(tmp_path: Path, monkeypatch):
    """No key in env -> post_cell is NEVER called, exit 0 (logging must not
    brick the eval job)."""
    mod = _load_log_eval()
    called = {"n": 0}

    def fake_post(*a, **k):
        called["n"] += 1
        return "x"

    monkeypatch.setattr(mod, "post_cell", fake_post)
    for v in ("X_FIELD_NOTES_KEY", "FIELD_NOTES_KEY", "FIELD_NOTES_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("FIELD_NOTES_KEY_FILE", raising=False)
    rc = mod.main([
        "--jsonl", str(_write_pi05(tmp_path)),
        "--job", "1", "--title", "x",
    ])
    assert rc == 0
    assert called["n"] == 0, "post must not be attempted without a key"
