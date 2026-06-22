"""Tests for lint_eval_launchers rule F (hardcoded-port detection, PR1).

Rule F flags a standalone ``80[0-9][0-9]`` PORT literal in a targeted launcher
while NOT flagging checkpoint-step path components (``/18000``, ``_v1/18000``),
env-var defaults written as ``-8000`` suffixes, or explanatory comments.
"""
from __future__ import annotations

from pathlib import Path

from robofactory.utils.lint_eval_launchers import (
    RE_HARDCODED_PORT,
    lint_one,
)


def test_regex_matches_real_port_usages():
    for s in (
        "--port 8000",
        "--ports 8000,8001",
        "start_server 0 8000 ARM",
        "('127.0.0.1',8000)",
        "probing ports 8000/8001",
    ):
        assert RE_HARDCODED_PORT.search(s), f"should flag port literal in {s!r}"


def test_regex_ignores_ckpt_steps_and_longer_numbers():
    for s in (
        "_v1/18000",                       # ckpt step (path)
        "/checkpoints/run/19999",          # ckpt step
        "step-8000",                       # ckpt-step wandb tag (preceded by -)
        "CKPT_STEP:-8000",                 # env-var default (preceded by -)
        "port 80000",                      # 5-digit, not an 80xx port
        "8200",                            # 82xx is out of the 80xx scope
    ):
        assert not RE_HARDCODED_PORT.search(s), f"should NOT flag {s!r}"


def test_lint_one_flags_hardcoded_port(tmp_path: Path):
    f = tmp_path / "bad_60seeds.sh"
    f.write_text(
        "#!/bin/bash\n"
        "# a comment mentioning 8000 is fine\n"
        "start_server 0 8000 cfg dir arm0\n"
    )
    rpt = lint_one(f)
    f_findings = [x for x in rpt.findings if x.rule == "F"]
    assert len(f_findings) == 1, f"exactly one rule-F finding expected, got {rpt.findings}"
    assert f_findings[0].line_no == 3, "must point at the code line, not the comment"


def test_lint_one_allows_free_ports_pattern(tmp_path: Path):
    f = tmp_path / "good_60seeds.sh"
    f.write_text(
        "#!/bin/bash\n"
        "# never collide on the hardcoded 8000/8001 (explanatory comment)\n"
        "source _lib/free_ports.sh\n"
        'read -r PORT0 PORT1 <<<"$(free_ports 2)"\n'
        'start_server 0 "$PORT0" cfg dir arm0\n'
        "CKPT_DIR=/ckpt/run_v1/18000\n"
    )
    rpt = lint_one(f)
    f_findings = [x for x in rpt.findings if x.rule == "F"]
    assert f_findings == [], f"no rule-F finding expected, got {f_findings}"
