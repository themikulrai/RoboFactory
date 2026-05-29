"""End-to-end PLUMBING smoke test for the hierarchical Lift-Barrier eval.

Validates the orchestration layer WITHOUT any trained checkpoints, by spanning the two
relevant envs via subprocess:

  - mock HL server  : memer env, `hl_server.py --mock`  (fixed dual subtask, no Qwen)
  - hierarchical eval: RoboFactory env, `eval_hierarchical_lift_barrier.py --mock-ll`
                       (zero-action LL chunks, no pi0.5 servers) + real SAPIEN env

Asserts (success RATE is irrelevant pre-training — only plumbing):
  1. eval ran to completion (exit 0) and wrote eval_LiftBarrier-rf_*.json
  2. the env actually stepped  (episode "steps" >= 1)
  3. HL was queried at least once  (n_hl_queries >= 1)
  4. BOTH arms received a subtask prompt  (HL log line shows left= and right=)
  5. a success bool was written  (episodes[0]["success"] is a bool)

This test runs SAPIEN, so it must execute on a COMPUTE node (not the login node) and in the
RoboFactory env. It self-skips if the env pythons are missing. Run with:

    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
        robofactory/policy/openpi_pi05/test_hierarchical_smoke.py -s
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

MEMER_PY = "/iris/u/mikulrai/envs/memer/bin/python"
RF_PY = "/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python"
HL_SERVER = "/iris/u/mikulrai/projects/memer/scripts/hl_server.py"
EVAL = "/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/openpi_pi05/eval_hierarchical_lift_barrier.py"

pytestmark = pytest.mark.skipif(
    not (Path(MEMER_PY).exists() and Path(RF_PY).exists() and Path(HL_SERVER).exists() and Path(EVAL).exists()),
    reason="requires memer + RoboFactory envs and the HL server / eval scripts",
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(port: int, retries: int = 60) -> bool:
    import http.client

    for _ in range(retries):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/healthz")
            r = c.getresponse()
            r.read()
            c.close()
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def test_hierarchical_plumbing_mock():
    hl_port = _free_port()
    tmpdir = Path(tempfile.mkdtemp(prefix="hl_smoke_"))
    results_dir = tmpdir / "results"

    hl_proc = subprocess.Popen(
        [MEMER_PY, HL_SERVER, "--mock", "--port", str(hl_port)],
        stdout=open(tmpdir / "hl.log", "w"),
        stderr=subprocess.STDOUT,
    )
    eval_log = tmpdir / "eval.log"
    try:
        assert _wait_health(hl_port), f"mock HL server never healthy on :{hl_port}"

        cmd = [
            RF_PY, EVAL,
            "--mock-ll",
            "--hl-host", "127.0.0.1", "--hl-port", str(hl_port),
            "--hl-query-interval", "10",
            "--max-episodes", "1",
            "--max-env-steps", "30",
            "--seed", "1000",
            "--results-dir", str(results_dir),
        ]
        with open(eval_log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=900)

        log_text = eval_log.read_text()
        # 1. eval completed cleanly
        assert rc == 0, f"eval exited {rc}\n--- eval log ---\n{log_text[-4000:]}"

        # find the results JSON
        jsons = sorted(results_dir.glob("eval_LiftBarrier-rf_*.json"))
        assert jsons, f"no results JSON written\n--- eval log ---\n{log_text[-4000:]}"
        results = json.loads(jsons[-1].read_text())

        assert results["num_episodes"] == 1
        ep = results["episodes"][0]

        # 2. env stepped
        assert ep["steps"] >= 1, f"env did not step: {ep}"
        # 3. HL queried at least once
        assert ep["n_hl_queries"] >= 1, f"HL never queried: {ep}"
        # 5. success bool written
        assert isinstance(ep["success"], bool), f"success not a bool: {ep!r}"
        # 4. both arms received a subtask prompt (from HL log line)
        assert "left=" in log_text and "right=" in log_text, "missing per-arm prompt log lines"
        # mock HL returns approach/lift; confirm both flowed through
        assert "'approach'" in log_text, "arm0 prompt (subtask_left) not injected"
        assert "'lift'" in log_text, "arm1 prompt (subtask_right) not injected"

        print(f"[smoke OK] steps={ep['steps']} n_hl_queries={ep['n_hl_queries']} "
              f"success={ep['success']} subtasks=({ep['last_subtask_left']},{ep['last_subtask_right']})")
    finally:
        hl_proc.terminate()
        try:
            hl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            hl_proc.kill()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
