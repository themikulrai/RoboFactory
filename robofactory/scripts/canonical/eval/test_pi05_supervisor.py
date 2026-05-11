"""Integration tests for Pi05ServerSupervisor.

These tests exercise the full spawn/wait/reap cycle against a tiny
TCP-listening fake serve_policy stub. They run on Linux only (the
PR_SET_PDEATHSIG path is Linux-specific); skipped elsewhere.
"""
from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from robofactory.scripts.canonical.eval._lib.pi05_server_supervisor import (
    Pi05ServerSupervisor,
    ServerSpec,
)
from robofactory.scripts.canonical.eval._lib.port_wait import (
    is_port_up,
    wait_for_ports,
)


pytestmark = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="supervisor uses Linux-only prctl(PR_SET_PDEATHSIG)",
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Fake serve_policy: parse `--port N`, bind it, loop. Mimics the openpi
# serve_policy CLI surface enough that the supervisor's _build_cmd works.
_FAKE_SERVE_POLICY = textwrap.dedent(
    """
    import argparse, socket, sys, time
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("policy_keyword", nargs="?")
    p.add_argument("--policy.config")
    p.add_argument("--policy.dir")
    args, unknown = p.parse_known_args()
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", args.port))
    s.listen(1)
    print(f"[fake-serve_policy] listening on {args.port}", flush=True)
    while True:
        time.sleep(0.5)
    """
).strip()


@pytest.fixture
def fake_openpi_tree(tmp_path: Path) -> Path:
    """Build a tmp openpi dir with a fake .venv/bin/python + serve_policy.py."""
    openpi = tmp_path / "openpi"
    (openpi / ".venv" / "bin").mkdir(parents=True)
    (openpi / "scripts").mkdir()
    (openpi / "scripts" / "serve_policy.py").write_text(_FAKE_SERVE_POLICY)
    # Symlink real python so the child has a real interpreter.
    (openpi / ".venv" / "bin" / "python").symlink_to(sys.executable)
    return openpi


def _make_supervisor(specs, log_dir, openpi_dir: Path):
    sup = Pi05ServerSupervisor(specs, log_dir=log_dir)
    sup.OPENPI_PYTHON = str(openpi_dir / ".venv" / "bin" / "python")
    sup.OPENPI_DIR = str(openpi_dir)
    sup.SERVE_POLICY_SCRIPT = "scripts/serve_policy.py"
    return sup


class TestSupervisorLifecycle:
    def test_single_server_spawn_and_reap(self, fake_openpi_tree, tmp_path):
        port = _free_port()
        spec = ServerSpec(
            port=port, gpu_index=0,
            policy_config="dummy", policy_dir="/dummy",
            xla_mem_fraction=0.1, name="solo",
        )
        sup = _make_supervisor([spec], tmp_path / "logs", fake_openpi_tree)
        with sup:
            wait_for_ports([port], deadline_s=15.0, poll_s=0.5)
            assert is_port_up("127.0.0.1", port)
            pid = sup._procs[0].pid
            # Sanity: child is alive.
            os.kill(pid, 0)
        # After __exit__: child should be dead. Wait briefly for kernel cleanup.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"server pid={pid} still alive after __exit__")
        # Port should be free.
        assert not is_port_up("127.0.0.1", port)

    def test_three_server_spawn_and_reap(self, fake_openpi_tree, tmp_path):
        ports = [_free_port() for _ in range(3)]
        specs = [
            ServerSpec(
                port=p, gpu_index=i,
                policy_config=f"cfg{i}", policy_dir=f"/d{i}",
                xla_mem_fraction=0.1, name=f"arm{i}",
            )
            for i, p in enumerate(ports)
        ]
        sup = _make_supervisor(specs, tmp_path / "logs", fake_openpi_tree)
        with sup:
            wait_for_ports(ports, deadline_s=15.0, poll_s=0.5)
            for p in ports:
                assert is_port_up("127.0.0.1", p)
            pids = [proc.pid for proc in sup._procs]
        # All children dead post-__exit__.
        deadline = time.time() + 3.0
        for pid in pids:
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                pytest.fail(f"server pid={pid} still alive after __exit__")
        for p in ports:
            assert not is_port_up("127.0.0.1", p)

    def test_log_files_and_pid_files_written(self, fake_openpi_tree, tmp_path):
        port = _free_port()
        spec = ServerSpec(
            port=port, gpu_index=0,
            policy_config="x", policy_dir="/y",
            xla_mem_fraction=0.1, name="lognames",
        )
        sup = _make_supervisor([spec], tmp_path / "logs", fake_openpi_tree)
        with sup:
            wait_for_ports([port], deadline_s=15.0, poll_s=0.5)
            assert (tmp_path / "logs" / "lognames.pid").exists()
            assert (tmp_path / "logs" / "lognames.log").exists()
            content = (tmp_path / "logs" / "lognames.log").read_text()
            # Fake script prints when listening.
            assert "listening on" in content


class TestSupervisorAbort:
    """Test that signal interruption tears children down (no orphans)."""

    def test_kill_parent_reaps_child_via_pdeathsig(self, fake_openpi_tree, tmp_path):
        """
        Spawn a parent Python process that brings up a supervisor and then
        sleeps. SIGKILL the parent. Verify the child server dies (PDEATHSIG
        does the work, not the parent's atexit/trap).
        """
        port = _free_port()
        parent_script = tmp_path / "parent.py"
        # Build a self-contained parent that imports our supervisor.
        parent_script.write_text(textwrap.dedent(f"""
            import os, sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
            from robofactory.scripts.canonical.eval._lib.pi05_server_supervisor import (
                Pi05ServerSupervisor, ServerSpec,
            )
            spec = ServerSpec(
                port={port}, gpu_index=0,
                policy_config="x", policy_dir="/y",
                xla_mem_fraction=0.1, name="abortme",
            )
            sup = Pi05ServerSupervisor([spec], log_dir={str(tmp_path / "logs2")!r})
            sup.OPENPI_PYTHON = {str(fake_openpi_tree / ".venv" / "bin" / "python")!r}
            sup.OPENPI_DIR = {str(fake_openpi_tree)!r}
            sup.SERVE_POLICY_SCRIPT = "scripts/serve_policy.py"
            sup.__enter__()
            # Print child PID so the test harness can verify it dies.
            print(sup._procs[0].pid, flush=True)
            time.sleep(300)  # Sleep until killed.
        """))
        # Launch parent. Capture stderr for debugging diagnostics.
        parent = subprocess.Popen(
            [sys.executable, str(parent_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Read the child PID the parent prints once setup is done.
        line = parent.stdout.readline().strip()
        try:
            child_pid = int(line)
        except ValueError:
            parent.kill()
            try:
                _, err = parent.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                err = "<parent didn't exit>"
            pytest.fail(
                f"parent did not print child pid; got {line!r}. stderr:\n{err}"
            )
        # Wait briefly for the fake server to bind.
        deadline = time.time() + 10.0
        while time.time() < deadline and not is_port_up("127.0.0.1", port):
            time.sleep(0.2)
        assert is_port_up("127.0.0.1", port), "fake server never bound port"
        # SIGKILL the parent — atexit and trap WILL NOT fire.
        parent.kill()
        parent.wait(timeout=5)
        # Child should die within ~2s via PR_SET_PDEATHSIG.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            # Force-kill so we don't leak.
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pytest.fail(
                f"child pid={child_pid} survived SIGKILL to parent — "
                "PR_SET_PDEATHSIG did not fire"
            )
