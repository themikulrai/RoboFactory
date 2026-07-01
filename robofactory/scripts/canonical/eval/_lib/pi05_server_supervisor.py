"""Spawn + supervise Pi0.5 serve_policy children with strict-reaping semantics.

Bash `trap cleanup EXIT INT TERM` works for graceful exit, but if the parent
shell is SIGKILL'd by Slurm at time-limit, the JAX server children are
orphaned. We use `prctl(PR_SET_PDEATHSIG, SIGTERM)` in the child's
`preexec_fn` so the kernel signals SIGTERM to the child the moment the
parent process dies — even via SIGKILL.

Linux-only (ctypes loads libc.so.6). On non-Linux, the supervisor falls
back to atexit + signal-handler cleanup, which catches everything except
SIGKILL.
"""
from __future__ import annotations

import atexit
import contextlib
import ctypes
import os
import platform
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Linux PR_SET_PDEATHSIG = 1; see <sys/prctl.h>.
_PR_SET_PDEATHSIG = 1
_IS_LINUX = platform.system() == "Linux"


def free_ports(n: int, host: str = "127.0.0.1") -> list[int]:
    """Return ``n`` DISTINCT, currently-free TCP ports on ``host``.

    The single source of truth for runtime port allocation in the eval
    pipeline (mirrors ``robofactory/scripts/_lib/free_ports.sh`` for bash
    launchers). Co-scheduled eval jobs on one node used to collide on the
    hardcoded 8000/8001/8002 server ports; a same-node clash silently routes a
    client to the wrong policy server. Allocating job-unique ports here removes
    that whole class (solution_E6.md §PR1).

    All ``n`` sockets are bound (to port 0 -> OS-assigned) BEFORE any is
    closed. Binding one-at-a-time would let the kernel re-hand the just-freed
    port, producing duplicates. There is an unavoidable TOCTOU window between
    closing here and the server re-binding, but two simultaneous binds never
    alias, which is the collision we actually saw in the wild.
    """
    if n < 1:
        raise ValueError(f"free_ports: n must be >= 1, got {n}")
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind((host, 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _set_pdeathsig_then_setsid() -> None:
    """preexec_fn: child receives SIGTERM when parent dies. Also new session."""
    if _IS_LINUX:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            rc = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
            if rc != 0:
                # Non-fatal: child still runs, just without auto-reap.
                err = ctypes.get_errno()
                print(
                    f"[pi05_supervisor] prctl(PR_SET_PDEATHSIG) failed: errno={err}",
                    file=sys.stderr,
                )
        except OSError:
            pass
    # Detach from controlling terminal — keeps SIGINT to the parent from
    # propagating via the tty group.
    try:
        os.setsid()
    except OSError:
        pass


@dataclass
class ServerSpec:
    """One Pi0.5 server: GPU + port + policy config + ckpt dir."""

    port: int
    gpu_index: int
    policy_config: str
    policy_dir: str
    xla_mem_fraction: float
    name: str  # log-line label (e.g. "arm0", "pm")


class Pi05ServerSupervisor:
    """Context manager that spawns + reaps Pi0.5 serve_policy children.

    On `__enter__`: spawns each ServerSpec via `subprocess.Popen` with
    `preexec_fn=_set_pdeathsig_then_setsid`. PIDs are recorded; logs stream
    to per-server files.

    On `__exit__` (or SIGTERM/SIGINT to the parent): sends SIGTERM to each
    process group, waits 2s, then SIGKILL. atexit is registered as a
    belt-and-suspenders fallback for unexpected interpreter exit paths.
    """

    OPENPI_PYTHON = "/iris/u/mikulrai/projects/openpi/.venv/bin/python"
    OPENPI_DIR = "/iris/u/mikulrai/projects/openpi"
    SERVE_POLICY_SCRIPT = "scripts/serve_policy.py"

    def __init__(
        self,
        specs: list[ServerSpec],
        log_dir: "Path | str",
    ) -> None:
        self.specs = specs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._procs: list[subprocess.Popen] = []
        self._signal_installed = False

    def _build_cmd(self, spec: ServerSpec) -> list[str]:
        return [
            self.OPENPI_PYTHON,
            self.SERVE_POLICY_SCRIPT,
            "--port",
            str(spec.port),
            "policy:checkpoint",
            f"--policy.config={spec.policy_config}",
            f"--policy.dir={spec.policy_dir}",
        ]

    def _spawn(self, spec: ServerSpec) -> subprocess.Popen:
        env = os.environ.copy()
        # An unset OPENPI_DATA_HOME makes serve_policy re-download the 11.6GB norm-stats /
        # asset bundle into the venv default, which blows the wait_for_ports health-check
        # window (600s) and cold-starts every job. Pin it to the shared cache.
        env.setdefault("OPENPI_DATA_HOME", "/iris/u/mikulrai/.cache/openpi")
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_index)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(spec.xla_mem_fraction)
        log_path = self.log_dir / f"{spec.name}.log"
        log_fh = open(log_path, "wb")
        print(
            f"[pi05_supervisor] starting {spec.name} GPU={spec.gpu_index} "
            f"port={spec.port} cfg={spec.policy_config} dir={spec.policy_dir}",
            file=sys.stderr,
            flush=True,
        )
        # cwd=OPENPI_DIR so serve_policy.py resolves its config tree.
        proc = subprocess.Popen(
            self._build_cmd(spec),
            cwd=self.OPENPI_DIR,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=_set_pdeathsig_then_setsid,
        )
        (self.log_dir / f"{spec.name}.pid").write_text(str(proc.pid))
        return proc

    def _terminate_all(self, sig: int = signal.SIGTERM) -> None:
        for proc in self._procs:
            if proc.poll() is not None:
                continue
            try:
                # Signal the whole process group (we did setsid in the child).
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                pass

    def _cleanup(self) -> None:
        if not self._procs:
            return
        print("[pi05_supervisor] terminating servers (SIGTERM)", file=sys.stderr, flush=True)
        self._terminate_all(signal.SIGTERM)
        # Wait up to 2s.
        for proc in self._procs:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        # Force-kill anything still alive.
        alive = [p for p in self._procs if p.poll() is None]
        if alive:
            print("[pi05_supervisor] SIGKILLing leftover servers", file=sys.stderr, flush=True)
            self._terminate_all(signal.SIGKILL)
        self._procs = []

    def _install_signal_handlers(self) -> None:
        if self._signal_installed:
            return

        def _handler(signum, frame):  # noqa: ARG001
            print(
                f"[pi05_supervisor] caught signal {signum}; cleaning up.",
                file=sys.stderr,
                flush=True,
            )
            self._cleanup()
            # Re-raise the default behavior for the signal.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError):
                signal.signal(sig, _handler)
        atexit.register(self._cleanup)
        self._signal_installed = True

    def __enter__(self) -> "Pi05ServerSupervisor":
        self._install_signal_handlers()
        for spec in self.specs:
            self._procs.append(self._spawn(spec))
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self._cleanup()
        return None

    @property
    def ports(self) -> list[int]:
        return [s.port for s in self.specs]


__all__ = ["ServerSpec", "Pi05ServerSupervisor", "free_ports"]
