"""TCP port-wait probe used to detect Pi0.5 server readiness."""
from __future__ import annotations

import socket
import sys
import time
from typing import Iterable


def is_port_up(host: str, port: int, timeout_s: float = 1.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout_s)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def wait_for_ports(
    ports: Iterable[int],
    host: str = "127.0.0.1",
    deadline_s: float = 600.0,
    poll_s: float = 5.0,
) -> None:
    """Block until every port is listening. Raises TimeoutError after `deadline_s`.

    Probes ports in order — i.e. if port 8001 comes up before 8000, we still
    wait on 8000 first. This matches the bash loop in the legacy launchers
    and keeps log output stable.
    """
    start = time.time()
    for port in ports:
        while not is_port_up(host, port):
            elapsed = time.time() - start
            if elapsed >= deadline_s:
                raise TimeoutError(
                    f"port {port} on {host} did not come up within "
                    f"{deadline_s:.0f}s (elapsed {elapsed:.0f}s)"
                )
            time.sleep(poll_s)
        print(f"[port_wait] {host}:{port} up", file=sys.stderr, flush=True)
