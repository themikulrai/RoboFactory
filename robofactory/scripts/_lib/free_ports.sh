#!/bin/bash
# free_ports.sh — single source of truth for job-unique free-port allocation.
#
# WHY: Co-scheduled eval jobs on the same node used to collide on the hardcoded
# 8000/8001/8002 server ports. A same-node port clash silently routes one
# client to the WRONG policy server (the casualty signature was
# `IndexError('index 15 is out of bounds for axis 0 with size 8')` plus 0%
# evals — solution_E6.md §PR1). This helper hands out N DISTINCT, currently
# free TCP ports so no two co-scheduled jobs ever pick the same one.
#
# USAGE (source it, then call):
#     source "$(dirname "$0")/_lib/free_ports.sh"
#     read -r PORT0 PORT1 <<<"$(free_ports 2)"
#
# Or run standalone:
#     robofactory/scripts/_lib/free_ports.sh 3   # -> "53124 41887 60233"
#
# Prints N space-separated ports on stdout. Exits non-zero on bad N.
#
# IMPLEMENTATION NOTE: all N sockets are bound to port 0 SIMULTANEOUSLY before
# any is closed. Binding one-at-a-time would let the OS hand back the
# just-freed port, yielding duplicates. Generalizes the inline bind-0 snippet
# that previously lived in eval_pi05_lb_wc_decent_60seeds.sh:49-59.

# Python used purely for the socket bind. Prefer the RoboFactory env python;
# fall back to whatever `python3`/`python` is on PATH so the helper works from
# any venv that sources it.
_FREE_PORTS_PY="${FREE_PORTS_PYTHON:-/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python}"
if [ ! -x "$_FREE_PORTS_PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        _FREE_PORTS_PY="$(command -v python3)"
    else
        _FREE_PORTS_PY="$(command -v python)"
    fi
fi

free_ports() {
    local n="${1:?free_ports: number of ports required}"
    "$_FREE_PORTS_PY" -c "
import socket, sys
n = int(sys.argv[1])
if n < 1:
    raise SystemExit('free_ports: n must be >= 1')
socks = [socket.socket() for _ in range(n)]
try:
    for s in socks:
        s.bind(('127.0.0.1', 0))   # bind ALL before closing any -> distinct
    print(*[s.getsockname()[1] for s in socks])
finally:
    for s in socks:
        s.close()
" "$n"
}

# Allow standalone execution: `free_ports.sh N`.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    free_ports "${1:-1}"
fi
