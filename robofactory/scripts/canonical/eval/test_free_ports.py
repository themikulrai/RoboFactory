"""Unit tests for the free_ports() helper (PR1 — kill hardcoded-port collisions).

free_ports(n) must return n DISTINCT, currently-bindable TCP ports. The
distinctness guarantee is what prevents two co-scheduled eval jobs from picking
the same server port and silently routing a client to the wrong policy.
"""
from __future__ import annotations

import socket

import pytest

from robofactory.scripts.canonical.eval._lib.pi05_server_supervisor import free_ports


def test_returns_n_distinct_bindable_ports():
    ports = free_ports(3)
    assert len(ports) == 3, "must return exactly 3 ports"
    assert all(isinstance(p, int) for p in ports), "ports must be ints"
    assert len(set(ports)) == 3, f"ports must be DISTINCT, got {ports}"
    # Each port must be re-bindable (it was free when allocated).
    for p in ports:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))  # raises OSError if not bindable
        finally:
            s.close()


def test_single_port():
    ports = free_ports(1)
    assert len(ports) == 1
    assert isinstance(ports[0], int)


def test_many_distinct():
    n = 16
    ports = free_ports(n)
    assert len(set(ports)) == n, f"all {n} ports must be distinct"


def test_rejects_zero():
    with pytest.raises(ValueError):
        free_ports(0)


def test_rejects_negative():
    with pytest.raises(ValueError):
        free_ports(-1)
