"""Unit tests for RIP identity-prefix prompt composition (pure, GPU-free).

Guards: single-space join, empty/whitespace prefix is a strict no-op (baseline
reproduction), and base is returned verbatim when un-prefixed.
"""
from __future__ import annotations

from robofactory.policy.openpi_pi05.identity_prompt import compose_identity_prompt

BASE = "stack the two cubes with the two robot arms"


def test_left_prefix_single_space():
    assert compose_identity_prompt("<left arm>", BASE) == f"<left arm> {BASE}"


def test_right_prefix_single_space():
    assert compose_identity_prompt("<right arm>", BASE) == f"<right arm> {BASE}"


def test_empty_prefix_is_noop():
    assert compose_identity_prompt("", BASE) == BASE


def test_none_prefix_is_noop():
    assert compose_identity_prompt(None, BASE) == BASE


def test_whitespace_prefix_is_noop():
    assert compose_identity_prompt("   ", BASE) == BASE


def test_prefix_is_stripped_no_double_space():
    # a padded prefix must not yield a double space
    assert compose_identity_prompt("  <left arm>  ", BASE) == f"<left arm> {BASE}"
    assert "  " not in compose_identity_prompt("  <left arm>  ", BASE)


def test_base_returned_verbatim_when_unprefixed():
    # base is NOT stripped/altered on the no-op path (bit-identical to pre-RIP)
    weird = "  trailing and leading  "
    assert compose_identity_prompt("", weird) == weird


def test_subtask_base_prefixed():
    assert compose_identity_prompt("<right arm>", "place on the goal") == "<right arm> place on the goal"
