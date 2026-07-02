"""Per-arm identity-prefix prompt composition (RIP: Robot-Identifying Prompt).

Pure, GPU-free helper shared by the flat (``eval_decent_pi05.py``) and hierarchical
(``eval_hierarchical_lift_barrier.py``) eval drivers so both compose the identity
prompt identically. RIP prepends a per-arm token (``"<left arm>"`` / ``"<right arm>"``)
to the base instruction with a SINGLE space; an empty/whitespace prefix is a strict
no-op (today's behavior), so passing empty prefixes reproduces the baseline byte-for-byte.

Vocab validation stays on the BARE base prompt at the call sites — the prefix is a
separate, un-validated token (it is out-of-vocab by design).
"""
from __future__ import annotations


def compose_identity_prompt(prefix: str, base: str) -> str:
    """Return ``"<prefix> <base>"`` (single space) or ``base`` if prefix is empty.

    ``prefix`` is stripped; a None/empty/whitespace-only prefix returns ``base``
    unchanged. ``base`` is returned verbatim (not stripped) so the un-prefixed path
    is bit-identical to the pre-RIP behavior.
    """
    p = (prefix or "").strip()
    return f"{p} {base}" if p else base
