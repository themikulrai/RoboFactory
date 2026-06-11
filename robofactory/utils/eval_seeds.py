"""Single source of truth for eval seed conventions (solution_E6.md §PR4).

Before this module, RoboFactory had THREE incompatible seed conventions baked
into the eval drivers, and the drivers themselves applied silent per-driver
transforms (``seed * 100_000 + ep_i`` in the pi0.5 drivers, ``(seed + ep) *
stride`` in the hierarchical driver). The consequence: NO DP-vs-pi0.5 number was
ever seed-paired, the "canonical 32/60" was run on raw bases 100..159 that
silently overlapped the datagen training pool after the transform, and the
"paired" manifest entry fed DP raw seeds into a ``x100_000`` path (audit_A3 §2).

This module makes the env seed the ONE quantity that matters: an env seed is
just an int handed to ``env.reset(seed=...)``. A "pool" is a named, frozen list
of FINAL env seeds. Drivers consume those env seeds DIRECTLY — there is no
downstream transform left anywhere. Because both DP and pi0.5 drivers reset the
same env with the same int, resolving the same pool name in both gives true
seed pairing.

Pools
-----
- ``train_datagen``    range(0, 183)             — the datagen pool. REFUSED for
                                                   eval by default (overlapping
                                                   it leaks training states).
- ``canonical_env_60`` [b*100_000 for b in 100..159] — the historical pi0.5
                                                   convention, now expressed as
                                                   FINAL env seeds (the x100_000
                                                   that used to live inside the
                                                   driver is folded in here).
- ``fresh_ood_60``     range(20_000, 20_060)     — the 05-29 fresh-wave OOD set.
- ``upstream_100``     range(1_000, 1_100)       — the paper protocol set
                                                   (blocker-1 controls).
- ``dp_legacy_60``     [10_000..10_029]+[1_000..1_029] — DEPRECATED. The old DP
                                                   convention; kept ONLY so old
                                                   result JSONs remain decodable.

Public API
----------
- ``get_pool(name) -> list[int]``    final env seeds, no transform applied.
- ``resolve_seeds(pool=None, env_seeds=None, allow_train=False) -> (list[int], dict)``
                                     the one entry point drivers call; returns
                                     the env seeds plus a provenance dict to
                                     embed verbatim in the result JSON.
- ``assert_no_train_overlap(seeds, allow_train=False)`` hard-fail on a datagen
                                     seed unless explicitly allowed (the caller
                                     records ``allow_train`` in the JSON).
- ``pool_sha(name) -> str``          stable sha256 over the pool's env seeds.
- ``POOL_NAMES``                     the set of valid pool names.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- pools

# Datagen pool: seeds 0..182 were used by the RoboFactory planner to GENERATE the
# training trajectories. Evaluating on any of these leaks a memorized init state,
# so eval REFUSES them unless the caller passes allow_train=True (recorded in JSON).
_TRAIN_DATAGEN_LO = 0
_TRAIN_DATAGEN_HI = 183  # exclusive -> 0..182 are train seeds

# Each value below is a FINAL env seed (the int passed to env.reset). There is NO
# remaining per-driver transform — what get_pool returns is what env.reset sees.
POOLS: Dict[str, List[int]] = {
    # datagen pool — REFUSED for eval by default.
    "train_datagen": list(range(_TRAIN_DATAGEN_LO, _TRAIN_DATAGEN_HI)),
    # historical pi0.5 convention. The drivers USED to compute base*100_000 from
    # bases 100..159; that x100_000 is folded in here so the driver consumes the
    # final env seed directly.
    "canonical_env_60": [b * 100_000 for b in range(100, 160)],
    # 05-29 fresh-wave OOD convention.
    "fresh_ood_60": list(range(20_000, 20_060)),
    # paper protocol (blocker-1 controls).
    "upstream_100": list(range(1_000, 1_100)),
    # DEPRECATED — the old DP convention. Archaeology only (old result JSONs).
    "dp_legacy_60": list(range(10_000, 10_030)) + list(range(1_000, 1_030)),
}

POOL_NAMES = frozenset(POOLS)

# Pools that are deprecated / refused by default — surfaced so callers and the
# lint can warn without re-encoding policy.
DEPRECATED_POOLS = frozenset({"dp_legacy_60"})
TRAIN_POOLS = frozenset({"train_datagen"})


# --------------------------------------------------------------------------- helpers
def get_pool(name: str) -> List[int]:
    """Return the FINAL env seeds for ``name`` (a fresh copy; no transform).

    Raises ``KeyError`` with the valid names on an unknown pool."""
    try:
        return list(POOLS[name])
    except KeyError:
        raise KeyError(
            "unknown seed pool {!r}; valid pools: {}".format(
                name, sorted(POOL_NAMES)
            )
        )


def _sha_of_seeds(seeds: List[int]) -> str:
    """sha256 over the seed list in a canonical, order-sensitive encoding.

    Compact JSON of the int list — stable across machines/python builds and
    independent of any file formatting (newline vs comma)."""
    payload = json.dumps([int(s) for s in seeds], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pool_sha(name: str) -> str:
    """Stable sha256 of pool ``name``'s env seeds (order-sensitive)."""
    return _sha_of_seeds(get_pool(name))


def seeds_sha(seeds: List[int]) -> str:
    """Stable sha256 of an arbitrary resolved env-seed list."""
    return _sha_of_seeds(list(seeds))


def is_train_seed(seed: int) -> bool:
    """True iff ``seed`` is in the datagen pool 0..182."""
    return _TRAIN_DATAGEN_LO <= int(seed) < _TRAIN_DATAGEN_HI


def assert_no_train_overlap(seeds: List[int], allow_train: bool = False) -> None:
    """Hard-fail if any seed overlaps the datagen pool (0..182).

    The datagen pool generated the training trajectories; evaluating on it leaks
    memorized init states (the TSC-cent-0/60-on-raw-100-159 class). Pass
    ``allow_train=True`` ONLY for a deliberate train-seed probe — the caller MUST
    record that flag in the result JSON so a train-overlapping run is never
    mistaken for a clean held-out eval."""
    if allow_train:
        return
    overlap = sorted({int(s) for s in seeds if is_train_seed(s)})
    if overlap:
        raise ValueError(
            "seed pool overlaps the datagen training pool 0..{}: {} "
            "(evaluating on datagen seeds leaks memorized init states). Use a "
            "held-out pool (canonical_env_60 / fresh_ood_60 / upstream_100) or, "
            "for a deliberate train-seed probe, pass allow_train=True — which is "
            "recorded in the result JSON.".format(_TRAIN_DATAGEN_HI - 1, overlap)
        )


def _parse_env_seeds(env_seeds: str) -> List[int]:
    """Parse a comma- or whitespace-separated ad-hoc env-seed list to ints."""
    toks = [t for t in env_seeds.replace(",", " ").split() if t.strip()]
    if not toks:
        raise ValueError("--env-seeds was empty")
    return [int(t) for t in toks]


def resolve_seeds(
    pool: Optional[str] = None,
    env_seeds: Optional[str] = None,
    allow_train: bool = False,
) -> Tuple[List[int], Dict[str, object]]:
    """Resolve the env seeds a driver will run + a provenance dict for the JSON.

    Exactly one of ``pool`` / ``env_seeds`` should be supplied:
      - ``pool``      a name in POOL_NAMES -> get_pool(name) (preferred).
      - ``env_seeds`` an ad-hoc comma/space list -> recorded as pool ``"adhoc"``.

    Returns ``(seeds, provenance)`` where ``provenance`` is a JSON-safe dict:
    ``{seed_pool, seed_pool_sha, n_seeds, allow_train}``. Always runs the
    train-overlap guard (unless ``allow_train``). Drivers embed ``provenance``
    verbatim next to ``args`` so every result names its seed convention."""
    if pool and env_seeds:
        raise ValueError("pass at most one of pool / env_seeds, not both")
    if pool:
        seeds = get_pool(pool)
        pool_name = pool
    elif env_seeds:
        seeds = _parse_env_seeds(env_seeds)
        pool_name = "adhoc"
    else:
        raise ValueError("must pass exactly one of pool / env_seeds")

    assert_no_train_overlap(seeds, allow_train=allow_train)
    provenance: Dict[str, object] = {
        "seed_pool": pool_name,
        "seed_pool_sha": _sha_of_seeds(seeds),
        "n_seeds": len(seeds),
        "allow_train": bool(allow_train),
    }
    return seeds, provenance


def _cli(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI so launchers / Makefiles can dump or pin a pool.

    ``python -m robofactory.utils.eval_seeds <pool>``            -> newline seeds
    ``python -m robofactory.utils.eval_seeds <pool> --sha``      -> pool sha256
    ``python -m robofactory.utils.eval_seeds <pool> --delim ,``  -> joined seeds
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pool", choices=sorted(POOL_NAMES))
    p.add_argument("--sha", action="store_true", help="print pool sha256 instead of seeds")
    p.add_argument("--delim", default="\n", help="join delimiter (default newline)")
    a = p.parse_args(argv)
    if a.sha:
        print(pool_sha(a.pool))
        return 0
    print(a.delim.join(str(s) for s in get_pool(a.pool)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
