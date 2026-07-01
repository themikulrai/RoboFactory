"""Merge seed-sharded eval result JSONs into one full-pool result.

Each shard is a ``run_eval.py --num-shards`` array task that evaluated a disjoint
round-robin slice of a seed pool (see ``shard_seeds`` in ``run_eval.py``). This
recombines the per-shard JSONs into a single result in the ``eval_decent_pi05.py``
schema, recomputing the aggregate metrics over the union of episodes.

    python -m robofactory.scripts.canonical.eval.merge_shards \\
        --inputs '/iris/u/mikulrai/logs/eval_pi05_decent_shard*of*/eval_decent_*.json' \\
        --pool canonical_env_60 \\
        --out /iris/u/mikulrai/logs/eval_pi05_decent/eval_decent_merged.json

A silently-wrong success rate is a P0, so every consistency check is a HARD failure
(raises ``MergeError``):
  (a) ``args`` must agree across shards, ignoring the fields that legitimately vary
      per shard/job (seeds, seed pool, ports, out/video dirs, run id, shard suffix);
  (b) checkpoint identity — the ``server_metadata`` block — must be byte-identical
      across shards when present;
  (c) the UNION of per-episode seeds must equal ``resolve_seeds(--pool)`` EXACTLY —
      no duplicate, no gap — derived the same way the driver derives per-episode
      seeds (``pool_seed + ep_i`` for ``ep_i in range(num_episodes)``).

Requeue story: if one shard directory holds several timestamped JSONs (an orion
``PreemptMode=REQUEUE`` job reran from scratch), the NEWEST is used and a warning
is emitted — the stale rerun is dropped, never double-counted.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from robofactory.utils.eval_seeds import resolve_seeds

# Recompute validity exactly like the driver (eval_decent_pi05.main).
from robofactory.utils.eval_validity import classify_validity


class MergeError(ValueError):
    """A shard-consistency check failed — merging would produce a wrong number."""


# args keys that legitimately differ across shards of the SAME eval; everything
# else must match. seeds/env_seeds/seed_pool: sharded. ports: job-unique free ports.
# out_dir/video_dir: _shard{i}of{n} suffixed. run_id/wandb_name: per array task.
# trajectory_*: per-run h5 paths.
_IGNORE_ARGS = frozenset({
    "seeds", "env_seeds", "seed_pool",
    "ports",
    "out_dir", "video_dir",
    "run_id", "wandb_name",
    "trajectory_log_path", "trajectory_root",
})


def _embedded_ts(path: Path) -> int:
    """Unix ts embedded in an ``eval_*_<ts>.json`` filename; file mtime as fallback."""
    m = re.search(r"_(\d+)\.json$", path.name)
    if m:
        return int(m.group(1))
    return int(path.stat().st_mtime)


def _expand_inputs(inputs: list[str]) -> list[Path]:
    """Expand each --inputs arg (a glob or a literal path) into existing JSON files."""
    files: list[Path] = []
    for pat in inputs:
        matched = sorted(glob.glob(pat))
        if matched:
            files.extend(Path(p) for p in matched)
        elif Path(pat).exists():
            files.append(Path(pat))
        else:
            raise FileNotFoundError(f"--inputs pattern matched nothing: {pat!r}")
    # Stable, de-duplicated.
    seen: dict[Path, None] = {}
    for f in files:
        seen.setdefault(f.resolve(), None)
    return list(seen)


def _dedup_by_shard_dir(files: list[Path]) -> list[Path]:
    """One JSON per shard directory; on multiple (orion requeue) keep newest + warn."""
    by_dir: dict[Path, list[Path]] = {}
    for f in files:
        by_dir.setdefault(f.parent, []).append(f)
    chosen: list[Path] = []
    for d in sorted(by_dir):
        group = by_dir[d]
        if len(group) > 1:
            newest = max(group, key=_embedded_ts)
            warnings.warn(
                f"[merge_shards] shard dir {d} has {len(group)} result JSONs "
                f"(orion requeue rerun?); using newest {newest.name} and dropping "
                f"{sorted(p.name for p in group if p != newest)}"
            )
            chosen.append(newest)
        else:
            chosen.append(group[0])
    return chosen


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_args_consistent(shards: list[dict], paths: list[Path]) -> None:
    ref_args = shards[0].get("args", {})
    ref = {k: v for k, v in ref_args.items() if k not in _IGNORE_ARGS}
    for js, p in zip(shards[1:], paths[1:]):
        cur = {k: v for k, v in js.get("args", {}).items() if k not in _IGNORE_ARGS}
        if cur != ref:
            diff = sorted(
                k for k in set(ref) | set(cur) if ref.get(k) != cur.get(k)
            )
            details = "\n".join(
                f"    {k}: {ref.get(k)!r} (shard0) != {cur.get(k)!r} ({p.name})"
                for k in diff
            )
            raise MergeError(
                "shard args disagree on non-shard fields (would merge incompatible "
                f"runs):\n{details}"
            )


def _assert_ckpt_identity(shards: list[dict], paths: list[Path]) -> None:
    ref = shards[0].get("server_metadata")
    if ref is None:
        return  # nothing to compare (DP or older schema)
    for js, p in zip(shards[1:], paths[1:]):
        cur = js.get("server_metadata")
        if cur != ref:
            raise MergeError(
                f"checkpoint identity mismatch: server_metadata in {p.name} differs "
                f"from shard0 — shards evaluated DIFFERENT checkpoints:\n"
                f"    shard0: {ref}\n    {p.name}: {cur}"
            )


def _episode_seeds(shards: list[dict]) -> list[int]:
    out: list[int] = []
    for js in shards:
        for r in js.get("results", []):
            out.append(int(r["seed"]))
    return out


def _assert_seed_coverage(
    shards: list[dict], pool: str, num_episodes: int, allow_train: bool
) -> list[int]:
    """Union of per-episode seeds must equal the pool (× num_episodes) EXACTLY."""
    pool_seeds, _ = resolve_seeds(pool=pool, allow_train=allow_train)
    # Mirror run_episode: per-episode seed = pool_seed + ep_i.
    expected = sorted(s + ep for s in pool_seeds for ep in range(num_episodes))
    got = sorted(_episode_seeds(shards))

    if len(got) != len(set(got)):
        seen: set[int] = set()
        dups = sorted({s for s in got if (s in seen) or seen.add(s)})
        raise MergeError(
            f"duplicate episode seeds across shards {dups} — a seed was evaluated "
            "more than once (overlapping shards or a double-counted requeue)."
        )
    if got != expected:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise MergeError(
            f"episode-seed union does not match pool {pool!r} "
            f"(expected {len(expected)}, got {len(got)}):\n"
            f"    missing (gaps): {missing}\n"
            f"    extra (not in pool): {extra}"
        )
    return expected


def merge_shards(
    inputs: list[str],
    pool: str,
    out: Optional[str] = None,
) -> dict:
    """Validate + merge sharded eval JSONs. Returns the merged result dict.

    Raises ``MergeError`` on any consistency failure. Writes ``out`` if given."""
    files = _expand_inputs(inputs)
    if not files:
        raise MergeError("no input files resolved from --inputs")
    chosen = _dedup_by_shard_dir(files)
    if len(chosen) < 1:
        raise MergeError("no shard JSONs after requeue de-duplication")
    shards = [_load(p) for p in chosen]

    _assert_args_consistent(shards, chosen)
    _assert_ckpt_identity(shards, chosen)

    ref_args = dict(shards[0].get("args", {}))
    num_episodes = int(ref_args.get("num_episodes", 1))
    allow_train = bool(ref_args.get("allow_train_seeds", False))
    _assert_seed_coverage(shards, pool, num_episodes, allow_train)

    # Union of episodes, sorted by (seed, then original order) for a stable result.
    merged_results: list[dict] = []
    for js in shards:
        merged_results.extend(js.get("results", []))
    merged_results.sort(key=lambda r: (int(r["seed"]),))

    # Recompute aggregate metrics EXACTLY as eval_decent_pi05.main does.
    n = len(merged_results)
    n_succ = sum(1 for r in merged_results if r.get("success"))
    n_sustained = sum(1 for r in merged_results if r.get("success_sustained_10") is True)
    validity = classify_validity(merged_results)
    pool_seeds, pool_provenance = resolve_seeds(pool=pool, allow_train=allow_train)

    merged = dict(shards[0])  # template: args, provenance, server_metadata, prompt_validation…
    merged["args"] = {
        **ref_args,
        "seeds": ",".join(str(s) for s in pool_seeds),
        "seed_pool": pool,
        "out_dir": str(Path(out).parent) if out else ref_args.get("out_dir", ""),
        "video_dir": "",
        "run_id": "merged",
    }
    merged["seed_provenance"] = pool_provenance  # full-pool provenance (name + sha)
    merged["validity"] = validity
    merged["results"] = merged_results
    merged["merged_from"] = [str(p) for p in chosen]
    merged["merged_shards"] = {
        "num_shards": len(shards),
        "pool": pool,
        "num_episodes": num_episodes,
        "merged_at": int(time.time()),
        "aggregate": {
            "n_episodes": n,
            "n_success": n_succ,
            "success_rate": (n_succ / n) if n else 0.0,
            "success_first_rate": (n_succ / n) if n else 0.0,
            "n_success_sustained_10": n_sustained,
            "success_sustained_10_rate": (n_sustained / n) if n else 0.0,
        },
    }

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(merged, indent=2))
    return merged


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Shard result-JSON paths or globs (one dir per shard).")
    p.add_argument("--pool", required=True,
                   help="Seed pool name understood by eval_seeds.resolve_seeds "
                        "(e.g. canonical_env_60); the union of shard seeds must equal it.")
    p.add_argument("--out", required=True, help="Merged result JSON output path.")
    args = p.parse_args(argv)

    try:
        merged = merge_shards(args.inputs, args.pool, out=args.out)
    except (MergeError, FileNotFoundError) as e:
        print(f"[merge_shards] FAIL: {e}", file=sys.stderr)
        return 1
    agg = merged["merged_shards"]["aggregate"]
    print(
        f"[merge_shards] merged {merged['merged_shards']['num_shards']} shards "
        f"({agg['n_episodes']} episodes) -> {args.out}\n"
        f"  success_first: {agg['n_success']}/{agg['n_episodes']} "
        f"({100.0 * agg['success_first_rate']:.1f}%)\n"
        f"  success_sustained_10: {agg['n_success_sustained_10']}/{agg['n_episodes']} "
        f"({100.0 * agg['success_sustained_10_rate']:.1f}%)\n"
        f"  valid={merged['validity']['valid']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
