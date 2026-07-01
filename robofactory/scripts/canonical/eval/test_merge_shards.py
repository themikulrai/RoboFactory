"""Tests for merge_shards: shard recombination + the hard consistency guards.

A silently-wrong merged success rate is a P0, so the failure paths (missing seed,
duplicate seed, args mismatch, checkpoint mismatch) must HARD-fail, and the orion
requeue path (multiple JSONs in one shard dir) must take the newest with a warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robofactory.scripts.canonical.eval import merge_shards as m
from robofactory.utils.eval_seeds import get_pool

POOL = "fresh_ood_60"  # 20000..20059, 60 seeds, no train-pool overlap


def _results(seeds, succ_map):
    return [
        {
            "seed": int(s),
            "success": bool(succ_map.get(s, False)),
            "success_first": bool(succ_map.get(s, False)),
            "steps": 100,
            "wall_s": 1.0,
            "success_sustained_10": bool(succ_map.get(s, False)),
        }
        for s in seeds
    ]


def _shard_dict(seeds, succ_map, args_over=None, server_metadata="default"):
    args = {
        "task": "ThreeRobotsStackCube-rf",
        "num_arms": 3,
        "num_episodes": 1,
        "max_env_steps": 400,
        "replan_after": 8,
        "seeds": ",".join(str(s) for s in seeds),
        "ports": "1,2,3",
        "out_dir": "/x",
        "video_dir": "/x/v",
        "run_id": "J",
    }
    if args_over:
        args.update(args_over)
    sm = (
        {"arm0": {"config_name": "c0"}, "arm1": {"config_name": "c1"}, "arm2": {"config_name": "c2"}}
        if server_metadata == "default"
        else server_metadata
    )
    return {"args": args, "server_metadata": sm, "results": _results(seeds, succ_map)}


def _write_shards(tmp_path, shard_seed_lists, succ_map, per_shard_args=None,
                  server_metadata="default", ts_base=1000):
    n = len(shard_seed_lists)
    paths = []
    for i, seeds in enumerate(shard_seed_lists):
        d = tmp_path / f"eval_pi05_decent_shard{i}of{n}"
        d.mkdir(parents=True, exist_ok=True)
        over = {"ports": f"{i},{i},{i}", "out_dir": str(d),
                "video_dir": str(d / "v"), "run_id": f"J{i}"}
        if per_shard_args and i in per_shard_args:
            over.update(per_shard_args[i])
        js = _shard_dict(seeds, succ_map, args_over=over, server_metadata=server_metadata)
        p = d / f"eval_decent_T_{ts_base + i}.json"
        p.write_text(json.dumps(js))
        paths.append(p)
    return paths


def _three_way(pool):
    return [pool[0::3], pool[1::3], pool[2::3]]


def test_merge_three_shards_correct_aggregate(tmp_path):
    pool = get_pool(POOL)
    succ = {s: (s % 2 == 0) for s in pool}  # evens succeed -> 30 successes
    paths = _write_shards(tmp_path, _three_way(pool), succ)
    merged = m.merge_shards([str(p) for p in paths], POOL, out=str(tmp_path / "merged.json"))

    agg = merged["merged_shards"]["aggregate"]
    exp_succ = sum(1 for s in pool if s % 2 == 0)
    assert agg["n_episodes"] == 60
    assert agg["n_success"] == exp_succ
    assert abs(agg["success_first_rate"] - exp_succ / 60) < 1e-9
    assert agg["n_success_sustained_10"] == exp_succ
    assert merged["validity"]["valid"] is True
    assert len(merged["merged_from"]) == 3
    assert merged["seed_provenance"]["seed_pool"] == POOL
    # union complete + sorted by seed
    assert [r["seed"] for r in merged["results"]] == sorted(pool)
    assert Path(tmp_path / "merged.json").exists()


def test_missing_seed_hard_error(tmp_path):
    pool = get_pool(POOL)
    shard_lists = [pool[0::3][:-1], pool[1::3], pool[2::3]]  # drop one seed
    paths = _write_shards(tmp_path, shard_lists, {})
    with pytest.raises(m.MergeError, match="union does not match"):
        m.merge_shards([str(p) for p in paths], POOL)


def test_duplicate_seed_hard_error(tmp_path):
    pool = get_pool(POOL)
    s0 = pool[0::3]
    shard_lists = [s0, pool[1::3], pool[2::3] + [s0[0]]]  # s0[0] duplicated
    paths = _write_shards(tmp_path, shard_lists, {})
    with pytest.raises(m.MergeError, match="duplicate"):
        m.merge_shards([str(p) for p in paths], POOL)


def test_args_mismatch_hard_error(tmp_path):
    pool = get_pool(POOL)
    paths = _write_shards(tmp_path, _three_way(pool), {},
                          per_shard_args={1: {"max_env_steps": 999}})
    with pytest.raises(m.MergeError, match="args disagree"):
        m.merge_shards([str(p) for p in paths], POOL)


def test_server_metadata_mismatch_hard_error(tmp_path):
    pool = get_pool(POOL)
    shard_lists = _three_way(pool)
    paths = []
    for i, seeds in enumerate(shard_lists):
        d = tmp_path / f"eval_pi05_decent_shard{i}of3"
        d.mkdir(parents=True)
        sm = {"arm0": {"config_name": "c0" if i == 0 else "WRONG_CKPT"}}
        js = _shard_dict(seeds, {}, args_over={"ports": f"{i},{i},{i}", "out_dir": str(d),
                                               "video_dir": str(d / "v"), "run_id": f"J{i}"},
                         server_metadata=sm)
        p = d / f"eval_decent_T_{1000 + i}.json"
        p.write_text(json.dumps(js))
        paths.append(p)
    with pytest.raises(m.MergeError, match="checkpoint identity"):
        m.merge_shards([str(p) for p in paths], POOL)


def test_requeue_multiple_json_newest_wins(tmp_path):
    pool = get_pool(POOL)
    shard_lists = _three_way(pool)
    paths = _write_shards(tmp_path, shard_lists, {s: False for s in pool})
    # Simulate an orion requeue rerun in shard0's dir: a NEWER json (ts 9999) where all
    # of shard0's seeds succeeded. The stale ts-1000 json (all-fail) must be dropped.
    d0 = paths[0].parent
    s0 = shard_lists[0]
    newer = _shard_dict(
        s0, {s: True for s in s0},
        args_over={"ports": "0,0,0", "out_dir": str(d0),
                   "video_dir": str(d0 / "v"), "run_id": "J0"},
    )
    (d0 / "eval_decent_T_9999.json").write_text(json.dumps(newer))

    inputs = [str(d0 / "*.json")] + [str(p) for p in paths[1:]]
    with pytest.warns(UserWarning, match="requeue"):
        merged = m.merge_shards(inputs, POOL)

    # newest (all-success) used for shard0; shards 1&2 all-fail -> n_success == len(s0).
    agg = merged["merged_shards"]["aggregate"]
    assert agg["n_episodes"] == 60
    assert agg["n_success"] == len(s0)
    succ_seeds = {r["seed"] for r in merged["results"] if r["success"]}
    assert succ_seeds == set(s0)


def test_missing_input_glob_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.merge_shards([str(tmp_path / "nope_*.json")], POOL)
