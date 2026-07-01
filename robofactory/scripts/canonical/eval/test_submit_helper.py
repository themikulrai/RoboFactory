"""Tests for the sbatch-CLI builder in submit_helper.

Subprocess-free: they build the `sbatch` argv in memory and assert the sharded
submit adds `--array`, rewrites `%j` -> `%A_%a` in the log paths, and forwards
`--num-shards` to run_eval.py. The single-shard path must stay untouched.
"""
from __future__ import annotations

from pathlib import Path

from robofactory.scripts.canonical.eval import submit_helper as sh
from robofactory.scripts.canonical.eval.manifest_schema import (
    LauncherCfg,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "eval" / "manifest.yaml"


def _cfg(launcher_id: str) -> LauncherCfg:
    return load_manifest(REAL_MANIFEST).launchers[launcher_id]


class TestShardedSubmit:
    def test_array_and_num_shards_added_for_shards_4(self):
        cfg = _cfg("pm_dp_in1k")
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, [], shards=4)
        assert "--array" in cmd
        assert cmd[cmd.index("--array") + 1] == "0-3"
        assert "--num-shards" in cmd
        assert cmd[cmd.index("--num-shards") + 1] == "4"
        # --num-shards must sit AFTER the launcher_id positional (run_eval.py argparse).
        assert cmd.index("--num-shards") > cmd.index("pm_dp_in1k")

    def test_output_error_j_rewritten_to_A_a(self):
        base = _cfg("pm_dp_in1k")
        cfg = base.model_copy(update={
            "sbatch": base.sbatch.model_copy(update={
                "output": "/iris/u/mikulrai/logs/eval/%x_%j.out",
                "error": "/iris/u/mikulrai/logs/eval/%x_%j.err",
            })
        })
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, [], shards=4)
        assert cmd[cmd.index("--output") + 1] == "/iris/u/mikulrai/logs/eval/%x_%A_%a.out"
        assert cmd[cmd.index("--error") + 1] == "/iris/u/mikulrai/logs/eval/%x_%A_%a.err"

    def test_single_shard_unchanged(self):
        base = _cfg("pm_dp_in1k")
        cfg = base.model_copy(update={
            "sbatch": base.sbatch.model_copy(update={
                "output": "/iris/u/mikulrai/logs/eval/%x_%j.out",
            })
        })
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, [], shards=1)
        assert "--array" not in cmd
        assert "--num-shards" not in cmd
        # %j is left intact for a non-array (single-job) submit.
        assert cmd[cmd.index("--output") + 1] == "/iris/u/mikulrai/logs/eval/%x_%j.out"

    def test_extra_argv_still_forwarded_with_shards(self):
        cfg = _cfg("pm_dp_in1k")
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, ["--dry-run"], shards=4)
        assert cmd[-1] == "--dry-run"
        # --num-shards precedes the forwarded extra argv.
        assert cmd.index("--num-shards") < cmd.index("--dry-run")
