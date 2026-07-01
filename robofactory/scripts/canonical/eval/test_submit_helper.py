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


class TestDependencySubmit:
    """Train->eval chaining: --dependency must reach sbatch verbatim, with
    --kill-on-invalid-dep=yes, and must compose with --shards."""

    def test_dependency_flag_and_kill_on_invalid_dep(self):
        cfg = _cfg("pm_dp_in1k")
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, [], dependency="afterok:12345")
        assert cmd[cmd.index("--dependency") + 1] == "afterok:12345"
        assert "--kill-on-invalid-dep=yes" in cmd
        # Both are sbatch flags: they must precede the run_eval.sh positional.
        script_idx = next(i for i, a in enumerate(cmd) if a.endswith("run_eval.sh"))
        assert cmd.index("--dependency") < script_idx
        assert cmd.index("--kill-on-invalid-dep=yes") < script_idx

    def test_multi_jobid_spec_passthrough(self):
        # Decent train pair: eval waits on BOTH arm jobs.
        cfg = _cfg("pm_dp_in1k")
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, [], dependency="afterok:111:222")
        assert cmd[cmd.index("--dependency") + 1] == "afterok:111:222"

    def test_dependency_composes_with_shards(self):
        # The dependency attaches to the --array job as a whole, so every array
        # task inherits it; both flags must coexist on one sbatch CLI.
        cfg = _cfg("pm_dp_in1k")
        cmd = sh._build_sbatch_cmd(
            "pm_dp_in1k", cfg, [], shards=4, dependency="afterok:12345"
        )
        assert cmd[cmd.index("--array") + 1] == "0-3"
        assert cmd[cmd.index("--dependency") + 1] == "afterok:12345"
        assert "--kill-on-invalid-dep=yes" in cmd
        assert cmd[cmd.index("--num-shards") + 1] == "4"

    def test_no_dependency_is_the_default_and_adds_nothing(self):
        cfg = _cfg("pm_dp_in1k")
        cmd = sh._build_sbatch_cmd("pm_dp_in1k", cfg, [])
        assert "--dependency" not in cmd
        assert "--kill-on-invalid-dep=yes" not in cmd

    def test_cli_dependency_flag_reaches_builder(self, capsys):
        # End-to-end through main() with --print: no sbatch is invoked; the printed
        # CLI must carry the dependency + shards composition.
        rc = sh.main(["pm_dp_in1k", "--print", "--shards", "4",
                      "--dependency", "afterok:777:888"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "--dependency afterok:777:888" in out
        assert "--kill-on-invalid-dep=yes" in out
        assert "--array 0-3" in out
