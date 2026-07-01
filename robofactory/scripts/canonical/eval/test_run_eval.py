"""Tests for the universal eval dispatcher's driver-argv builders.

These tests are subprocess-free: they build argv lists in memory and assert
specific flags appear in the right order. The point is to lock the
contract between manifest entries and the eval_*.py CLIs that run_eval.py
shells out to.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from robofactory.scripts.canonical.eval import run_eval as r
from robofactory.scripts.canonical.eval.manifest_schema import (
    LauncherCfg,
    PolicyType,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "eval" / "manifest.yaml"


def _cfg(launcher_id: str) -> LauncherCfg:
    return load_manifest(REAL_MANIFEST).launchers[launcher_id]


# ---------------------------------------------------------------------------
# Helpers: _load_seeds, _resolve_train_cfg
# ---------------------------------------------------------------------------
class TestLoadSeeds:
    def test_comma_delim_for_pi05(self, tmp_path):
        cfg = _cfg("pm_pi05_paired_dp_seeds")
        seeds = r._load_seeds(cfg)
        # 60 seeds, comma-joined.
        assert "," in seeds
        assert " " not in seeds
        assert len(seeds.split(",")) == 60

    def test_space_delim_for_dp(self):
        cfg = _cfg("pm_dp_in1k")
        seeds = r._load_seeds(cfg)
        assert "," not in seeds
        assert len(seeds.split()) == 60


class TestResolveTrainCfg:
    def test_returns_none_when_hydra_config_missing(self, tmp_path):
        # Synthesize a cfg pointing at a directory without .hydra_config.yaml.
        cfg = _cfg("pm_dp_in1k")  # ckpt.path under a real dir
        out = r._resolve_train_cfg(cfg)
        # The .ckpt file's parent dir has no .hydra_config.yaml -> None
        # (unless the user has one cached — accept None or an existing file)
        if out is not None:
            assert out.exists()

    def test_returns_path_when_hydra_config_present(self, tmp_path):
        # Build a fake cfg whose ckpt parent contains .hydra_config.yaml.
        hydra_path = tmp_path / ".hydra_config.yaml"
        hydra_path.write_text("task: {env_runner: {env_id: X}}\n")
        cfg = _cfg("pm_dp_in1k").model_copy(update={
            "ckpt": _cfg("pm_dp_in1k").ckpt.model_copy(
                update={"path": str(tmp_path / "ckpt.ckpt")}
            )
        })
        out = r._resolve_train_cfg(cfg)
        assert out == hydra_path


# ---------------------------------------------------------------------------
# Per-policy_type driver builders
# ---------------------------------------------------------------------------
class TestBuildPi05SingleArgv:
    def test_required_flags_present(self):
        cfg = _cfg("pm_pi05_paired_dp_seeds")
        seeds = r._load_seeds(cfg)
        # PR1: builders take runtime-allocated real ports, NOT manifest indices.
        argv = r.build_pi05_single_argv(cfg, seeds, "JOB123", [54321])
        assert "eval_pi05.py" in " ".join(argv)
        assert "--task" in argv and "PickMeat-rf" in argv
        assert "--port" in argv and "54321" in argv  # the runtime-allocated port
        assert "8000" not in argv  # no hardcoded port leaks through
        assert "--seeds" in argv and seeds in argv
        assert "--max-env-steps" in argv and "400" in argv
        assert "--robot-uid" in argv and "panda" in argv
        assert "--prompt" in argv
        assert "pick the meat with the gripper" in argv
        assert "--wandb-project" in argv

    def test_no_prompt_when_unset(self):
        cfg = _cfg("pm_pi05_paired_dp_seeds")
        cfg2 = cfg.model_copy(update={
            "task": cfg.task.model_copy(update={"prompt": None})
        })
        argv = r.build_pi05_single_argv(cfg2, "1,2", "J", [12345])
        assert "--prompt" not in argv


class TestBuildPi05DecentArgv:
    def test_multi_port_csv(self):
        cfg = _cfg("tsc_pi05_d1_decent")
        seeds = r._load_seeds(cfg)
        # PR1: real ports are passed at runtime; manifest holds only indices.
        argv = r.build_pi05_decent_argv(cfg, seeds, "JOB123", [40001, 40002, 40003])
        assert "eval_decent_pi05.py" in " ".join(argv)
        assert "--ports" in argv and "40001,40002,40003" in argv
        assert "8000,8001,8002" not in argv  # no hardcoded ports leak through
        assert "--num-arms" in argv and "3" in argv
        assert "--robot-uids-csv" in argv
        # No --prompt for decent (joint task).
        assert "--prompt" not in argv


class TestBuildDpSingleArgv:
    def test_ckpt_path_and_max_steps(self):
        cfg = _cfg("pm_dp_in1k")
        seeds = r._load_seeds(cfg)
        argv = r.build_dp_single_argv(cfg, seeds, "JOB")
        flat = " ".join(argv)
        assert "eval_dp.py" in flat
        assert "--ckpt-path=/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt" in flat
        assert "--max-steps=200" in flat
        # -s expansion: 60 seed tokens after -s, all integers
        s_idx = argv.index("-s")
        for tok in argv[s_idx + 1: s_idx + 61]:
            int(tok)  # raises if non-int

    def test_dp_backend_defaults_to_cpu(self):
        cfg = _cfg("pm_dp_in1k")
        argv = r.build_dp_single_argv(cfg, "100", "J")
        b_idx = argv.index("-b")
        assert argv[b_idx + 1] == "cpu"


class TestBuildDpDecentMultiArgv:
    def test_freezeenc_ckpt_suffix(self):
        cfg = _cfg("tsc_dp_d1_freezeenc_decent")
        argv = r.build_dp_decent_multi_argv(cfg, "100 101", "J")
        flat = " ".join(argv)
        assert "eval_multi_dp.py" in flat
        assert "--ckpt-suffix=freezeenc" in flat
        assert "--no-include-global" in argv
        assert "--img-height=240" in argv
        assert "--img-width=320" in argv

    def test_d2_wristcam_reeval_uses_dp_backend_gpu(self):
        cfg = _cfg("tsc_dp_d2_wristcam_reeval")
        argv = r.build_dp_decent_multi_argv(cfg, "100", "J")
        b_idx = argv.index("-b")
        assert argv[b_idx + 1] == "gpu"
        # include-global flips on for d2_wristcam (224x224 head_cam_global)
        assert "--include-global" in argv
        assert "--robot-uids=panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi" in argv

    def test_robot_uids_omitted_when_unset(self):
        cfg = _cfg("tsc_dp_d1_decent_in1k")
        argv = r.build_dp_decent_multi_argv(cfg, "100", "J")
        # d1_decent_in1k doesn't set robot_uids_csv -> no --robot-uids flag
        assert not any(a.startswith("--robot-uids") for a in argv)


class TestBuildDpJointCentArgv:
    def test_relative_ckpt_path_for_workspace(self):
        cfg = _cfg("tsc_dp_d1_workspace_cent")
        argv = r.build_dp_joint_cent_argv(cfg, "100 101", "J")
        assert "eval_joint_dp.py" in " ".join(argv)
        # Relative path required by the `./` + path quirk in eval_joint_dp.py.
        ckpt_idx = argv.index("--ckpt-path")
        assert argv[ckpt_idx + 1] == "checkpoints/ThreeRobotsStackCube-rf_joint_d1_workspace_150_in1k/300.ckpt"
        # workspace launcher doesn't need --robot-uids
        assert "--robot-uids" not in argv

    def test_wristcam_cent_includes_robot_uids(self):
        cfg = _cfg("tsc_dp_d2_wristcam_cent")
        argv = r.build_dp_joint_cent_argv(cfg, "100", "J")
        assert "--robot-uids" in argv

    def test_seed_flag_singular(self):
        cfg = _cfg("tsc_dp_d1_workspace_cent")
        argv = r.build_dp_joint_cent_argv(cfg, "100 101 102", "J")
        # eval_joint_dp uses --seed (singular) nargs='+'
        assert "--seed" in argv


# ---------------------------------------------------------------------------
# extra_argv pass-through + env_hooks dry-run
# ---------------------------------------------------------------------------
class TestExtraAndHooks:
    def test_extra_argv_appended(self):
        cfg = _cfg("pm_dp_in1k").model_copy(update={
            "extra_argv": ["--debug-flag", "value"]
        })
        argv = r.build_driver_argv(cfg, "100", "J")
        assert argv[-2:] == ["--debug-flag", "value"]

    def test_env_hooks_dry_run_does_not_subprocess(self):
        cfg = _cfg("pm_dp_r3m")
        with mock.patch.object(r.subprocess, "run") as run_mock:
            r.run_env_hooks(cfg, dry_run=True)
        assert run_mock.call_count == 0

    def test_env_hooks_invoked_when_not_dry_run(self):
        cfg = _cfg("pm_dp_r3m")
        with mock.patch.object(r.subprocess, "run") as run_mock:
            r.run_env_hooks(cfg, dry_run=False)
        assert run_mock.call_count == 1
        cmd = run_mock.call_args[0][0]
        assert cmd[0] == "bash" and cmd[1] == "-lc"


# ---------------------------------------------------------------------------
# _build_server_specs
# ---------------------------------------------------------------------------
class TestBuildServerSpecs:
    def test_single_spec_for_pi05_single(self):
        cfg = _cfg("pm_pi05_paired_dp_seeds")
        # PR1: real ports allocated at runtime, passed into _build_server_specs.
        specs = r._build_server_specs(cfg, [54321])
        assert len(specs) == 1
        assert specs[0].name == "pm"
        assert specs[0].port == 54321
        assert specs[0].gpu_index == 0

    def test_per_arm_specs_for_pi05_decent(self):
        cfg = _cfg("tsc_pi05_d1_decent")
        specs = r._build_server_specs(cfg, [40001, 40002, 40003])
        assert [s.name for s in specs] == ["arm0", "arm1", "arm2"]
        assert [s.port for s in specs] == [40001, 40002, 40003]
        assert [s.gpu_index for s in specs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# run_launcher dry-run end-to-end (no subprocess invocations)
# ---------------------------------------------------------------------------
class TestDryRunFlow:
    @pytest.mark.parametrize("launcher_id", [
        "pm_pi05_paired_dp_seeds",
        "tsc_pi05_d1_decent",
        "pm_dp_in1k",
        "pm_dp_r3m",
        "tsc_dp_d1_decent_in1k",
        "tsc_dp_d1_freezeenc_decent",
        "tsc_dp_d2_wristcam_reeval",
        "tsc_dp_d1_workspace_cent",
        "tsc_dp_d2_wristcam_cent",
    ])
    def test_dry_run_does_not_invoke_subprocess(self, launcher_id):
        cfg = _cfg(launcher_id)
        with mock.patch.object(r.subprocess, "run") as run_mock:
            rc = r.run_launcher(cfg, launcher_id, slurm_job_id="DRY", dry_run=True)
        assert rc == 0
        assert run_mock.call_count == 0

    def test_preflight_only_runs_preflight_then_stops(self):
        """--preflight-only should subprocess.run once (the preflight) and stop."""
        cfg = _cfg("pm_dp_in1k")
        with mock.patch.object(r.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            rc = r.run_launcher(
                cfg, "pm_dp_in1k",
                slurm_job_id="LOCAL", dry_run=False, preflight_only=True,
            )
        assert rc == 0
        # Exactly one subprocess call: the preflight. No driver, no server.
        assert run_mock.call_count == 1
        called_cmd = run_mock.call_args[0][0]
        assert "preflight_eval_guards" in " ".join(called_cmd)


# ---------------------------------------------------------------------------
# PR5: N_REPEATS resolution + repeat loop
# ---------------------------------------------------------------------------
class TestResolveNRepeats:
    def setup_method(self):
        # Make sure no ambient N_REPEATS env leaks into the precedence tests.
        os.environ.pop("N_REPEATS", None)

    def teardown_method(self):
        os.environ.pop("N_REPEATS", None)

    def test_canonical_default_is_3(self):
        cfg = _cfg("pm_dp_in1k")
        assert r.resolve_n_repeats(cfg, "pm_dp_in1k") == 3

    def test_ablation_prefix_default_is_1(self):
        cfg = _cfg("pm_dp_in1k")
        assert r.resolve_n_repeats(cfg, "ablation_dino_blora") == 1
        assert r.resolve_n_repeats(cfg, "abl_in1k_crop") == 1

    def test_manifest_field_overrides_default(self):
        cfg = _cfg("pm_dp_in1k").model_copy(update={"n_repeats": 5})
        assert r.resolve_n_repeats(cfg, "pm_dp_in1k") == 5

    def test_env_var_overrides_everything(self):
        os.environ["N_REPEATS"] = "7"
        cfg = _cfg("pm_dp_in1k").model_copy(update={"n_repeats": 5})
        # env (7) beats cfg.n_repeats (5) beats default (3).
        assert r.resolve_n_repeats(cfg, "pm_dp_in1k") == 7

    def test_zero_or_negative_rejected(self):
        os.environ["N_REPEATS"] = "0"
        cfg = _cfg("pm_dp_in1k")
        with pytest.raises(ValueError):
            r.resolve_n_repeats(cfg, "pm_dp_in1k")


class TestRepeatLoop:
    def setup_method(self):
        os.environ.pop("N_REPEATS", None)

    def teardown_method(self):
        os.environ.pop("N_REPEATS", None)

    def test_dp_runs_driver_once_per_rep(self):
        """N_REPEATS=3 => preflight (1) + driver (3) = 4 subprocess.run calls; each
        driver argv carries a distinct _rep{k} jsonl path."""
        os.environ["N_REPEATS"] = "3"
        cfg = _cfg("pm_dp_in1k")
        with mock.patch.object(r.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            rc = r.run_launcher(cfg, "pm_dp_in1k", slurm_job_id="JOB", dry_run=False)
        assert rc == 0
        assert run_mock.call_count == 4  # 1 preflight + 3 driver reps
        driver_cmds = [
            c.args[0] for c in run_mock.call_args_list
            if "eval_dp.py" in " ".join(c.args[0])
        ]
        assert len(driver_cmds) == 3
        rep_paths = []
        for cmd in driver_cmds:
            jp = [a for a in cmd if a.startswith("--jsonl-path=")]
            assert len(jp) == 1, f"expected one --jsonl-path, got {jp}"
            rep_paths.append(jp[0])
        # All three rep paths are distinct (rep0/rep1/rep2).
        assert len(set(rep_paths)) == 3
        assert any("rep0" in p for p in rep_paths)
        assert any("rep2" in p for p in rep_paths)

    def test_single_rep_keeps_legacy_naming(self):
        """N_REPEATS=1 => no _rep suffix injected (legacy timestamp filename preserved)."""
        os.environ["N_REPEATS"] = "1"
        cfg = _cfg("pm_dp_in1k")
        with mock.patch.object(r.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            rc = r.run_launcher(cfg, "pm_dp_in1k", slurm_job_id="JOB", dry_run=False)
        assert rc == 0
        assert run_mock.call_count == 2  # 1 preflight + 1 driver
        driver_cmds = [
            c.args[0] for c in run_mock.call_args_list
            if "eval_dp.py" in " ".join(c.args[0])
        ]
        assert len(driver_cmds) == 1
        # No injected per-rep jsonl path in single-rep mode.
        assert not any(a.startswith("--jsonl-path=") for a in driver_cmds[0])

    def test_nonzero_rc_surfaces_but_all_reps_run(self):
        """A failing rep sets the final rc nonzero but doesn't abort remaining reps."""
        os.environ["N_REPEATS"] = "3"
        cfg = _cfg("pm_dp_in1k")
        calls = {"n": 0}

        def fake_run(cmd, *a, **k):
            m = mock.Mock()
            # preflight (call 0) ok; one driver rep fails.
            if "eval_dp.py" in " ".join(cmd):
                calls["n"] += 1
                m.returncode = 1 if calls["n"] == 2 else 0
            else:
                m.returncode = 0
            return m

        with mock.patch.object(r.subprocess, "run", side_effect=fake_run):
            rc = r.run_launcher(cfg, "pm_dp_in1k", slurm_job_id="JOB", dry_run=False)
        assert rc == 1            # nonzero surfaced
        assert calls["n"] == 3    # all three driver reps still ran


# ---------------------------------------------------------------------------
# Seed sharding: shard_seeds() round-robin partition + _inject_shard_output()
# ---------------------------------------------------------------------------
class TestShardSeeds:
    def test_four_shard_disjoint_partition_preserves_order(self):
        pool = ",".join(str(s) for s in range(60))
        shards = [r.shard_seeds(pool, i, 4, ",") for i in range(4)]
        parsed = [s.split(",") for s in shards]
        # each shard is exactly pool[i::4] (order preserved within shard)
        for i in range(4):
            assert parsed[i] == [str(s) for s in range(60)][i::4]
        # union is the whole pool with no duplicates and no gaps
        flat = [tok for sh in parsed for tok in sh]
        assert len(flat) == 60
        assert len(set(flat)) == 60
        assert sorted(int(x) for x in flat) == list(range(60))

    def test_num_shards_one_is_identity(self):
        pool = "1,2,3,4,5"
        assert r.shard_seeds(pool, 0, 1, ",") == pool

    def test_space_delim_roundtrip(self):
        # split on whitespace, rejoin with the launcher's space delim
        assert r.shard_seeds("10 11 12 13", 1, 2, " ") == "11 13"

    def test_comma_input_space_delim_output(self):
        # input delim and output delim can differ (split handles both)
        assert r.shard_seeds("10,11,12,13", 0, 2, " ") == "10 12"

    def test_invalid_shard_raises(self):
        # Explicit SystemExit, NOT a bare `assert`: a `raise` statement is never
        # stripped by `python -O` (only `assert`/`__debug__` blocks are), so an
        # out-of-bounds shard fails loudly under optimization instead of silently
        # returning an EMPTY slice. Asserting SystemExit here (not AssertionError,
        # which an `assert` would raise in this non-`-O` test process) proves the
        # check is the explicit raise, not a strippable assert.
        with pytest.raises(SystemExit):
            r.shard_seeds("1,2,3", 0, 5, ",")   # num_shards > n_seeds
        with pytest.raises(SystemExit):
            r.shard_seeds("1,2,3", 3, 3, ",")   # shard_index == num_shards
        with pytest.raises(SystemExit):
            r.shard_seeds("1,2,3", -1, 3, ",")  # negative shard_index

    def test_valid_edge_num_shards_equals_n_seeds(self):
        # Boundary: num_shards == n_seeds is valid (each shard gets exactly 1 seed).
        assert r.shard_seeds("1,2,3", 2, 3, ",") == "3"
        assert r.shard_seeds("1,2,3", 0, 3, ",") == "1"


class TestInjectShardOutput:
    def test_out_and_video_dirs_suffixed_and_created(self, tmp_path):
        argv = [
            "python", "x.py",
            "--out-dir", str(tmp_path / "out"),
            "--video-dir", str(tmp_path / "out" / "videos_JOB"),
        ]
        out = r._inject_shard_output(argv, 2, 4)
        oi = out.index("--out-dir")
        vi = out.index("--video-dir")
        assert out[oi + 1].endswith("_shard2of4")
        assert out[vi + 1].endswith("_shard2of4")
        assert Path(out[oi + 1]).is_dir()  # out-dir is created

    def test_jsonl_path_stem_suffixed(self):
        argv = ["python", "y.py", "--jsonl-path=/logs/eval/PickMeat-rf_rep0.jsonl"]
        out = r._inject_shard_output(argv, 1, 3)
        assert out[-1] == "--jsonl-path=/logs/eval/PickMeat-rf_rep0_shard1of3.jsonl"

    def test_num_shards_one_is_noop(self):
        argv = ["--out-dir", "/a/b", "--video-dir", "/a/b/videos"]
        assert r._inject_shard_output(argv, 0, 1) == argv


class TestShardedDispatch:
    """run_launcher end-to-end (mocked subprocess) proving the driver receives the
    sharded seed slice and a shard-suffixed jsonl path, and that num_shards=1 is a
    byte-identical regression of the pre-sharding argv."""

    def setup_method(self):
        os.environ["N_REPEATS"] = "1"  # isolate from the 3-rep canonical default

    def teardown_method(self):
        os.environ.pop("N_REPEATS", None)

    def _driver_seeds(self, argv):
        s_idx = argv.index("-s")
        seeds = []
        for tok in argv[s_idx + 1:]:
            if tok.startswith("-"):
                break
            seeds.append(tok)
        return seeds

    def _run_capture(self, launcher_id, num_shards, shard_index):
        cfg = _cfg(launcher_id)
        with mock.patch.object(r.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            r.run_launcher(
                cfg, launcher_id, slurm_job_id="JOB",
                num_shards=num_shards, shard_index=shard_index,
            )
        return [
            c.args[0] for c in run_mock.call_args_list
            if "eval_dp.py" in " ".join(c.args[0])
        ][0]

    def test_four_shards_partition_dp_seeds(self):
        expected = r._load_seeds(_cfg("pm_dp_in1k")).split()
        collected = []
        for k in range(4):
            argv = self._run_capture("pm_dp_in1k", 4, k)
            shard = self._driver_seeds(argv)
            assert shard == expected[k::4]                 # order-preserving stride
            collected += shard
        assert sorted(collected, key=int) == sorted(expected, key=int)
        assert len(collected) == len(expected)             # disjoint partition

    def test_num_shards_one_byte_identical(self):
        argv = self._run_capture("pm_dp_in1k", 1, 0)
        # full seed pool, no shard suffix anywhere.
        assert self._driver_seeds(argv) == r._load_seeds(_cfg("pm_dp_in1k")).split()
        assert not any("_shard" in a for a in argv)

    def _run_capture_pi05(self, launcher_id, num_shards, shard_index):
        """Capture the decent-pi0.5 driver argv without spawning real servers."""
        cfg = _cfg(launcher_id)
        with mock.patch.object(r, "Pi05ServerSupervisor"), \
                mock.patch.object(r, "wait_for_ports"), \
                mock.patch.object(r, "run_registry_drift_guard"), \
                mock.patch.object(r.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            r.run_launcher(
                cfg, launcher_id, slurm_job_id="JOB",
                num_shards=num_shards, shard_index=shard_index,
            )
        return [
            c.args[0] for c in run_mock.call_args_list
            if "eval_decent_pi05.py" in " ".join(c.args[0])
        ][0]

    def test_pi05_out_and_video_dirs_shard_suffixed(self):
        full = r._load_seeds(_cfg("tsc_pi05_d1_decent"))          # comma-joined pool
        expected_n = len(r.shard_seeds(full, 1, 4, ",").split(","))
        argv = self._run_capture_pi05("tsc_pi05_d1_decent", 4, 1)
        assert argv[argv.index("--out-dir") + 1].endswith("_shard1of4")
        assert argv[argv.index("--video-dir") + 1].endswith("_shard1of4")
        # driver receives exactly this shard's slice of the 60-seed pool (15 seeds).
        got = argv[argv.index("--seeds") + 1].split(",")
        assert got == full.split(",")[1::4]
        assert len(got) == expected_n
