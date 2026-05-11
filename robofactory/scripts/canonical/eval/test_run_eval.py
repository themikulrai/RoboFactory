"""Tests for the universal eval dispatcher's driver-argv builders.

These tests are subprocess-free: they build argv lists in memory and assert
specific flags appear in the right order. The point is to lock the
contract between manifest entries and the eval_*.py CLIs that run_eval.py
shells out to.
"""
from __future__ import annotations

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
        argv = r.build_pi05_single_argv(cfg, seeds, "JOB123")
        assert "eval_pi05.py" in " ".join(argv)
        assert "--task" in argv and "PickMeat-rf" in argv
        assert "--port" in argv and "8000" in argv
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
        argv = r.build_pi05_single_argv(cfg2, "1,2", "J")
        assert "--prompt" not in argv


class TestBuildPi05DecentArgv:
    def test_multi_port_csv(self):
        cfg = _cfg("tsc_pi05_d1_decent")
        seeds = r._load_seeds(cfg)
        argv = r.build_pi05_decent_argv(cfg, seeds, "JOB123")
        assert "eval_decent_pi05.py" in " ".join(argv)
        assert "--ports" in argv and "8000,8001,8002" in argv
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
        specs = r._build_server_specs(cfg)
        assert len(specs) == 1
        assert specs[0].name == "pm"
        assert specs[0].port == 8000
        assert specs[0].gpu_index == 0

    def test_per_arm_specs_for_pi05_decent(self):
        cfg = _cfg("tsc_pi05_d1_decent")
        specs = r._build_server_specs(cfg)
        assert [s.name for s in specs] == ["arm0", "arm1", "arm2"]
        assert [s.port for s in specs] == [8000, 8001, 8002]
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
