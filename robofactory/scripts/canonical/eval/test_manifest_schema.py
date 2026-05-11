"""Tests for the eval-launcher manifest schema."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robofactory.scripts.canonical.eval.manifest_schema import (
    CkptCfg,
    LauncherCfg,
    Manifest,
    PolicyType,
    SbatchCfg,
    SeedsCfg,
    ServerCfg,
    TaskCfg,
    WandbCfg,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "eval" / "manifest.yaml"


def _minimal_pi05_single(**overrides) -> dict:
    d = {
        "policy_type": "pi05_single",
        "ckpt": {"dir": "/x", "train_config": "cfg"},
        "task": {
            "env_id": "PickMeat-rf",
            "scene_config": "configs/table/pick_meat.yaml",
            "camera_family": "head_camera",
        },
        "seeds": {
            "file": "/iris/u/mikulrai/runs/eval_seeds_60_dp.txt",
            "sha256_file": "/iris/u/mikulrai/runs/eval_seeds_60_dp.sha256",
            "delim": ",",
        },
        "server": {
            "ports": [8000],
            "gpu_indices": [0],
            "xla_mem_fraction": 0.4,
        },
        "wandb": {"tags": ["eval"]},
        "sbatch": {},
    }
    d.update(overrides)
    return d


def _minimal_dp_single(**overrides) -> dict:
    d = {
        "policy_type": "dp_single",
        "ckpt": {"path": "/x/300.ckpt"},
        "task": {
            "env_id": "PickMeat-rf",
            "scene_config": "configs/table/pick_meat.yaml",
            "camera_family": "in1k",
        },
        "seeds": {
            "file": "/iris/u/mikulrai/runs/eval_seeds_60_dp.txt",
            "sha256_file": "/iris/u/mikulrai/runs/eval_seeds_60_dp.sha256",
            "delim": " ",
        },
        "wandb": {"tags": ["eval"]},
        "sbatch": {},
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Real manifest must always parse — guards against accidental YAML breakage.
# ---------------------------------------------------------------------------
class TestRealManifest:
    def test_real_manifest_validates(self):
        mfst = load_manifest(REAL_MANIFEST)
        assert len(mfst.launchers) >= 12

    def test_real_manifest_has_all_5_policy_types(self):
        mfst = load_manifest(REAL_MANIFEST)
        seen = {cfg.policy_type for cfg in mfst.launchers.values()}
        assert seen == {
            PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT,
            PolicyType.DP_SINGLE, PolicyType.DP_DECENT_MULTI,
            PolicyType.DP_JOINT_CENT,
        }


# ---------------------------------------------------------------------------
# Strict-extra-forbid guarantee — the YAML-typo-protection that makes B work.
# ---------------------------------------------------------------------------
class TestExtraForbid:
    def test_unknown_root_key_fails(self):
        bad = {"launchers": {}, "unknown_key": 42}
        with pytest.raises(ValidationError):
            Manifest(**bad)

    def test_unknown_launcher_key_fails(self):
        bad = {"launchers": {"x": {**_minimal_pi05_single(), "unknown": 1}}}
        with pytest.raises(ValidationError):
            Manifest(**bad)

    def test_unknown_nested_key_fails(self):
        d = _minimal_pi05_single()
        d["wandb"]["tagss"] = ["typo"]  # double-s typo, exact case from advocate
        with pytest.raises(ValidationError):
            LauncherCfg(**d)


# ---------------------------------------------------------------------------
# Policy-type cross-validation.
# ---------------------------------------------------------------------------
class TestPolicyCrossValidation:
    def test_pi05_single_requires_server(self):
        d = _minimal_pi05_single()
        del d["server"]
        with pytest.raises(ValidationError) as exc:
            LauncherCfg(**d)
        assert "server config" in str(exc.value)

    def test_dp_single_must_not_have_server(self):
        d = _minimal_dp_single()
        d["server"] = {"ports": [1], "gpu_indices": [0], "xla_mem_fraction": 0.4}
        with pytest.raises(ValidationError):
            LauncherCfg(**d)

    def test_pi05_decent_requires_arm_configs(self):
        d = _minimal_pi05_single(policy_type="pi05_decent")
        # server has no arm_configs/arm_dirs
        with pytest.raises(ValidationError) as exc:
            LauncherCfg(**d)
        assert "arm_configs" in str(exc.value)

    def test_pi05_decent_with_arm_configs_validates(self):
        d = _minimal_pi05_single(policy_type="pi05_decent")
        d["server"] = {
            "ports": [8000, 8001, 8002],
            "gpu_indices": [0, 1, 2],
            "xla_mem_fraction": 0.3,
            "arm_configs": ["a", "b", "c"],
            "arm_dirs": ["/d/a", "/d/b", "/d/c"],
        }
        cfg = LauncherCfg(**d)
        assert cfg.policy_type == PolicyType.PI05_DECENT

    def test_server_ports_gpus_length_mismatch_fails(self):
        with pytest.raises(ValidationError):
            ServerCfg(ports=[8000, 8001], gpu_indices=[0], xla_mem_fraction=0.4)

    def test_dp_joint_cent_requires_ckpt_path(self):
        d = _minimal_dp_single(policy_type="dp_joint_cent")
        d["ckpt"] = {}  # neither path nor path_relative
        with pytest.raises(ValidationError) as exc:
            LauncherCfg(**d)
        assert "path" in str(exc.value)

    def test_pi05_requires_ckpt_dir_and_train_config(self):
        d = _minimal_pi05_single()
        d["ckpt"] = {}
        with pytest.raises(ValidationError) as exc:
            LauncherCfg(**d)
        assert "ckpt.dir" in str(exc.value) or "train_config" in str(exc.value)


# ---------------------------------------------------------------------------
# Defaults and pass-through fields.
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_sbatch_defaults(self):
        sb = SbatchCfg()
        assert sb.time == "4:00:00"
        assert sb.gres == "gpu:1"
        assert sb.mail_type == "END,FAIL"

    def test_dp_backend_defaults_to_cpu(self):
        t = TaskCfg(env_id="x", scene_config="c", camera_family="head_camera")
        assert t.dp_backend == "cpu"

    def test_dp_backend_accepts_gpu(self):
        t = TaskCfg(
            env_id="x", scene_config="c", camera_family="head_camera",
            dp_backend="gpu",
        )
        assert t.dp_backend == "gpu"

    def test_dp_backend_rejects_invalid(self):
        with pytest.raises(ValidationError):
            TaskCfg(
                env_id="x", scene_config="c", camera_family="head_camera",
                dp_backend="tpu",
            )

    def test_seeds_delim_must_be_space_or_comma(self):
        with pytest.raises(ValidationError):
            SeedsCfg(file="/x", sha256_file="/y", delim=";")


# ---------------------------------------------------------------------------
# YAML loading edge cases.
# ---------------------------------------------------------------------------
class TestLoading:
    def test_load_manifest_rejects_non_mapping_root(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.yaml"
            p.write_text("- 1\n- 2\n")
            with pytest.raises(TypeError):
                load_manifest(p)

    def test_load_manifest_rejects_unknown_root_field(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.yaml"
            p.write_text(yaml.safe_dump({"launchers": {}, "extra": 1}))
            with pytest.raises(ValidationError):
                load_manifest(p)
