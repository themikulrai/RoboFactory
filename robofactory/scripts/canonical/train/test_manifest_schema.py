"""Tests for the DP-train-launcher manifest schema."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from robofactory.scripts.canonical.train.manifest_schema import (
    AgentMode,
    EncoderCfg,
    EncoderFamily,
    Manifest,
    ResumeCfg,
    SbatchCfg,
    TaskCfg,
    TrainLauncherCfg,
    TrainMode,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "train" / "manifest.yaml"


def _minimal_robot(**overrides) -> dict:
    d = {
        "agent_mode": "robot",
        "arm_id": 0,
        "task": {
            "task_config": "default_task",
            "env_name": "PickMeat-rf",
            "num_arms": 1,
            "cam_family": "workspace",
        },
        "encoder": {"family": "resnet18_in1k"},
        "training": {"num_epochs": 300, "seed": 100, "batch_size": 64},
        "zarr_path": "/d/PickMeat-rf_150.zarr",
        "preflight": {"cheap": {"enabled": False, "scene_config": "configs/table/pick_meat.yaml"}},
        "wandb": {"project": "diffusion-robofactory", "tags": ["robot_dp"]},
    }
    d.update(overrides)
    return d


def _minimal_joint(**overrides) -> dict:
    d = {
        "agent_mode": "joint",
        "task": {
            "task_config": "joint_task_3arm",
            "env_name": "ThreeRobotsStackCube-rf",
            "num_arms": 3,
            "cam_family": "workspace",
        },
        "encoder": {"family": "resnet18_in1k"},
        "training": {"num_epochs": 300},
        "zarr_path": "/d/ThreeRobotsStackCube-rf_workspace_cent_150.zarr",
        "preflight": {"cheap": {"enabled": False, "scene_config": "configs/table/three_robots_stack_cube.yaml"}},
        "wandb": {"project": "diffusion-robofactory", "tags": ["joint"]},
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

    def test_real_manifest_has_resume_launchers(self):
        mfst = load_manifest(REAL_MANIFEST)
        resumes = [l for l in mfst.launchers if l.startswith("resume_")]
        assert len(resumes) == 3
        for lid in resumes:
            assert mfst.launchers[lid].resume.from_ckpt is not None


# ---------------------------------------------------------------------------
# Strict-extra-forbid guarantee.
# ---------------------------------------------------------------------------
class TestExtraForbid:
    def test_unknown_root_key_fails(self):
        with pytest.raises(ValidationError):
            Manifest(**{"launchers": {}, "unknown_key": 42})

    def test_unknown_launcher_key_fails(self):
        bad = {"launchers": {"x": {**_minimal_robot(), "unknown": 1}}}
        with pytest.raises(ValidationError):
            Manifest(**bad)

    def test_unknown_nested_key_fails(self):
        d = _minimal_robot()
        d["wandb"]["tagss"] = ["typo"]
        with pytest.raises(ValidationError):
            TrainLauncherCfg(**d)

    def test_unknown_encoder_key_fails(self):
        d = _minimal_robot()
        d["encoder"]["frozen"] = True  # typo for `freeze`
        with pytest.raises(ValidationError):
            TrainLauncherCfg(**d)


# ---------------------------------------------------------------------------
# Encoder cross-field validators.
# ---------------------------------------------------------------------------
class TestEncoderValidators:
    def test_dinov2_requires_resize_224(self):
        d = _minimal_robot()
        d["encoder"] = {"family": "dinov2_lora", "use_group_norm": False}  # missing resize
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "resize_shape" in str(exc.value)

    def test_dinov2_requires_group_norm_false(self):
        d = _minimal_robot()
        d["encoder"] = {"family": "dinov2_lora", "resize_shape": [224, 224]}  # group_norm None
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "use_group_norm" in str(exc.value)

    def test_dinov2_valid(self):
        d = _minimal_robot()
        d["encoder"] = {"family": "dinov2_patchattn", "resize_shape": [224, 224], "use_group_norm": False}
        cfg = TrainLauncherCfg(**d)
        assert cfg.encoder.is_dinov2()

    def test_crop_within_workspace_image_ok(self):
        d = _minimal_robot()
        d["encoder"] = {"family": "resnet18_in1k", "crop_shape": [216, 288]}
        cfg = TrainLauncherCfg(**d)  # 216<=240, 288<=320
        assert cfg.encoder.crop_shape == [216, 288]

    def test_crop_exceeds_workspace_image_fails(self):
        d = _minimal_robot()
        d["encoder"] = {"family": "resnet18_in1k", "crop_shape": [300, 400]}
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "exceeds task image dims" in str(exc.value)

    def test_crop_exceeds_wristcam_image_fails(self):
        d = _minimal_robot()
        d["task"] = {"task_config": "default_task_wristcam", "env_name": "x", "num_arms": 1, "cam_family": "wristcam"}
        d["zarr_path"] = "/d/x_150.zarr"
        d["encoder"] = {"family": "resnet18_in1k", "crop_shape": [240, 240]}  # >224
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "exceeds task image dims" in str(exc.value)

    def test_crop_3dim_form_ok(self):
        d = _minimal_robot()
        d["encoder"] = {"family": "resnet18_in1k", "crop_shape": [3, 216, 288]}
        cfg = TrainLauncherCfg(**d)
        assert cfg.encoder.crop_shape == [3, 216, 288]


# ---------------------------------------------------------------------------
# agent_mode cross-field validators.
# ---------------------------------------------------------------------------
class TestAgentModeValidators:
    def test_robot_requires_arm_id(self):
        d = _minimal_robot()
        del d["arm_id"]
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "arm_id" in str(exc.value)

    def test_robot_single_arm_no_agent_token_ok(self):
        # PM single-arm legitimately uses PickMeat-rf_150 (no agentN token).
        cfg = TrainLauncherCfg(**_minimal_robot())
        assert cfg.arm_id == 0

    def test_robot_decent_requires_agent_token_in_stem(self):
        d = _minimal_robot()
        d["task"] = {"task_config": "default_task_wristcam", "env_name": "x", "num_arms": 3, "cam_family": "wristcam"}
        d["arm_id"] = 1
        d["zarr_path"] = "/d/x_d2_wristcam_150.zarr"  # no agent1
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "agent1" in str(exc.value)

    def test_robot_decent_with_agent_token_ok(self):
        d = _minimal_robot()
        d["task"] = {"task_config": "default_task_wristcam", "env_name": "x", "num_arms": 3, "cam_family": "wristcam"}
        d["arm_id"] = 1
        d["zarr_path"] = "/d/x_agent1_d2_wristcam_150.zarr"
        cfg = TrainLauncherCfg(**d)
        assert cfg.arm_id == 1

    def test_joint_forbids_arm_id(self):
        d = _minimal_joint()
        d["arm_id"] = 0
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "forbids arm_id" in str(exc.value)

    def test_joint_requires_matching_task_config(self):
        d = _minimal_joint()
        d["task"]["num_arms"] = 2  # but task_config still joint_task_3arm
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "joint_task_2arm" in str(exc.value)

    def test_joint_valid(self):
        cfg = TrainLauncherCfg(**_minimal_joint())
        assert cfg.agent_mode == AgentMode.JOINT


# ---------------------------------------------------------------------------
# resume validators.
# ---------------------------------------------------------------------------
class TestResume:
    def test_from_ckpt_and_auto_mutually_exclusive(self):
        with pytest.raises(ValidationError):
            ResumeCfg(from_ckpt="/x/285.ckpt", auto=True)

    def test_from_ckpt_only(self):
        r = ResumeCfg(from_ckpt="/x/285.ckpt")
        assert r.from_ckpt == "/x/285.ckpt"
        assert r.auto is False

    def test_auto_only(self):
        assert ResumeCfg(auto=True).auto is True


# ---------------------------------------------------------------------------
# mode preset defaults (explicit fields win).
# ---------------------------------------------------------------------------
class TestModeDefaults:
    def test_overfit_mode_fills_defaults(self):
        d = _minimal_robot()
        d["mode"] = "overfit"
        d["training"] = {}  # nothing explicit
        cfg = TrainLauncherCfg(**d)
        assert cfg.training.num_epochs == 2000
        assert cfg.training.max_train_episodes == 1
        assert cfg.training.batch_size == 32

    def test_explicit_field_beats_mode_default(self):
        d = _minimal_robot()
        d["mode"] = "overfit"
        d["training"] = {"num_epochs": 500}  # explicit wins
        cfg = TrainLauncherCfg(**d)
        assert cfg.training.num_epochs == 500
        assert cfg.training.max_train_episodes == 1  # still from mode

    def test_baseline_mode_defaults(self):
        d = _minimal_robot()
        d["mode"] = "baseline"
        d["training"] = {}
        cfg = TrainLauncherCfg(**d)
        assert cfg.training.num_epochs == 300
        assert cfg.training.rollout_every == 10000


# ---------------------------------------------------------------------------
# ckpt-dir collision (manifest-level validator).
# ---------------------------------------------------------------------------
class TestCkptCollision:
    def test_two_launchers_same_zarr_stem_collide(self):
        a = _minimal_robot()
        b = _minimal_robot()  # same zarr_path -> same stem
        with pytest.raises(ValidationError) as exc:
            Manifest(launchers={"a": a, "b": b})
        assert "collision" in str(exc.value)

    def test_ckpt_alias_resolves_collision(self):
        a = _minimal_robot()
        b = _minimal_robot(ckpt_alias="/d/PickMeat-rf_150_variant.zarr")
        mfst = Manifest(launchers={"a": a, "b": b})
        assert len(mfst.launchers) == 2

    def test_distinct_zarr_no_collision(self):
        a = _minimal_robot()
        b = _minimal_robot(zarr_path="/d/PickMeat-rf_150_in1k_crop.zarr")
        mfst = Manifest(launchers={"a": a, "b": b})
        assert len(mfst.launchers) == 2


# ---------------------------------------------------------------------------
# Defaults & misc.
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_sbatch_train_defaults(self):
        sb = SbatchCfg()
        assert sb.time == "24:00:00"
        assert sb.gres == "gpu:a40:1"
        assert sb.mem == "96G"

    def test_encoder_freeze_default_false(self):
        e = EncoderCfg(family="resnet18_in1k")
        assert e.freeze is False

    def test_cheap_preflight_requires_scene_when_enabled(self):
        d = _minimal_robot()
        d["preflight"] = {"cheap": {"enabled": True}}  # no scene_config
        with pytest.raises(ValidationError) as exc:
            TrainLauncherCfg(**d)
        assert "scene_config" in str(exc.value)


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
