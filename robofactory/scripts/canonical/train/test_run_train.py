"""Tests for the universal DP-train dispatcher's override-dict builder.

Subprocess-free: build the resolved override dict in memory and assert
specific keys/values. Locks the contract between manifest entries and the
train.py Hydra CLI.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from robofactory.scripts.canonical.train import run_train as r
from robofactory.scripts.canonical.train.manifest_schema import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "train" / "manifest.yaml"


def _cfg(lid: str):
    return load_manifest(REAL_MANIFEST).launchers[lid]


def _ov(lid: str) -> dict:
    return r.build_override_dict(_cfg(lid))


# ---------------------------------------------------------------------------
# config-name selection.
# ---------------------------------------------------------------------------
class TestConfigName:
    def test_robot_uses_robot_dp(self):
        argv = r.build_train_argv(_cfg("pm_d1_ep300_in1k"))
        assert "--config-name=robot_dp.yaml" in argv

    def test_robot_argv_starts_with_train_py(self):
        argv = r.build_train_argv(_cfg("pm_d1_ep300_in1k"))
        assert argv[0] == "python"
        assert argv[1].endswith("train.py")


# ---------------------------------------------------------------------------
# Encoder expansion.
# ---------------------------------------------------------------------------
class TestEncoderExpansion:
    def test_in1k_weights_triple(self):
        ov = _ov("pm_d1_ep300_in1k")
        assert ov["policy.obs_encoder.rgb_model.weights"] == "IMAGENET1K_V1"
        assert "policy.obs_encoder.rgb_model._target_" not in ov

    def test_r3m_weights(self):
        ov = _ov("pm_d1_ep300_r3m")
        assert ov["policy.obs_encoder.rgb_model.weights"] == "r3m"

    def test_scratch_omits_weights(self):
        ov = _ov("pm_d1_ep300_crop")
        # random-init: no weights key at all (matches legacy crop ablation).
        assert "policy.obs_encoder.rgb_model.weights" not in ov
        assert ov["policy.obs_encoder.crop_shape"] == [216, 288]

    def test_dinov2_lora_target(self):
        ov = _ov("pm_d1_ep300_dinov2_blora")
        assert ov["policy.obs_encoder.rgb_model._target_"].endswith("get_dinov2_lora")
        assert ov["policy.obs_encoder.rgb_model.name"] == "vit_base_patch14_dinov2"
        assert ov["policy.obs_encoder.resize_shape"] == [224, 224]
        assert ov["policy.obs_encoder.use_group_norm"] is False
        assert "policy.obs_encoder.rgb_model.weights" not in ov

    def test_dinov2_patchattn_target(self):
        ov = _ov("pm_d1_ep300_dinov2_spatch")
        assert ov["policy.obs_encoder.rgb_model._target_"].endswith("get_dinov2_patchattn")
        assert ov["policy.obs_encoder.rgb_model.name"] == "vit_small_patch14_dinov2"


# ---------------------------------------------------------------------------
# Resume expansion.
# ---------------------------------------------------------------------------
class TestResumeExpansion:
    def test_from_ckpt_emits_plus_load_and_resume_false(self):
        ov = _ov("resume_tsc_d2_in1k_a0_from285")
        assert ov["training.resume"] is False
        assert "+training.load_ckpt" in ov
        assert ov["+training.load_ckpt"].endswith("285.ckpt")

    def test_no_resume_emits_resume_false(self):
        ov = _ov("pm_d1_ep300_in1k")
        assert ov["training.resume"] is False
        assert "+training.load_ckpt" not in ov


# ---------------------------------------------------------------------------
# argv formatting of lists/bools.
# ---------------------------------------------------------------------------
class TestArgvFormatting:
    def test_list_to_hydra_syntax(self):
        argv = r.override_dict_to_argv({"policy.obs_encoder.crop_shape": [216, 288]})
        assert argv == ["policy.obs_encoder.crop_shape=[216,288]"]

    def test_bool_capitalized(self):
        argv = r.override_dict_to_argv({"training.resume": False})
        assert argv == ["training.resume=False"]

    def test_plus_prefix_preserved(self):
        argv = r.override_dict_to_argv({"+training.load_ckpt": "/x/285.ckpt"})
        assert argv == ["+training.load_ckpt=/x/285.ckpt"]


# ---------------------------------------------------------------------------
# Symlinks.
# ---------------------------------------------------------------------------
class TestSymlinks:
    def test_no_alias_no_symlinks(self):
        # TSC launchers have no ckpt_alias / eval_name_symlink.
        assert r.declared_symlinks(_cfg("tsc_d2_ep300_in1k_a0")) == []

    def test_alias_declares_symlink(self):
        # PM in1k has ckpt_alias=PickMeat-rf_150_in1k -> symlink to real zarr.
        links = r.declared_symlinks(_cfg("pm_d1_ep300_in1k"))
        assert len(links) == 1
        link, target = links[0]
        assert link.endswith("PickMeat-rf_150_in1k.zarr")
        assert target == "PickMeat-rf_150.zarr"

    def test_dry_run_creates_no_symlinks(self):
        cfg = _cfg("pm_d1_ep300_in1k")
        with mock.patch.object(Path, "symlink_to") as sym:
            r.create_symlinks(cfg, dry_run=True)
        assert sym.call_count == 0


# ---------------------------------------------------------------------------
# Dry-run end-to-end (no subprocess).
# ---------------------------------------------------------------------------
class TestDryRunFlow:
    @pytest.mark.parametrize("lid", [
        "pm_d1_ep300_in1k",
        "pm_d1_ep300_in1k_crop",
        "pm_d1_ep300_crop",
        "pm_d1_ep300_r3m",
        "pm_d1_ep300_dinov2_blora",
        "pm_d1_ep300_dinov2_spatch",
        "tsc_d2_ep300_in1k_a0",
        "tsc_d2_ep300_in1k_a1",
        "tsc_d2_ep300_in1k_a2",
        "resume_tsc_d2_in1k_a0_from285",
        "resume_tsc_d2_in1k_a1_from285",
        "resume_tsc_d2_in1k_a2_from285",
    ])
    def test_dry_run_no_subprocess(self, lid):
        cfg = _cfg(lid)
        with mock.patch.object(r.subprocess, "run") as run_mock:
            rc = r.run_launcher(cfg, lid, slurm_job_id="DRY", dry_run=True)
        assert rc == 0
        assert run_mock.call_count == 0


# ---------------------------------------------------------------------------
# PARITY: resolved override dict must match the legacy script's Hydra
# overrides (masking exp_name/logging.* prefix). The legacy values are
# transcribed from the .sh files at migration time.
# ---------------------------------------------------------------------------
# Masked keys are compared by PREFIX presence only (static, no date/uuid here).
MASKED = ("exp_name", "logging.mode", "logging.project", "logging.group", "logging.tags")


def _strip_masked(ov: dict) -> dict:
    return {k: v for k, v in ov.items() if k not in MASKED}


# Expected non-masked override dicts transcribed verbatim from each legacy .sh.
LEGACY = {
    "pm_d1_ep300_in1k": {
        "task": "default_task",
        "task.name": "PickMeat-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_in1k.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "pm_d1_ep300_in1k_crop": {
        "task": "default_task",
        "task.name": "PickMeat-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_in1k_crop.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "policy.obs_encoder.crop_shape": [216, 288],
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "pm_d1_ep300_crop": {
        "task": "default_task",
        "task.name": "PickMeat-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_scratch_crop.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.crop_shape": [216, 288],
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "pm_d1_ep300_r3m": {
        "task": "default_task",
        "task.name": "PickMeat-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_r3m.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model.weights": "r3m",
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "pm_d1_ep300_dinov2_blora": {
        "task": "default_task",
        "task.name": "PickMeat-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_dinob.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model._target_": "diffusion_policy.model.vision.model_getter.get_dinov2_lora",
        "policy.obs_encoder.rgb_model.name": "vit_base_patch14_dinov2",
        "policy.obs_encoder.resize_shape": [224, 224],
        "policy.obs_encoder.use_group_norm": False,
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "pm_d1_ep300_dinov2_spatch": {
        "task": "default_task",
        "task.name": "PickMeat-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/PickMeat-rf_150_dinos.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model._target_": "diffusion_policy.model.vision.model_getter.get_dinov2_patchattn",
        "policy.obs_encoder.rgb_model.name": "vit_small_patch14_dinov2",
        "policy.obs_encoder.resize_shape": [224, 224],
        "policy.obs_encoder.use_group_norm": False,
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "tsc_d2_ep300_in1k_a0": {
        "task": "default_task_wristcam",
        "task.name": "ThreeRobotsStackCube-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent0_d2_wristcam_150.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "tsc_d2_ep300_in1k_a1": {
        "task": "default_task_wristcam",
        "task.name": "ThreeRobotsStackCube-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent1_d2_wristcam_150.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 1,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "tsc_d2_ep300_in1k_a2": {
        "task": "default_task_wristcam",
        "task.name": "ThreeRobotsStackCube-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 2,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "resume_tsc_d2_in1k_a0_from285": {
        "task": "default_task_wristcam",
        "task.name": "ThreeRobotsStackCube-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent0_d2_wristcam_150.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 0,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "+training.load_ckpt": "/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent0_d2_wristcam_150/285.ckpt",
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "resume_tsc_d2_in1k_a1_from285": {
        "task": "default_task_wristcam",
        "task.name": "ThreeRobotsStackCube-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent1_d2_wristcam_150.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 1,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "+training.load_ckpt": "/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent1_d2_wristcam_150/285.ckpt",
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
    "resume_tsc_d2_in1k_a2_from285": {
        "task": "default_task_wristcam",
        "task.name": "ThreeRobotsStackCube-rf",
        "task.dataset.zarr_path": "/iris/u/mikulrai/projects/RoboFactory/robofactory/data/zarr_data/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150.zarr",
        "task.dataset.max_train_episodes": 150,
        "current_agent_id": 2,
        "policy.obs_encoder.rgb_model.weights": "IMAGENET1K_V1",
        "training.debug": False,
        "training.resume": False,
        "+training.load_ckpt": "/iris/u/mikulrai/checkpoints/RoboFactory/ThreeRobotsStackCube-rf_agent2_d2_wristcam_150/285.ckpt",
        "training.seed": 100,
        "training.device": "cuda:0",
        "training.num_epochs": 300,
        "training.rollout_every": 10000,
        "dataloader.batch_size": 64,
        "val_dataloader.batch_size": 64,
    },
}


class TestParity:
    @pytest.mark.parametrize("lid", sorted(LEGACY))
    def test_override_dict_matches_legacy(self, lid):
        got = _strip_masked(_ov(lid))
        expected = LEGACY[lid]
        assert got == expected, (
            f"{lid} parity mismatch\n"
            f"  only-in-got: { {k: got[k] for k in got.keys() - expected.keys()} }\n"
            f"  only-in-exp: { {k: expected[k] for k in expected.keys() - got.keys()} }\n"
            f"  diff-vals:   { {k: (got[k], expected[k]) for k in got.keys() & expected.keys() if got[k] != expected[k]} }"
        )

    # exp_name is masked in the dict-parity compare; here we assert it is
    # emitted at all and carries the expected stable prefix per launcher.
    EXP_PREFIX = {
        "pm_d1_ep300_in1k": "pm-d1-ep300-in1k",
        "pm_d1_ep300_in1k_crop": "pm-d1-ep300-in1k-crop",
        "pm_d1_ep300_crop": "pm-d1-ep300-crop",
        "pm_d1_ep300_r3m": "pm-d1-ep300-r3m",
        "pm_d1_ep300_dinov2_blora": "pm-d1-ep300-dino-blora",
        "pm_d1_ep300_dinov2_spatch": "pm-d1-ep300-dino-spatch",
        "tsc_d2_ep300_in1k_a0": "tsc-d2-ep300-in1k-a0",
        "tsc_d2_ep300_in1k_a1": "tsc-d2-ep300-in1k-a1",
        "tsc_d2_ep300_in1k_a2": "tsc-d2-ep300-in1k-a2",
        "resume_tsc_d2_in1k_a0_from285": "tsc-d2-ep300-in1k-a0",
        "resume_tsc_d2_in1k_a1_from285": "tsc-d2-ep300-in1k-a1",
        "resume_tsc_d2_in1k_a2_from285": "tsc-d2-ep300-in1k-a2",
    }

    @pytest.mark.parametrize("lid", sorted(LEGACY))
    def test_masked_prefix_present(self, lid):
        # exp_name + logging.* must be emitted (from extra_overrides), stable prefix.
        ov = _ov(lid)
        assert "exp_name" in ov, f"{lid} missing exp_name"
        assert ov["exp_name"] == self.EXP_PREFIX[lid]
        assert "logging.mode" in ov
