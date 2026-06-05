"""Pydantic schema for the DP-train-launcher manifest.

Mirror of the eval manifest schema (`..eval.manifest_schema`): one entry per
`<launcher_id>` that `run_train.py` knows how to dispatch to
`policy/Diffusion-Policy/train.py`. `extra="forbid"` on every model means a
typo in a YAML key fails at load time, not 18h into a retrain — this is the
drift-prevention guarantee that lets us delete the hand-copied `.sh`
launchers without losing the per-launcher contract.

Scope: Diffusion-Policy (DP) ONLY. Pi0.5 training lives in openpi and is out
of scope for this dispatcher.

Run the schema check standalone to validate a manifest before submitting:

    python -m robofactory.scripts.canonical.train.manifest_schema \\
        scripts/canonical/train/manifest.yaml
"""
from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AgentMode(str, Enum):
    """Picks the Hydra `--config-name`.

    robot -> robot_dp.yaml  (single-arm / per-arm decentralised DP)
    joint -> joint_dp.yaml  (centralised multi-arm DP)
    """

    ROBOT = "robot"
    JOINT = "joint"


class CamFamily(str, Enum):
    WORKSPACE = "workspace"
    WRISTCAM = "wristcam"
    WRISTCAM_NO_GLOBAL = "wristcam_no_global"


class EncoderFamily(str, Enum):
    RESNET18_IN1K = "resnet18_in1k"
    RESNET18_R3M = "resnet18_r3m"
    RESNET18_SCRATCH = "resnet18_scratch"
    DINOV2_LORA = "dinov2_lora"
    DINOV2_PATCHATTN = "dinov2_patchattn"


class TrainMode(str, Enum):
    """Optional preset that supplies DEFAULTS for the training-loop fields.

    Any explicitly-set training field WINS over the mode default (see
    `_apply_mode_defaults`). `baseline` = canonical ep300 retrain;
    `retrain` = same as baseline (kept distinct for log clarity);
    `overfit` = single-demo memorisation knobs (rollout/val/sample off).
    """

    BASELINE = "baseline"
    RETRAIN = "retrain"
    OVERFIT = "overfit"


# DINOv2 encoder _target_ + name expansion (no weights triple).
_DINOV2_TARGETS = {
    EncoderFamily.DINOV2_LORA: (
        "diffusion_policy.model.vision.model_getter.get_dinov2_lora",
        "vit_base_patch14_dinov2",
    ),
    EncoderFamily.DINOV2_PATCHATTN: (
        "diffusion_policy.model.vision.model_getter.get_dinov2_patchattn",
        "vit_small_patch14_dinov2",
    ),
}
# ResNet weights triple expansion: weights value (None means rgb_model.weights=null).
_RESNET_WEIGHTS = {
    EncoderFamily.RESNET18_IN1K: "IMAGENET1K_V1",
    EncoderFamily.RESNET18_R3M: "r3m",
    EncoderFamily.RESNET18_SCRATCH: None,
}

# mode -> training-loop defaults (only filled when field left unset by the user).
_MODE_DEFAULTS: dict[TrainMode, dict] = {
    TrainMode.BASELINE: {
        "num_epochs": 300,
        "rollout_every": 10000,
    },
    TrainMode.RETRAIN: {
        "num_epochs": 300,
        "rollout_every": 10000,
    },
    TrainMode.OVERFIT: {
        "num_epochs": 2000,
        "max_train_episodes": 1,
        "checkpoint_every": 400,
        "val_every": 10000,
        "val_ratio": 0,
        "rollout_every": 10000,
        "sample_every": 10000,
        "batch_size": 32,
    },
}


class StrictModel(BaseModel):
    """Base class: any unknown YAML key fails the load."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class TaskCfg(StrictModel):
    """Task / scene / camera selection shared across DP configs."""

    task_config: str  # yaml name under policy/.../config/task/ (e.g. default_task)
    env_name: str  # task.name override (e.g. PickMeat-rf)
    num_arms: int = 1
    cam_family: CamFamily

    def image_dims(self) -> list[int]:
        """[C,H,W] of the obs images for this task config.

        default_task (robot, workspace single) -> 240x320.
        Everything wristcam / joint -> 224x224.
        """
        if self.task_config == "default_task":
            return [3, 240, 320]
        return [3, 224, 224]


class EncoderCfg(StrictModel):
    """Vision-encoder selection. `family` expands the rgb_model triple."""

    family: EncoderFamily
    freeze: bool = False  # -> training.freeze_encoder
    resize_shape: Optional[list[int]] = None
    crop_shape: Optional[list[int]] = None
    random_crop: Optional[bool] = None
    use_group_norm: Optional[bool] = None

    def is_dinov2(self) -> bool:
        return self.family in _DINOV2_TARGETS


class TrainingCfg(StrictModel):
    """Explicit training-loop overrides. Each maps to a Hydra key.

    Left as Optional so a `mode` preset can supply defaults via the parent's
    model_validator; an explicitly-set value always wins.
    """

    num_epochs: Optional[int] = None
    max_train_episodes: Optional[int] = None  # -> task.dataset.max_train_episodes
    checkpoint_every: Optional[int] = None
    val_every: Optional[int] = None
    val_ratio: Optional[float] = None  # -> task.dataset.val_ratio
    rollout_every: Optional[int] = None
    sample_every: Optional[int] = None
    batch_size: Optional[int] = None  # -> dataloader.batch_size + val_dataloader.batch_size
    seed: Optional[int] = None


class ResumeCfg(StrictModel):
    """Resume regime.

    from_ckpt -> emit `+training.load_ckpt=<path>` AND `training.resume=False`.
        The `+` prefix is REQUIRED: load_ckpt is NOT a key in robot_dp.yaml.
    auto      -> emit `training.resume=True` (workspace auto-picks latest).
    Mutually exclusive. from_ckpt is existence-checked at run-time (cheap
    preflight), not at schema-load (the ckpt may not exist until a prior
    chained job finishes).
    """

    from_ckpt: Optional[str] = None
    auto: bool = False

    @model_validator(mode="after")
    def _mutually_exclusive(self):
        if self.from_ckpt is not None and self.auto:
            raise ValueError("resume.from_ckpt and resume.auto are mutually exclusive")
        return self


class CheapPreflightCfg(StrictModel):
    """Config-only checks run by run_train.py BEFORE exec'ing train.py."""

    enabled: bool = True
    task_config: Optional[str] = None  # default: task.task_config
    scene_config: Optional[str] = None  # required when enabled
    agent_id: Optional[int] = None  # default: arm_id or 0

    @model_validator(mode="after")
    def _scene_required_when_enabled(self):
        if self.enabled and not self.scene_config:
            raise ValueError("preflight.cheap.scene_config required when cheap.enabled")
        return self


class HeavyPreflightCfg(StrictModel):
    """Submit-time orchestration inputs (consumed by submit_train.sh).

    run_train.py must NOT try to do submit-time orchestration; these fields
    only reproduce the afterok chain of submit_with_preflights.sh.
    """

    gated: bool = False
    needs_overfit: bool = False
    scene_config: Optional[str] = None
    baseline_ckpt: Optional[str] = None

    @model_validator(mode="after")
    def _scene_required_when_gated(self):
        if self.gated and not self.scene_config:
            raise ValueError("preflight.heavy.scene_config required when heavy.gated")
        return self


class PreflightCfg(StrictModel):
    cheap: CheapPreflightCfg = Field(default_factory=CheapPreflightCfg)
    heavy: HeavyPreflightCfg = Field(default_factory=HeavyPreflightCfg)


class WandbCfg(StrictModel):
    project: str
    group: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class SbatchCfg(StrictModel):
    """SBATCH directives data-fed into `sbatch <flags> run_train.sh <id>`.

    Copy of eval's SbatchCfg with train-appropriate defaults (24h, 96G).
    """

    job_name: Optional[str] = None  # default: <launcher_id>
    time: str = "24:00:00"
    gres: str = "gpu:a40:1"
    cpus_per_task: int = 8
    mem: str = "96G"
    exclude: Optional[str] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    nice: Optional[int] = None
    requeue: bool = False
    output: Optional[str] = None  # default: /iris/u/mikulrai/logs/phase2_debug/<id>_%j.out
    error: Optional[str] = None
    mail_type: Optional[str] = "END,FAIL"
    mail_user: Optional[str] = "mikulrai+spam@gmail.com"


class TrainLauncherCfg(StrictModel):
    """One DP train run — agent_mode × task × encoder × training × resume."""

    agent_mode: AgentMode
    task: TaskCfg
    encoder: EncoderCfg
    mode: Optional[TrainMode] = None
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    resume: ResumeCfg = Field(default_factory=ResumeCfg)
    arm_id: Optional[int] = None
    zarr_path: str
    ckpt_alias: Optional[str] = None  # zarr-stem symlink alias (private ckpt dir)
    eval_name_symlink: Optional[str] = None  # checkpoints/<eval_name> -> <alias_stem>
    extra_overrides: list[str] = Field(default_factory=list)
    preflight: PreflightCfg = Field(default_factory=PreflightCfg)
    wandb: WandbCfg
    sbatch: SbatchCfg = Field(default_factory=SbatchCfg)

    # ----- mode default application (explicit fields win) -----
    @model_validator(mode="after")
    def _apply_mode_defaults(self):
        if self.mode is not None:
            for key, val in _MODE_DEFAULTS[self.mode].items():
                cur = getattr(self.training, key)
                if cur is None:
                    setattr(self.training, key, val)
        return self

    # ----- encoder cross-field validators -----
    @model_validator(mode="after")
    def _dinov2_constraints(self):
        if self.encoder.is_dinov2():
            if self.encoder.resize_shape != [224, 224]:
                raise ValueError(
                    f"dinov2 family requires resize_shape=[224,224], "
                    f"got {self.encoder.resize_shape}"
                )
            if self.encoder.use_group_norm is not False:
                raise ValueError(
                    "dinov2 family requires use_group_norm=False, "
                    f"got {self.encoder.use_group_norm}"
                )
        return self

    @model_validator(mode="after")
    def _crop_shape_within_image(self):
        if self.encoder.crop_shape is not None:
            dims = self.task.image_dims()  # [C,H,W]
            crop = self.encoder.crop_shape
            # crop_shape may be [H,W] (len 2) or [C,H,W] (len 3).
            if len(crop) == 2:
                ch, cw = crop
                ih, iw = dims[1], dims[2]
            elif len(crop) == 3:
                _, ch, cw = crop
                _, ih, iw = dims
            else:
                raise ValueError(f"crop_shape must have 2 or 3 dims, got {crop}")
            if ch > ih or cw > iw:
                raise ValueError(
                    f"crop_shape {crop} exceeds task image dims {dims} "
                    f"(task_config={self.task.task_config})"
                )
        return self

    # ----- agent_mode cross-field validators -----
    @model_validator(mode="after")
    def _agent_mode_constraints(self):
        if self.agent_mode == AgentMode.ROBOT:
            if self.arm_id is None:
                raise ValueError("agent_mode=robot requires arm_id")
            # Decentralised multi-arm runs MUST encode the arm in the zarr stem
            # (the per-arm-ckpt-dir contract). Single-arm runs (num_arms==1) use
            # an arm-agnostic stem (e.g. PickMeat-rf_150) and are exempt.
            if self.task.num_arms > 1:
                stem = Path(self.zarr_path).stem
                if f"agent{self.arm_id}" not in stem:
                    raise ValueError(
                        f"agent_mode=robot decentralised (num_arms>1): zarr stem "
                        f"{stem!r} must contain 'agent{self.arm_id}'"
                    )
        elif self.agent_mode == AgentMode.JOINT:
            if self.arm_id is not None:
                raise ValueError("agent_mode=joint forbids arm_id")
            expected = f"joint_task_{self.task.num_arms}arm"
            if self.task.task_config != expected:
                raise ValueError(
                    f"agent_mode=joint requires task_config=={expected!r}, "
                    f"got {self.task.task_config!r}"
                )
        return self


class Manifest(StrictModel):
    launchers: dict[str, TrainLauncherCfg]

    @model_validator(mode="after")
    def _no_ckpt_dir_collisions(self):
        """Two launchers must not resolve to the same ckpt dir (zarr-stem bug).

        train.py derives ckpt dir from the zarr stem: checkpoints/<zarr_stem>.
        ckpt_alias overrides the effective stem. We assert the resolved
        (alias_stem or zarr_stem) is unique across launchers.
        """
        seen: dict[str, str] = {}
        for lid, cfg in self.launchers.items():
            # Resume launchers (resume.from_ckpt) intentionally continue an
            # existing ckpt dir in place — they are NOT competing new targets,
            # so they are exempt from the collision check.
            if cfg.resume.from_ckpt is not None:
                continue
            if cfg.ckpt_alias:
                stem = Path(cfg.ckpt_alias).stem
            else:
                stem = Path(cfg.zarr_path).stem
            if stem in seen:
                raise ValueError(
                    f"ckpt-dir collision: launchers {seen[stem]!r} and {lid!r} "
                    f"both resolve to checkpoints/{stem} "
                    f"(set distinct ckpt_alias)"
                )
            seen[stem] = lid
        return self


def load_manifest(path: Path) -> Manifest:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"manifest root must be a mapping, got {type(data).__name__}")
    return Manifest(**data)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate a train manifest YAML against the schema.")
    p.add_argument("manifest", type=Path)
    args = p.parse_args(argv)
    try:
        mfst = load_manifest(args.manifest)
    except Exception as exc:
        print(f"[manifest_schema] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[manifest_schema] OK: {len(mfst.launchers)} launcher(s) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
