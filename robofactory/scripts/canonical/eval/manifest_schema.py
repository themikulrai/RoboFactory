"""Pydantic schema for the eval-launcher manifest.

The manifest is the single source of truth for every `<launcher_id>` that
`run_eval.py` knows how to dispatch. `extra="forbid"` on every model means a
typo in a YAML key fails at load time, not at run time — this is the
drift-prevention guarantee that lets us delete the hand-copied `.sh`
launchers without losing the per-launcher contract.

Run the schema check standalone to validate a manifest before submitting:

    python -m robofactory.scripts.canonical.eval.manifest_schema \\
        scripts/canonical/eval/manifest.yaml
"""
from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PolicyType(str, Enum):
    PI05_SINGLE = "pi05_single"
    PI05_DECENT = "pi05_decent"
    DP_SINGLE = "dp_single"
    DP_DECENT_MULTI = "dp_decent_multi"
    DP_JOINT_CENT = "dp_joint_cent"


class StrictModel(BaseModel):
    """Base class: any unknown YAML key fails the load."""

    model_config = ConfigDict(extra="forbid")


class CkptCfg(StrictModel):
    """Where the policy lives and how to identify it for preflight.

    Pi0.5 ckpts are directory trees (`dir`). DP ckpts are single `.ckpt`
    files (`path` or `path_relative`). `path_relative` is the workaround for
    `eval_joint_dp.py`'s `open('./' + ckpt_path)` quirk — that driver needs
    a path relative to the project root.
    """

    dir: Optional[str] = None
    path: Optional[str] = None
    path_relative: Optional[str] = None
    # Preflight inputs (Stage-3 guards). When absent, `_resolve_train_cfg.sh`
    # falls back to the eval cfg with a soft warn.
    train_config: Optional[str] = None  # Pi0.5 config name (e.g. pi05_robofactory_pm_lora_finetune)
    preflight_path: Optional[str] = None  # `_resolve_train_cfg.sh` input; absolute path
    # DP-only: --data-num/--checkpoint-num/--ckpt-suffix
    suffix: str = ""
    data_num: int = 150
    checkpoint_num: int = 300


class TaskCfg(StrictModel):
    """Task / scene / camera / driver flags shared across drivers."""

    env_id: str
    scene_config: str
    camera_family: str
    camera_mapping: Optional[str] = None  # Pi0.5 only
    num_arms: int = 1
    robot_uid: Optional[str] = None  # Pi0.5 --robot-uid, DP --robot-uids
    robot_uids_csv: Optional[str] = None  # DP joint/decent --robot-uids comma-list
    max_steps: int = 400  # DP --max-steps; Pi0.5 --max-env-steps. We translate.
    replan_after: int = 8  # Pi0.5 only
    prompt: Optional[str] = None  # Pi0.5 single-arm only
    img_height: Optional[int] = None  # DP eval_multi_dp only
    img_width: Optional[int] = None  # DP eval_multi_dp only
    obs_cam_family: Optional[str] = None  # DP eval_multi_dp --obs-cam-family
    include_global: Optional[bool] = None  # DP eval_multi_dp --(no-)include-global
    dp_backend: Literal["cpu", "gpu"] = "cpu"  # DP eval_*.py `-b` flag


class SeedsCfg(StrictModel):
    """File-driven seed regime. `delim` controls how SEEDS is joined for argv."""

    file: str
    sha256_file: str
    delim: Literal[",", " "]


class ServerCfg(StrictModel):
    """Pi0.5 server bringup (single or decent multi-server).

    NOTE (PR1): `ports` are LOGICAL ARM INDICES (e.g. [0], [0, 1], [0, 1, 2]),
    NOT real TCP ports. `run_eval.py` allocates job-unique real ports at runtime
    via `free_ports()` and maps each logical index -> a real port. This removes
    the hardcoded-port collision class where co-scheduled evals clashed on
    8000/8001/8002 and silently routed a client to the wrong policy server.
    The list still defines arm COUNT and ORDER, so the length checks below hold.
    """

    ports: list[int]  # logical arm indices; real ports allocated at runtime
    gpu_indices: list[int]
    xla_mem_fraction: float
    # Decent multi-server: per-arm config + dir lists. Same length as `ports`.
    arm_configs: Optional[list[str]] = None
    arm_dirs: Optional[list[str]] = None

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.ports) != len(self.gpu_indices):
            raise ValueError(
                f"ports={self.ports} and gpu_indices={self.gpu_indices} "
                "must have the same length"
            )
        if self.arm_configs is not None:
            if len(self.arm_configs) != len(self.ports):
                raise ValueError("arm_configs must match ports in length")
            if self.arm_dirs is None or len(self.arm_dirs) != len(self.ports):
                raise ValueError(
                    "arm_dirs must be provided and match arm_configs in length"
                )
        return self


class WandbCfg(StrictModel):
    project: str = "openpi-robofactory"
    tags: list[str]


class SbatchCfg(StrictModel):
    """SBATCH directives data-fed into `sbatch <flags> run_eval.sh <id>`."""

    job_name: Optional[str] = None  # default: <launcher_id>
    time: str = "4:00:00"
    gres: str = "gpu:1"
    cpus_per_task: int = 8
    mem: str = "64G"
    exclude: Optional[str] = "iris-hp-z8,iris-hgx-1,iris-hgx-2"
    partition: Optional[str] = None
    output: Optional[str] = None  # default: /iris/u/mikulrai/logs/eval/<launcher_id>_%j.out
    error: Optional[str] = None
    mail_type: Optional[str] = "END,FAIL"
    mail_user: Optional[str] = "mikulrai+spam@gmail.com"


class LauncherCfg(StrictModel):
    """One eval run — ckpt × task × seed-regime × scheduling."""

    policy_type: PolicyType
    ckpt: CkptCfg
    task: TaskCfg
    seeds: SeedsCfg
    server: Optional[ServerCfg] = None
    extra_argv: list[str] = Field(default_factory=list)
    env_hooks: list[str] = Field(default_factory=list)
    wandb: WandbCfg
    sbatch: SbatchCfg
    # PR5: number of repeat eval runs (same seeds, different sampler RNG is pinned per
    # episode so reps differ only by GPU-kernel nondeterminism). Default 3 for canonical
    # rows; set 1 for ablations. The N_REPEATS env var overrides this at dispatch time.
    n_repeats: Optional[int] = None

    @model_validator(mode="after")
    def _check_policy_consistency(self):
        is_pi05 = self.policy_type in (PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT)
        if is_pi05 and self.server is None:
            raise ValueError(
                f"policy_type={self.policy_type.value} requires server config"
            )
        if not is_pi05 and self.server is not None:
            raise ValueError(
                f"policy_type={self.policy_type.value} must not have server config"
            )
        if is_pi05 and not (self.ckpt.dir or self.ckpt.train_config):
            raise ValueError(
                "Pi0.5 launcher needs ckpt.dir and ckpt.train_config"
            )
        if self.policy_type == PolicyType.PI05_DECENT:
            assert self.server is not None  # for mypy
            if self.server.arm_configs is None:
                raise ValueError(
                    "pi05_decent requires server.arm_configs + server.arm_dirs"
                )
        if self.policy_type == PolicyType.DP_JOINT_CENT:
            if not (self.ckpt.path or self.ckpt.path_relative):
                raise ValueError(
                    "dp_joint_cent requires ckpt.path or ckpt.path_relative"
                )
        if self.policy_type in (PolicyType.DP_SINGLE, PolicyType.DP_DECENT_MULTI):
            # DP single uses ckpt.path; DP decent multi uses data_num/checkpoint_num
            # (eval_multi_dp.py reconstructs paths internally).
            pass
        return self


class Manifest(StrictModel):
    launchers: dict[str, LauncherCfg]


def load_manifest(path: Path) -> Manifest:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"manifest root must be a mapping, got {type(data).__name__}")
    return Manifest(**data)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate a manifest YAML against the schema.")
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
