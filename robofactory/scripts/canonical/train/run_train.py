"""Universal DP-train dispatcher.

    python -m robofactory.scripts.canonical.train.run_train \\
        <launcher_id> [--manifest <path>] [--dry-run] [--slurm-job-id <id>]

Resolves the manifest entry for `<launcher_id>`, runs the cheap config-only
preflight (dump_train_cfg + train_eval_consistency), creates the
ckpt_alias / eval_name symlinks, expands encoder/training/resume/extra
overrides into the Hydra override argv, and execs
`./policy/Diffusion-Policy/train.py --config-name=... <overrides>`.

DP-ONLY. Pi0.5 training lives in openpi and is out of scope. The heavy
preflight afterok chain is submit-time orchestration handled by
submit_train.sh, NOT here.

`--dry-run` prints the resolved override dict + declared symlinks (date/uuid
fields masked) without invoking anything — useful for diffing against the
legacy `.sh` invocations during migration.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from robofactory.scripts.canonical.train.manifest_schema import (
    AgentMode,
    EncoderFamily,
    TrainLauncherCfg,
    _DINOV2_TARGETS,
    _RESNET_WEIGHTS,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]  # robofactory/
PREFLIGHT_PYTHON = "/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python"
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "train" / "manifest.yaml"
CKPT_ROOT = Path("/iris/u/mikulrai/checkpoints/RoboFactory")

CONFIG_NAME_FOR_MODE = {
    AgentMode.ROBOT: "robot_dp.yaml",
    AgentMode.JOINT: "joint_dp.yaml",
}


# ---------------------------------------------------------------------------
# zarr alias resolution
# ---------------------------------------------------------------------------
def effective_zarr_path(cfg: TrainLauncherCfg) -> str:
    """The zarr_path train.py actually sees.

    When ckpt_alias is set, the override points at the alias path (a symlink
    to the real zarr in the same dir), so train.py derives a private ckpt dir
    from the alias stem. Otherwise the raw zarr_path is used verbatim.
    """
    if cfg.ckpt_alias:
        zarr_dir = Path(cfg.zarr_path).parent
        return str(zarr_dir / (Path(cfg.ckpt_alias).stem + ".zarr"))
    return cfg.zarr_path


# ---------------------------------------------------------------------------
# Override-dict construction (the load-bearing contract)
# ---------------------------------------------------------------------------
def build_override_dict(cfg: TrainLauncherCfg) -> dict[str, object]:
    """Resolve the launcher into an ordered key->value Hydra override dict.

    Keys with a leading `+` (e.g. `+training.load_ckpt`) denote Hydra
    append-overrides. Returned as a plain dict; argv-ification happens in
    `override_dict_to_argv`. Order is insertion order = emission order.
    """
    ov: dict[str, object] = {}

    # ----- task -----
    ov["task"] = cfg.task.task_config
    ov["task.name"] = cfg.task.env_name
    # ckpt_alias overrides the effective zarr_path so train.py's
    # zarr-stem-derived ckpt dir (checkpoints/<stem>) is private. The alias is
    # a symlink to the real zarr (created by create_symlinks); data identical.
    ov["task.dataset.zarr_path"] = effective_zarr_path(cfg)

    # joint workspace: env_runner camera_family must flip wristcam->workspace.
    if cfg.agent_mode == AgentMode.JOINT and cfg.task.cam_family.value == "workspace":
        ov["task.env_runner.camera_family"] = "workspace"

    # ----- current_agent_id -----
    if cfg.agent_mode == AgentMode.ROBOT:
        ov["current_agent_id"] = cfg.arm_id

    # ----- encoder -----
    fam = cfg.encoder.family
    if cfg.encoder.is_dinov2():
        target, name = _DINOV2_TARGETS[fam]
        ov["policy.obs_encoder.rgb_model._target_"] = target
        ov["policy.obs_encoder.rgb_model.name"] = name
    else:
        weights = _RESNET_WEIGHTS[fam]
        # scratch -> weights=null (omitted from explicit override; matches the
        # legacy crop-on-scratch script which simply never set weights).
        if weights is not None:
            ov["policy.obs_encoder.rgb_model.weights"] = weights
    if cfg.encoder.resize_shape is not None:
        ov["policy.obs_encoder.resize_shape"] = cfg.encoder.resize_shape
    if cfg.encoder.crop_shape is not None:
        ov["policy.obs_encoder.crop_shape"] = cfg.encoder.crop_shape
    if cfg.encoder.random_crop is not None:
        ov["policy.obs_encoder.random_crop"] = cfg.encoder.random_crop
    if cfg.encoder.use_group_norm is not None:
        ov["policy.obs_encoder.use_group_norm"] = cfg.encoder.use_group_norm
    if cfg.encoder.freeze:
        ov["training.freeze_encoder"] = True

    # ----- dataset training fields -----
    t = cfg.training
    if t.max_train_episodes is not None:
        ov["task.dataset.max_train_episodes"] = t.max_train_episodes
    if t.val_ratio is not None:
        ov["task.dataset.val_ratio"] = t.val_ratio

    # ----- training loop -----
    ov["training.debug"] = False
    # resume / load_ckpt
    if cfg.resume.from_ckpt is not None:
        ov["training.resume"] = False
        ov["+training.load_ckpt"] = cfg.resume.from_ckpt
    else:
        ov["training.resume"] = bool(cfg.resume.auto)
    if t.seed is not None:
        ov["training.seed"] = t.seed
    ov["training.device"] = "cuda:0"
    if t.num_epochs is not None:
        ov["training.num_epochs"] = t.num_epochs
    if t.checkpoint_every is not None:
        ov["training.checkpoint_every"] = t.checkpoint_every
    if t.val_every is not None:
        ov["training.val_every"] = t.val_every
    if t.rollout_every is not None:
        ov["training.rollout_every"] = t.rollout_every
    if t.sample_every is not None:
        ov["training.sample_every"] = t.sample_every

    # ----- batch size (both dataloaders, matching legacy scripts) -----
    if t.batch_size is not None:
        ov["dataloader.batch_size"] = t.batch_size
        ov["val_dataloader.batch_size"] = t.batch_size

    # ----- extra escape hatch (verbatim, last so it can override anything) -----
    for raw in cfg.extra_overrides:
        if "=" not in raw:
            raise ValueError(f"extra_overrides entry missing '=': {raw!r}")
        k, v = raw.split("=", 1)
        ov[k] = v

    return ov


def _fmt_value(v: object) -> str:
    """Hydra argv formatting. Lists become [a,b] with no spaces (Hydra list syntax)."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def override_dict_to_argv(ov: dict[str, object]) -> list[str]:
    return [f"{k}={_fmt_value(v)}" for k, v in ov.items()]


# ---------------------------------------------------------------------------
# Symlinks (ckpt_alias + eval_name)
# ---------------------------------------------------------------------------
def declared_symlinks(cfg: TrainLauncherCfg) -> list[tuple[str, str]]:
    """Return (link_path, target) pairs this launcher would create.

    1. zarr alias: <ckpt_alias>.zarr -> <zarr_path basename> (so train.py's
       zarr-stem-derived ckpt dir is private).
    2. eval-name symlink: checkpoints/<eval_name_symlink> -> <alias_stem>.
    """
    links: list[tuple[str, str]] = []
    if cfg.ckpt_alias:
        zarr_dir = Path(cfg.zarr_path).parent
        alias_path = zarr_dir / (Path(cfg.ckpt_alias).stem + ".zarr")
        links.append((str(alias_path), Path(cfg.zarr_path).name))
    if cfg.eval_name_symlink:
        alias_stem = (
            Path(cfg.ckpt_alias).stem if cfg.ckpt_alias else Path(cfg.zarr_path).stem
        )
        link = CKPT_ROOT / cfg.eval_name_symlink
        links.append((str(link), alias_stem))
    return links


def create_symlinks(cfg: TrainLauncherCfg, *, dry_run: bool) -> None:
    for link, target in declared_symlinks(cfg):
        print(f"[run_train] symlink: {link} -> {target}", file=sys.stderr, flush=True)
        if dry_run:
            continue
        link_p = Path(link)
        link_p.parent.mkdir(parents=True, exist_ok=True)
        if not link_p.exists() and not link_p.is_symlink():
            link_p.symlink_to(target)


# ---------------------------------------------------------------------------
# Cheap preflight (config-only; run BEFORE exec)
# ---------------------------------------------------------------------------
def run_cheap_preflight(cfg: TrainLauncherCfg, slurm_job_id: str, *, dry_run: bool) -> None:
    cp = cfg.preflight.cheap
    if not cp.enabled:
        print("[run_train] cheap preflight disabled", file=sys.stderr, flush=True)
        return

    # from_ckpt existence guard (cheap, caught here not 18h in).
    if cfg.resume.from_ckpt is not None and not dry_run:
        if not Path(cfg.resume.from_ckpt).is_file():
            raise FileNotFoundError(
                f"resume.from_ckpt does not exist: {cfg.resume.from_ckpt}"
            )

    task_config = cp.task_config or cfg.task.task_config
    agent_id = cp.agent_id if cp.agent_id is not None else (cfg.arm_id or 0)
    run_dir = Path(f"/iris/u/mikulrai/runs/preflight/{slurm_job_id}")
    dump_cmd = [
        PREFLIGHT_PYTHON, "-m", "robofactory.scripts.preflight.dump_train_cfg",
        "--task-config", task_config,
        "--scene-config", cp.scene_config,
        "--agent-id", str(agent_id),
        "--out-train", str(run_dir / "train_cheap.yaml"),
        "--out-eval", str(run_dir / "eval_cheap.yaml"),
    ]
    consistency_cmd = [
        PREFLIGHT_PYTHON, "-m", "robofactory.scripts.preflight.train_eval_consistency",
        "--train-cfg", str(run_dir / "train_cheap.yaml"),
        "--eval-cfg", str(run_dir / "eval_cheap.yaml"),
        "--out", str(run_dir / "preflight_consistency.json"),
    ]
    print(f"[run_train] cheap-preflight dump: {shlex.join(dump_cmd)}", file=sys.stderr, flush=True)
    print(f"[run_train] cheap-preflight check: {shlex.join(consistency_cmd)}", file=sys.stderr, flush=True)
    if dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(dump_cmd, check=True, cwd=REPO_ROOT)
    subprocess.run(consistency_cmd, check=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def build_train_argv(cfg: TrainLauncherCfg) -> list[str]:
    config_name = CONFIG_NAME_FOR_MODE[cfg.agent_mode]
    argv = [
        "python",
        "./policy/Diffusion-Policy/train.py",
        f"--config-name={config_name}",
    ]
    argv.extend(override_dict_to_argv(build_override_dict(cfg)))
    return argv


# masked keys: never compared verbatim (date/uuid/now-derived); compared by
# stable prefix only. Train manifests set exp_name/logging.* in extra_overrides
# if at all, so this set covers the same surface as eval's masking.
MASKED_PREFIXES = ("exp_name",)


def run_launcher(
    cfg: TrainLauncherCfg,
    launcher_id: str,
    *,
    slurm_job_id: str,
    dry_run: bool = False,
) -> int:
    print(
        f"[run_train] launcher={launcher_id} agent_mode={cfg.agent_mode.value} "
        f"encoder={cfg.encoder.family.value} task={cfg.task.task_config}",
        file=sys.stderr, flush=True,
    )

    run_cheap_preflight(cfg, slurm_job_id, dry_run=dry_run)
    create_symlinks(cfg, dry_run=dry_run)

    argv = build_train_argv(cfg)
    if dry_run:
        ov = build_override_dict(cfg)
        print("[run_train] --- resolved override dict ---")
        for k, v in ov.items():
            print(f"  {k} = {_fmt_value(v)}")
        print("[run_train] --- declared symlinks ---")
        for link, target in declared_symlinks(cfg):
            print(f"  {link} -> {target}")
        if cfg.extra_overrides:
            print("[run_train] --- extra_overrides (verbatim, LOUD) ---")
            for raw in cfg.extra_overrides:
                print(f"  !! {raw}")
        print(f"[run_train] config-name={CONFIG_NAME_FOR_MODE[cfg.agent_mode]}")
        print(f"[run_train] driver argv: {shlex.join(argv)}", file=sys.stderr, flush=True)
        return 0

    print(f"[run_train] driver argv: {shlex.join(argv)}", file=sys.stderr, flush=True)
    return subprocess.run(argv, cwd=REPO_ROOT).returncode


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("launcher_id", help="ID of a launcher in the manifest")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved override dict + symlinks without executing.")
    p.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    args = p.parse_args(argv)

    mfst = load_manifest(args.manifest)
    if args.launcher_id not in mfst.launchers:
        print(
            f"[run_train] unknown launcher_id={args.launcher_id!r}; "
            f"known: {sorted(mfst.launchers)}",
            file=sys.stderr,
        )
        return 2
    cfg = mfst.launchers[args.launcher_id]
    return run_launcher(
        cfg, args.launcher_id,
        slurm_job_id=args.slurm_job_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
