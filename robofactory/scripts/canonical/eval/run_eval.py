"""Universal eval dispatcher.

    python -m robofactory.scripts.canonical.eval.run_eval \\
        --launcher-id <id> [--manifest <path>] [--dry-run]

Resolves the manifest entry for `<launcher_id>`, runs the Stage-3 preflight
guards, brings up Pi0.5 server(s) if needed (with strict-reaping via
PR_SET_PDEATHSIG), and dispatches to the matching eval_*.py via subprocess.

Existing eval entry points (eval_pi05.py, eval_decent_pi05.py, eval_dp.py,
eval_multi_dp.py, eval_joint_dp.py) are NOT modified — this dispatcher just
centralizes the launcher boilerplate that was previously duplicated across
12 hand-copied `.sh` files.

`--dry-run` prints what would be executed without invoking anything. Useful
for diffing against the legacy `.sh` invocations during migration.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from robofactory.scripts.canonical.eval._lib.pi05_server_supervisor import (
    Pi05ServerSupervisor,
    ServerSpec,
)
from robofactory.scripts.canonical.eval._lib.port_wait import wait_for_ports
from robofactory.scripts.canonical.eval.manifest_schema import (
    LauncherCfg,
    PolicyType,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]  # robofactory/
PREFLIGHT_PYTHON = "/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python"
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "eval" / "manifest.yaml"


# ---------------------------------------------------------------------------
# Preflight + env hooks
# ---------------------------------------------------------------------------
def _load_seeds(cfg: LauncherCfg) -> str:
    """Read the seed file and join according to `seeds.delim`."""
    raw = Path(cfg.seeds.file).read_text().split()
    return cfg.seeds.delim.join(raw)


def _resolve_train_cfg(cfg: LauncherCfg) -> Optional[Path]:
    """Replicate `_resolve_train_cfg.sh`: prefer .hydra_config.yaml next to ckpt.

    Looks in dirname(ckpt_path) for `.hydra_config.yaml`. If absent, returns
    None so the caller can fall back to the eval cfg.
    """
    ckpt_ref = (
        cfg.ckpt.preflight_path
        or cfg.ckpt.dir
        or cfg.ckpt.path
        or (str(REPO_ROOT / cfg.ckpt.path_relative) if cfg.ckpt.path_relative else None)
    )
    if not ckpt_ref:
        return None
    parent = Path(ckpt_ref).parent
    cand = parent / ".hydra_config.yaml"
    return cand if cand.exists() else None


def run_preflight(cfg: LauncherCfg, argv_seeds: str, *, dry_run: bool = False) -> None:
    """Invoke Stage-3 preflight guards as subprocess (same binary the .sh used)."""
    train_cfg = _resolve_train_cfg(cfg)
    if train_cfg is None:
        # Legacy fallback: use the eval cfg as the train cfg input. Matches
        # `_resolve_train_cfg.sh`'s soft-warn behavior.
        train_cfg = REPO_ROOT / cfg.task.scene_config
    cmd = [
        PREFLIGHT_PYTHON,
        "-m",
        "robofactory.utils.preflight_eval_guards",
        "--train-cfg",
        str(train_cfg),
        "--eval-cfg",
        str(REPO_ROOT / cfg.task.scene_config),
        "--seed-file",
        cfg.seeds.file,
        "--expected-sha256-file",
        cfg.seeds.sha256_file,
        "--argv-seeds",
        argv_seeds,
    ]
    print(f"[run_eval] preflight: {shlex.join(cmd)}", file=sys.stderr, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_env_hooks(cfg: LauncherCfg, *, dry_run: bool = False) -> None:
    """Run any bash one-liners declared in env_hooks (e.g. R3M symlink)."""
    for hook in cfg.env_hooks:
        print(f"[run_eval] env_hook: {hook}", file=sys.stderr, flush=True)
        if dry_run:
            continue
        subprocess.run(["bash", "-lc", hook], check=True)


# ---------------------------------------------------------------------------
# Driver argv builders — one per policy_type
# ---------------------------------------------------------------------------
def _video_out_dirs(cfg: LauncherCfg, slurm_job_id: str) -> tuple[Path, Path]:
    """Mirror legacy launcher convention for video + output dirs."""
    base = Path("/iris/u/mikulrai/logs/eval_pi05")
    if cfg.policy_type == PolicyType.PI05_DECENT:
        base = Path("/iris/u/mikulrai/logs/eval_pi05_decent")
    video_dir = base / f"videos_{slurm_job_id}"
    out_dir = base
    return out_dir, video_dir


def build_pi05_single_argv(cfg: LauncherCfg, seeds: str, slurm_job_id: str) -> list[str]:
    assert cfg.server is not None
    out_dir, video_dir = _video_out_dirs(cfg, slurm_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        PREFLIGHT_PYTHON,
        "-u",
        "./policy/openpi_pi05/eval_pi05.py",
        "--task", cfg.task.env_id,
        "--config", str(REPO_ROOT / cfg.task.scene_config),
        "--host", "127.0.0.1",
        "--port", str(cfg.server.ports[0]),
        "--num-arms", str(cfg.task.num_arms),
        "--num-episodes", "1",
        "--seeds", seeds,
        "--max-env-steps", str(cfg.task.max_steps),
        "--replan-after", str(cfg.task.replan_after),
        "--robot-uid", cfg.task.robot_uid or "panda",
        "--camera-mapping", cfg.task.camera_mapping or "",
        "--out-dir", str(out_dir),
        "--video-dir", str(video_dir),
        "--video-max", "3",
        "--run-id", slurm_job_id,
        "--wandb",
        "--wandb-project", cfg.wandb.project,
        "--wandb-tags", ",".join(cfg.wandb.tags),
    ]
    if cfg.task.prompt:
        argv.extend(["--prompt", cfg.task.prompt])
    return argv


def build_pi05_decent_argv(cfg: LauncherCfg, seeds: str, slurm_job_id: str) -> list[str]:
    assert cfg.server is not None
    out_dir, video_dir = _video_out_dirs(cfg, slurm_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        PREFLIGHT_PYTHON,
        "-u",
        "./policy/openpi_pi05/eval_decent_pi05.py",
        "--task", cfg.task.env_id,
        "--config", str(REPO_ROOT / cfg.task.scene_config),
        "--host", "127.0.0.1",
        "--ports", ",".join(str(p) for p in cfg.server.ports),
        "--num-arms", str(cfg.task.num_arms),
        "--num-episodes", "1",
        "--seeds", seeds,
        "--max-env-steps", str(cfg.task.max_steps),
        "--replan-after", str(cfg.task.replan_after),
        "--robot-uid", cfg.task.robot_uid or "panda_wristcam_multi",
        "--robot-uids-csv", cfg.task.robot_uids_csv or "",
        "--camera-mapping", cfg.task.camera_mapping or "",
        "--out-dir", str(out_dir),
        "--video-dir", str(video_dir),
        "--video-max", "3",
        "--run-id", slurm_job_id,
        "--wandb",
        "--wandb-project", cfg.wandb.project,
        "--wandb-tags", ",".join(cfg.wandb.tags),
    ]
    return argv


def build_dp_single_argv(cfg: LauncherCfg, seeds: str, slurm_job_id: str) -> list[str]:
    if not cfg.ckpt.path:
        raise ValueError("dp_single requires ckpt.path")
    argv = [
        "python",
        "-u",
        "./policy/Diffusion-Policy/eval_dp.py",
        f"--config={REPO_ROOT / cfg.task.scene_config}",
        f"--ckpt-path={cfg.ckpt.path}",
        f"--data-num={cfg.ckpt.data_num}",
        f"--checkpoint-num={cfg.ckpt.checkpoint_num}",
        "-o", "rgb",
        "-b", cfg.task.dp_backend,
        "-n", "1",
        "--render-mode=sensors",
        "-s", *seeds.split(),
        "--quiet",
        f"--max-steps={cfg.task.max_steps}",
        "--wandb",
        f"--wandb-tags={','.join(cfg.wandb.tags)}",
    ]
    return argv


def build_dp_decent_multi_argv(cfg: LauncherCfg, seeds: str, slurm_job_id: str) -> list[str]:
    argv = [
        "python",
        "-u",
        "./policy/Diffusion-Policy/eval_multi_dp.py",
        f"--config={REPO_ROOT / cfg.task.scene_config}",
        f"--data-num={cfg.ckpt.data_num}",
        f"--checkpoint-num={cfg.ckpt.checkpoint_num}",
        f"--ckpt-suffix={cfg.ckpt.suffix}",
        f"--obs-cam-family={cfg.task.obs_cam_family or cfg.task.camera_family}",
    ]
    if cfg.task.include_global is True:
        argv.append("--include-global")
    elif cfg.task.include_global is False:
        argv.append("--no-include-global")
    if cfg.task.img_height is not None:
        argv.append(f"--img-height={cfg.task.img_height}")
    if cfg.task.img_width is not None:
        argv.append(f"--img-width={cfg.task.img_width}")
    if cfg.task.robot_uids_csv:
        argv.append(f"--robot-uids={cfg.task.robot_uids_csv}")
    argv.extend([
        "-o", "rgb",
        "-b", cfg.task.dp_backend,
        "-n", "1",
        "--render-mode=sensors",
        "-s", *seeds.split(),
        "--quiet",
        f"--max-steps={cfg.task.max_steps}",
        "--wandb",
        f"--wandb-tags={','.join(cfg.wandb.tags)}",
    ])
    return argv


def build_dp_joint_cent_argv(cfg: LauncherCfg, seeds: str, slurm_job_id: str) -> list[str]:
    ckpt = cfg.ckpt.path_relative or cfg.ckpt.path
    if not ckpt:
        raise ValueError("dp_joint_cent requires ckpt.path_relative or ckpt.path")
    argv = [
        "python",
        "-u",
        "./policy/Diffusion-Policy/eval_joint_dp.py",
        "--ckpt-path", ckpt,
        "--config", str(REPO_ROOT / cfg.task.scene_config),
        "--env-id", cfg.task.env_id,
        "--camera-family", cfg.task.camera_family,
        "--n-agents", str(cfg.task.num_arms),
    ]
    if cfg.task.robot_uids_csv:
        argv.extend(["--robot-uids", cfg.task.robot_uids_csv])
    argv.extend([
        "--max-steps", str(cfg.task.max_steps),
        "--quiet",
        "--wandb",
        "--wandb-tags", ",".join(cfg.wandb.tags),
        "--seed", *seeds.split(),
    ])
    return argv


DRIVER_BUILDERS: dict[PolicyType, Callable[..., list[str]]] = {
    PolicyType.PI05_SINGLE: build_pi05_single_argv,
    PolicyType.PI05_DECENT: build_pi05_decent_argv,
    PolicyType.DP_SINGLE: build_dp_single_argv,
    PolicyType.DP_DECENT_MULTI: build_dp_decent_multi_argv,
    PolicyType.DP_JOINT_CENT: build_dp_joint_cent_argv,
}


def build_driver_argv(cfg: LauncherCfg, seeds: str, slurm_job_id: str) -> list[str]:
    builder = DRIVER_BUILDERS[cfg.policy_type]
    argv = builder(cfg, seeds, slurm_job_id)
    argv.extend(cfg.extra_argv)
    return argv


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def _build_server_specs(cfg: LauncherCfg) -> list[ServerSpec]:
    assert cfg.server is not None
    if cfg.policy_type == PolicyType.PI05_SINGLE:
        # Single arm: train_config + ckpt.dir.
        if not (cfg.ckpt.train_config and cfg.ckpt.dir):
            raise ValueError("pi05_single needs ckpt.train_config and ckpt.dir")
        return [
            ServerSpec(
                port=cfg.server.ports[0],
                gpu_index=cfg.server.gpu_indices[0],
                policy_config=cfg.ckpt.train_config,
                policy_dir=cfg.ckpt.dir,
                xla_mem_fraction=cfg.server.xla_mem_fraction,
                name="pm",
            )
        ]
    # PI05_DECENT
    assert cfg.server.arm_configs and cfg.server.arm_dirs
    specs = []
    for i, (port, gpu, acfg, adir) in enumerate(zip(
        cfg.server.ports,
        cfg.server.gpu_indices,
        cfg.server.arm_configs,
        cfg.server.arm_dirs,
    )):
        specs.append(ServerSpec(
            port=port,
            gpu_index=gpu,
            policy_config=acfg,
            policy_dir=adir,
            xla_mem_fraction=cfg.server.xla_mem_fraction,
            name=f"arm{i}",
        ))
    return specs


def run_launcher(
    cfg: LauncherCfg,
    launcher_id: str,
    *,
    slurm_job_id: str,
    dry_run: bool = False,
) -> int:
    seeds = _load_seeds(cfg)
    print(
        f"[run_eval] launcher={launcher_id} policy={cfg.policy_type.value} "
        f"seeds={cfg.seeds.file} (n={len(seeds.replace(',', ' ').split())})",
        file=sys.stderr, flush=True,
    )

    run_env_hooks(cfg, dry_run=dry_run)
    run_preflight(cfg, argv_seeds=seeds, dry_run=dry_run)

    is_pi05 = cfg.policy_type in (PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT)
    if is_pi05:
        assert cfg.server is not None
        server_log_dir = Path(
            f"/iris/u/mikulrai/logs/eval_pi05/run_eval_{slurm_job_id}_servers"
        )
        specs = _build_server_specs(cfg)
        argv = build_driver_argv(cfg, seeds, slurm_job_id)
        print(f"[run_eval] driver argv: {shlex.join(argv)}", file=sys.stderr, flush=True)
        if dry_run:
            for spec in specs:
                print(
                    f"[run_eval] would spawn {spec.name} gpu={spec.gpu_index} "
                    f"port={spec.port} cfg={spec.policy_config} dir={spec.policy_dir}",
                    file=sys.stderr, flush=True,
                )
            return 0
        with Pi05ServerSupervisor(specs, log_dir=server_log_dir):
            wait_for_ports(cfg.server.ports, deadline_s=600.0, poll_s=5.0)
            print("[run_eval] all server ports up; launching eval driver", file=sys.stderr, flush=True)
            rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
            return rc

    # DP path: no server bringup.
    argv = build_driver_argv(cfg, seeds, slurm_job_id)
    print(f"[run_eval] driver argv: {shlex.join(argv)}", file=sys.stderr, flush=True)
    if dry_run:
        return 0
    return subprocess.run(argv, cwd=REPO_ROOT).returncode


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("launcher_id", help="ID of a launcher in the manifest")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dry-run", action="store_true",
                   help="Print argv + preflight cmd without executing.")
    p.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    args = p.parse_args(argv)

    mfst = load_manifest(args.manifest)
    if args.launcher_id not in mfst.launchers:
        print(
            f"[run_eval] unknown launcher_id={args.launcher_id!r}; "
            f"known: {sorted(mfst.launchers)}",
            file=sys.stderr,
        )
        return 2
    cfg = mfst.launchers[args.launcher_id]
    return run_launcher(cfg, args.launcher_id, slurm_job_id=args.slurm_job_id, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
