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
import pathlib
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from robofactory.scripts.canonical.eval._lib.pi05_server_supervisor import (
    Pi05ServerSupervisor,
    ServerSpec,
    free_ports,
)
from robofactory.scripts.canonical.eval._lib.port_wait import wait_for_ports
from robofactory.scripts.canonical.eval.manifest_schema import (
    LauncherCfg,
    PolicyType,
    load_manifest,
)
from robofactory.utils.ckpt_registry import CkptRegistry, assert_registry_or_exit


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


def shard_seeds(seeds: str, shard_index: int, num_shards: int, delim: str) -> str:
    """Round-robin slice ``seeds`` into shard ``shard_index`` of ``num_shards``.

    Splits on comma OR whitespace (either delim the launcher uses), takes the
    ``[shard_index::num_shards]`` stride, and rejoins with ``delim``. The stride
    keeps within-shard order stable and makes the N shards a disjoint partition
    of the pool (every seed lands in exactly one shard).

    MUST be called AFTER the sha-guard preflight, which has to see the FULL pool.
    ``num_shards <= 1`` is the identity (returns ``seeds`` unchanged) so the
    single-shard path is byte-identical to the pre-sharding behavior.
    """
    if num_shards <= 1:
        return seeds
    toks = seeds.replace(",", " ").split()
    # A plain `assert` here is compiled OUT under `python -O`, which would let a
    # shard past the pool bounds (e.g. num_shards > n_seeds) silently return an EMPTY
    # slice — an eval on zero seeds with no error. Use an explicit runtime check that
    # fires regardless of optimization level.
    if not (0 <= shard_index < num_shards <= len(toks)):
        raise SystemExit(
            f"[run_eval] invalid shard: shard_index={shard_index}, "
            f"num_shards={num_shards}, n_seeds={len(toks)} "
            "(need 0 <= shard_index < num_shards <= n_seeds; num_shards must not "
            "exceed the pool size, else some shards get zero seeds)."
        )
    return delim.join(toks[shard_index::num_shards])


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


def run_registry_drift_guard(cfg: LauncherCfg, *, dry_run: bool = False) -> None:
    """Week-2 L3 (decision-table row #8): hard-fail BEFORE GPU/server bringup if a
    pi0.5 launcher resolves a checkpoint that disagrees with the pinned canonical
    in ``scripts/canonical/eval/ckpt_registry.yaml``.

    This is the manifest-side half of the guard (catches a wrong `dir:` in the YAML
    up front, even in --dry-run). The server-side half — comparing the PR2
    server-identity metadata the served policy actually announces — is enforced by
    the eval driver via ``check_from_server_metadata`` after connect. Both consult
    the same registry.

    Only pi0.5 launchers carry a `train_config` + resolvable ckpt dir; DP launchers
    are out of L3 scope and skipped. An unregistered config (PM/2SC/LP) is a no-op.
    """
    is_pi05 = cfg.policy_type in (PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT)
    if not is_pi05:
        return
    try:
        registry = CkptRegistry.load()
    except OSError as e:
        # Missing registry file should not silently disable the guard; warn loudly.
        print(f"[run_eval] WARN: ckpt registry not loaded ({e}); drift guard skipped",
              file=sys.stderr, flush=True)
        return

    pairs: list[tuple[str, str, str]] = []  # (config_name, ckpt_dir, label)
    if cfg.policy_type == PolicyType.PI05_SINGLE and cfg.ckpt.train_config and cfg.ckpt.dir:
        pairs.append((cfg.ckpt.train_config, cfg.ckpt.dir, "single"))
    elif cfg.policy_type == PolicyType.PI05_DECENT and cfg.server is not None:
        for i, (acfg, adir) in enumerate(
            zip(cfg.server.arm_configs or [], cfg.server.arm_dirs or [])
        ):
            pairs.append((acfg, adir, f"arm{i}"))

    for config_name, ckpt_dir, label in pairs:
        step = pathlib.PurePath(ckpt_dir).name
        resolved_step = int(step) if step.isdigit() else None
        print(f"[run_eval] ckpt-registry guard: {label} config={config_name} "
              f"dir={ckpt_dir}", file=sys.stderr, flush=True)
        if dry_run:
            # Still run the pure comparison in dry-run; it never touches a GPU.
            from robofactory.utils.ckpt_registry import check_registry_drift
            ok, diag, _ = check_registry_drift(
                registry, config_name, ckpt_dir, resolved_step, label=label
            )
            if not ok:
                print(f"[run_eval] DRY-RUN would hard-fail: {diag}",
                      file=sys.stderr, flush=True)
            elif diag:
                print(f"[run_eval] DRY-RUN warn: {diag}", file=sys.stderr, flush=True)
            continue
        assert_registry_or_exit(
            registry, config_name, ckpt_dir, resolved_step, label=label
        )


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


def build_pi05_single_argv(
    cfg: LauncherCfg, seeds: str, slurm_job_id: str, real_ports: list[int]
) -> list[str]:
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
        "--port", str(real_ports[0]),
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
    # PR2 server-identity handshake: pass the config name the supervisor will serve so
    # the client hard-fails before episode 1 on a wrong-policy-behind-the-port mismatch.
    if cfg.ckpt.train_config:
        argv.extend(["--expect-config", cfg.ckpt.train_config])
    if cfg.task.prompt:
        argv.extend(["--prompt", cfg.task.prompt])
    return argv


def build_pi05_decent_argv(
    cfg: LauncherCfg, seeds: str, slurm_job_id: str, real_ports: list[int]
) -> list[str]:
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
        "--ports", ",".join(str(p) for p in real_ports),
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
    # PR2 server-identity handshake: one expected config per arm, in --ports order.
    # The supervisor serves cfg.server.arm_configs[i] on real_ports[i] (see
    # _build_server_specs), so the alignment matches by construction.
    if cfg.server.arm_configs:
        argv.extend(["--expect-config", ",".join(cfg.server.arm_configs)])
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


def build_driver_argv(
    cfg: LauncherCfg,
    seeds: str,
    slurm_job_id: str,
    real_ports: Optional[list[int]] = None,
) -> list[str]:
    builder = DRIVER_BUILDERS[cfg.policy_type]
    if cfg.policy_type in (PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT):
        # Pi0.5 builders take the runtime-allocated real ports (manifest stores
        # only logical arm indices now — PR1).
        if real_ports is None:
            raise ValueError("pi0.5 driver argv requires runtime-allocated real_ports")
        argv = builder(cfg, seeds, slurm_job_id, real_ports)
    else:
        argv = builder(cfg, seeds, slurm_job_id)
    argv.extend(cfg.extra_argv)
    return argv


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def _build_server_specs(cfg: LauncherCfg, real_ports: list[int]) -> list[ServerSpec]:
    assert cfg.server is not None
    if cfg.policy_type == PolicyType.PI05_SINGLE:
        # Single arm: train_config + ckpt.dir.
        if not (cfg.ckpt.train_config and cfg.ckpt.dir):
            raise ValueError("pi05_single needs ckpt.train_config and ckpt.dir")
        return [
            ServerSpec(
                port=real_ports[0],
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
        real_ports,
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


def resolve_n_repeats(cfg: LauncherCfg, launcher_id: str) -> int:
    """PR5: N_REPEATS precedence — env var N_REPEATS > cfg.n_repeats > policy default.

    Policy default is 3 for canonical rows and 1 for ablations (the headline contract
    is mean±Wilson-CI over >=3 reps; single-run deltas <15pp are not findings). We treat
    a launcher whose id starts with `ablation`/`abl_` or that lives under scripts/ablations
    as an ablation; everything else defaults to 3.
    """
    env_val = os.environ.get("N_REPEATS")
    if env_val:
        n = int(env_val)
        if n < 1:
            raise ValueError(f"N_REPEATS must be >=1, got {n}")
        return n
    if cfg.n_repeats is not None:
        if cfg.n_repeats < 1:
            raise ValueError(f"cfg.n_repeats must be >=1, got {cfg.n_repeats}")
        return cfg.n_repeats
    is_ablation = launcher_id.startswith(("ablation", "abl_", "abl-"))
    return 1 if is_ablation else 3


def _inject_rep_output(argv: list[str], cfg: LauncherCfg, rep: int) -> list[str]:
    """Route this rep's result file to a `*_rep{k}.json[l]` path so reps never collide
    and `summarize_eval_runs.py` can glob them. pi0.5 drivers take --out-dir; DP drivers
    take --jsonl-path. Idempotent: only appended when the manifest didn't already set one.
    """
    argv = list(argv)
    is_pi05 = cfg.policy_type in (PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT)
    if is_pi05:
        # Point --out-dir at a per-rep subdir (driver still timestamps the filename;
        # one JSON per subdir => one rep per dir, unambiguous for the summarizer).
        if "--out-dir" in argv:
            i = argv.index("--out-dir")
            argv[i + 1] = str(Path(argv[i + 1]) / f"rep{rep}")
        Path(argv[argv.index("--out-dir") + 1]).mkdir(parents=True, exist_ok=True)
    else:
        # DP: give the driver an explicit per-rep jsonl path (only if none set yet).
        if "--jsonl-path" not in argv and not any(a.startswith("--jsonl-path=") for a in argv):
            base = Path("/iris/u/mikulrai/logs/eval") / f"{cfg.task.env_id}_rep{rep}.jsonl"
            base.parent.mkdir(parents=True, exist_ok=True)
            argv.append(f"--jsonl-path={base}")
    return argv


def _inject_shard_output(argv: list[str], shard_index: int, num_shards: int) -> list[str]:
    """Suffix this shard's result/video dirs with ``_shard{i}of{n}`` so co-running
    array-task shards never clobber each other's output. Mirrors ``_inject_rep_output``:
    edits the value of --out-dir / --video-dir (pi0.5) and --jsonl-path (DP) in place.
    Idempotent no-op when ``num_shards <= 1`` (keeps the single-shard argv identical).
    """
    if num_shards <= 1:
        return argv
    argv = list(argv)
    suf = f"_shard{shard_index}of{num_shards}"
    for flag in ("--out-dir", "--video-dir"):
        if flag in argv:
            i = argv.index(flag)
            p = Path(argv[i + 1])
            argv[i + 1] = str(p.with_name(p.name + suf))
            if flag == "--out-dir":
                Path(argv[i + 1]).mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(argv):
        if a.startswith("--jsonl-path="):
            p = Path(a.split("=", 1)[1])
            argv[i] = f"--jsonl-path={p.with_name(p.stem + suf + p.suffix)}"
    return argv


def run_launcher(
    cfg: LauncherCfg,
    launcher_id: str,
    *,
    slurm_job_id: str,
    dry_run: bool = False,
    preflight_only: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
) -> int:
    seeds = _load_seeds(cfg)
    n_repeats = resolve_n_repeats(cfg, launcher_id)
    print(
        f"[run_eval] launcher={launcher_id} policy={cfg.policy_type.value} "
        f"seeds={cfg.seeds.file} (n={len(seeds.replace(',', ' ').split())}) "
        f"n_repeats={n_repeats}",
        file=sys.stderr, flush=True,
    )

    run_env_hooks(cfg, dry_run=dry_run)
    run_preflight(cfg, argv_seeds=seeds, dry_run=dry_run)
    # Week-2 L3: hard-fail on a wrong-checkpoint-for-config before GPU/server bringup.
    run_registry_drift_guard(cfg, dry_run=dry_run)
    if preflight_only:
        print(
            f"[run_eval] preflight-only mode: stopping before driver/servers.",
            file=sys.stderr, flush=True,
        )
        return 0

    # Seed sharding: preflight above saw the FULL pool (the sha guard must). Now that
    # the guards passed, slice the pool round-robin so each array task (SLURM_ARRAY_TASK_ID
    # -> shard_index) evaluates a disjoint subset. merge_shards.py recombines them.
    if num_shards > 1:
        full_n = len(seeds.replace(",", " ").split())
        seeds = shard_seeds(seeds, shard_index, num_shards, cfg.seeds.delim)
        shard_toks = seeds.replace(",", " ").split()
        print(
            f"[run_eval] shard {shard_index}/{num_shards}: "
            f"{len(shard_toks)}/{full_n} seeds -> {seeds}",
            file=sys.stderr, flush=True,
        )

    is_pi05 = cfg.policy_type in (PolicyType.PI05_SINGLE, PolicyType.PI05_DECENT)
    rc_final = 0
    for rep in range(n_repeats):
        rep_job_id = f"{slurm_job_id}_rep{rep}" if n_repeats > 1 else slurm_job_id
        print(f"[run_eval] === rep {rep + 1}/{n_repeats} (run_id={rep_job_id}) ===",
              file=sys.stderr, flush=True)
        if is_pi05:
            assert cfg.server is not None
            # Manifest stores logical arm indices in `cfg.server.ports`; allocate one
            # job-unique real port per arm at runtime so co-scheduled evals never
            # collide (PR1). The index list defines only arm count/order.
            real_ports = free_ports(len(cfg.server.ports))
            print(
                f"[run_eval] allocated runtime ports {real_ports} for "
                f"{len(cfg.server.ports)} arm(s) (manifest indices "
                f"{cfg.server.ports})",
                file=sys.stderr, flush=True,
            )
            server_log_dir = Path(
                f"/iris/u/mikulrai/logs/eval_pi05/run_eval_{rep_job_id}_servers"
            )
            specs = _build_server_specs(cfg, real_ports)
            argv = build_driver_argv(cfg, seeds, rep_job_id, real_ports)
            argv = _inject_shard_output(argv, shard_index, num_shards)
            argv = _inject_rep_output(argv, cfg, rep) if n_repeats > 1 else argv
            print(f"[run_eval] driver argv: {shlex.join(argv)}", file=sys.stderr, flush=True)
            if dry_run:
                for spec in specs:
                    print(
                        f"[run_eval] would spawn {spec.name} gpu={spec.gpu_index} "
                        f"port={spec.port} cfg={spec.policy_config} dir={spec.policy_dir}",
                        file=sys.stderr, flush=True,
                    )
                continue
            with Pi05ServerSupervisor(specs, log_dir=server_log_dir):
                wait_for_ports(real_ports, deadline_s=600.0, poll_s=5.0)
                print("[run_eval] all server ports up; launching eval driver", file=sys.stderr, flush=True)
                rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
        else:
            # DP path: no server bringup.
            argv = build_driver_argv(cfg, seeds, rep_job_id)
            argv = _inject_shard_output(argv, shard_index, num_shards)
            argv = _inject_rep_output(argv, cfg, rep) if n_repeats > 1 else argv
            print(f"[run_eval] driver argv: {shlex.join(argv)}", file=sys.stderr, flush=True)
            if dry_run:
                continue
            rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
        if rc != 0:
            rc_final = rc  # keep going through reps but surface a nonzero exit
    return rc_final


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("launcher_id", help="ID of a launcher in the manifest")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dry-run", action="store_true",
                   help="Print argv + preflight cmd without executing.")
    p.add_argument("--preflight-only", action="store_true",
                   help="Run preflight + env_hooks for real, then stop. "
                        "Useful for verifying scene/seed/wandb without burning a GPU.")
    p.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the seed pool into N disjoint round-robin shards "
                        "(one array task each). Preflight still sees the full pool.")
    p.add_argument("--shard-index", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
                   help="Which shard this task runs (0-based). Defaults to "
                        "$SLURM_ARRAY_TASK_ID so an sbatch --array wires it automatically.")
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
    return run_launcher(
        cfg, args.launcher_id,
        slurm_job_id=args.slurm_job_id,
        dry_run=args.dry_run,
        preflight_only=args.preflight_only,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
