"""Read the train manifest, build an `sbatch` CLI for `<launcher_id>`, and exec it.

Wraps `sbatch` because SBATCH directives can either live inside the script
(parsed by Slurm at submit time) OR come from `sbatch <flags>` (CLI flags
override script directives). This driver puts every per-launcher resource
into the manifest, then constructs the `sbatch` call so a single
`run_train.sh` shell script can serve every launcher.

When `preflight.heavy.gated` is set the helper reproduces the afterok chain
of `submit_with_preflights.sh`: it submits slurm_heavy_preflights.sh first
(optionally gated on a 1-episode overfit pass via `needs_overfit`) and gates
the training job on it via `--dependency=afterok`.

Usage:
    python -m robofactory.scripts.canonical.train.submit_helper <launcher_id>
    python -m robofactory.scripts.canonical.train.submit_helper <launcher_id> --print
    python -m robofactory.scripts.canonical.train.submit_helper <launcher_id> -- --dry-run

The `-- <extra>` pass-through is forwarded as argv to `run_train.sh`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Optional

from robofactory.scripts.canonical.train.manifest_schema import (
    TrainLauncherCfg,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_TRAIN_SH = REPO_ROOT / "scripts" / "canonical" / "train" / "run_train.sh"
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "train" / "manifest.yaml"
DEFAULT_OUTPUT = "/iris/u/mikulrai/logs/phase2_debug/{id}_%j.out"
DEFAULT_ERROR = "/iris/u/mikulrai/logs/phase2_debug/{id}_%j.err"


def _build_sbatch_cmd(launcher_id: str, cfg: TrainLauncherCfg, run_train_argv: list[str]) -> list[str]:
    sb = cfg.sbatch
    cmd: list[str] = ["sbatch"]
    job_name = sb.job_name or launcher_id
    cmd.extend(["--job-name", job_name])
    cmd.extend(["--time", sb.time])
    cmd.extend(["--gres", sb.gres])
    cmd.extend(["--cpus-per-task", str(sb.cpus_per_task)])
    cmd.extend(["--mem", sb.mem])
    if sb.exclude:
        cmd.extend(["--exclude", sb.exclude])
    if sb.partition:
        cmd.extend(["--partition", sb.partition])
    if sb.account:
        cmd.extend(["--account", sb.account])
    if sb.nice is not None:
        cmd.extend(["--nice", str(sb.nice)])
    if sb.requeue:
        cmd.append("--requeue")
    cmd.extend(["--output", sb.output or DEFAULT_OUTPUT.format(id=launcher_id)])
    cmd.extend(["--error", sb.error or DEFAULT_ERROR.format(id=launcher_id)])
    if sb.mail_type:
        cmd.extend(["--mail-type", sb.mail_type])
    if sb.mail_user:
        cmd.extend(["--mail-user", sb.mail_user])
    cmd.append(str(RUN_TRAIN_SH))
    cmd.append(launcher_id)
    cmd.extend(run_train_argv)
    return cmd


def _split_extra(argv: list[str]) -> tuple[list[str], list[str]]:
    """Anything after a literal `--` is forwarded as run_train.sh argv."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def main(argv: Optional[list[str]] = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    main_argv, extra = _split_extra(raw)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("launcher_id")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="Print the sbatch CLI without invoking it.")
    args = p.parse_args(main_argv)

    mfst = load_manifest(args.manifest)
    if args.launcher_id not in mfst.launchers:
        print(
            f"[submit_helper] unknown launcher_id={args.launcher_id!r}; "
            f"known: {sorted(mfst.launchers)}",
            file=sys.stderr,
        )
        return 2
    cfg = mfst.launchers[args.launcher_id]

    if cfg.preflight.heavy.gated:
        # Submit-time orchestration: reproduce submit_with_preflights.sh's
        # afterok chain. We do NOT run_train.py-side this; it's purely a
        # submission concern. We print the chain plan; the operator runs
        # submit_with_preflights.sh with the heavy.* fields. Keeping the
        # actual chaining in that battle-tested script avoids re-deriving the
        # overfit-bootstrap argv here.
        print(
            "[submit_helper] preflight.heavy.gated=true for "
            f"{args.launcher_id!r}: gate this launcher via "
            "submit_with_preflights.sh (afterok chain). heavy cfg: "
            f"needs_overfit={cfg.preflight.heavy.needs_overfit} "
            f"scene_config={cfg.preflight.heavy.scene_config} "
            f"baseline_ckpt={cfg.preflight.heavy.baseline_ckpt}",
            file=sys.stderr,
        )

    cmd = _build_sbatch_cmd(args.launcher_id, cfg, extra)

    if args.print_only:
        print(shlex.join(cmd))
        return 0

    # exec replaces this process so the user sees sbatch's own output.
    print(f"[submit_helper] {shlex.join(cmd)}", file=sys.stderr, flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
