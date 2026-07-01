"""Read the manifest, build an `sbatch` CLI for `<launcher_id>`, and exec it.

Wraps `sbatch` because SBATCH directives can either live inside the script
(parsed by Slurm at submit time) OR come from `sbatch <flags>` (CLI flags
override script directives). This driver puts every per-launcher resource
into the manifest, then constructs the `sbatch` call so a single
`run_eval.sh` shell script can serve every launcher.

Usage:
    python -m robofactory.scripts.canonical.eval.submit_helper <launcher_id>
    python -m robofactory.scripts.canonical.eval.submit_helper <launcher_id> --print
    python -m robofactory.scripts.canonical.eval.submit_helper <launcher_id> -- --dry-run

The `-- <extra>` pass-through is forwarded as argv to `run_eval.sh`.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Optional

from robofactory.scripts.canonical.eval.manifest_schema import (
    LauncherCfg,
    load_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_EVAL_SH = REPO_ROOT / "scripts" / "canonical" / "eval" / "run_eval.sh"
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "canonical" / "eval" / "manifest.yaml"


def _build_sbatch_cmd(
    launcher_id: str,
    cfg: LauncherCfg,
    run_eval_argv: list[str],
    shards: int = 1,
) -> list[str]:
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
    # Sharded submit: one array task per shard. %j (single-job id) can't disambiguate
    # array tasks, so rewrite it to %A_%a (array-job id + task id) in the log paths.
    if shards > 1:
        cmd.extend(["--array", f"0-{shards - 1}"])
    if sb.output:
        cmd.extend(["--output", sb.output.replace("%j", "%A_%a") if shards > 1 else sb.output])
    if sb.error:
        cmd.extend(["--error", sb.error.replace("%j", "%A_%a") if shards > 1 else sb.error])
    if sb.mail_type:
        cmd.extend(["--mail-type", sb.mail_type])
    if sb.mail_user:
        cmd.extend(["--mail-user", sb.mail_user])
    cmd.append(str(RUN_EVAL_SH))
    cmd.append(launcher_id)
    # --num-shards flows to run_eval.py; --shard-index defaults to $SLURM_ARRAY_TASK_ID
    # inside each array task, so it needs no explicit flag here.
    if shards > 1:
        cmd.extend(["--num-shards", str(shards)])
    cmd.extend(run_eval_argv)
    return cmd


def _split_extra(argv: list[str]) -> tuple[list[str], list[str]]:
    """Anything after a literal `--` is forwarded as run_eval.sh argv."""
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
    p.add_argument("--shards", type=int, default=1,
                   help="Submit as an sbatch --array of N shards; each array task evals a "
                        "disjoint round-robin slice of the seed pool (merge with merge_shards.py).")
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
    cmd = _build_sbatch_cmd(args.launcher_id, cfg, extra, shards=args.shards)

    if args.print_only:
        print(shlex.join(cmd))
        return 0

    # exec replaces this process so the user sees sbatch's own output.
    print(f"[submit_helper] {shlex.join(cmd)}", file=sys.stderr, flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
