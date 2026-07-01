"""Chain eval(s) after training via sbatch --dependency=afterok — no human in the loop.

Measured gap this removes: ~5h between train-finish and a human noticing and
submitting the eval. Instead, submit both at once; Slurm starts the eval the
moment the training job(s) COMPLETE with exit 0.

    python -m robofactory.scripts.canonical.chain_train_eval \\
        --train <train_launcher_id> [--train <id2> ...] \\
        --eval <eval_launcher_id>  [--eval  <id2> ...] \\
        [--shards N] [--after <existing_jobid> ...] [--dry-run]

Behavior:
  * Every --train id is validated against the TRAIN manifest and every --eval id
    against the EVAL manifest BEFORE anything is submitted (a typo aborts the
    whole chain with zero jobs queued).
  * Each train launcher is submitted via the existing train submit_helper sbatch
    builder (with --parsable injected so the job id is machine-readable).
  * Each eval launcher is then submitted gated on afterok:<jid1>[:<jid2>...]
    over ALL train job ids (+ any --after ids), via the existing eval
    submit_helper builder — so --shards composes exactly like a manual
    `submit_eval.sh --shards N` (one --array job; every array task inherits the
    dependency).
  * --after <jobid> chains onto ALREADY-SUBMITTED jobs (e.g. pi0.5 trains
    launched from openpi, which the DP-only train manifest cannot express).
    --train may be omitted entirely when --after is given.
  * If any submission fails mid-chain, we ABORT immediately and print the
    `scancel` line that cleans up the jobs already queued.
  * --dry-run prints the exact sbatch commands without submitting anything.

Semantics notes:
  * afterok waits for the dependency job's FINAL state. Preempted-and-requeued
    train jobs (--requeue partitions like orion) go back to PENDING under the
    same job id, so the eval keeps waiting across preemptions and only fires
    on a true successful completion. --kill-on-invalid-dep=yes (added by the eval
    builder whenever a dependency is set) cancels the eval if training FAILs.
  * Do NOT extend this pattern downstream of the eval: eval Slurm exit codes
    are unreliable (job 16045963: State=FAILED ExitCode=2 with a complete
    results JSON already on disk). Anything consuming eval output must key on
    results-JSON-exists, never on Slurm job state.
  * Checkpoint contract: the eval manifest pins ckpt dirs/steps statically
    (e.g. .../<config>/<run>/19999 for pi0.5 = num_train_steps-1, or
    checkpoint_num: 300 for DP = training.num_epochs). This tool does NOT
    rewrite them — the run-time guards (ckpt_registry drift guard + preflight)
    hard-fail the eval if the pinned ckpt is absent or wrong, which is exactly
    what happens if you chain an eval whose manifest entry predates the train
    run's output path. Verify the eval entry points at what the train launcher
    will produce.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from robofactory.scripts.canonical.eval import submit_helper as eval_sh
from robofactory.scripts.canonical.eval.manifest_schema import (
    load_manifest as load_eval_manifest,
)
from robofactory.scripts.canonical.train import submit_helper as train_sh
from robofactory.scripts.canonical.train.manifest_schema import (
    load_manifest as load_train_manifest,
)


def parse_sbatch_jobid(output: str) -> int:
    """Extract the job id from sbatch stdout.

    Handles both `--parsable` output ("12345" or "12345;cluster") and the
    human format ("Submitted batch job 12345").
    """
    text = output.strip()
    m = re.match(r"^(\d+)(?:;[\w.-]+)?$", text)
    if m:
        return int(m.group(1))
    m = re.search(r"Submitted batch job (\d+)", text)
    if m:
        return int(m.group(1))
    raise ValueError(f"cannot parse sbatch job id from output: {output!r}")


def _submit(cmd: list[str], *, dry_run: bool, fake_jobid: int) -> int:
    """Run one sbatch command and return its job id.

    In --dry-run, print the exact command and return a fake id so the printed
    downstream --dependency specs show the chain shape.
    """
    if dry_run:
        print(f"DRYRUN: {shlex.join(cmd)}")
        return fake_jobid
    print(f"[chain_train_eval] {shlex.join(cmd)}", file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return parse_sbatch_jobid(proc.stdout)


def run_chain(
    train_ids: list[str],
    eval_ids: list[str],
    *,
    shards: int = 1,
    after: Optional[list[int]] = None,
    dry_run: bool = False,
    train_manifest: Optional[Path] = None,
    eval_manifest: Optional[Path] = None,
) -> int:
    after = list(after or [])
    if not train_ids and not after:
        print("[chain_train_eval] need at least one --train or --after", file=sys.stderr)
        return 2
    if not eval_ids:
        print("[chain_train_eval] need at least one --eval", file=sys.stderr)
        return 2

    # ---- validate EVERYTHING before submitting ANYTHING --------------------
    train_mfst = load_train_manifest(train_manifest or train_sh.DEFAULT_MANIFEST)
    eval_mfst = load_eval_manifest(eval_manifest or eval_sh.DEFAULT_MANIFEST)
    unknown_train = [t for t in train_ids if t not in train_mfst.launchers]
    unknown_eval = [e for e in eval_ids if e not in eval_mfst.launchers]
    if unknown_train or unknown_eval:
        if unknown_train:
            print(f"[chain_train_eval] unknown train launcher(s): {unknown_train}; "
                  f"known: {sorted(train_mfst.launchers)}", file=sys.stderr)
        if unknown_eval:
            print(f"[chain_train_eval] unknown eval launcher(s): {unknown_eval}; "
                  f"known: {sorted(eval_mfst.launchers)}", file=sys.stderr)
        return 2

    submitted: list[tuple[str, int]] = []  # (label, jobid) for the scancel hint
    fake = 900001  # deterministic fake ids for --dry-run chain display

    # ---- 1. submit train job(s) --------------------------------------------
    train_jids: list[int] = []
    for tid in train_ids:
        cfg = train_mfst.launchers[tid]
        if cfg.preflight.heavy.gated:
            print(f"[chain_train_eval] WARN: {tid!r} has preflight.heavy.gated=true; "
                  "this tool submits it UNgated — gate via submit_with_preflights.sh "
                  "and chain the eval with --after instead.", file=sys.stderr)
        cmd = train_sh._build_sbatch_cmd(tid, cfg, [])
        cmd.insert(1, "--parsable")
        try:
            jid = _submit(cmd, dry_run=dry_run, fake_jobid=fake)
        except (RuntimeError, ValueError) as e:
            print(f"[chain_train_eval] ABORT: train submit failed for {tid!r}: {e}",
                  file=sys.stderr)
            if submitted:
                jids = " ".join(str(j) for _, j in submitted)
                print(f"[chain_train_eval] cleanup already-queued jobs with: scancel {jids}",
                      file=sys.stderr)
            return 1
        fake += 1
        train_jids.append(jid)
        submitted.append((f"train:{tid}", jid))

    dep = "afterok:" + ":".join(str(j) for j in train_jids + after)

    # ---- 2. submit eval job(s) gated on ALL train ids -----------------------
    for eid in eval_ids:
        cmd = eval_sh._build_sbatch_cmd(
            eid, eval_mfst.launchers[eid], [], shards=shards, dependency=dep
        )
        cmd.insert(1, "--parsable")
        try:
            jid = _submit(cmd, dry_run=dry_run, fake_jobid=fake)
        except (RuntimeError, ValueError) as e:
            print(f"[chain_train_eval] ABORT: eval submit failed for {eid!r}: {e}",
                  file=sys.stderr)
            jids = " ".join(str(j) for _, j in submitted)
            print(f"[chain_train_eval] cleanup already-queued jobs with: scancel {jids}",
                  file=sys.stderr)
            return 1
        fake += 1
        submitted.append((f"eval:{eid}", jid))

    # ---- summary -------------------------------------------------------------
    print("\nChain queued:" if not dry_run else "\nChain plan (dry-run, nothing submitted):")
    for label, jid in submitted:
        gate = f"  (waits {dep})" if label.startswith("eval:") else ""
        print(f"  {jid}  {label}{gate}")
    if submitted and not dry_run:
        print(f"Cancel all: scancel {' '.join(str(j) for _, j in submitted)}")
    if shards > 1:
        print(f"After the eval finishes, merge shards with merge_shards.py (N={shards}).")
    print("Reminder: eval success = results-JSON-exists, not Slurm state (job 16045963).")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", action="append", default=[], metavar="TRAIN_ID",
                   help="Train launcher id (repeatable; e.g. one per decent arm).")
    p.add_argument("--eval", action="append", default=[], metavar="EVAL_ID",
                   dest="evals",
                   help="Eval launcher id (repeatable). Each is gated afterok on "
                        "ALL --train jobs (+ --after ids).")
    p.add_argument("--shards", type=int, default=1,
                   help="Forwarded to the eval submit as --shards N (seed-sharded array).")
    p.add_argument("--after", action="append", type=int, default=[], metavar="JOBID",
                   help="Also gate the eval(s) on an ALREADY-submitted job id "
                        "(e.g. a pi0.5 train launched from openpi). Repeatable. "
                        "--train may be omitted when this is given.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the exact sbatch commands without submitting.")
    p.add_argument("--train-manifest", type=Path, default=None)
    p.add_argument("--eval-manifest", type=Path, default=None)
    args = p.parse_args(argv)
    return run_chain(
        args.train, args.evals,
        shards=args.shards, after=args.after, dry_run=args.dry_run,
        train_manifest=args.train_manifest, eval_manifest=args.eval_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
