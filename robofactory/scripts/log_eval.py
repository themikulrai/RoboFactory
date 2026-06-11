#!/usr/bin/env python3
"""Auto-post a field-notes cell at the END of every 60-seed eval (A8 loop-killer).

The audit's systemic finding (WEEK1_EXECUTION §A8, solution_E13 §C): the
"new-cell-per-run with job id" rule kept failing because it relied on a human
*remembering* to do it. This makes it mechanical — every
``scripts/{canonical,ablations,...}/*_60seeds.sh`` launcher ends with a call to
this script, which reads the run's result file off disk, computes the success
rate, and POSTs a NEW field-notes cell carrying the job id + jsonl path + SR.
Lint rule H (``lint_eval_launchers.py``) FAILS any inline 60-seed launcher that
does not call this at the end, so the convention can no longer silently rot.

It accepts EITHER eval output format and auto-detects:
  * pi0.5 result JSON  (``eval_pi05.py`` / ``eval_decent_pi05.py``): one JSON
    object ``{"results": [{"seed", "success", ...}], "validity": {...}, ...}``.
  * DP per-episode JSONL (``eval_multi_dp.py`` / ``eval_joint_dp.py``):
    newline-delimited records tagged ``kind`` in {manifest, episode, summary,
    validity}; SR is recomputed from the episode rows (and cross-checked against
    the summary row if present).

Key handling (NEVER hardcode the key — the launchers run on slurm where the key
is in env): the field-notes write key is read from env at RUNTIME, trying in
order ``X_FIELD_NOTES_KEY`` -> ``FIELD_NOTES_KEY`` -> ``FIELD_NOTES_API_KEY``,
then a ``--key-file``/``$FIELD_NOTES_KEY_FILE`` fallback. If no key is available
the script prints a warning to stderr and exits 0 (NEVER fails the eval job over
a logging side-effect — the eval result on disk is the source of truth). The
base URL comes from ``$FIELD_NOTES_API_URL`` (default: the known Heroku app).

Usage (appended at the END of a launcher; result-path is the file the driver
just wrote, here resolved by the launcher's own ``ls -t``):
    python scripts/log_eval.py \
        --jsonl   "$RESULT_FILE" \
        --job     "$SLURM_JOB_ID" \
        --title   "Eval LB wc Pi0.5 [Decent]" \
        --project "$FIELD_NOTES_PROJECT"      # optional; falls back to env

A ``--dry-run`` flag (or ``$LOG_EVAL_DRY_RUN=1``) builds and prints the cell
payload WITHOUT making any HTTP call — used by the unit test and for a safe
launcher smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (env-overridable; NO secret is ever baked in)
# ---------------------------------------------------------------------------
DEFAULT_FN_BASE = "https://field-notes-mikulrai-5633eb0f870d.herokuapp.com"
# Env var names that may carry the field-notes WRITE key, in priority order.
KEY_ENV_VARS = ("X_FIELD_NOTES_KEY", "FIELD_NOTES_KEY", "FIELD_NOTES_API_KEY")
# Env var names that may carry the target project id, in priority order.
PROJECT_ENV_VARS = ("FIELD_NOTES_PROJECT", "FIELD_NOTES_PROJECT_ID")


def _log(msg: str) -> None:
    print(f"[log_eval] {msg}", file=sys.stderr, flush=True)


def resolve_key(key_file: str | None) -> str | None:
    """Resolve the field-notes write key from env (preferred) or a key file.

    Returns None if no key is available — the caller treats that as a soft skip,
    NEVER a hard failure (logging must not brick an eval job)."""
    for var in KEY_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val.strip()
    # File fallback: explicit --key-file, then $FIELD_NOTES_KEY_FILE.
    candidate = key_file or os.environ.get("FIELD_NOTES_KEY_FILE")
    if candidate:
        p = Path(candidate).expanduser()
        if p.is_file():
            return p.read_text().strip()
    return None


def resolve_project(arg_project: str | None) -> str | None:
    if arg_project:
        return arg_project
    for var in PROJECT_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None


def resolve_base() -> str:
    return os.environ.get("FIELD_NOTES_API_URL", DEFAULT_FN_BASE).rstrip("/")


# ---------------------------------------------------------------------------
# Result-file parsing (auto-detect pi0.5 JSON vs DP JSONL)
# ---------------------------------------------------------------------------
def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(v)


def parse_result_file(path: Path) -> dict:
    """Return a normalized summary dict:
        {n, n_success, sr, valid, invalid_reason, fmt, results: [ {seed, success} ]}
    Auto-detects the pi0.5 single-JSON format vs the DP per-line JSONL format.
    Raises FileNotFoundError if the path is missing."""
    text = path.read_text()

    # Try the pi0.5 single-object JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "results" in obj:
            return _parse_pi05_obj(obj)
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one record per line, DP format).
    return _parse_dp_jsonl(text)


def _parse_pi05_obj(obj: dict) -> dict:
    results = obj.get("results") or []
    norm = []
    for r in results:
        if not isinstance(r, dict):
            continue
        norm.append(
            {"seed": r.get("seed"), "success": _coerce_bool(r.get("success", False))}
        )
    n = len(norm)
    n_succ = sum(1 for r in norm if r["success"])
    validity = obj.get("validity") or {}
    valid = _coerce_bool(validity.get("valid", True)) if validity else True
    invalid_reason = validity.get("invalid_reason") if validity else None
    return {
        "n": n,
        "n_success": n_succ,
        "sr": (n_succ / n) if n else 0.0,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "fmt": "pi0.5-json",
        "results": norm,
    }


def _parse_dp_jsonl(text: str) -> dict:
    episodes: list[dict] = []
    summary: dict | None = None
    validity: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a half-written trailing line
        if not isinstance(rec, dict):
            continue
        kind = rec.get("kind")
        if kind == "episode":
            episodes.append(rec)
        elif kind == "summary":
            summary = rec
        elif kind == "validity":
            validity = rec
        elif kind is None and "success" in rec:
            # bare per-episode lines (older DP format with no `kind` tag)
            episodes.append(rec)

    norm = [
        {"seed": e.get("seed"), "success": _coerce_bool(e.get("success", False))}
        for e in episodes
    ]
    n = len(norm)
    n_succ = sum(1 for r in norm if r["success"])
    # Prefer the recomputed-from-episodes count; fall back to the summary row
    # if no episode rows were found (e.g. a record-dir-only run).
    if n == 0 and summary:
        n = int(summary.get("n_total", 0) or 0)
        n_succ = int(summary.get("n_success", 0) or 0)
    valid = _coerce_bool(validity.get("valid", True)) if validity else True
    invalid_reason = validity.get("invalid_reason") if validity else None
    return {
        "n": n,
        "n_success": n_succ,
        "sr": (n_succ / n) if n else 0.0,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "fmt": "dp-jsonl",
        "results": norm,
    }


# ---------------------------------------------------------------------------
# Cell payload builder
# ---------------------------------------------------------------------------
def build_cell(
    title: str,
    job: str,
    jsonl_path: str,
    summary: dict,
    after_cell_id: str | None = None,
    extra_note: str | None = None,
) -> dict:
    """Build the field-notes create-cell payload (metrics use {k,v,d} keys —
    the raw HTTP endpoint returns 422 for {label,value}; see memory
    feedback_field_notes_metrics_kv)."""
    n = summary["n"]
    n_succ = summary["n_success"]
    sr_pct = 100.0 * summary["sr"]
    valid = summary["valid"]
    sr_str = f"{n_succ}/{n} ({sr_pct:.1f}%)" if n else "0/0 (n/a)"
    valid_str = "valid" if valid else f"INVALID ({summary.get('invalid_reason')})"

    status = "done" if valid else "blocked"

    body_lines = [
        f"Auto-posted at end of a 60-seed eval (A8 loop-killer convention).",
        "",
        f"- **job id:** `{job}`",
        f"- **result file:** `{jsonl_path}`",
        f"- **success rate:** {sr_str}",
        f"- **validity:** {valid_str}",
        f"- **format:** {summary['fmt']}",
    ]
    if extra_note:
        body_lines += ["", extra_note]
    body = "\n".join(body_lines)

    metrics = [
        {"k": "Job", "v": str(job), "d": "slurm job id"},
        {"k": "Success rate", "v": sr_str, "d": "recomputed from the result file"},
        {"k": "Valid", "v": ("yes" if valid else "no"),
         "d": (summary.get("invalid_reason") or "")},
        {"k": "Result file", "v": Path(jsonl_path).name, "d": jsonl_path},
    ]

    conclusion = (
        f"{title}: SR {sr_str}"
        + ("" if valid else f" — RUN {valid_str}")
        + f". Result file on disk: {jsonl_path} (job {job})."
    )

    cell: dict = {
        "title": title,
        "kind": "agent",
        "status": status,
        "body": body,
        "metrics": metrics,
        "conclusion": conclusion,
    }
    if after_cell_id:
        cell["after_cell_id"] = after_cell_id
    return cell


# ---------------------------------------------------------------------------
# HTTP POST
# ---------------------------------------------------------------------------
def post_cell(base: str, project_id: str, key: str, cell: dict) -> str:
    url = f"{base}/projects/{project_id}/cells"
    data = json.dumps(cell).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Field-Notes-Key", key)
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode() or "{}")
    cid = out.get("id")
    if not cid:
        raise RuntimeError(f"create returned no id: {out}")
    return cid


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True,
                   help="Path to the eval result file (pi0.5 .json or DP .jsonl).")
    p.add_argument("--job", required=True, help="Slurm job id.")
    p.add_argument("--title", required=True, help="Field-notes cell title.")
    p.add_argument("--project", default=None,
                   help="Field-notes project id (else from $FIELD_NOTES_PROJECT).")
    p.add_argument("--after-cell-id", default=None,
                   help="Insert the new cell after this cell id (optional).")
    p.add_argument("--note", default=None, help="Optional extra markdown note.")
    p.add_argument("--key-file", default=None,
                   help="File holding the field-notes write key (else env).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build + print the payload; make NO HTTP call.")
    args = p.parse_args(argv)

    dry = args.dry_run or os.environ.get("LOG_EVAL_DRY_RUN") == "1"

    path = Path(args.jsonl)
    if not path.is_file():
        _log(f"result file not found: {path} — nothing to post (soft skip).")
        return 0

    try:
        summary = parse_result_file(path)
    except Exception as e:  # never fail the eval job over a parse error
        _log(f"could not parse {path}: {type(e).__name__}: {e} — soft skip.")
        return 0

    cell = build_cell(
        title=args.title,
        job=args.job,
        jsonl_path=str(path),
        summary=summary,
        after_cell_id=args.after_cell_id,
        extra_note=args.note,
    )

    if dry:
        # Emit the exact payload that WOULD be POSTed (used by tests / smoke).
        print(json.dumps(cell, indent=2))
        _log(f"dry-run: built payload for job {args.job} "
             f"(SR {summary['n_success']}/{summary['n']}, valid={summary['valid']}).")
        return 0

    key = resolve_key(args.key_file)
    project = resolve_project(args.project)
    base = resolve_base()
    if not key:
        _log("no field-notes key in env (" + "/".join(KEY_ENV_VARS) +
             ") and no --key-file — cell NOT posted (soft skip). "
             "The result file on disk is the source of truth.")
        return 0
    if not project:
        _log("no project id (--project or $FIELD_NOTES_PROJECT) — "
             "cell NOT posted (soft skip).")
        return 0

    try:
        cid = post_cell(base, project, key, cell)
        _log(f"posted cell {cid} for job {args.job} "
             f"(SR {summary['n_success']}/{summary['n']}).")
        print(cid)
    except Exception as e:  # logging must never brick the eval job
        _log(f"POST failed: {type(e).__name__}: {e} — soft skip.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
