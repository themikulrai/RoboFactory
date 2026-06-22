"""Summarize N_REPEATS eval result files into a mean ± Wilson-95%-CI headline (PR5).

Why this exists
---------------
A single 60-seed eval run has a binomial-noise band of roughly ±10–14 pp on the
success rate (84% ± 14 on one seed × 25 — A9). Reporting one run's number as a
finding manufactured a whole class of phantom "reproduction failures" and "fix
regressions" in this project (digest A T10). The honest contract is:

    headline number = mean ± Wilson 95% CI pooled over >= 3 repeat runs;
    single-run deltas < 15 pp are NOT findings.

This tool pools per-episode successes across reps and reports the Wilson interval.
It REFUSES to print a bare single-run number — a single run is always tagged
`single-run (noise ±10-14pp)` so it can never be mistaken for a stable estimate.

Bitwise determinism is NOT promised across GPUs (no cudnn.deterministic; A40/A5000/
A6000 kernels differ — A9 §5). PR5 pins the sampler RNG per episode so the SAME seed
on the SAME node/GPU reproduces; repeats quantify the residual cross-run spread.

Input formats (auto-detected per file)
--------------------------------------
* pi0.5 driver JSON:  a dict with top-level ``results`` = list of per-episode dicts,
  each having a ``success`` (0/1/bool) field.
* DP driver JSONL:    newline-delimited objects; per-episode rows ``{"kind":"episode",
  "success":...}`` and an optional ``{"kind":"summary","n_total":..,"n_success":..}``.
  We prefer counting episode rows; fall back to the summary row.

Usage
-----
    python -m robofactory.tools.summarize_eval_runs \
        eval_foo_rep0.json eval_foo_rep1.json eval_foo_rep2.json
    python -m robofactory.tools.summarize_eval_runs --glob '/iris/.../rep*/eval_*.json'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Z for a 95% two-sided interval.
_Z95 = 1.959963984540054

# The honest single-run noise band (binomial, ~25-60 seeds) — see module docstring.
SINGLE_RUN_SUFFIX = "single-run (noise ±10-14pp)"


@dataclass
class RepResult:
    """One repeat: how many episodes succeeded out of how many ran."""

    path: str
    n_total: int
    n_success: int

    @property
    def sr(self) -> float:
        return (self.n_success / self.n_total) if self.n_total else float("nan")


def wilson_ci(n_success: int, n_total: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lo, hi) in [0, 1]. Wilson (not normal-approx) because it stays inside
    [0,1] and behaves at the extremes (0/N, N/N) where the normal interval is wrong.
    """
    if n_total == 0:
        return (float("nan"), float("nan"))
    p = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p + z2 / (2 * n_total)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n_total)) / n_total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _coerce_success(v) -> int:
    """Map a success field (bool / 0-1 int / '1') to a 0/1 int."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v else 0
    if isinstance(v, str):
        return 1 if v.strip() in ("1", "true", "True") else 0
    raise ValueError(f"unrecognized success value: {v!r}")


def parse_rep_file(path: str) -> RepResult:
    """Parse one rep file (pi0.5 JSON or DP JSONL) into (n_total, n_success)."""
    text = Path(path).read_text().strip()
    if not text:
        raise ValueError(f"{path}: empty file")

    # Try whole-file JSON first (pi0.5 format).
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict) and "results" in obj:
        results = obj["results"]
        if not isinstance(results, list):
            raise ValueError(f"{path}: 'results' is not a list")
        n_total = len(results)
        n_success = sum(_coerce_success(r.get("success", 0)) for r in results)
        return RepResult(path=path, n_total=n_total, n_success=n_success)

    if isinstance(obj, dict) and {"n_total", "n_success"} <= obj.keys():
        # A bare DP summary dict (single JSON object, not JSONL).
        return RepResult(path=path, n_total=int(obj["n_total"]), n_success=int(obj["n_success"]))

    # Fall back to JSONL (DP driver writes manifest/episode/summary lines).
    episodes = 0
    succ = 0
    summary_row: Optional[dict] = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind == "episode":
            episodes += 1
            succ += _coerce_success(row.get("success", 0))
        elif kind == "summary":
            summary_row = row
    if episodes > 0:
        return RepResult(path=path, n_total=episodes, n_success=succ)
    if summary_row is not None and {"n_total", "n_success"} <= summary_row.keys():
        return RepResult(
            path=path,
            n_total=int(summary_row["n_total"]),
            n_success=int(summary_row["n_success"]),
        )
    raise ValueError(
        f"{path}: could not extract per-episode successes "
        "(no pi0.5 'results' list, no DP episode/summary rows)"
    )


@dataclass
class Summary:
    reps: list[RepResult]

    @property
    def n_reps(self) -> int:
        return len(self.reps)

    @property
    def pooled_total(self) -> int:
        return sum(r.n_total for r in self.reps)

    @property
    def pooled_success(self) -> int:
        return sum(r.n_success for r in self.reps)

    @property
    def mean_sr(self) -> float:
        """Mean of per-rep SRs (each rep weighted equally). Falls back to pooled SR
        if reps have differing episode counts? No — we report the simple per-rep mean
        as the headline, and pool only for the CI width."""
        srs = [r.sr for r in self.reps if r.n_total]
        return sum(srs) / len(srs) if srs else float("nan")

    @property
    def pooled_ci(self) -> tuple[float, float]:
        return wilson_ci(self.pooled_success, self.pooled_total)


def summarize(paths: list[str]) -> Summary:
    if not paths:
        raise ValueError("no result files given")
    reps = [parse_rep_file(p) for p in paths]
    return Summary(reps=reps)


def format_report(summary: Summary) -> str:
    """Human-readable report. A single rep is ALWAYS suffixed single-run(noise)."""
    lines: list[str] = []
    lines.append("=== eval repeat summary ===")
    for i, r in enumerate(summary.reps):
        lines.append(
            f"  rep{i}: {r.n_success}/{r.n_total} = {100.0 * r.sr:.1f}%  "
            f"({Path(r.path).name})"
        )
    lo, hi = summary.pooled_ci
    mean_pct = 100.0 * summary.mean_sr
    if summary.n_reps < 3:
        # REFUSE to present a bare headline; tag it so it can never read as a finding.
        suffix = SINGLE_RUN_SUFFIX if summary.n_reps == 1 else f"only {summary.n_reps} reps (<3)"
        lines.append(
            f"HEADLINE: {mean_pct:.1f}%  [{suffix}]  "
            f"(pooled Wilson 95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f}%, "
            f"{summary.pooled_success}/{summary.pooled_total} pooled)"
        )
    else:
        lines.append(
            f"HEADLINE: {mean_pct:.1f}%  "
            f"(mean of {summary.n_reps} reps; pooled Wilson 95% CI "
            f"{100.0 * lo:.1f}-{100.0 * hi:.1f}%, "
            f"{summary.pooled_success}/{summary.pooled_total} pooled episodes)"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="*", help="per-rep result JSON/JSONL files")
    p.add_argument("--glob", default=None, help="glob pattern for rep files (alternative to listing)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    args = p.parse_args(argv)

    paths = list(args.files)
    if args.glob:
        paths.extend(sorted(_glob.glob(args.glob)))
    if not paths:
        print("[summarize_eval_runs] no result files given (pass files or --glob)", file=sys.stderr)
        return 2

    summary = summarize(paths)
    if args.json:
        lo, hi = summary.pooled_ci
        out = {
            "n_reps": summary.n_reps,
            "per_rep": [
                {"path": r.path, "n_total": r.n_total, "n_success": r.n_success, "sr": r.sr}
                for r in summary.reps
            ],
            "mean_sr": summary.mean_sr,
            "pooled_total": summary.pooled_total,
            "pooled_success": summary.pooled_success,
            "wilson_ci_lo": lo,
            "wilson_ci_hi": hi,
            "single_run": summary.n_reps < 3,
            "single_run_suffix": SINGLE_RUN_SUFFIX if summary.n_reps == 1 else None,
        }
        print(json.dumps(out, indent=2))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
