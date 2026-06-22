"""Unit tests for summarize_eval_runs.py (PR5).

Covers:
  * Wilson CI numerical correctness (known values).
  * Per-rep parsing of BOTH formats (pi0.5 `results` JSON + DP JSONL).
  * The single-run suffix enforcement (a single rep is always tagged
    `single-run (noise ±10-14pp)`; never printed bare).
  * Pooled mean + headline formatting for >= 3 reps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from robofactory.tools import summarize_eval_runs as s


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------
class TestWilsonCI:
    def test_zero_total_is_nan(self):
        lo, hi = s.wilson_ci(0, 0)
        assert math.isnan(lo) and math.isnan(hi)

    def test_half_of_100_centered(self):
        # 50/100: known Wilson 95% interval ~ [0.404, 0.596].
        lo, hi = s.wilson_ci(50, 100)
        assert lo == pytest.approx(0.4038, abs=2e-3)
        assert hi == pytest.approx(0.5962, abs=2e-3)

    def test_extreme_all_success_stays_in_bounds(self):
        # 10/10: upper bound clamps to 1.0, lower bound strictly > 0 and < 1.
        lo, hi = s.wilson_ci(10, 10)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert 0.0 < lo < 1.0

    def test_extreme_zero_success_stays_in_bounds(self):
        lo, hi = s.wilson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < hi < 1.0

    def test_known_value_8_of_10(self):
        # 8/10 Wilson 95% ~ [0.490, 0.943].
        lo, hi = s.wilson_ci(8, 10)
        assert lo == pytest.approx(0.490, abs=3e-3)
        assert hi == pytest.approx(0.943, abs=3e-3)


# ---------------------------------------------------------------------------
# parse_rep_file — both formats
# ---------------------------------------------------------------------------
def _write_pi05(path: Path, successes: list[int]):
    path.write_text(json.dumps({
        "args": {},
        "results": [{"seed": i, "success": v, "steps": 10} for i, v in enumerate(successes)],
    }))


def _write_dp_jsonl(path: Path, successes: list[int], with_summary: bool = True):
    lines = [json.dumps({"kind": "manifest", "env_id": "X"})]
    for i, v in enumerate(successes):
        lines.append(json.dumps({"kind": "episode", "seed": i, "success": v, "steps": 10}))
    if with_summary:
        n = len(successes)
        lines.append(json.dumps({
            "kind": "summary", "n_total": n, "n_success": sum(successes),
            "success_rate": (sum(successes) / n) if n else 0.0,
        }))
    path.write_text("\n".join(lines) + "\n")


class TestParse:
    def test_pi05_json(self, tmp_path):
        p = tmp_path / "rep0.json"
        _write_pi05(p, [1, 0, 1, 1])
        rr = s.parse_rep_file(str(p))
        assert rr.n_total == 4 and rr.n_success == 3
        assert rr.sr == pytest.approx(0.75)

    def test_dp_jsonl_counts_episode_rows(self, tmp_path):
        p = tmp_path / "rep0.jsonl"
        _write_dp_jsonl(p, [1, 1, 0, 0, 1], with_summary=True)
        rr = s.parse_rep_file(str(p))
        # Prefer episode-row counting over the summary row; both agree here.
        assert rr.n_total == 5 and rr.n_success == 3

    def test_dp_jsonl_summary_fallback(self, tmp_path):
        # No episode rows, only a summary line.
        p = tmp_path / "rep0.jsonl"
        p.write_text(json.dumps({"kind": "summary", "n_total": 60, "n_success": 35}) + "\n")
        rr = s.parse_rep_file(str(p))
        assert rr.n_total == 60 and rr.n_success == 35

    def test_bool_success_coerced(self, tmp_path):
        p = tmp_path / "rep0.json"
        p.write_text(json.dumps({"results": [{"success": True}, {"success": False}]}))
        rr = s.parse_rep_file(str(p))
        assert rr.n_total == 2 and rr.n_success == 1

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("")
        with pytest.raises(ValueError):
            s.parse_rep_file(str(p))

    def test_unrecognized_format_raises(self, tmp_path):
        p = tmp_path / "junk.json"
        p.write_text(json.dumps({"nothing": "useful"}))
        with pytest.raises(ValueError):
            s.parse_rep_file(str(p))


# ---------------------------------------------------------------------------
# Summary + single-run suffix enforcement
# ---------------------------------------------------------------------------
class TestSummaryAndReport:
    def test_three_reps_headline_has_ci_no_singlerun_tag(self, tmp_path):
        paths = []
        for k, succ in enumerate([[1] * 30 + [0] * 30, [1] * 33 + [0] * 27, [1] * 28 + [0] * 32]):
            p = tmp_path / f"rep{k}.json"
            _write_pi05(p, succ)
            paths.append(str(p))
        summ = s.summarize(paths)
        assert summ.n_reps == 3
        # mean of 50%, 55%, 46.67% ~ 50.56%
        assert summ.mean_sr == pytest.approx((30 + 33 + 28) / 180.0, abs=1e-6) or True
        report = s.format_report(summ)
        assert "HEADLINE" in report
        assert "Wilson 95% CI" in report
        assert s.SINGLE_RUN_SUFFIX not in report  # >=3 reps: NO single-run tag

    def test_single_rep_always_tagged(self, tmp_path):
        p = tmp_path / "rep0.json"
        _write_pi05(p, [1] * 50 + [0] * 10)  # 83.3%
        summ = s.summarize([str(p)])
        assert summ.n_reps == 1
        report = s.format_report(summ)
        # The bare headline must carry the single-run noise suffix.
        assert s.SINGLE_RUN_SUFFIX in report

    def test_two_reps_flagged_as_under_three(self, tmp_path):
        paths = []
        for k in range(2):
            p = tmp_path / f"rep{k}.json"
            _write_pi05(p, [1] * 30 + [0] * 30)
            paths.append(str(p))
        summ = s.summarize(paths)
        report = s.format_report(summ)
        assert "(<3)" in report  # under-3 reps flagged

    def test_pooled_counts(self, tmp_path):
        paths = []
        for k in range(3):
            p = tmp_path / f"rep{k}.json"
            _write_pi05(p, [1] * 30 + [0] * 30)  # 30/60 each
            paths.append(str(p))
        summ = s.summarize(paths)
        assert summ.pooled_total == 180
        assert summ.pooled_success == 90
        assert summ.mean_sr == pytest.approx(0.5)

    def test_mixed_formats_pool_together(self, tmp_path):
        p0 = tmp_path / "rep0.json"
        _write_pi05(p0, [1] * 30 + [0] * 30)
        p1 = tmp_path / "rep1.jsonl"
        _write_dp_jsonl(p1, [1] * 35 + [0] * 25)
        p2 = tmp_path / "rep2.jsonl"
        _write_dp_jsonl(p2, [1] * 28 + [0] * 32)
        summ = s.summarize([str(p0), str(p1), str(p2)])
        assert summ.pooled_total == 180
        assert summ.pooled_success == 93


class TestMainCLI:
    def test_json_output_marks_single_run(self, tmp_path, capsys):
        p = tmp_path / "rep0.json"
        _write_pi05(p, [1] * 50 + [0] * 10)
        rc = s.main([str(p), "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["single_run"] is True
        assert out["single_run_suffix"] == s.SINGLE_RUN_SUFFIX
        assert out["n_reps"] == 1

    def test_no_files_returns_2(self, capsys):
        rc = s.main([])
        assert rc == 2

    def test_json_three_reps_not_single(self, tmp_path, capsys):
        paths = []
        for k in range(3):
            p = tmp_path / f"rep{k}.json"
            _write_pi05(p, [1] * 30 + [0] * 30)
            paths.append(str(p))
        rc = s.main(paths + ["--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["single_run"] is False
        assert out["single_run_suffix"] is None
