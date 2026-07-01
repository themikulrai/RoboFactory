"""Tests for chain_train_eval (train -> eval afterok chaining).

No real sbatch is ever invoked: subprocess.run is monkeypatched with a recorder
that returns canned --parsable job ids (or failures). Launcher ids come from the
REAL manifests so the builder paths exercised here are the ones production uses.

Covers:
  * parse_sbatch_jobid over --parsable / human / cluster-suffixed / garbage output
  * dependency spec construction (one train, decent pair, extra --after ids)
  * --shards composition on the chained eval submit
  * --dry-run prints the exact sbatch commands and calls NO subprocess
  * abort semantics: failed train submit stops the chain before the eval submit
    and prints a scancel cleanup hint for jobs already queued
  * unknown launcher ids abort BEFORE any submission
"""
from __future__ import annotations

import subprocess

import pytest

from robofactory.scripts.canonical import chain_train_eval as ct


# Two NON-gated train launchers: the happy paths exercise the real train submit
# builder without tripping the gated-launcher guard (which is covered separately in
# TestGatedGuard). Both are single-arm DP retrains, enough to test the afterok spec.
TRAIN_A = "pm_d1_ep300_crop"
TRAIN_B = "pm_d1_ep300_r3m"
# A gated launcher (preflight.heavy.gated + needs_overfit): must be refused ungated.
GATED_TRAIN = "tsc_d2_ep300_in1k_a0"
EVAL_ID = "pm_dp_in1k"


class _FakeSbatch:
    """Recorder standing in for subprocess.run; yields canned sbatch results."""

    def __init__(self, jobids=(111, 222, 333), fail_at=None):
        self.jobids = list(jobids)
        self.fail_at = fail_at  # 0-based call index that returns rc=1
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        idx = len(self.calls)
        self.calls.append(list(cmd))
        if self.fail_at is not None and idx == self.fail_at:
            return subprocess.CompletedProcess(cmd, 1, stdout="",
                                               stderr="sbatch: error: invalid partition")
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{self.jobids[idx]}\n", stderr="")


class TestParseSbatchJobid:
    def test_parsable_plain(self):
        assert ct.parse_sbatch_jobid("12345\n") == 12345

    def test_parsable_with_cluster_suffix(self):
        assert ct.parse_sbatch_jobid("12345;iris\n") == 12345

    def test_human_format(self):
        assert ct.parse_sbatch_jobid("Submitted batch job 16045963\n") == 16045963

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            ct.parse_sbatch_jobid("sbatch: error: something\n")


class TestChainHappyPath:
    def test_single_train_single_eval_dependency(self, monkeypatch, capsys):
        fake = _FakeSbatch(jobids=(111, 222))
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A], [EVAL_ID])
        assert rc == 0
        assert len(fake.calls) == 2
        train_cmd, eval_cmd = fake.calls
        # Both submits are machine-readable.
        assert train_cmd[:2] == ["sbatch", "--parsable"]
        assert eval_cmd[:2] == ["sbatch", "--parsable"]
        # Train cmd carries NO dependency; eval waits afterok on the train jid.
        assert "--dependency" not in train_cmd
        assert eval_cmd[eval_cmd.index("--dependency") + 1] == "afterok:111"
        assert "--kill-on-invalid-dep=yes" in eval_cmd
        out = capsys.readouterr().out
        assert "scancel 111 222" in out  # cancel-all hint

    def test_decent_pair_gates_eval_on_both(self, monkeypatch):
        fake = _FakeSbatch(jobids=(111, 222, 333))
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A, TRAIN_B], [EVAL_ID], shards=4)
        assert rc == 0
        assert len(fake.calls) == 3
        eval_cmd = fake.calls[2]
        assert eval_cmd[eval_cmd.index("--dependency") + 1] == "afterok:111:222"
        # --shards composed onto the chained eval: array + --num-shards.
        assert eval_cmd[eval_cmd.index("--array") + 1] == "0-3"
        assert eval_cmd[eval_cmd.index("--num-shards") + 1] == "4"

    def test_after_ids_appended_to_spec(self, monkeypatch):
        # Chain onto already-submitted jobs (e.g. pi0.5 trains from openpi):
        # --after ids join the afterok spec after the fresh train jids.
        fake = _FakeSbatch(jobids=(111, 222))
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A], [EVAL_ID], after=[999, 1000])
        assert rc == 0
        eval_cmd = fake.calls[1]
        assert eval_cmd[eval_cmd.index("--dependency") + 1] == "afterok:111:999:1000"

    def test_after_only_no_train(self, monkeypatch):
        fake = _FakeSbatch(jobids=(111,))
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([], [EVAL_ID], after=[4242])
        assert rc == 0
        assert len(fake.calls) == 1  # eval only
        eval_cmd = fake.calls[0]
        assert eval_cmd[eval_cmd.index("--dependency") + 1] == "afterok:4242"


class TestDryRun:
    def test_dry_run_prints_cmds_and_never_submits(self, monkeypatch, capsys):
        def _boom(*a, **k):  # any subprocess call is a test failure
            raise AssertionError("subprocess.run called during --dry-run")
        monkeypatch.setattr(ct.subprocess, "run", _boom)
        rc = ct.run_chain([TRAIN_A, TRAIN_B], [EVAL_ID], shards=4, dry_run=True)
        assert rc == 0
        out = capsys.readouterr().out
        dry_lines = [ln for ln in out.splitlines() if ln.startswith("DRYRUN: sbatch")]
        assert len(dry_lines) == 3  # 2 trains + 1 eval
        # Every printed command is a real sbatch CLI with --parsable.
        assert all("--parsable" in ln for ln in dry_lines)
        # The eval line shows the chain shape using the fake dry-run job ids.
        assert "--dependency afterok:900001:900002" in dry_lines[2]
        assert "--array 0-3" in dry_lines[2]
        # Results-JSON caveat is surfaced to the operator (job 16045963).
        assert "16045963" in out


class TestAbortSemantics:
    def test_first_train_submit_failure_aborts_before_anything_else(
        self, monkeypatch, capsys
    ):
        fake = _FakeSbatch(fail_at=0)
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A, TRAIN_B], [EVAL_ID])
        assert rc == 1
        assert len(fake.calls) == 1  # second train + eval never submitted
        err = capsys.readouterr().err
        assert "ABORT" in err
        # Nothing was queued before the failure -> no scancel hint needed.
        assert "scancel" not in err

    def test_second_train_failure_prints_scancel_hint_for_first(
        self, monkeypatch, capsys
    ):
        fake = _FakeSbatch(jobids=(111,), fail_at=1)
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A, TRAIN_B], [EVAL_ID])
        assert rc == 1
        assert len(fake.calls) == 2  # eval never submitted
        err = capsys.readouterr().err
        assert "scancel 111" in err

    def test_eval_submit_failure_prints_scancel_hint_for_trains(
        self, monkeypatch, capsys
    ):
        fake = _FakeSbatch(jobids=(111, 222), fail_at=2)
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A, TRAIN_B], [EVAL_ID])
        assert rc == 1
        err = capsys.readouterr().err
        assert "scancel 111 222" in err

    def test_unparsable_jobid_aborts(self, monkeypatch, capsys):
        class _Garbage(_FakeSbatch):
            def __call__(self, cmd, capture_output=True, text=True, timeout=None):
                self.calls.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, stdout="???\n", stderr="")
        fake = _Garbage()
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([TRAIN_A], [EVAL_ID])
        assert rc == 1
        assert len(fake.calls) == 1  # eval never submitted on unparsable train jid

    def test_sbatch_timeout_aborts_with_scancel_hint(self, monkeypatch, capsys):
        """A hung sbatch (TimeoutExpired) aborts the chain and prints a scancel hint."""
        calls = {"n": 0}

        def _timeout(cmd, capture_output=True, text=True, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return subprocess.CompletedProcess(cmd, 0, stdout="111\n", stderr="")
            raise subprocess.TimeoutExpired(cmd, timeout or 60)

        monkeypatch.setattr(ct.subprocess, "run", _timeout)
        # First train submits (jid 111); second train's sbatch hangs -> abort.
        rc = ct.run_chain([TRAIN_A, TRAIN_B], [EVAL_ID])
        assert rc == 1
        err = capsys.readouterr().err
        assert "ABORT" in err
        assert "60s" in err            # the timeout message is surfaced
        assert "scancel 111" in err    # cleanup hint for the already-queued train


class TestGatedGuard:
    """Item 1: a gated train launcher (preflight.heavy.gated/needs_overfit) is refused
    UNgated (nonzero exit, zero sbatch calls) unless --force-ungated is passed."""

    def test_gated_train_without_flag_submits_nothing(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError("subprocess.run called for a gated launcher")
        monkeypatch.setattr(ct.subprocess, "run", _boom)
        rc = ct.run_chain([GATED_TRAIN], [EVAL_ID])
        assert rc == 2
        err = capsys.readouterr().err
        assert "REFUSING" in err and GATED_TRAIN in err

    def test_gated_mixed_with_nongated_still_refused(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError("subprocess.run called despite a gated launcher")
        monkeypatch.setattr(ct.subprocess, "run", _boom)
        # One non-gated + one gated -> whole chain refused, nothing submitted.
        rc = ct.run_chain([TRAIN_A, GATED_TRAIN], [EVAL_ID])
        assert rc == 2
        assert GATED_TRAIN in capsys.readouterr().err

    def test_force_ungated_proceeds(self, monkeypatch, capsys):
        fake = _FakeSbatch(jobids=(111, 222))
        monkeypatch.setattr(ct.subprocess, "run", fake)
        rc = ct.run_chain([GATED_TRAIN], [EVAL_ID], force_ungated=True)
        assert rc == 0
        assert len(fake.calls) == 2  # gated train + eval both submitted
        err = capsys.readouterr().err
        assert "force-ungated" in err.lower()  # the skip-preflights warning fired


class TestValidationBeforeSubmission:
    def test_unknown_train_id_submits_nothing(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError("subprocess.run called despite bad launcher id")
        monkeypatch.setattr(ct.subprocess, "run", _boom)
        rc = ct.run_chain(["definitely_not_a_launcher"], [EVAL_ID])
        assert rc == 2
        assert "unknown train launcher" in capsys.readouterr().err

    def test_unknown_eval_id_submits_nothing(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError("subprocess.run called despite bad launcher id")
        monkeypatch.setattr(ct.subprocess, "run", _boom)
        rc = ct.run_chain([TRAIN_A], ["definitely_not_a_launcher"])
        assert rc == 2
        assert "unknown eval launcher" in capsys.readouterr().err

    def test_no_eval_is_an_error(self):
        assert ct.run_chain([TRAIN_A], []) == 2

    def test_no_train_and_no_after_is_an_error(self):
        assert ct.run_chain([], [EVAL_ID]) == 2


class TestMainCli:
    def test_cli_maps_flags_to_run_chain(self, monkeypatch):
        seen = {}
        def _fake_run_chain(train_ids, eval_ids, *, shards, after, dry_run,
                            force_ungated, train_manifest, eval_manifest):
            seen.update(train=train_ids, evals=eval_ids, shards=shards,
                        after=after, dry_run=dry_run, force_ungated=force_ungated)
            return 0
        monkeypatch.setattr(ct, "run_chain", _fake_run_chain)
        rc = ct.main(["--train", TRAIN_A, "--train", TRAIN_B,
                      "--eval", EVAL_ID, "--shards", "4",
                      "--after", "999", "--dry-run", "--force-ungated"])
        assert rc == 0
        assert seen == {"train": [TRAIN_A, TRAIN_B], "evals": [EVAL_ID],
                        "shards": 4, "after": [999], "dry_run": True,
                        "force_ungated": True}
