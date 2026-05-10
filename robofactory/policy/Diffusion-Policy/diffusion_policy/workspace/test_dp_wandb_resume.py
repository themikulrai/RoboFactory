"""Pytest-style tests for `_init_wandb_with_resume` in robotworkspace.py.

CPU-only: stubs `wandb.init` and `wandb.run`; no network or model loads.

Run from `/iris/u/mikulrai/projects/RoboFactory/`:
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \\
        robofactory/policy/Diffusion-Policy/diffusion_policy/workspace/test_dp_wandb_resume.py -v
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

# Diffusion-Policy has a hyphen in the dirname; add it to sys.path so dotted
# imports of `diffusion_policy.*` resolve.
_DP_DIR = Path(__file__).resolve().parents[2]
if str(_DP_DIR) not in sys.path:
    sys.path.insert(0, str(_DP_DIR))

pytest.importorskip("wandb", reason="wandb required for resume tests")
from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy.workspace import robotworkspace  # noqa: E402


def _cfg(resume=True, run_id=None):
    """Mirrors cfg.logging shape from configs/robot_dp.yaml."""
    return OmegaConf.create({
        "logging": {"project": "test-proj", "resume": resume, "mode": "online",
                    "name": "test-run", "tags": ["t"], "id": run_id, "group": None},
        "training": {"seed": 0},
    })


def _patch(returned_id):
    fake_run = mock.MagicMock()
    fake_run.id = returned_id
    init_mock = mock.MagicMock(return_value=fake_run)
    return fake_run, init_mock


def _run(tmp_path, cfg, returned_id):
    fake_run, init_mock = _patch(returned_id)
    with mock.patch.object(robotworkspace.wandb, "init", init_mock), \
         mock.patch.object(robotworkspace.wandb, "run", fake_run):
        robotworkspace._init_wandb_with_resume(tmp_path, cfg)
    return init_mock


# ---------------------------------------------------------------------------

def test_first_launch_writes_id_file(tmp_path):
    """No wandb_id.txt -> init WITHOUT resume='must'; id persisted to disk."""
    init_mock = _run(tmp_path, _cfg(resume=True), "deadbeef")
    assert init_mock.call_count == 1
    kwargs = init_mock.call_args.kwargs
    assert "id" not in kwargs
    # cfg.logging.resume=True is normalised to 'allow' on a fresh launch.
    assert kwargs["resume"] == "allow"
    assert (tmp_path / "wandb_id.txt").read_text().strip() == "deadbeef"


def test_resume_reads_id_from_file_and_passes_resume_must(tmp_path):
    """Existing wandb_id.txt='abc123' -> init id='abc123', resume='must'."""
    (tmp_path / "wandb_id.txt").write_text("abc123\n")
    init_mock = _run(tmp_path, _cfg(resume=True), "abc123")
    kwargs = init_mock.call_args.kwargs
    assert kwargs["id"] == "abc123"
    assert kwargs["resume"] == "must"


def test_idempotent_double_call(tmp_path):
    """Second call after first-launch reads saved id and resumes (no fresh run)."""
    cfg = _cfg(resume=True)
    first = _run(tmp_path, cfg, "run_42")
    assert "id" not in first.call_args.kwargs
    second = _run(tmp_path, cfg, "run_42")
    assert second.call_args.kwargs["id"] == "run_42"
    assert second.call_args.kwargs["resume"] == "must"


def test_id_file_atomic_write_documented(tmp_path):
    """Document write strategy: impl uses Path.write_text() (NOT tmp+rename).
    Sentinel: if a `.tmp` sibling appears, the impl gained atomic-write
    semantics and this test should be updated to verify the rename."""
    _run(tmp_path, _cfg(resume=True), "atomic_check_id")
    assert (tmp_path / "wandb_id.txt").read_text().strip() == "atomic_check_id"
    siblings = [p.name for p in tmp_path.iterdir()]
    assert not any(s.endswith(".tmp") for s in siblings)


@pytest.mark.parametrize("contents,desc", [
    ("", "empty"),
    ("   \n\t  \n", "whitespace-only"),
    ("  abc123  \n\n", "padded"),
    ("abc123\nstray-second-line\n", "multi-line"),
])
def test_resume_with_corrupt_id_file(tmp_path, contents, desc):
    """Impl: `read_text().strip() or None`.
    - empty/whitespace -> falsy -> fresh init.
    - padded -> stripped, treated as resume.
    - multi-line -> .strip() leaves first+second line; passed verbatim."""
    (tmp_path / "wandb_id.txt").write_text(contents)
    init_mock = _run(tmp_path, _cfg(resume=True), "fresh_xxx")
    kwargs = init_mock.call_args.kwargs
    stripped = contents.strip()
    if not stripped:
        assert "id" not in kwargs, f"{desc}: empty/whitespace must yield fresh launch"
        assert kwargs["resume"] == "allow"
    else:
        assert kwargs["id"] == stripped, f"{desc}: id should equal stripped contents"
        assert kwargs["resume"] == "must"


def test_concurrent_resume_race_two_threads(tmp_path):
    """Devil's Advocate: two callers simultaneously hit the same output_dir.
    Contract: one wins (writes id file), the other reads it and resumes.

    Caveats / known gaps:
    - True multi-process is out of scope: mock.patch is process-global, and we
      don't want a real wandb network call.
    - Threads serialised by `threading.Event` so A's write_text() happens-
      before B's is_file() check. Without that, two threads racing on
      mock.patch.object would clobber each other's patches — that's a mock
      hazard, not a production race in `_init_wandb_with_resume`.
    - For real concurrency stress, run two real `train.py` processes against
      the same output_dir on the cluster.
    """
    cfg = _cfg(resume=True)
    a_done = threading.Event()
    state = {"a_kwargs": None, "b_kwargs": None}

    def recorder(slot):
        def _init(**kwargs):
            state[slot] = dict(kwargs)
            run = mock.MagicMock()
            run.id = kwargs.get("id", "winner_id")
            return run
        return _init

    def thread_a():
        fake_run = mock.MagicMock(); fake_run.id = "winner_id"
        with mock.patch.object(robotworkspace.wandb, "init", side_effect=recorder("a_kwargs")), \
             mock.patch.object(robotworkspace.wandb, "run", fake_run):
            robotworkspace._init_wandb_with_resume(tmp_path, cfg)
        a_done.set()

    def thread_b():
        if not a_done.wait(timeout=5):
            raise RuntimeError("thread A never completed")
        fake_run = mock.MagicMock(); fake_run.id = "winner_id"
        with mock.patch.object(robotworkspace.wandb, "init", side_effect=recorder("b_kwargs")), \
             mock.patch.object(robotworkspace.wandb, "run", fake_run):
            robotworkspace._init_wandb_with_resume(tmp_path, cfg)

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start(); tb.start()
    ta.join(timeout=5); tb.join(timeout=5)
    assert not ta.is_alive() and not tb.is_alive(), "thread hung"

    # A took fresh-launch path; B took resume path with A's id.
    assert state["a_kwargs"] is not None and "id" not in state["a_kwargs"]
    assert state["a_kwargs"]["resume"] == "allow"
    assert state["b_kwargs"] is not None
    assert state["b_kwargs"]["id"] == "winner_id"
    assert state["b_kwargs"]["resume"] == "must"
    assert (tmp_path / "wandb_id.txt").read_text().strip() == "winner_id"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
