"""Regression: a single motion-planner failure must NOT kill the whole shard.

Shard 2 of the LB combined gen died at 11/20 with an uncaught
``RuntimeError: Fail to parameterize path`` raised by mplib's TOPP DURING the
rollout (primitives plan screw paths lazily from live pose, so the failure fires
inside ``run_program`` -- not at ``spec.build``). ``_run_one_variant`` only
guarded ``spec.build``, so the rollout exception propagated up through
``_process_group`` -> the seed loop -> killed the process, losing every remaining
episode (the entire clean slice in shard 2's case).

The fix wraps ``run_program`` too, honouring ``_run_one_variant``'s documented
contract: "(None, None) if planning OR the rollout raised". These tests pin that
contract AND the downstream rollback flow it feeds (a dropped variant is isolated
per-member / rolls the atomic group back, never aborting the loop).

Pure (no SAPIEN/GPU): the planner / recorder / run_program are monkeypatched.
"""
from __future__ import annotations

import os.path as osp
import sys

import pytest

_DART_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
_SCRIPTS_DIR = osp.dirname(_DART_DIR)
for _p in (_DART_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_subtask_rollouts as R  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeSpec:
    def __init__(self, name="simultaneous", seed=7, variant_id=0, num_arms=2,
                 build_raises=None):
        self.name = name
        self.seed = seed
        self.variant_id = variant_id
        self.num_arms = num_arms
        self.family = "baseline"
        self._build_raises = build_raises

    def build(self, planner, env):
        if self._build_raises is not None:
            raise self._build_raises
        return ["program"]  # opaque; run_program is monkeypatched


class _FakePlanner:
    control_mode = "pd_joint_pos"


def _patch_variant_deps(monkeypatch):
    """Stub everything _run_one_variant touches except run_program (the unit under
    test decides run_program's behaviour per case)."""
    monkeypatch.setattr(R, "_build_planner", lambda env, seed: _FakePlanner())
    monkeypatch.setattr(R, "SubtaskRecorder", lambda num_arms: ("REC", num_arms))
    monkeypatch.setattr(R, "_make_boundary_hook",
                        lambda cfg, seed, vid: None)


# ---------------------------------------------------------------------------
# _run_one_variant: the contract
# ---------------------------------------------------------------------------
def test_rollout_runtimeerror_returns_none_none(monkeypatch):
    """THE regression: TOPP-style RuntimeError mid-rollout -> (None, None)."""
    _patch_variant_deps(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("Fail to parameterize path")
    monkeypatch.setattr(R, "run_program", _boom)

    rec, out = R._run_one_variant(env=object(), spec=_FakeSpec(),
                                  max_steps=600, dart_cfg=None)
    assert rec is None and out is None


@pytest.mark.parametrize("exc", [
    RuntimeError("Fail to parameterize path"),
    ValueError("bad ik"),
    IndexError("planner oob"),
    KeyError("missing"),
])
def test_rollout_any_exception_isolated(monkeypatch, exc):
    """Any rollout exception (not just TOPP) is swallowed to (None, None) so one
    flaky seed can never abort the shard."""
    _patch_variant_deps(monkeypatch)
    monkeypatch.setattr(R, "run_program",
                        lambda *a, **k: (_ for _ in ()).throw(exc))
    rec, out = R._run_one_variant(object(), _FakeSpec(), 600, None)
    assert rec is None and out is None


def test_build_exception_still_returns_none_none(monkeypatch):
    """The pre-existing build-time guard is preserved (no regression)."""
    _patch_variant_deps(monkeypatch)
    monkeypatch.setattr(R, "run_program", lambda *a, **k: {"steps": 5})
    spec = _FakeSpec(build_raises=RuntimeError("plan -1"))
    rec, out = R._run_one_variant(object(), spec, 600, None)
    assert rec is None and out is None


def test_clean_rollout_returns_rec_and_out(monkeypatch):
    """A clean run returns (recorder, run_out) unchanged."""
    _patch_variant_deps(monkeypatch)
    sentinel = {"steps": 42, "all_success": True}
    monkeypatch.setattr(R, "run_program", lambda *a, **k: sentinel)
    rec, out = R._run_one_variant(object(), _FakeSpec(num_arms=3), 600, None)
    assert rec == ("REC", 3)        # the recorder we constructed
    assert out is sentinel


def test_keyboardinterrupt_not_swallowed(monkeypatch):
    """except Exception must NOT trap KeyboardInterrupt/SystemExit (BaseException)
    -- a Ctrl-C / scancel should still stop the run, not be logged as a dropped
    variant."""
    _patch_variant_deps(monkeypatch)
    monkeypatch.setattr(R, "run_program",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        R._run_one_variant(object(), _FakeSpec(), 600, None)


# ---------------------------------------------------------------------------
# _process_group: the rollback path the (None,None) feeds
# ---------------------------------------------------------------------------
class _FakeEnv:
    """Minimal env exposing what _process_group reads: a flush that hands out a
    fresh _episode_id, plus the _h5_file/_json_* attrs the atomic rollback uses."""
    def __init__(self):
        self._episode_id = -1
        self._next = 100
        self.discarded = 0

        class _H5:
            def flush(self_inner):
                pass
        self._h5_file = _H5()
        self._json_data = {"episodes": []}
        self._json_path = None

    def flush_trajectory(self, save=True):
        if not save:
            self.discarded += 1
            return
        self._episode_id = self._next
        self._next += 1


def _patch_group_deps(monkeypatch, written_calls, deleted_calls):
    # member passes iff out is a truthy dict with ok=True
    monkeypatch.setattr(R, "_member_passes",
                        lambda out: bool(out) and out.get("ok", False))
    monkeypatch.setattr(
        R, "_write_member",
        lambda stream, meta, tag, seed, eid, rec, T, spec:
            written_calls.append(eid))
    monkeypatch.setattr(
        R, "_delete_episodes",
        lambda ids, *a, **k: deleted_calls.extend(int(i) for i in ids))


def test_per_member_drops_failed_keeps_rest(monkeypatch):
    """per-member: a member whose rollout raised (now -> (None,None)) is dropped;
    the good members are still committed. Mirrors the merged slice."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)
    env = _FakeEnv()

    outcomes = iter([
        (("REC", 2), {"ok": True, "steps": 5}),   # good
        (None, None),                              # the TOPP-failed variant
        (("REC", 2), {"ok": True, "steps": 5}),   # good
    ])
    monkeypatch.setattr(R, "_run_one_variant",
                        lambda *a, **k: next(outcomes))

    members = [_FakeSpec(name=n) for n in ("a", "b", "c")]
    kept, group_ok = R._process_group(
        env, members, max_steps=600, per_attempt_timeout=5, dart_cfg=None,
        stream_h5={}, episodes_meta=[], slice_tag="merged", seed=7, gid=0,
        kept_so_far=0, per_member=True)

    assert kept == 2                  # two good members survived
    assert group_ok is False          # not a full clean group
    assert len(written) == 2          # exactly the two goods committed
    assert deleted == []              # per-member never rolls back
    assert env.discarded == 1         # the failed member's buffer was discarded


def test_atomic_rolls_back_whole_group_on_failure(monkeypatch):
    """atomic (clean slice): if any member's rollout raised, the WHOLE matched
    group is rolled back -- nothing committed, the already-flushed member deleted."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)
    env = _FakeEnv()

    outcomes = iter([
        (("REC", 2), {"ok": True, "steps": 5}),   # member 0 flushes -> id 100
        (None, None),                              # member 1 rollout raised
        (("REC", 2), {"ok": True, "steps": 5}),   # never reached (break)
    ])
    monkeypatch.setattr(R, "_run_one_variant",
                        lambda *a, **k: next(outcomes))

    members = [_FakeSpec(name=n) for n in ("a", "b", "c")]
    kept, group_ok = R._process_group(
        env, members, max_steps=600, per_attempt_timeout=5, dart_cfg=None,
        stream_h5={}, episodes_meta=[], slice_tag="clean", seed=7, gid=0,
        kept_so_far=0, per_member=False)

    assert kept == 0 and group_ok is False
    assert written == []              # atomic: nothing committed
    assert deleted == [100]           # member 0's flushed traj rolled back
    assert env.discarded == 1         # member 1's open buffer discarded


def test_loop_survives_failure_no_propagation(monkeypatch):
    """Belt-and-suspenders: _process_group RETURNS normally (no raise) even when a
    member's rollout failed -- the property the shard loop relies on to keep going."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)
    monkeypatch.setattr(R, "_run_one_variant", lambda *a, **k: (None, None))
    # must not raise (the property the shard loop relies on to keep going):
    kept, ok = R._process_group(
        _FakeEnv(), [_FakeSpec()], 600, 5, None, {}, [], "merged", 7, 0, 0,
        per_member=True)
    assert kept == 0 and ok is False  # the lone member was dropped, not committed


# ---------------------------------------------------------------------------
# DartCfg defaults (shove_joints)
# ---------------------------------------------------------------------------
def test_dartcfg_shove_joints_default_is_proximal():
    """DartCfg defaults to the proximal base/shoulder/elbow joints (wrist untouched)."""
    assert R.DartCfg().shove_joints == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# per-variant skew tracking (variant_kept)
# ---------------------------------------------------------------------------
def test_variant_kept_accumulates_per_member(monkeypatch):
    """per-member: variant_kept counts each COMMITTED variant by spec.name; a dropped
    (None,None) member does NOT increment its name."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)
    env = _FakeEnv()

    outcomes = iter([
        (("REC", 2), {"ok": True, "steps": 5}),   # 'a' kept
        (None, None),                              # 'b' dropped (no bump)
        (("REC", 2), {"ok": True, "steps": 5}),   # 'c' kept
    ])
    monkeypatch.setattr(R, "_run_one_variant", lambda *a, **k: next(outcomes))

    members = [_FakeSpec(name=n) for n in ("a", "b", "c")]
    variant_kept = {}
    R._process_group(
        env, members, max_steps=600, per_attempt_timeout=5, dart_cfg=None,
        stream_h5={}, episodes_meta=[], slice_tag="merged", seed=7, gid=0,
        kept_so_far=0, per_member=True, variant_kept=variant_kept)

    assert variant_kept == {"a": 1, "c": 1}  # 'b' dropped -> absent


def test_variant_kept_accumulates_atomic_only_on_full_group(monkeypatch):
    """atomic: variant_kept increments ONLY when the WHOLE group passes (a rolled-back
    atomic member must never be counted)."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)

    # Case 1: full group passes -> both names counted once.
    env = _FakeEnv()
    ok_outcomes = iter([
        (("REC", 2), {"ok": True, "steps": 5}),
        (("REC", 2), {"ok": True, "steps": 5}),
    ])
    monkeypatch.setattr(R, "_run_one_variant", lambda *a, **k: next(ok_outcomes))
    vk = {}
    R._process_group(
        env, [_FakeSpec(name="x"), _FakeSpec(name="y")], 600, 5, None,
        {}, [], "clean", 7, 0, 0, per_member=False, variant_kept=vk)
    assert vk == {"x": 1, "y": 1}

    # Case 2: one member fails -> whole group rolled back -> NOTHING counted.
    env2 = _FakeEnv()
    fail_outcomes = iter([
        (("REC", 2), {"ok": True, "steps": 5}),   # x flushes then gets rolled back
        (None, None),                              # y fails -> atomic drop
    ])
    monkeypatch.setattr(R, "_run_one_variant", lambda *a, **k: next(fail_outcomes))
    vk2 = {}
    R._process_group(
        env2, [_FakeSpec(name="x"), _FakeSpec(name="y")], 600, 5, None,
        {}, [], "clean", 7, 0, 0, per_member=False, variant_kept=vk2)
    assert vk2 == {}  # atomic rollback -> no variant counted


def test_variant_kept_accumulates_across_groups(monkeypatch):
    """The SAME dict, reused across groups, sums repeated variant names."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)
    monkeypatch.setattr(R, "_run_one_variant",
                        lambda *a, **k: (("REC", 2), {"ok": True, "steps": 5}))
    vk = {}
    for _ in range(3):  # three single-member 'baseline' groups
        R._process_group(
            _FakeEnv(), [_FakeSpec(name="baseline")], 600, 5, None,
            {}, [], "recovery", 7, 0, 0, per_member=False, variant_kept=vk)
    assert vk == {"baseline": 3}


def test_variant_kept_none_is_noop(monkeypatch):
    """variant_kept=None (default) -> no tracking, no error (back-compat)."""
    written, deleted = [], []
    _patch_group_deps(monkeypatch, written, deleted)
    monkeypatch.setattr(R, "_run_one_variant",
                        lambda *a, **k: (("REC", 2), {"ok": True, "steps": 5}))
    kept, ok = R._process_group(
        _FakeEnv(), [_FakeSpec(name="a")], 600, 5, None, {}, [], "merged", 7, 0, 0,
        per_member=True)  # no variant_kept passed
    assert kept == 1 and ok is True
