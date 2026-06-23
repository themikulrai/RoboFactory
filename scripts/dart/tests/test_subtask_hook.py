"""PURE tests for the REAL DART boundary_hook + the single-pass-with-delete H5
rollback helper in ``run_subtask_rollouts.py``.

No sim, no gym.make, no SAPIEN. The hook is exercised against a FAKE interpreter
``state`` stream (fake _ArmState + fake planner) and ``dart_perturb.
inject_joint_disturbance`` is MONKEYPATCHED to a recorder so we can assert exactly
which calls the hook makes (determinism, force-first-inject, frozen grip carry,
target-arm selection). The delete helper is unit-tested against a REAL temp h5py
file (write 2 traj groups, drop one, assert survivor remains).

Run:
    cd /iris/u/mikulrai/RoboFactory-subtask-wt && PYTHONPATH=. \
        /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
        scripts/dart/tests/test_subtask_hook.py -q
"""
from __future__ import annotations

import os.path as osp
import sys

import numpy as np
import pytest

# make scripts/dart importable so run_subtask_rollouts is reachable
_DART_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
_SCRIPTS_DIR = osp.dirname(_DART_DIR)
for _p in (_DART_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_subtask_rollouts as R  # noqa: E402
from robofactory.planner import dart_perturb  # noqa: E402


# ===========================================================================
# Fakes for the interpreter ``state`` the hook consumes.
# ===========================================================================
class _FakeArm:
    """Minimal stand-in for subtask_interpreter._ArmState exposing only what the
    hook reads: ``.current`` (truthy=has a queued recipe / None=done), ``.ti``
    (tick index), ``.built`` (None until built), ``.frozen_grip`` (last grip)."""

    def __init__(self, current=True, ti=0, built=None, frozen_grip=1.0):
        self.current = current
        self.ti = ti
        self.built = built
        self.frozen_grip = frozen_grip


class _FakePlanner:
    def __init__(self, num_arms=2, control_mode="pd_joint_pos"):
        self.control_mode = control_mode
        # robot[i] is just an opaque token; inject is monkeypatched so it's unused.
        self.robot = [object() for _ in range(num_arms)]


def _entering_state(moving_arms, num_arms=2, grips=None):
    """Build a fake interpreter ``state`` where ``moving_arms`` are about to ENTER a
    primitive (current truthy, ti==0, built is None) and the rest are done."""
    grips = grips if grips is not None else [1.0] * num_arms
    arms = {}
    for i in range(num_arms):
        if i in moving_arms:
            arms[i] = _FakeArm(current=True, ti=0, built=None, frozen_grip=grips[i])
        else:
            # a "done"/non-entering arm: current None so it is NOT in `moving`.
            arms[i] = _FakeArm(current=None, ti=0, built=None, frozen_grip=grips[i])
    return {"step": 0, "arms": arms, "planner": _FakePlanner(num_arms)}


def _install_recorder(monkeypatch):
    """Monkeypatch dart_perturb.inject_joint_disturbance to record its call args.

    The hook calls it as ``dart_perturb.inject_joint_disturbance(...)`` (module
    attribute lookup), so patching the attribute on the imported module catches it.
    Returns the list it appends each call's kwargs/args to.
    """
    calls = []

    def _fake_inject(base_env, robots, grips, move_ids, rng, sigma, K, floor,
                     control_mode="pd_joint_pos", action_prefix="panda", sink=None):
        # snapshot the RNG state-consuming draw so two identical streams are
        # provably identical (capture a deterministic sample from the SAME rng).
        probe = float(rng.random())
        calls.append({
            "move_ids": list(move_ids),
            "grips": list(grips),
            "sigma": float(sigma),
            "K": int(K),
            "floor": float(floor),
            "control_mode": control_mode,
            "rng_probe": probe,
        })

    monkeypatch.setattr(dart_perturb, "inject_joint_disturbance", _fake_inject)
    return calls


# ===========================================================================
# HOOK: returns None when disabled.
# ===========================================================================
def test_hook_none_when_cfg_falsy_or_sigma_nonpositive():
    assert R._make_boundary_hook(None, 1, 2) is None
    assert R._make_boundary_hook(R.DartCfg(sigma=0.0), 1, 2) is None
    assert R._make_boundary_hook(R.DartCfg(sigma=-0.5), 1, 2) is None
    assert callable(R._make_boundary_hook(R.DartCfg(sigma=0.3), 1, 2))


# ===========================================================================
# HOOK: first transition ALWAYS injects (force-first).
# ===========================================================================
def test_hook_first_transition_always_injects(monkeypatch):
    calls = _install_recorder(monkeypatch)
    # p_inject=0 -> only the forced first call should ever inject.
    cfg = R.DartCfg(sigma=0.3, p_inject=0.0, dart_seed=7)
    hook = R._make_boundary_hook(cfg, env_seed=11, variant_id=0)
    state = _entering_state(moving_arms=[0, 1])
    hook(object(), state)               # transition counter 0 -> forced inject
    assert len(calls) == 1
    # a second transition with p_inject=0 must NOT inject.
    hook(object(), _entering_state(moving_arms=[0, 1]))
    assert len(calls) == 1


def test_hook_no_moving_arms_no_inject(monkeypatch):
    calls = _install_recorder(monkeypatch)
    cfg = R.DartCfg(sigma=0.3, p_inject=1.0)
    hook = R._make_boundary_hook(cfg, 1, 2)
    # NO arm is entering a primitive -> the hook must early-return, no inject.
    state = _entering_state(moving_arms=[])
    hook(object(), state)
    assert calls == []


# ===========================================================================
# HOOK: determinism — same (env_seed, variant_id) over the SAME state stream
# yields an IDENTICAL sequence of inject calls; different ids differ.
# ===========================================================================
def _drive(hook, n_transitions, moving=(0, 1), grips=None):
    for _ in range(n_transitions):
        hook(object(), _entering_state(moving_arms=list(moving), grips=grips))


def test_hook_determinism_same_ids_identical_calls(monkeypatch):
    cfg = R.DartCfg(sigma=0.4, p_inject=0.5, k_min=5, k_max=15, floor=0.15,
                    dart_seed=3)

    calls_a = _install_recorder(monkeypatch)
    hook_a = R._make_boundary_hook(cfg, env_seed=42, variant_id=1)
    _drive(hook_a, 12)

    calls_b = _install_recorder(monkeypatch)
    hook_b = R._make_boundary_hook(cfg, env_seed=42, variant_id=1)
    _drive(hook_b, 12)

    assert len(calls_a) == len(calls_b) > 0
    assert calls_a == calls_b, "identical (env_seed,variant_id) must replay exactly"


def test_hook_different_variant_ids_diverge(monkeypatch):
    cfg = R.DartCfg(sigma=0.4, p_inject=0.5, dart_seed=3)

    calls_a = _install_recorder(monkeypatch)
    hook_a = R._make_boundary_hook(cfg, env_seed=42, variant_id=1)
    _drive(hook_a, 12)

    calls_b = _install_recorder(monkeypatch)
    hook_b = R._make_boundary_hook(cfg, env_seed=42, variant_id=2)
    _drive(hook_b, 12)

    # different variant ids must produce a DIFFERENT call stream (the seed includes
    # variant_id). They share the forced first inject but should diverge by the K /
    # target-arm / probe draws and/or the inject decisions.
    assert calls_a != calls_b


def test_hook_different_env_seeds_diverge(monkeypatch):
    cfg = R.DartCfg(sigma=0.4, p_inject=0.5, dart_seed=3)

    calls_a = _install_recorder(monkeypatch)
    _drive(R._make_boundary_hook(cfg, env_seed=1, variant_id=0), 12)

    calls_b = _install_recorder(monkeypatch)
    _drive(R._make_boundary_hook(cfg, env_seed=2, variant_id=0), 12)

    assert calls_a != calls_b


# ===========================================================================
# HOOK: frozen grip carry + forwarded params (sigma/floor/K-range/control_mode).
# ===========================================================================
def test_hook_passes_frozen_grip_and_params(monkeypatch):
    calls = _install_recorder(monkeypatch)
    cfg = R.DartCfg(sigma=0.4, p_inject=1.0, k_min=6, k_max=6, floor=0.2,
                    dart_seed=0)
    hook = R._make_boundary_hook(cfg, env_seed=5, variant_id=0)
    grips = [-1.0, 0.7]  # arm0 closed, arm1 partly open (frozen grips)
    hook(object(), _entering_state(moving_arms=[0, 1], grips=grips))
    assert len(calls) == 1
    c = calls[0]
    assert c["grips"] == grips, "grips passed must equal each arm's frozen_grip"
    assert c["sigma"] == pytest.approx(0.4)
    assert c["floor"] == pytest.approx(0.2)
    assert c["K"] == 6                      # k_min==k_max -> exactly that
    assert c["control_mode"] == "pd_joint_pos"
    assert c["move_ids"] and c["move_ids"][0] in (0, 1)


def test_hook_control_mode_follows_planner(monkeypatch):
    calls = _install_recorder(monkeypatch)
    cfg = R.DartCfg(sigma=0.4, p_inject=1.0, dart_seed=0)
    hook = R._make_boundary_hook(cfg, 5, 0)
    state = _entering_state(moving_arms=[0, 1])
    state["planner"].control_mode = "pd_joint_pos_vel"
    hook(object(), state)
    assert calls[0]["control_mode"] == "pd_joint_pos_vel"


def test_hook_target_arm_only_from_moving(monkeypatch):
    """The perturbed arm must be one that is ENTERING a primitive this transition."""
    calls = _install_recorder(monkeypatch)
    cfg = R.DartCfg(sigma=0.4, p_inject=1.0, dart_seed=0)
    # only arm1 is moving (arm0 done) -> every inject must target arm1.
    hook = R._make_boundary_hook(cfg, 5, 0)
    for _ in range(8):
        hook(object(), _entering_state(moving_arms=[1]))
    assert calls, "expected at least the forced first inject"
    assert all(c["move_ids"] == [1] for c in calls), [c["move_ids"] for c in calls]


def test_hook_does_not_touch_global_numpy_rng(monkeypatch):
    """The hook must derive its own Generator (SeedSequence) and never perturb the
    GLOBAL numpy RNG (so env reset / sampler RNG stay untouched)."""
    _install_recorder(monkeypatch)
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    hook = R._make_boundary_hook(R.DartCfg(sigma=0.4, p_inject=1.0), 5, 0)
    _drive(hook, 10)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after), "hook perturbed the GLOBAL numpy RNG"


# ===========================================================================
# HOOK: end-to-end determinism THROUGH the real dart_perturb (no monkeypatch) —
# the unwrapped env's step stream must be byte-identical for identical ids.
# ===========================================================================
class _FakeStepEnv:
    """Records every action dict pushed by dart_perturb.inject_joint_disturbance."""

    def __init__(self):
        self.steps = []

    def step(self, action_dict):
        self.steps.append({k: np.asarray(v).copy() for k, v in action_dict.items()})


class _FakeQposRobot:
    def __init__(self, q7):
        self._q = np.concatenate([np.asarray(q7, np.float64), [0.04, 0.04]])

    def get_qpos(self):
        return self._q[None, :]


def _planner_with_qpos(num_arms=2):
    pl = _FakePlanner(num_arms)
    pl.robot = [_FakeQposRobot(np.zeros(7) + 0.1 * i) for i in range(num_arms)]
    return pl


def test_hook_real_inject_step_stream_deterministic():
    """Drive the REAL dart_perturb through the hook (no monkeypatch) and assert two
    identical-id hooks produce byte-identical unwrapped-env action streams."""
    cfg = R.DartCfg(sigma=0.4, p_inject=0.5, k_min=4, k_max=9, floor=0.15,
                    dart_seed=2)

    def run_once():
        env = _FakeStepEnv()
        hook = R._make_boundary_hook(cfg, env_seed=99, variant_id=4)
        for _ in range(10):
            state = {"step": 0, "arms": {
                0: _FakeArm(current=True, ti=0, built=None, frozen_grip=-1.0),
                1: _FakeArm(current=True, ti=0, built=None, frozen_grip=1.0),
            }, "planner": _planner_with_qpos(2)}
            hook(env, state)
        return env.steps

    a = run_once()
    b = run_once()
    assert len(a) == len(b) > 0
    for sa, sb in zip(a, b):
        assert set(sa) == set(sb)
        for k in sa:
            assert np.array_equal(sa[k], sb[k]), k


# ===========================================================================
# DELETE HELPER: write 2 traj groups to a REAL temp h5, drop one, assert.
# ===========================================================================
def _make_h5(path, keys):
    import h5py
    f = h5py.File(path, "w")
    for k in keys:
        g = f.create_group(k)
        g.create_dataset("data", data=np.arange(3))
    return f


def test_delete_episodes_removes_only_target(tmp_path):
    import h5py
    traj_p = str(tmp_path / "T.h5")
    strm_p = str(tmp_path / "T_subtask_stream.h5")
    traj = _make_h5(traj_p, ["traj_0", "traj_1"])
    strm = _make_h5(strm_p, ["traj_0", "traj_1"])

    json_data = {"episodes": [{"episode_id": 0}, {"episode_id": 1}]}
    meta = [{"episode_id": 0, "variant": "a"}, {"episode_id": 1, "variant": "b"}]

    n = R._delete_episodes([1], traj, strm, json_data=json_data,
                           json_path=None, meta_list=meta)
    assert n == 1
    # survivor remains, deleted key is GONE in BOTH h5 files
    assert "traj_0" in traj and "traj_1" not in traj
    assert "traj_0" in strm and "traj_1" not in strm
    # json + in-memory meta rows for episode 1 dropped, episode 0 kept
    assert json_data["episodes"] == [{"episode_id": 0}]
    assert meta == [{"episode_id": 0, "variant": "a"}]
    traj.close()
    strm.close()

    # the on-disk survivor is still readable
    with h5py.File(traj_p, "r") as f:
        assert list(f.keys()) == ["traj_0"]


def test_delete_episodes_multiple_and_missing(tmp_path):
    traj = _make_h5(str(tmp_path / "T.h5"), ["traj_5", "traj_6", "traj_7"])
    strm = _make_h5(str(tmp_path / "S.h5"), ["traj_5", "traj_6"])  # traj_7 stream absent
    json_data = {"episodes": [{"episode_id": 5}, {"episode_id": 6}, {"episode_id": 7}]}
    meta = [{"episode_id": 5}, {"episode_id": 6}, {"episode_id": 7}]

    # delete 6 and 7 (7's stream group is missing -> tolerated, no crash)
    n = R._delete_episodes([6, 7], traj, strm, json_data=json_data, meta_list=meta)
    assert n == 2  # both traj_6 and traj_7 unlinked from the trajectory h5
    assert set(traj.keys()) == {"traj_5"}
    assert set(strm.keys()) == {"traj_5"}
    assert json_data["episodes"] == [{"episode_id": 5}]
    assert meta == [{"episode_id": 5}]
    traj.close()
    strm.close()


def test_delete_episodes_writes_json_sidecar(tmp_path):
    import json as _json
    traj = _make_h5(str(tmp_path / "T.h5"), ["traj_0", "traj_1"])
    strm = _make_h5(str(tmp_path / "S.h5"), ["traj_0", "traj_1"])
    json_path = str(tmp_path / "T.json")
    json_data = {"episodes": [{"episode_id": 0}, {"episode_id": 1}], "source_type": "x"}
    with open(json_path, "w") as f:
        _json.dump(json_data, f)

    R._delete_episodes([0], traj, strm, json_data=json_data, json_path=json_path,
                       meta_list=None)
    traj.close()
    strm.close()
    # the re-dumped sidecar on disk no longer references episode 0
    with open(json_path) as f:
        reread = _json.load(f)
    assert reread["episodes"] == [{"episode_id": 1}]
    assert reread["source_type"] == "x"  # other keys preserved


# ===========================================================================
# MIX plumbing (pure): suffix splice, mix parse, budget split.
# ===========================================================================
def test_splice_suffix():
    assert R._splice_suffix("table/lift_barrier.yaml", "_aug") == \
        "table/lift_barrier_aug.yaml"
    assert R._splice_suffix("table/lift_barrier.yaml", "") == \
        "table/lift_barrier.yaml"
    assert R._splice_suffix("a/b/c.yaml", "_x") == "a/b/c_x.yaml"


def test_parse_mix_and_budget():
    mix = R._parse_mix("recovery=0.3,merged=0.4,clean=0.3")
    assert mix == [("recovery", 0.3), ("merged", 0.4), ("clean", 0.3)]
    assert R._parse_mix("") == []
    # budget split sums to exactly num for several num values
    for num in (10, 30, 31, 100, 7):
        b = R._split_budget(num, mix)
        assert sum(b.values()) == num, (num, b)
        assert set(b) == {"recovery", "merged", "clean"}


def test_parse_mix_rejects_unknown_slice():
    with pytest.raises(SystemExit):
        R._parse_mix("bogus=1.0")
    with pytest.raises(SystemExit):
        R._parse_mix("recovery")  # no '='
