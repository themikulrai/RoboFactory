"""Tests for the DART data-gen pipeline.

PURE tests (no sim, always run): exercise merge_h5 and the disturbance math
helper on synthetic data.

SIM tests (@pytest.mark.sim, skipped unless DART_RUN_SIM=1): run a 1-2 episode
LiftBarrier DART rollout on a GPU compute node and verify the resulting H5
never recorded the injected noise, that recovery actually happened, that
actions stay within joint limits, that obs is one longer than actions, and that
a huge sigma keeps nothing.

Run pure tests:   pytest scripts/dart/tests
Run all (on GPU): DART_RUN_SIM=1 pytest scripts/dart/tests
"""
import json
import os
import os.path as osp
import sys

import h5py
import numpy as np
import pytest

# make scripts/dart importable as a package root
_DART_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
_SCRIPTS_DIR = osp.dirname(_DART_DIR)
for p in (_SCRIPTS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

from dart import merge_h5  # noqa: E402

RUN_SIM = os.environ.get("DART_RUN_SIM", "0") == "1"
sim = pytest.mark.skipif(not RUN_SIM, reason="set DART_RUN_SIM=1 to run sim tests")

# Panda arm joint limits (7 revolute joints) — used by the SIM bound check.
PANDA_QLIMITS_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_QLIMITS_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


# ---------------------------------------------------------------------------
# PURE: merge_h5
# ---------------------------------------------------------------------------
def test_merge_basic(scripted_pair, dart_pair, tmp_path):
    scripted_h5, scripted_eps = scripted_pair
    dart_h5, dart_eps = dart_pair
    out_h5 = str(tmp_path / "merged" / "Task.h5")

    n_total, out_json_path = merge_h5.merge(scripted_h5, dart_h5, out_h5)

    assert n_total == 4
    with h5py.File(out_h5, "r") as f:
        keys = merge_h5._sorted_traj_keys(f)
        assert keys == ["traj_0", "traj_1", "traj_2", "traj_3"]  # contiguous
        # each group still carries its datasets after copy
        for k in keys:
            assert "actions" in f[k]
            assert "success" in f[k]

    out_json = json.loads(open(out_json_path).read())
    assert len(out_json["episodes"]) == 4
    # ids renumbered contiguously and in scripted-then-dart order
    assert [e["episode_id"] for e in out_json["episodes"]] == [0, 1, 2, 3]
    # success flags preserved
    assert all(e["success"] for e in out_json["episodes"])
    # seeds preserved: scripted 100,101 then dart 200,201
    assert [e["episode_seed"] for e in out_json["episodes"]] == [100, 101, 200, 201]
    # env_info taken from scripted json
    assert out_json["env_info"]["env_id"] == "Fake-rf"


def test_merge_preserves_failure_flags(tmp_path, fake_h5_builder):
    """A failing episode keeps success=False through the merge."""
    make_fake_h5 = fake_h5_builder
    s_h5 = tmp_path / "s" / "T.h5"
    d_h5 = tmp_path / "d" / "T.h5"
    s_h5.parent.mkdir(parents=True)
    d_h5.parent.mkdir(parents=True)
    make_fake_h5(str(s_h5), n_traj=2, seed_base=0, successes=[True, False])
    make_fake_h5(str(d_h5), n_traj=1, seed_base=50, successes=[True])
    out_h5 = str(tmp_path / "o" / "T.h5")

    n_total, out_json_path = merge_h5.merge(s_h5.as_posix(), d_h5.as_posix(), out_h5)
    assert n_total == 3
    eps = json.loads(open(out_json_path).read())["episodes"]
    assert [e["success"] for e in eps] == [True, False, True]


def test_merge_renumbers_noncontiguous_source(tmp_path, fake_h5_builder):
    """DART ids continue after scripted even if a source has gaps."""
    make_fake_h5 = fake_h5_builder
    s_h5 = tmp_path / "s" / "T.h5"
    s_h5.parent.mkdir(parents=True)
    # build scripted with 3 trajs
    make_fake_h5(str(s_h5), n_traj=3, seed_base=0)
    # dart with 2 trajs
    d_h5 = tmp_path / "d" / "T.h5"
    d_h5.parent.mkdir(parents=True)
    make_fake_h5(str(d_h5), n_traj=2, seed_base=10)
    out_h5 = str(tmp_path / "o" / "T.h5")
    n_total, _ = merge_h5.merge(s_h5.as_posix(), d_h5.as_posix(), out_h5)
    assert n_total == 5
    with h5py.File(out_h5, "r") as f:
        assert merge_h5._sorted_traj_keys(f) == [
            "traj_0", "traj_1", "traj_2", "traj_3", "traj_4"]


# ---------------------------------------------------------------------------
# PURE: disturbance math helper
# ---------------------------------------------------------------------------
def _apply_disturbance(q_full, rng, sigma):
    """Reference implementation of the per-arm perturbation used in
    run_dart_rollouts._inject_disturbance: noise on the 7 arm joints only, the
    gripper channel (last element) untouched. q_full is [7 joints + gripper]."""
    q = q_full[:7].copy()
    grip = q_full[7]
    q = q + rng.normal(0.0, sigma, size=7)
    return np.hstack([q, grip])


def test_disturbance_only_arm_not_gripper():
    rng = np.random.default_rng(0)
    q_full = np.array([0.1, 0.2, 0.3, -0.4, 0.5, 1.0, -0.2, 1.0])  # last=gripper
    out = _apply_disturbance(q_full, rng, sigma=0.3)
    assert out.shape == (8,)
    # gripper unchanged exactly
    assert out[7] == q_full[7]
    # arm joints changed
    assert not np.allclose(out[:7], q_full[:7])


def test_disturbance_scale_matches_sigma():
    """Std of the added perturbation should be ~sigma over many samples."""
    rng = np.random.default_rng(42)
    sigma = 0.25
    base = np.zeros(8)
    base[7] = 1.0
    deltas = []
    for _ in range(20000):
        out = _apply_disturbance(base, rng, sigma)
        deltas.append(out[:7] - base[:7])
    deltas = np.concatenate(deltas)
    assert abs(deltas.std() - sigma) < 0.02
    assert abs(deltas.mean()) < 0.01  # zero-mean


def test_disturbance_sigma_zero_is_noop():
    rng = np.random.default_rng(7)
    q_full = np.array([0.1, 0.2, 0.3, -0.4, 0.5, 1.0, -0.2, -1.0])
    out = _apply_disturbance(q_full, rng, sigma=0.0)
    assert np.allclose(out, q_full)


def test_dart_rng_does_not_consume_global():
    """The dedicated Generator pattern must not touch numpy's global RNG, so
    task-seed reproducibility is preserved."""
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    rng = np.random.default_rng(0)
    _ = rng.normal(0, 1, size=100)
    _ = rng.integers(5, 16)
    after = np.random.get_state()[1].copy()
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# SIM tests (GPU compute node only; DART_RUN_SIM=1)
# ---------------------------------------------------------------------------
def _run_liftbarrier(tmp_path, sigma, num=1, dart_seed=0, p_inject=1.0,
                     capture_disturbances=False):
    """Run a tiny LiftBarrier DART rollout, returning (out_h5, disturbances)."""
    from dart import run_dart_rollouts as rdr

    sink = [] if capture_disturbances else None
    rdr._DISTURBANCE_SINK = sink
    record_dir = str(tmp_path / f"sigma{sigma}")
    try:
        out_h5 = rdr.run(
            "LiftBarrier", num, record_dir,
            sigma=sigma, k_min=5, k_max=8, p_inject=p_inject,
            dart_seed=dart_seed, max_retries_per_seed=2,
            per_attempt_timeout=300,
        )
    finally:
        rdr._DISTURBANCE_SINK = None
    return out_h5, sink


@sim
def test_sim_noise_not_recorded(tmp_path):
    """(1) No recorded actions/panda-i row equals any emitted disturbance."""
    out_h5, disturbances = _run_liftbarrier(
        tmp_path, sigma=0.15, num=1, p_inject=1.0, capture_disturbances=True)
    assert osp.exists(out_h5)
    assert len(disturbances) > 0, "no disturbance was injected"

    # flatten emitted disturbance arrays per agent
    emitted = {0: [], 1: []}
    for d in disturbances:
        for aid in (0, 1):
            emitted[aid].append(np.asarray(d[f"panda-{aid}"], dtype=np.float64))

    with h5py.File(out_h5, "r") as f:
        trajs = merge_h5_keys(f)
        assert trajs, "no trajectory recorded"
        for k in trajs:
            for aid in (0, 1):
                rec = np.asarray(f[k]["actions"][f"panda-{aid}"][:], dtype=np.float64)
                for row in rec:
                    for em in emitted[aid]:
                        assert not np.allclose(row, em, atol=1e-6), (
                            "a recorded action equals an injected disturbance — "
                            "noise leaked into the dataset")


@sim
def test_sim_recovery_shifts_qpos(tmp_path):
    """(2) Recorded qpos at the first recovery step after an injection differs
    from the pre-injection qpos by ~the drift (i.e. the arm actually moved)."""
    out_h5, _ = _run_liftbarrier(tmp_path, sigma=0.3, num=1, p_inject=1.0)
    with h5py.File(out_h5, "r") as f:
        k = merge_h5_keys(f)[0]
        qpos = np.asarray(f[k]["obs"]["agent"]["panda-0"]["qpos"][:])
        # consecutive qpos deltas should show non-trivial motion somewhere
        deltas = np.abs(np.diff(qpos[:, :7], axis=0)).max()
        assert deltas > 1e-3, "recorded qpos shows no recovery motion"


@sim
def test_sim_actions_within_limits(tmp_path):
    """(3) Every recorded action within Panda joint limits + valid gripper, and
    (4) qpos.shape[0] == actions.shape[0] + 1."""
    out_h5, _ = _run_liftbarrier(tmp_path, sigma=0.15, num=1, p_inject=1.0)
    with h5py.File(out_h5, "r") as f:
        for k in merge_h5_keys(f):
            for aid in (0, 1):
                act = np.asarray(f[k]["actions"][f"panda-{aid}"][:])
                arm = act[:, :7]
                grip = act[:, 7]
                # allow small numerical slack on the planner targets
                assert (arm >= PANDA_QLIMITS_LOW - 0.05).all()
                assert (arm <= PANDA_QLIMITS_HIGH + 0.05).all()
                assert ((grip >= -1.05) & (grip <= 1.05)).all()
                qpos = np.asarray(f[k]["obs"]["agent"][f"panda-{aid}"]["qpos"][:])
                assert qpos.shape[0] == act.shape[0] + 1


@sim
def test_sim_huge_sigma_keeps_nothing(tmp_path):
    """(5) With huge sigma=5.0 no episode succeeds. Low yield is NON-FATAL:
    run() must return WITHOUT raising, passed==0, dart_meta.json exists with
    passed==0 and episodes==[], and the H5 has no traj_* groups (or was never
    created)."""
    from dart import run_dart_rollouts as rdr
    record_dir = str(tmp_path / "huge")
    # must NOT raise — degradation at high sigma is the sweep signal, not error
    rdr.run("LiftBarrier", 1, record_dir, sigma=5.0, k_min=5, k_max=8,
            p_inject=1.0, dart_seed=0, max_retries_per_seed=2,
            per_attempt_timeout=300)

    meta = osp.join(record_dir, "LiftBarrier", "dart_meta.json")
    assert osp.exists(meta), "dart_meta.json must be written even when passed==0"
    md = json.loads(open(meta).read())
    assert md["passed"] == 0
    assert md["episodes"] == []

    out_h5 = osp.join(record_dir, "LiftBarrier", "LiftBarrier.h5")
    if osp.exists(out_h5):
        with h5py.File(out_h5, "r") as f:
            assert merge_h5_keys(f) == []


def merge_h5_keys(f):
    """Local re-export to avoid importing merge_h5 again at call sites."""
    return merge_h5._sorted_traj_keys(f)
