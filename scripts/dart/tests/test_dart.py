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
# PURE: AUG1 — guaranteed >=1 injection (force-first-move) + magnitude floor
#
# These drive the ACTUAL run_dart_rollouts helpers (_make_dart_move and
# _inject_disturbance) — no sim. A tiny fake "solver" mimics the bits the
# disturbance hook touches (base_env / env.unwrapped / robot qpos / gripper),
# so the real floor + force-first-move logic is exercised, not a copy of it.
# ---------------------------------------------------------------------------
class _FakeBaseEnv:
    """Stand-in for env.unwrapped: records every action handed to .step()."""

    def __init__(self):
        self.steps = []

    def step(self, action):
        # store a deep copy so later mutations can't alias the captured value
        if isinstance(action, dict):
            self.steps.append({k: np.asarray(v, dtype=np.float64).copy()
                               for k, v in action.items()})
        else:
            self.steps.append(np.asarray(action, dtype=np.float64).copy())
        return None


class _FakeRobot:
    """Stand-in for solver.robot[aid]: get_qpos()[0, :-2] -> 7 arm joints.

    Returns a torch-like object exposing .cpu().numpy(); we just hand back a
    numpy view since _inject_disturbance only calls .cpu().numpy()."""

    class _QposHandle:
        def __init__(self, arr):
            self._arr = arr  # shape (1, 9): 7 joints + 2 gripper fingers

        def __getitem__(self, idx):
            return _FakeRobot._CpuNumpy(self._arr[idx])

    class _CpuNumpy:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    def __init__(self, q7):
        # full qpos row = 7 arm joints + 2 finger joints
        self._full = np.hstack([np.asarray(q7, dtype=np.float64), [0.04, 0.04]])

    def get_qpos(self):
        return _FakeRobot._QposHandle(self._full[None, :])  # (1, 9)


class _FakeEnvWrap:
    """Wrapped env whose .unwrapped is the shared fake base env."""

    def __init__(self, base):
        self.unwrapped = base


class _FakeSolver:
    """Minimal object exposing exactly the attributes _inject_disturbance reads."""

    def __init__(self, q7_list, multi_agent):
        base = _FakeBaseEnv()
        self.base_env = base
        self.env = _FakeEnvWrap(base)
        self.is_multi_agent = multi_agent
        self.agent_num = len(q7_list)
        self.robot = [_FakeRobot(q7) for q7 in q7_list]
        # gripper_state is indexed per agent; a scalar per agent is fine.
        self.gripper_state = [1.0 for _ in q7_list]


def _rdr():
    from dart import run_dart_rollouts as rdr  # noqa: E402
    return rdr


def test_force_first_move_injects_even_at_p_inject_zero():
    """AUG1 (1): with p_inject=0.0 the FIRST real (not dry_run) move STILL
    injects, guaranteeing n_injects >= 1. dry_run moves never inject. We
    monkeypatch the heavy bits (_ORIG_MOVE no-op, _inject_disturbance a counter)
    so this is a pure logic test of the closure / ep_counter gate."""
    rdr = _rdr()
    ep_counter = {"n_calls": 0, "n_injects": 0}
    injected_calls = []

    orig_move = rdr._ORIG_MOVE
    orig_inject = rdr._inject_disturbance
    try:
        # planning probe (the real method) becomes a no-op so we test only gating
        rdr._ORIG_MOVE = lambda self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1: None
        rdr._inject_disturbance = lambda self, rng, sigma, K, move_id, floor: injected_calls.append(K)

        rng = np.random.default_rng(0)
        # p_inject=0.0 -> only the force-first-move rule can ever inject
        dart_move = rdr._make_dart_move(rng, sigma=0.3, k_min=5, k_max=15,
                                        p_inject=0.0, floor=0.15,
                                        ep_counter=ep_counter)

        fake_self = object()  # _ORIG_MOVE/_inject are no-ops; self is unused here
        # a dry_run probe BEFORE any real move must NOT inject and must NOT count
        dart_move(fake_self, pose=None, dry_run=True, move_id=0)
        assert ep_counter["n_calls"] == 0
        assert ep_counter["n_injects"] == 0
        assert injected_calls == []

        # first REAL move -> forced injection even though p_inject == 0.0
        dart_move(fake_self, pose=None, dry_run=False, move_id=0)
        assert ep_counter["n_calls"] == 1
        assert ep_counter["n_injects"] == 1
        assert len(injected_calls) == 1

        # subsequent real moves at p_inject=0.0 must NOT inject again
        for _ in range(20):
            dart_move(fake_self, pose=None, dry_run=False, move_id=0)
        assert ep_counter["n_injects"] == 1  # still exactly one
        assert ep_counter["n_calls"] == 21
    finally:
        rdr._ORIG_MOVE = orig_move
        rdr._inject_disturbance = orig_inject


def test_force_first_move_resets_per_episode():
    """AUG1 (1): resetting ep_counter to zero (as run() does before each solver
    call) re-arms the force-first-move rule, so EACH episode gets >= 1 injection
    even at p_inject=0.0."""
    rdr = _rdr()
    ep_counter = {"n_calls": 0, "n_injects": 0}
    n_inject = [0]
    orig_move = rdr._ORIG_MOVE
    orig_inject = rdr._inject_disturbance
    try:
        rdr._ORIG_MOVE = lambda self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1: None
        rdr._inject_disturbance = lambda self, rng, sigma, K, move_id, floor: n_inject.__setitem__(0, n_inject[0] + 1)
        rng = np.random.default_rng(1)
        dart_move = rdr._make_dart_move(rng, sigma=0.3, k_min=5, k_max=15,
                                        p_inject=0.0, floor=0.15,
                                        ep_counter=ep_counter)
        fake_self = object()
        for _ in range(3):  # 3 simulated episodes
            ep_counter["n_calls"] = 0  # run() does this before each solver call
            ep_counter["n_injects"] = 0
            for _ in range(5):  # several real moves per episode
                dart_move(fake_self, pose=None, dry_run=False, move_id=0)
            assert ep_counter["n_injects"] == 1  # exactly the forced first move
        assert n_inject[0] == 3  # one forced injection per episode
    finally:
        rdr._ORIG_MOVE = orig_move
        rdr._inject_disturbance = orig_inject


def test_inject_floor_enforced_single_agent():
    """AUG1 (2): a per-arm offset whose raw draw is below the floor is scaled UP
    so the commanded net joint displacement has L2 >= inject_floor. Driven
    through the REAL _inject_disturbance with a fake single-agent solver."""
    rdr = _rdr()
    floor = 0.15
    q7 = np.array([0.1, 0.2, 0.3, -0.4, 0.5, 1.0, -0.2])
    # sigma tiny so the raw draw is almost always below the floor -> floor binds
    norms = []
    for s in range(200):
        solver = _FakeSolver([q7], multi_agent=False)
        rng = np.random.default_rng(s)
        rdr._inject_disturbance(solver, rng, sigma=1e-4, K=3, move_id=0, floor=floor)
        # the HELD target is whatever was stepped; recover net displacement
        stepped = solver.base_env.steps[0]            # flat action [7 + gripper]
        target = np.asarray(stepped)[:7]
        disp = target - q7
        norms.append(float(np.linalg.norm(disp)))
        # all K steps must HOLD the identical target (no per-step re-sampling)
        for st in solver.base_env.steps:
            assert np.allclose(np.asarray(st)[:7], target)
        # gripper channel stays clean
        assert np.asarray(stepped)[7] == solver.gripper_state[0]
    norms = np.array(norms)
    # Floor enforced. The production scaling is off/(mag+1e-9)*floor, so when the
    # raw draw mag is tiny the +1e-9 guard against div-by-zero pulls the result a
    # hair UNDER the floor by ~floor*1e-9/mag (sub-microradian). Tolerate that
    # documented epsilon; the floor is still effectively binding.
    assert (norms >= floor - 1e-4).all(), f"min floored norm {norms.min()} < {floor}"
    assert (norms <= floor + 1e-6).all(), f"max floored norm {norms.max()} > {floor}"
    # randomness preserved above the floor: directions differ across draws
    assert norms.std() < 1e-3  # all pinned ~at the floor (tiny sigma)
    # but the DIRECTION must vary across seeds (not a constant vector)
    dirs = []
    for s in range(50):
        solver = _FakeSolver([q7], multi_agent=False)
        rng = np.random.default_rng(s)
        rdr._inject_disturbance(solver, rng, sigma=1e-4, K=1, move_id=0, floor=floor)
        d = np.asarray(solver.base_env.steps[0])[:7] - q7
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    dirs = np.array(dirs)
    assert dirs.std(axis=0).max() > 1e-2, "floored offset direction does not vary"


def test_inject_floor_randomness_above_floor():
    """AUG1 (2): when sigma is large the raw draw exceeds the floor and is used
    UNCHANGED, so magnitude varies across draws (randomness preserved above the
    floor — the floor only lifts trivially-small draws)."""
    rdr = _rdr()
    floor = 0.15
    q7 = np.zeros(7)
    norms = []
    for s in range(300):
        solver = _FakeSolver([q7], multi_agent=False)
        rng = np.random.default_rng(s)
        rdr._inject_disturbance(solver, rng, sigma=0.5, K=1, move_id=0, floor=floor)
        d = np.asarray(solver.base_env.steps[0])[:7] - q7
        norms.append(float(np.linalg.norm(d)))
    norms = np.array(norms)
    assert (norms >= floor - 1e-6).all()         # never below floor
    assert norms.std() > 0.1                       # genuinely varying magnitude
    assert norms.max() > floor + 0.3               # large draws pass through


def test_inject_multi_agent_only_moving_arm_perturbed():
    """AUG1 (2): in the multi-agent path only arms in move_id drift; other arms
    HOLD their captured qpos exactly; every arm's gripper channel stays clean."""
    rdr = _rdr()
    q0a = np.array([0.0, 0.1, 0.2, -0.3, 0.4, 0.9, -0.1])
    q0b = np.array([0.5, -0.2, 0.1, -0.6, 0.3, 0.8, 0.2])
    solver = _FakeSolver([q0a, q0b], multi_agent=True)
    rng = np.random.default_rng(3)
    rdr._inject_disturbance(solver, rng, sigma=0.4, K=4, move_id=[0], floor=0.15)
    step0 = solver.base_env.steps[0]
    a0 = np.asarray(step0["panda-0"])
    a1 = np.asarray(step0["panda-1"])
    # agent 0 (in move_id) is perturbed; agent 1 is held at its captured qpos
    assert not np.allclose(a0[:7], q0a)
    assert np.allclose(a1[:7], q0b)
    # grippers clean on both
    assert a0[7] == solver.gripper_state[0]
    assert a1[7] == solver.gripper_state[1]
    # held identically for all K steps
    for st in solver.base_env.steps:
        assert np.allclose(np.asarray(st["panda-0"]), a0)
        assert np.allclose(np.asarray(st["panda-1"]), a1)


def test_inject_disturbance_uses_unwrapped_env_and_fills_sink():
    """PURE noise-not-recorded contract: _inject_disturbance steps ONLY the
    UNWRAPPED env (base is self.env.unwrapped — asserted in-code) and appends to
    _DISTURBANCE_SINK exactly the actions it stepped, so the sim leak check has
    something to compare against. Driven without sim via the fake solver."""
    rdr = _rdr()
    solver = _FakeSolver([np.zeros(7)], multi_agent=False)
    # contract precondition the production code asserts:
    assert solver.base_env is solver.env.unwrapped
    sink = []
    orig_sink = rdr._DISTURBANCE_SINK
    rdr._DISTURBANCE_SINK = sink
    try:
        rng = np.random.default_rng(0)
        rdr._inject_disturbance(solver, rng, sigma=0.3, K=5, move_id=0, floor=0.15)
    finally:
        rdr._DISTURBANCE_SINK = orig_sink
    # K steps -> K stepped actions -> K sink entries, each matching the step
    assert len(solver.base_env.steps) == 5
    assert len(sink) == 5
    for stepped, captured in zip(solver.base_env.steps, sink):
        assert np.allclose(np.asarray(stepped), np.asarray(captured))


def test_inject_disturbance_asserts_unwrapped():
    """If base_env is NOT env.unwrapped the in-code assert MUST fire (this is the
    guard that keeps injected noise out of the recorded buffer)."""
    rdr = _rdr()
    solver = _FakeSolver([np.zeros(7)], multi_agent=False)
    # break the contract: point the wrapped env's .unwrapped at a different obj
    solver.env.unwrapped = _FakeBaseEnv()
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        rdr._inject_disturbance(solver, rng, sigma=0.3, K=2, move_id=0, floor=0.15)


# ---------------------------------------------------------------------------
# PURE: AUG2 — aug yamls + scene_builder randyaw_deg gating
# ---------------------------------------------------------------------------
import yaml  # noqa: E402

# _DART_DIR == <repo>/scripts/dart -> repo root is two dirnames up.
_REPO_ROOT = osp.dirname(osp.dirname(_DART_DIR))
_CONFIG_DIR = osp.join(_REPO_ROOT, "robofactory", "configs", "table")


def _load_yaml(name):
    with open(osp.join(_CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def _barrier_obj(cfg):
    return cfg["objects"][0]  # LiftBarrier has a single object: the barrier


def _cubes(cfg):
    prims = cfg["scene"]["primitives"]
    return [p for p in prims if p["name"] in ("cubeA", "cubeB", "cubeC")]


def test_aug_yamls_exist_and_parse():
    """AUG2: both _aug yamls exist and parse as valid YAML with the expected
    top-level task_name."""
    lb = _load_yaml("lift_barrier_aug.yaml")
    tsc = _load_yaml("three_robots_stack_cube_aug.yaml")
    assert lb["task_name"] == "LiftBarrier"
    assert tsc["task_name"] == "ThreeRobotsStackCube"


def test_aug_lift_barrier_wider_and_has_yaw():
    """AUG2: the aug barrier's randp_scale is STRICTLY wider (>=, with at least
    one strictly greater) than canonical, and carries randyaw_deg == 30."""
    canon = _barrier_obj(_load_yaml("lift_barrier.yaml"))["pos"]
    aug = _barrier_obj(_load_yaml("lift_barrier_aug.yaml"))["pos"]
    c = np.asarray(canon["randp_scale"], dtype=np.float64)
    a = np.asarray(aug["randp_scale"], dtype=np.float64)
    assert (a >= c).all(), f"aug randp_scale {a} not >= canonical {c}"
    assert (a > c).any(), f"aug randp_scale {a} is not strictly wider than {c}"
    assert aug.get("randyaw_deg") == 30
    # canonical must have NO yaw key (isolation; double-checked below too)
    assert "randyaw_deg" not in canon


def test_aug_stack_cube_wider_and_has_yaw():
    """AUG2: every aug cube has randp_scale strictly wider than canonical (by
    absolute magnitude — canonical uses a negative scale on cubeA's y) and
    randyaw_deg == 30 (replacing the uncapped random_quaternions)."""
    canon_cubes = {c["name"]: c["pos"] for c in _cubes(_load_yaml("three_robots_stack_cube.yaml"))}
    aug_cubes = {c["name"]: c["pos"] for c in _cubes(_load_yaml("three_robots_stack_cube_aug.yaml"))}
    assert set(aug_cubes) == {"cubeA", "cubeB", "cubeC"}
    for name in ("cubeA", "cubeB", "cubeC"):
        c = np.abs(np.asarray(canon_cubes[name]["randp_scale"], dtype=np.float64))
        a = np.abs(np.asarray(aug_cubes[name]["randp_scale"], dtype=np.float64))
        assert (a >= c).all(), f"{name}: aug |randp_scale| {a} not >= canonical {c}"
        assert (a > c).any(), f"{name}: aug |randp_scale| {a} not strictly wider than {c}"
        assert aug_cubes[name].get("randyaw_deg") == 30, f"{name} missing randyaw_deg==30"
        # aug replaced the uncapped 360deg yaw with a bounded one
        assert "random_quaternions" not in aug_cubes[name], \
            f"{name}: aug must NOT keep uncapped random_quaternions"


def test_canonical_yamls_unchanged_isolation():
    """AUG2 ISOLATION: canonical yamls must be byte-equivalent to upstream in the
    randomization fields — NO randyaw_deg key anywhere, original randp_scale, and
    cubes still use the uncapped random_quaternions. Re-read from disk."""
    lb = _load_yaml("lift_barrier.yaml")
    tsc = _load_yaml("three_robots_stack_cube.yaml")
    # canonical barrier: original randp_scale, no yaw key
    bpos = _barrier_obj(lb)["pos"]
    assert bpos["randp_scale"] == [0.3, 0.05, 0.]
    assert "randyaw_deg" not in bpos
    # canonical cubes: original randp_scale, random_quaternions present, no yaw key
    canon = {c["name"]: c["pos"] for c in _cubes(tsc)}
    assert canon["cubeA"]["randp_scale"] == [0.05, -0.05, 0.]
    assert canon["cubeB"]["randp_scale"] == [0.05, 0.05, 0.]
    assert canon["cubeC"]["randp_scale"] == [0.05, 0.05, 0.]
    for name in ("cubeA", "cubeB", "cubeC"):
        assert canon[name].get("random_quaternions") == [1, 1, 0]
        assert "randyaw_deg" not in canon[name]


def _apply_randyaw(qpos, randyaw_deg, rng_val):
    """Reference of the scene_builder randyaw_deg block (gated identically in the
    primitive and object branches): yaw in [-deg, +deg] applied as a LOCAL-frame
    post-multiply of the base quaternion."""
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import qmult
    yaw = np.deg2rad((rng_val * 2 - 1) * randyaw_deg)
    dq = euler2quat(0, 0, yaw)
    return qmult(np.array(qpos), dq)


def test_randyaw_gating_no_key_is_noop():
    """AUG2: the randyaw_deg block is KEY-GATED. A pos dict WITHOUT the key takes
    the original code path unchanged (we assert by exercising the in-code guard
    directly on a config dict: 'randyaw_deg' in pos is False -> qpos untouched)."""
    pos_no_key = {"randp_scale": [0.1, 0.1, 0.0], "qpos": [1.0, 0.0, 0.0, 0.0]}
    qpos = np.array(pos_no_key["qpos"])
    # this mirrors the exact in-code guard: only mutate when the key is present
    if "randyaw_deg" in pos_no_key:
        qpos = _apply_randyaw(qpos, pos_no_key["randyaw_deg"], 0.7)
    assert np.allclose(qpos, [1.0, 0.0, 0.0, 0.0])  # unchanged -> gated off


def test_randyaw_deg_zero_is_noop():
    """AUG2: randyaw_deg == 0 (or rng giving the midpoint) yields the identity
    rotation, a safe no-op even when the key is present."""
    qpos = [1.0, 0.0, 0.0, 0.0]
    out = _apply_randyaw(qpos, 0.0, 0.42)        # deg=0 -> yaw=0 -> identity
    assert np.allclose(out, qpos)
    out2 = _apply_randyaw(qpos, 30.0, 0.5)        # rng_val=0.5 -> (1-1)... yaw=0
    assert np.allclose(out2, qpos)


def test_randyaw_pure_z_rotation_and_bounds():
    """AUG2: applied to an identity base quat, randyaw_deg gives a PURE world-z
    rotation (roll==pitch==0) bounded to [-deg, +deg], and the result stays a
    unit quaternion."""
    from transforms3d.euler import quat2euler
    yaws = []
    for s in range(500):
        rng = np.random.default_rng(s)
        q = _apply_randyaw([1.0, 0.0, 0.0, 0.0], 30.0, float(rng.random()))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9     # unit quaternion
        roll, pitch, yaw = quat2euler(q)
        assert abs(roll) < 1e-9 and abs(pitch) < 1e-9  # pure z rotation
        yaws.append(np.rad2deg(yaw))
    yaws = np.array(yaws)
    assert yaws.min() >= -30.0 - 1e-6 and yaws.max() <= 30.0 + 1e-6
    assert yaws.std() > 5.0                              # genuinely random yaw
    assert abs(yaws.mean()) < 3.0                        # symmetric about 0


def test_scene_builder_randyaw_block_is_key_guarded():
    """AUG2: source-level guard — both the primitive and object branches of
    RFSceneBuilder.initialize mutate qpos ONLY inside `if 'randyaw_deg' in
    ...['pos']:`. Asserts the gate exists twice so canonical (no-key) yamls keep
    the byte-for-byte original path."""
    import inspect
    import robofactory.utils.scenes.scene_builder as sb
    src = inspect.getsource(sb.RFSceneBuilder.initialize)
    # exactly the two gated insertions (primitive + object branches)
    assert src.count("'randyaw_deg' in primitive_cfg['pos']") == 1
    assert src.count("'randyaw_deg' in asset_cfg['pos']") == 1
    # both guarded blocks use the local-frame post-multiply
    assert src.count("qmult(np.array(qpos), dq)") == 2


# ---------------------------------------------------------------------------
# PURE: DENSE scheme — jitter (single-arm, jitter_sigma) + frequent single-arm
# shove+replan, both via the UNWRAPPED env (noise-not-recorded contract).
# ---------------------------------------------------------------------------
class _RecordingWrapEnv:
    """Wrapped env: .step() RECORDS (mimics RecordEpisodeMA), and .unwrapped is
    the shared fake base env whose .step() does NOT record. The follow_path
    test asserts the recorded action equals the CLEAN waypoint (not any nudge)."""

    def __init__(self, base):
        self.unwrapped = base
        self.recorded = []  # actions handed to the WRAPPED (recorded) step

    def step(self, action):
        if isinstance(action, dict):
            self.recorded.append({k: np.asarray(v, dtype=np.float64).copy()
                                  for k, v in action.items()})
        else:
            self.recorded.append(np.asarray(action, dtype=np.float64).copy())
        # follow_path unpacks 5 values from env.step
        return None, 0.0, False, False, {}


class _DenseFakeSolver:
    """Fake solver exposing what BOTH _jitter_nudge and the wrapped follow_path
    read: base_env / env(.unwrapped, .step) / robot qpos / gripper_state /
    is_multi_agent / agent_num / control_mode / elapsed_steps / print_env_info /
    vis. The wrapped env records; the base env does not."""

    def __init__(self, q7_list, multi_agent, control_mode="pd_joint_pos"):
        base = _FakeBaseEnv()
        self.base_env = base
        self.env = _RecordingWrapEnv(base)
        self.is_multi_agent = multi_agent
        self.agent_num = len(q7_list)
        self.robot = [_FakeRobot(q7) for q7 in q7_list]
        self.gripper_state = [1.0 for _ in q7_list]
        self.control_mode = control_mode
        self.elapsed_steps = 0
        self.print_env_info = False
        self.vis = False


def _result_group(positions_list):
    """Build a follow_path result_group: one dict per moving arm with a
    'position' array of shape (T, 7)."""
    return [{"position": np.asarray(p, dtype=np.float64)} for p in positions_list]


def test_dense_jitter_single_arm_and_uses_jitter_sigma():
    """(a) The dense jitter nudge perturbs exactly ONE arm (not all), uses
    jitter_sigma, and is centered on that arm's CLEAN WAYPOINT (not current qpos)
    so it is bounded around the path (no compounding random walk). The chosen
    arm's target must equal waypoint + small noise, NOT qpos + noise."""
    rdr = _rdr()
    q0a = np.array([0.0, 0.1, 0.2, -0.3, 0.4, 0.9, -0.1])
    q0b = np.array([0.5, -0.2, 0.1, -0.6, 0.3, 0.8, 0.2])   # current qpos arm 1
    q0c = np.array([-0.3, 0.4, -0.1, -0.5, 0.2, 1.0, 0.0])
    # waypoint for the perturbed arm, deliberately FAR from its current qpos so
    # we can prove the nudge tracks the waypoint, not the drifted qpos.
    waypoint = np.array([2.0, -1.0, 0.7, -2.0, 1.3, 2.5, -0.9])
    solver = _DenseFakeSolver([q0a, q0b, q0c], multi_agent=True)
    rng = np.random.default_rng(0)
    rdr._jitter_nudge(solver, rng, jitter_sigma=0.05, arm_id=1, waypoint7=waypoint)
    step = solver.base_env.steps[0]
    a0, a1, a2 = (np.asarray(step[f"panda-{i}"]) for i in range(3))
    # only arm 1 perturbed; 0 and 2 held at current qpos
    assert np.allclose(a0[:7], q0a)
    assert np.allclose(a2[:7], q0c)
    # arm 1 target is centered on the WAYPOINT (within a few sigma), NOT qpos
    assert np.linalg.norm(a1[:7] - waypoint) < 0.05 * 7  # close to waypoint
    assert np.linalg.norm(a1[:7] - q0b) > 1.0            # far from current qpos
    assert not np.allclose(a1[:7], waypoint)             # but genuinely perturbed
    # grippers all clean
    for a, g in zip((a0, a1, a2), solver.gripper_state):
        assert a[7] == g
    # nudge went to the UNWRAPPED env, NOT the recorded wrapped env
    assert len(solver.env.recorded) == 0
    assert len(solver.base_env.steps) == 1

    # jitter_sigma scale: std of (target - waypoint) over many draws ~ sigma,
    # and centered on the waypoint regardless of (differing) current qpos.
    wp = np.zeros(7)
    deltas = []
    for s in range(4000):
        sv = _DenseFakeSolver([np.ones(7) * 0.3], multi_agent=False)  # qpos != wp
        r = np.random.default_rng(s)
        rdr._jitter_nudge(sv, r, jitter_sigma=0.05, arm_id=0, waypoint7=wp)
        deltas.append(np.asarray(sv.base_env.steps[0])[:7] - wp)
    deltas = np.concatenate(deltas)
    assert abs(deltas.std() - 0.05) < 0.01
    assert abs(deltas.mean()) < 0.01


def test_dense_pick_single_arm_is_one_of_move_id():
    """(a)/(b) _pick_single_arm returns exactly ONE id, always drawn from the
    move_id set (never 'all'); over many draws it covers every member."""
    rdr = _rdr()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(500):
        a = rdr._pick_single_arm(rng, [0, 2, 3])
        assert a in (0, 2, 3)
        seen.add(a)
    assert seen == {0, 2, 3}  # all reachable, but one at a time
    # scalar move_id is normalized to a single-element list
    assert rdr._pick_single_arm(rng, 1) == 1
    # empty move_id -> None (no arm to perturb)
    assert rdr._pick_single_arm(rng, []) is None


def test_dense_shove_event_is_single_arm():
    """(b) A dense shove event perturbs exactly ONE arm from the move's move_id
    even when the move drives several arms. Driven through the REAL _make_dense_move
    with p_shove=1.0 (always shove) and a stubbed _ORIG_MOVE / _inject_disturbance
    capturing the move_id list handed to the injection."""
    rdr = _rdr()
    captured = []
    orig_move = rdr._ORIG_MOVE
    orig_inject = rdr._inject_disturbance
    try:
        rdr._ORIG_MOVE = lambda self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1: None
        rdr._inject_disturbance = (
            lambda self, rng, sigma, K, move_id, floor: captured.append(list(move_id)))
        ep_counter = {"n_calls": 0, "n_injects": 0,
                      "n_shove_events": 0, "n_jitter_steps": 0}
        rng = np.random.default_rng(0)
        dense_move = rdr._make_dense_move(
            rng, sigma_min=0.05, sigma_max=0.09, K=10, floor=0.0,
            p_shove=1.0, ep_counter=ep_counter)
        fake_self = object()
        # a multi-arm move: move_id=[0,1] — the shove must pick ONE of them
        for _ in range(50):
            dense_move(fake_self, pose=None, dry_run=False, move_id=[0, 1])
        assert ep_counter["n_shove_events"] == 50
        # every captured injection targeted exactly ONE arm, from {0,1}
        assert all(len(m) == 1 for m in captured)
        assert {m[0] for m in captured} == {0, 1}  # both reachable over draws
        # dry_run never shoves
        before = len(captured)
        dense_move(fake_self, pose=None, dry_run=True, move_id=[0, 1])
        assert len(captured) == before
    finally:
        rdr._ORIG_MOVE = orig_move
        rdr._inject_disturbance = orig_inject


def test_dense_shove_sigma_in_uniform_range():
    """(b) Per-event shove intensity is drawn U(sigma_min, sigma_max)."""
    rdr = _rdr()
    sigmas = []
    orig_move = rdr._ORIG_MOVE
    orig_inject = rdr._inject_disturbance
    try:
        rdr._ORIG_MOVE = lambda self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1: None
        rdr._inject_disturbance = (
            lambda self, rng, sigma, K, move_id, floor: sigmas.append(sigma))
        ep_counter = {"n_calls": 0, "n_injects": 0,
                      "n_shove_events": 0, "n_jitter_steps": 0}
        rng = np.random.default_rng(0)
        dense_move = rdr._make_dense_move(
            rng, sigma_min=0.05, sigma_max=0.09, K=10, floor=0.0,
            p_shove=1.0, ep_counter=ep_counter)
        for _ in range(2000):
            dense_move(object(), pose=None, dry_run=False, move_id=[0])
    finally:
        rdr._ORIG_MOVE = orig_move
        rdr._inject_disturbance = orig_inject
    sigmas = np.array(sigmas)
    assert sigmas.min() >= 0.05 - 1e-9
    assert sigmas.max() <= 0.09 + 1e-9
    assert sigmas.std() > 0.005  # genuinely spread across the range


def test_dense_jitter_goes_through_unwrapped_and_asserts():
    """(c) noise-not-recorded contract for the JITTER path: _jitter_nudge steps
    ONLY env.unwrapped (asserted in-code) and never the recorded wrapped env."""
    rdr = _rdr()
    solver = _DenseFakeSolver([np.zeros(7)], multi_agent=False)
    assert solver.base_env is solver.env.unwrapped  # precondition
    rng = np.random.default_rng(0)
    rdr._jitter_nudge(solver, rng, jitter_sigma=0.05, arm_id=0, waypoint7=np.zeros(7))
    assert len(solver.base_env.steps) == 1     # unwrapped stepped
    assert len(solver.env.recorded) == 0       # recorded buffer untouched
    # break the contract -> the in-code assert fires
    solver.env.unwrapped = _FakeBaseEnv()
    with pytest.raises(AssertionError):
        rdr._jitter_nudge(solver, rng, jitter_sigma=0.05, arm_id=0, waypoint7=np.zeros(7))


def test_dense_shove_goes_through_unwrapped():
    """(c) noise-not-recorded contract for the SHOVE path: the dense shove routes
    through _inject_disturbance, which asserts base is env.unwrapped and steps
    only the unwrapped env (not the recorded wrapped env)."""
    rdr = _rdr()
    solver = _DenseFakeSolver([np.zeros(7), np.zeros(7)], multi_agent=True)
    orig_move = rdr._ORIG_MOVE
    try:
        rdr._ORIG_MOVE = lambda self, pose, dry_run=False, refine_steps=0, move_id=0, jump=1: None
        ep_counter = {"n_calls": 0, "n_injects": 0,
                      "n_shove_events": 0, "n_jitter_steps": 0}
        rng = np.random.default_rng(0)
        dense_move = rdr._make_dense_move(
            rng, sigma_min=0.05, sigma_max=0.09, K=10, floor=0.0,
            p_shove=1.0, ep_counter=ep_counter)
        dense_move(solver, pose=None, dry_run=False, move_id=[0, 1])
    finally:
        rdr._ORIG_MOVE = orig_move
    assert len(solver.base_env.steps) == 10    # K unwrapped steps
    assert len(solver.env.recorded) == 0       # recorded buffer untouched
    # break the contract on a fresh solver -> _inject_disturbance assert fires
    bad = _DenseFakeSolver([np.zeros(7)], multi_agent=False)
    bad.env.unwrapped = _FakeBaseEnv()
    with pytest.raises(AssertionError):
        rdr._inject_disturbance(bad, np.random.default_rng(0), sigma=0.05, K=2,
                                move_id=[0], floor=0.0)


def test_dense_follow_path_records_clean_waypoint_not_nudge():
    """(d) The wrapped follow_path preserves the original recorded-step semantics:
    every RECORDED action equals the CLEAN waypoint from result_group (NOT the
    jittered target). With jitter_frac=1.0 a nudge fires before EVERY step, yet
    the recorded actions are byte-for-byte the clean waypoints; the nudges land
    only on the unwrapped (non-recorded) env."""
    rdr = _rdr()
    # single-agent, 3 waypoints (7 joints each)
    wps = np.array([[0.1, 0.2, 0.3, -0.4, 0.5, 1.0, -0.2],
                    [0.15, 0.25, 0.35, -0.45, 0.55, 1.05, -0.25],
                    [0.2, 0.3, 0.4, -0.5, 0.6, 1.1, -0.3]])
    solver = _DenseFakeSolver([np.zeros(7)], multi_agent=False)
    ep_counter = {"n_calls": 0, "n_injects": 0,
                  "n_shove_events": 0, "n_jitter_steps": 0}
    rng = np.random.default_rng(0)
    fp = rdr._make_dart_jitter_follow_path(
        rng, jitter_frac=1.0, jitter_sigma=0.05, ep_counter=ep_counter)
    fp(solver, _result_group([wps]), move_id=[0])
    # exactly one recorded step per waypoint, each == the CLEAN waypoint + gripper
    assert len(solver.env.recorded) == wps.shape[0]
    for rec, wp in zip(solver.env.recorded, wps):
        rec = np.asarray(rec)
        assert np.allclose(rec[:7], wp), "recorded action is NOT the clean waypoint"
        assert rec[7] == solver.gripper_state[0]  # gripper clean
    # jitter_frac=1.0 -> one unwrapped nudge before each of the 3 steps
    assert ep_counter["n_jitter_steps"] == 3
    assert len(solver.base_env.steps) == 3       # all nudges on unwrapped env
    # and NONE of the recorded actions equals a nudge target
    for rec in solver.env.recorded:
        for nudge in solver.base_env.steps:
            assert not np.allclose(np.asarray(rec)[:7], np.asarray(nudge)[:7]), \
                "a nudge target leaked into the recorded buffer"


def test_dense_follow_path_zero_jitter_matches_clean_loop():
    """(d) With jitter_frac=0.0 the wrapped follow_path is behaviorally identical
    to the original clean loop: no unwrapped steps, recorded actions == clean
    waypoints, multi-agent non-moving arms held at their qpos."""
    rdr = _rdr()
    q0a = np.zeros(7)
    q0b = np.array([0.5, -0.2, 0.1, -0.6, 0.3, 0.8, 0.2])  # held (not moving)
    wps = np.array([[0.1, 0.2, 0.3, -0.4, 0.5, 1.0, -0.2],
                    [0.2, 0.3, 0.4, -0.5, 0.6, 1.1, -0.3]])
    solver = _DenseFakeSolver([q0a, q0b], multi_agent=True)
    ep_counter = {"n_calls": 0, "n_injects": 0,
                  "n_shove_events": 0, "n_jitter_steps": 0}
    fp = rdr._make_dart_jitter_follow_path(
        np.random.default_rng(0), jitter_frac=0.0, jitter_sigma=0.05,
        ep_counter=ep_counter)
    fp(solver, _result_group([wps]), move_id=[0])  # only arm 0 moves
    assert ep_counter["n_jitter_steps"] == 0
    assert len(solver.base_env.steps) == 0  # NO unwrapped steps at all
    assert len(solver.env.recorded) == wps.shape[0]
    for rec, wp in zip(solver.env.recorded, wps):
        rec = rec  # dict
        a0 = np.asarray(rec["panda-0"])
        a1 = np.asarray(rec["panda-1"])
        assert np.allclose(a0[:7], wp)        # moving arm follows the waypoint
        assert np.allclose(a1[:7], q0b)       # non-moving arm holds its qpos
        assert a0[7] == solver.gripper_state[0]
        assert a1[7] == solver.gripper_state[1]


# ---------------------------------------------------------------------------
# SIM tests (GPU compute node only; DART_RUN_SIM=1)
# ---------------------------------------------------------------------------
def _run_liftbarrier_recovered(tmp_path, sigma=0.05, p_inject=0.5, dart_seed=0,
                               capture_disturbances=True):
    """Run a tiny LiftBarrier DART rollout that RELIABLY yields >=1 recorded,
    successful episode containing >=1 captured disturbance.

    Design rationale (the prior p_inject=1.0 helper was so disruptive the
    scripted solver never recovered -> 0 trajs recorded -> the real
    noise-not-recorded contract was never reached):
      - small sigma=0.05 (~3deg/joint) -> the planner re-solves easily,
      - p_inject=0.5 -> disturbances DO fire (so the sink is non-empty) but not
        on every move,
      - num=1 with override_seeds=range(8) and max_retries_per_seed=2 -> run()
        loops seeds, discarding failed-recovery episodes, until ONE success is
        recorded (up to 8 candidate seeds * 2 retries = 16 chances).

    Returns (out_h5, disturbances) where disturbances is the captured sink list
    (each entry a dict {"panda-0": (8,), "panda-1": (8,)} of emitted noisy
    action targets) or None if capture was disabled.
    """
    from dart import run_dart_rollouts as rdr

    sink = [] if capture_disturbances else None
    rdr._DISTURBANCE_SINK = sink
    record_dir = str(tmp_path / f"sigma{sigma}")
    try:
        out_h5 = rdr.run(
            "LiftBarrier", 1, record_dir,
            sigma=sigma, k_min=5, k_max=8, p_inject=p_inject,
            dart_seed=dart_seed, max_retries_per_seed=2,
            per_attempt_timeout=300,
            override_seeds=list(range(8)),
        )
    finally:
        rdr._DISTURBANCE_SINK = None
    return out_h5, sink


def _run_liftbarrier(tmp_path, sigma, num=1, dart_seed=0, p_inject=1.0,
                     capture_disturbances=False):
    """Thin wrapper kept for the limits/huge-sigma sim tests that still want a
    raw single-run handle. Uses override_seeds=range(num*2) for a little
    head-room but does NOT guarantee a recovered episode (those tests tolerate
    zero/low yield)."""
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
    """CORE CONTRACT: no recorded actions/panda-i row equals any emitted
    disturbance action. Redesigned to RELIABLY produce a recovered episode
    (sigma=0.05, p_inject=0.5) before asserting the contract."""
    out_h5, disturbances = _run_liftbarrier_recovered(
        tmp_path, sigma=0.05, p_inject=0.5, capture_disturbances=True)
    assert osp.exists(out_h5)

    with h5py.File(out_h5, "r") as f:
        trajs = merge_h5_keys(f)
        # (a) at least one recorded traj — else the env couldn't recover even at
        # sigma=0.05; fail loudly, do NOT silently skip.
        assert trajs, (
            "no trajectory was recorded even at sigma=0.05/p_inject=0.5 across "
            "8 candidate seeds x 2 retries — the LiftBarrier solver could not "
            "recover from a tiny disturbance; the contract could not be tested")

        # (b) at least one disturbance must have actually been emitted, else the
        # contract is vacuously true and we've tested nothing.
        assert disturbances is not None and len(disturbances) > 0, (
            "no disturbance was captured — with p_inject=0.5 over a full "
            "episode at least one injection is expected; cannot verify contract")

        # flatten emitted disturbance arrays per agent
        emitted = {0: [], 1: []}
        for d in disturbances:
            for aid in (0, 1):
                emitted[aid].append(np.asarray(d[f"panda-{aid}"], dtype=np.float64))

        # (c) for EVERY recorded action row across all trajs/agents, it must NOT
        # match ANY emitted disturbance action for that agent (tight tol). A
        # match means the unwrapped-env noise leaked into the recorded dataset.
        for k in trajs:
            for aid in (0, 1):
                rec = np.asarray(
                    f[k]["actions"][f"panda-{aid}"][:], dtype=np.float64)
                for ri, row in enumerate(rec):
                    for di, em in enumerate(emitted[aid]):
                        if np.allclose(row, em, atol=1e-4):
                            raise AssertionError(
                                "NOISE LEAKED: recorded "
                                f"{k}/actions/panda-{aid}[{ri}]={row.tolist()} "
                                f"matches emitted disturbance #{di} "
                                f"{em.tolist()} within atol=1e-4")


@sim
def test_sim_recovery_shifts_qpos(tmp_path):
    """Recovery left a real fingerprint in the recorded qpos: the arm moved off
    the clean path and the planner re-solved. Uses the same robust helper
    (sigma=0.05) so a recovered episode is guaranteed first.

    Assertion strategy (robust, not brittle): the exact "first recovery step
    after injection" index cannot be recovered from the H5 alone (the noise
    steps run on the unwrapped env and are not recorded, so there is no marker
    row). Instead we establish a CLEAN baseline (sigma=0, p_inject=0) on the
    same seed pool, take its max per-step qpos delta, then require the disturbed
    episode to contain at least one per-step delta NOTICEABLY larger than that
    clean ceiling — i.e. a recovery jump that the undisturbed planner never
    produces."""
    # disturbed run (guaranteed recovered episode)
    out_h5, _ = _run_liftbarrier_recovered(
        tmp_path / "dist", sigma=0.05, p_inject=0.5, capture_disturbances=False)
    with h5py.File(out_h5, "r") as f:
        k = merge_h5_keys(f)[0]
        qpos_d = np.asarray(f[k]["obs"]["agent"]["panda-0"]["qpos"][:])
    assert qpos_d.shape[0] >= 2, "recorded qpos too short to measure motion"
    dist_step_deltas = np.abs(np.diff(qpos_d[:, :7], axis=0)).max(axis=1)
    assert dist_step_deltas.max() > 1e-3, "recorded qpos shows no motion at all"

    # clean baseline (no injection) over the same seed pool
    clean_h5, _ = _run_liftbarrier_recovered(
        tmp_path / "clean", sigma=0.0, p_inject=0.0, capture_disturbances=False)
    with h5py.File(clean_h5, "r") as f:
        ck = merge_h5_keys(f)
        assert ck, "clean baseline produced no trajectory"
        qpos_c = np.asarray(f[ck[0]]["obs"]["agent"]["panda-0"]["qpos"][:])
    clean_max_step = np.abs(np.diff(qpos_c[:, :7], axis=0)).max()

    # the disturbed episode must contain a recovery jump clearly above the
    # clean planner's largest single-step motion (1.5x margin for robustness).
    assert dist_step_deltas.max() > 1.5 * clean_max_step, (
        f"disturbed max per-step qpos delta {dist_step_deltas.max():.4f} is not "
        f"clearly above the clean-baseline max {clean_max_step:.4f} (1.5x "
        f"threshold {1.5 * clean_max_step:.4f}) — no recovery shift detected")


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
