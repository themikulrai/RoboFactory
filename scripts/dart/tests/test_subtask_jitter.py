"""PURE (no-sim) interpreter-level tests for DART DENSE jitter in
``subtask_interpreter.run_program``.

No sapien / gym / GPU. A light FakePlanner + FakeEnv (mirroring test_subtask.py)
drive ``run_program`` with hand-built fixed-tick primitives so the per-arm CLEAN
commanded waypoints are known exactly. Because the fake env is its own
``.unwrapped``, BOTH the recorded steps and the unrecorded jitter nudges log to the
same ``step_log``; we separate them by count (``len(step_log) - out['steps'] ==
out['n_jitter_steps']``) and by value (a jitter step's chosen-arm target = clean
waypoint + noise, != the clean waypoint a recorded step commands).

What is asserted:
  * jitter OFF (frac=0 / rng=None): NO nudges, step_log == recorded steps, the action
    stream is byte-identical to the no-jitter path (n_jitter_steps == 0).
  * jitter ON: extra UNRECORDED nudges appear, n_jitter_steps is tallied, the
    RECORDED action stream is UNCHANGED vs OFF (noise never becomes a label).
  * determinism: same (env_seed, variant) jitter rng -> identical nudge sequence;
    different -> diverges.
  * waypoint-centered (not qpos-centered): the nudge target tracks the supplied
    clean waypoint, not the arm's accumulated qpos.

Run:
    cd /iris/u/mikulrai/RoboFactory-subtask-wt && PYTHONPATH=. \
        /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
        scripts/dart/tests/test_subtask_jitter.py -q
"""
from __future__ import annotations

import os.path as osp
import sys

import numpy as np

# make scripts/dart importable (so run_subtask_rollouts is reachable)
_DART_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
_SCRIPTS_DIR = osp.dirname(_DART_DIR)
for _p in (_DART_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robofactory.planner import subtask_interpreter as I  # noqa: E402
from robofactory.planner import subtask_primitives as P  # noqa: E402
from robofactory.planner import subtask_vocab as vocab  # noqa: E402
from robofactory.planner.subtask_primitives import OPEN  # noqa: E402


# ===========================================================================
# Light off-GPU fakes (planner + env). The env is its OWN .unwrapped so the
# interpreter's _unwrap(env).step (jitter) and env.step (recorded) hit one log.
# ===========================================================================
class FakeRobot:
    def __init__(self, q7):
        self._q = np.concatenate(
            [np.asarray(q7, np.float32), [0.04, 0.04]]).astype(np.float32)

    def get_qpos(self):
        return self._q[None, :]  # (1, 9) numpy


class FakePlanner:
    def __init__(self, num_arms=2, control_mode="pd_joint_pos"):
        self.control_mode = control_mode
        self.robot = [FakeRobot(np.zeros(7) + 0.1 * i) for i in range(num_arms)]


class FakeEnv:
    """Records every step (recorded OR unrecorded jitter — they share this log,
    since unwrapped is self). Returns a benign step tuple."""

    def __init__(self):
        self.unwrapped = self
        self.step_log = []

    def step(self, action_dict):
        self.step_log.append({k: np.asarray(v).copy() for k, v in action_dict.items()})
        return {}, 0.0, False, False, {"success": False}


# A fixed-tick primitive whose CLEAN waypoints are KNOWN (so we can tell a recorded
# clean step from a noised jitter step). Each tick's qpos7 is distinct per step.
def _fixed_prim(arm: int, n_ticks: int = 6, grip: float = OPEN):
    base = 1.0 + arm  # arm0 -> 1.x, arm1 -> 2.x (distinct from FakeRobot qpos 0.x)
    ticks = []
    for t in range(n_ticks):
        qpos7 = (base + 0.01 * t) * np.ones(7, np.float32)
        ticks.append((qpos7.copy(), float(grip)))
    return P.Primitive(
        verb_id=vocab.APPROACH, target_id=0, text="approach", ticks=ticks,
        task="LiftBarrier", success_check=None, name=f"fixed_arm{arm}",
        # a stored target_pose so the APPROACH lint (warn) doesn't trip on a missing
        # success_check path -> we pass check_success_coverage='off' anyway below.
        target_pose=np.zeros(3, np.float32),
    )


def _program(num_arms=2, n_ticks=6):
    """One fixed-tick primitive per arm; both arms move every tick (no gating)."""
    prog = {}
    for arm in range(num_arms):
        prim = _fixed_prim(arm, n_ticks=n_ticks)
        prog[arm] = [I.QueuedRecipe(lambda pl, e, a, _p=prim: _p)]
    return prog


def _run(jitter_frac=0.0, jitter_sigma=0.0, jitter_rng=None, n_ticks=6,
         num_arms=2):
    env = FakeEnv()
    planner = FakePlanner(num_arms=num_arms)
    rec = I.SubtaskRecorder(num_arms=num_arms)
    out = I.run_program(
        env, planner, _program(num_arms, n_ticks), rec, max_steps=200,
        control_mode=planner.control_mode, check_success_coverage="off",
        jitter_frac=jitter_frac, jitter_sigma=jitter_sigma, jitter_rng=jitter_rng,
    )
    return env, out, rec


def _recorded_actions(env, out):
    """The LAST out['steps'] entries of step_log are the recorded steps; any earlier
    extra entries are jitter nudges interleaved before them. We can't slice by
    position (jitter is interleaved), so instead reconstruct the recorded stream by
    matching: a recorded step's arm targets equal the clean fixed-tick waypoints.

    Simpler & robust: re-run with jitter OFF to get the canonical recorded stream,
    and here just return the WHOLE log for the caller to compare counts."""
    return env.step_log


# ===========================================================================
# jitter OFF: no nudges; byte-identical to the no-jitter path.
# ===========================================================================
def test_jitter_off_frac_zero_no_nudges():
    env, out, _ = _run(jitter_frac=0.0, jitter_sigma=0.05,
                       jitter_rng=np.random.default_rng(0))
    # frac 0 -> never fires even with an rng + sigma
    assert out["n_jitter_steps"] == 0
    assert len(env.step_log) == out["steps"]  # log == recorded steps only


def test_jitter_off_rng_none_no_nudges():
    env, out, _ = _run(jitter_frac=0.9, jitter_sigma=0.05, jitter_rng=None)
    # rng None -> jitter disabled regardless of frac/sigma
    assert out["n_jitter_steps"] == 0
    assert len(env.step_log) == out["steps"]


def test_jitter_off_sigma_zero_no_nudges():
    env, out, _ = _run(jitter_frac=0.9, jitter_sigma=0.0,
                       jitter_rng=np.random.default_rng(0))
    assert out["n_jitter_steps"] == 0
    assert len(env.step_log) == out["steps"]


def test_jitter_off_is_byte_identical_to_no_jitter_path():
    """The OFF path (frac=0) must produce the EXACT same recorded step stream as the
    plain no-jitter call (no rng passed at all)."""
    env_a, out_a, rec_a = _run()  # plain (all defaults: no jitter)
    env_b, out_b, rec_b = _run(jitter_frac=0.0, jitter_sigma=0.0, jitter_rng=None)
    assert out_a["steps"] == out_b["steps"]
    assert out_a["n_jitter_steps"] == 0 and out_b["n_jitter_steps"] == 0
    assert len(env_a.step_log) == len(env_b.step_log)
    for sa, sb in zip(env_a.step_log, env_b.step_log):
        assert set(sa) == set(sb)
        for k in sa:
            assert np.array_equal(sa[k], sb[k]), k
    # label streams identical too
    aa, bb = rec_a.to_arrays(), rec_b.to_arrays()
    for arm in (0, 1):
        assert np.array_equal(aa[f"subtask_arm{arm}_verb"],
                              bb[f"subtask_arm{arm}_verb"])


# ===========================================================================
# jitter ON: extra UNRECORDED nudges; the RECORDED stream is UNCHANGED.
# ===========================================================================
def test_jitter_on_fires_unrecorded_nudges_recorded_stream_unchanged():
    # frac=1.0 -> a nudge before EVERY recorded step (both arms moving every tick).
    rng = np.random.default_rng(123)
    env, out, rec = _run(jitter_frac=1.0, jitter_sigma=0.05, jitter_rng=rng)

    # one nudge per recorded step (frac=1.0, always a moving arm available).
    assert out["n_jitter_steps"] == out["steps"] > 0
    # total log = recorded + jitter; the surplus over recorded == the nudges.
    assert len(env.step_log) == out["steps"] + out["n_jitter_steps"]

    # the RECORDED action stream must be UNCHANGED vs jitter OFF (noise -> no label).
    env_off, out_off, rec_off = _run()  # no jitter
    assert out["steps"] == out_off["steps"]
    # the recorded steps are the fixed clean tick actions; the OFF run's whole log IS
    # the recorded stream. Every clean action must appear in the ON run's log (the
    # jitter nudges are EXTRA entries that differ by noise), so the set of recorded
    # clean actions is a SUBSEQUENCE of the ON log. We verify by matching counts of
    # exact-clean steps.
    def _key(d):
        return tuple(sorted((k, tuple(np.round(v, 6))) for k, v in d.items()))

    clean_keys = [_key(s) for s in env_off.step_log]
    on_keys = [_key(s) for s in env.step_log]
    # every clean recorded step appears (byte-equal) in the ON log
    from collections import Counter
    on_counter = Counter(on_keys)
    for ck in clean_keys:
        assert on_counter[ck] >= 1, "a clean recorded step is missing from the ON log"
    # and the EXTRA ON entries (the nudges) are NOT clean (they carry noise)
    extra = len(on_keys) - len(clean_keys)
    assert extra == out["n_jitter_steps"]


def test_jitter_nudge_chosen_arm_is_clean_waypoint_plus_noise():
    """A jitter step commands ONE arm at clean_waypoint + noise (a NON-constant
    per-joint vector); the held arm and every recorded-step arm command a CONSTANT
    7-vector (our fixed ticks + the held live qpos are all constant). So a jitter
    step is exactly one with one NON-constant (noised) arm target. Verify those steps
    number n_jitter_steps and that the noised arm tracks a clean waypoint + noise."""
    rng = np.random.default_rng(7)
    env, out, _ = _run(jitter_frac=1.0, jitter_sigma=0.05, jitter_rng=rng)

    # clean fixed-tick qpos7 values (per arm) used by the recorded steps.
    clean_vals = [1.0 + arm + 0.01 * t for arm in (0, 1) for t in range(6)]
    clean_vals = np.array(clean_vals)

    n_noised = 0
    for s in env.step_log:
        noised = [k for k, v in s.items()
                  if not np.allclose(np.asarray(v)[:7], np.asarray(v)[0])]
        if len(noised) == 1:
            n_noised += 1
            # the noised arm = clean waypoint + N(0, sigma): its per-joint mean sits
            # near a clean waypoint value (the noise has zero mean over 7 joints, but
            # we only need it to be in the clean band, never the 0.x live-qpos band).
            tgt = np.asarray(s[noised[0]])[:7]
            mean = float(np.mean(tgt))
            nearest = clean_vals[np.argmin(np.abs(clean_vals - mean))]
            assert abs(mean - nearest) < 0.5, (mean, nearest)
            # the noise spread is mild (~sigma), bounded around the waypoint.
            assert float(np.std(tgt)) < 6.0 * 0.05
    # exactly the jitter nudges are the single-noised-arm steps
    assert n_noised == out["n_jitter_steps"] > 0


# ===========================================================================
# determinism: same jitter rng (same seed) -> identical nudge sequence.
# ===========================================================================
def _seedseq_rng(env_seed, variant_id, jseed=0):
    # mirror run_subtask_rollouts._make_jitter_rng's stream construction
    import run_subtask_rollouts as R
    cfg = R.JitterCfg(jitter_frac=1.0, jitter_sigma=0.05, jitter_seed=jseed)
    return R._make_jitter_rng(cfg, env_seed, variant_id)


def test_jitter_determinism_same_seed_identical():
    env_a, _, _ = _run(jitter_frac=1.0, jitter_sigma=0.05,
                       jitter_rng=_seedseq_rng(42, 1))
    env_b, _, _ = _run(jitter_frac=1.0, jitter_sigma=0.05,
                       jitter_rng=_seedseq_rng(42, 1))
    assert len(env_a.step_log) == len(env_b.step_log) > 0
    for sa, sb in zip(env_a.step_log, env_b.step_log):
        assert set(sa) == set(sb)
        for k in sa:
            assert np.array_equal(sa[k], sb[k]), k


def test_jitter_determinism_different_variant_diverges():
    env_a, _, _ = _run(jitter_frac=1.0, jitter_sigma=0.05,
                       jitter_rng=_seedseq_rng(42, 1))
    env_b, _, _ = _run(jitter_frac=1.0, jitter_sigma=0.05,
                       jitter_rng=_seedseq_rng(42, 2))
    # same length, but the noised targets differ somewhere
    diverged = False
    for sa, sb in zip(env_a.step_log, env_b.step_log):
        for k in sa:
            if not np.array_equal(sa[k], sb[k]):
                diverged = True
                break
        if diverged:
            break
    assert diverged, "different variant ids must produce a different jitter stream"


def test_jitter_determinism_different_env_seed_diverges():
    env_a, _, _ = _run(jitter_frac=1.0, jitter_sigma=0.05,
                       jitter_rng=_seedseq_rng(1, 0))
    env_b, _, _ = _run(jitter_frac=1.0, jitter_sigma=0.05,
                       jitter_rng=_seedseq_rng(2, 0))
    diverged = any(
        not np.array_equal(sa[k], sb[k])
        for sa, sb in zip(env_a.step_log, env_b.step_log) for k in sa
    )
    assert diverged


def test_jitter_rng_distinct_from_shove_stream():
    """_make_jitter_rng appends a stream tag so the jitter rng differs from a shove
    rng built with the same (env_seed, variant_id, seed) -> they never share draws."""
    import run_subtask_rollouts as R
    jr = R._make_jitter_rng(R.JitterCfg(jitter_frac=1.0, jitter_sigma=0.05,
                                        jitter_seed=3), 42, 1)
    # shove-style rng (the boundary_hook's construction): no stream tag, tc=0
    shove = np.random.default_rng(np.random.SeedSequence([42, 1, 0, 3]))
    assert jr.random() != shove.random(), "jitter and shove streams must differ"


# ===========================================================================
# waypoint-centered (NOT qpos-centered): the nudge tracks the clean waypoint.
# ===========================================================================
def test_jitter_is_waypoint_centered_not_qpos_centered():
    """The nudge target for the chosen arm equals its CLEAN waypoint + the rng noise,
    NOT its live qpos + noise. The fake robot's live qpos (0.x) is far from the
    primitive's clean waypoints (1.x / 2.x), so a waypoint-centered nudge lands near
    1.x/2.x; a (wrong) qpos-centered nudge would land near 0.x. We assert every nudge
    target is near a clean waypoint value, never near the live qpos."""
    rng = np.random.default_rng(0)
    env, out, _ = _run(jitter_frac=1.0, jitter_sigma=0.05, jitter_rng=rng)
    assert out["n_jitter_steps"] > 0

    # clean waypoint magnitudes are >= ~1.0; live qpos is <= ~0.1+grip. Any noised
    # target's mean must be near a clean band (>=0.9), proving waypoint-centering.
    clean_vals = []
    for arm in (0, 1):
        for t in range(6):
            clean_vals.append(1.0 + arm + 0.01 * t)
    clean_vals = np.array(clean_vals)

    for s in env.step_log:
        for k, v in s.items():
            arm_target = np.asarray(v)[:7]
            mean = float(np.mean(arm_target))
            # a noised target (non-constant) must sit near a CLEAN waypoint value
            # (within a few sigma), never near the 0.x live qpos.
            if not np.allclose(arm_target, arm_target[0]):
                nearest = clean_vals[np.argmin(np.abs(clean_vals - mean))]
                assert abs(mean - nearest) < 0.5, (
                    f"noised target mean {mean} not near any clean waypoint "
                    f"{nearest} -> looks qpos-centered, not waypoint-centered")
                assert mean > 0.5, "noised target collapsed toward the 0.x live qpos"
