"""Tests for the PR7 success-persistence probe + step-budget unification.

Two GPU-free synthetic tests required by solution_E6.md §PR7:
  - persistence: a transient success at step t that is LOST by t+3 must yield
    success_first=True but success_sustained_10=False.
  - DP budget: the env-step-capped loop terminates at exactly the 400-env-step cap
    (driven through a synthetic env that NEVER reports success/truncated).

The persistence test uses success_persistence.probe_sustained directly (the one tested
unit the three drivers call). The budget test drives a stripped synthetic replica of the
eval_multi_dp inner accounting (the real driver imports torch/sapien, unavailable on the
CPU test host) — it asserts the env-step cap, the per-env-step success break, and that
steps == env steps.
"""
from __future__ import annotations

from robofactory.utils.success_persistence import probe_sustained, SUSTAIN_K


# --------------------------------------------------------------------- persistence
class TestProbeSustained:
    def test_transient_success_lost_is_not_sustained(self):
        """Success at the entry frame, then the criterion drops at the FIRST probe step
        and never returns -> sustained=False, min_success_during=False, sustained_full=False.
        Mirrors a barrier bobble that falls back the moment the policy stops acting."""
        # Criterion holds for the first 2 probe steps then is lost by step 3 (t+3).
        seq = [True, True, False] + [False] * 20
        calls = {"i": 0}

        def step_fn(_action):
            i = calls["i"]
            calls["i"] += 1
            return seq[i], False, False

        out = probe_sustained(lambda: "hold", step_fn, k=10)
        assert out["sustained"] is False          # not held at +10
        assert out["min_success_during"] is False  # dropped at some point
        assert out["sustained_full"] is False
        assert out["sustained_steps"] == 10        # full horizon ran (env never ended)

    def test_genuinely_held_is_sustained(self):
        """Criterion holds at every probe step -> sustained + sustained_full + min hold."""
        def step_fn(_a):
            return True, False, False

        out = probe_sustained(lambda: "hold", step_fn, k=10)
        assert out["sustained"] is True
        assert out["sustained_full"] is True
        assert out["min_success_during"] is True
        assert out["sustained_steps"] == 10

    def test_dropped_then_recovered_min_false_but_sustained_true(self):
        """Criterion dips mid-probe but is back up at +10: sustained=True (held at end) yet
        min_success_during=False and sustained_full=False (it did not hold continuously)."""
        seq = [True, True, False, True, True, True, True, True, True, True]

        calls = {"i": 0}

        def step_fn(_a):
            i = calls["i"]
            calls["i"] += 1
            return seq[i], False, False

        out = probe_sustained(lambda: "hold", step_fn, k=10)
        assert out["sustained"] is True
        assert out["min_success_during"] is False
        assert out["sustained_full"] is False

    def test_budget_caps_probe_steps(self):
        """budget_left < k -> probe runs only budget_left steps; sustained_full=False even
        if every run step held (we could not measure the full horizon)."""
        def step_fn(_a):
            return True, False, False

        out = probe_sustained(lambda: "hold", step_fn, k=10, budget_left=3)
        assert out["sustained_steps"] == 3
        assert out["sustained"] is True
        assert out["sustained_full"] is False  # only 3 of 10 ran

    def test_zero_budget_records_unmeasured(self):
        out = probe_sustained(lambda: "hold", lambda a: (True, False, False), k=10, budget_left=0)
        assert out["sustained_steps"] == 0
        assert out["sustained_full"] is False

    def test_env_terminates_during_probe_stops(self):
        """If the env truncates mid-probe the loop stops; sustained reflects the last step."""
        calls = {"i": 0}

        def step_fn(_a):
            i = calls["i"]
            calls["i"] += 1
            # held at step 0, then env truncates at step 1 while still successful.
            return True, False, (i == 1)

        out = probe_sustained(lambda: "hold", step_fn, k=10)
        assert out["sustained_steps"] == 2
        assert out["sustained"] is True


# --------------------------------------------------------------------- DP env-step budget
class _SyntheticEnv:
    """Minimal env stand-in: counts steps, NEVER succeeds, NEVER truncates."""

    def __init__(self):
        self.n_steps = 0

    def step(self, _action):
        self.n_steps += 1
        return {}, 0.0, False, False, {"success": False}


def _dp_budget_loop(env, max_env_steps, max_chunk_actions=6, steps_per_action=3):
    """Stripped replica of the PR7 eval_multi_dp accounting: count ENV steps, cap at
    max_env_steps, check success per env step, break on truncated. Returns (success, env_steps).

    This mirrors the exact control flow the real driver uses (outer while not success ->
    chunk loop -> TOPP-expanded env.step loop) without the torch/sapien dependencies."""
    env_steps = 0
    success = False
    while not success:
        if env_steps >= max_env_steps:
            break
        budget_hit = False
        for _i in range(max_chunk_actions):
            for _j in range(steps_per_action):
                _obs, _r, terminated, truncated, info = env.step({"panda-0": [0]})
                env_steps += 1
                if bool(info.get("success", False)):
                    success = True
                    break
                if bool(truncated) or env_steps >= max_env_steps:
                    budget_hit = True
                    break
            if success or budget_hit:
                break
    return success, env_steps


class TestDPEnvStepBudget:
    def test_terminates_at_env_step_cap_400(self):
        env = _SyntheticEnv()
        success, env_steps = _dp_budget_loop(env, max_env_steps=400)
        assert success is False
        # Cap is enforced at the ENV-step granularity: never exceeds 400.
        assert env_steps == 400
        assert env.n_steps == 400

    def test_cap_not_exceeded_with_uneven_chunks(self):
        # steps_per_action that does not divide the cap -> still stops at exactly the cap.
        env = _SyntheticEnv()
        _success, env_steps = _dp_budget_loop(env, max_env_steps=400, steps_per_action=7)
        assert env_steps == 400

    def test_steps_equal_env_steps_on_early_success(self):
        # An env that succeeds at the 5th env step -> loop stops at 5 env steps.
        class _SucceedAt5:
            def __init__(self):
                self.n = 0

            def step(self, _a):
                self.n += 1
                return {}, 0.0, False, False, {"success": self.n >= 5}

        env = _SucceedAt5()
        success, env_steps = _dp_budget_loop(env, max_env_steps=400)
        assert success is True
        assert env_steps == 5


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
