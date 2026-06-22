"""Success-persistence probe for eval drivers (solution_E6.md §PR7).

The LiftBarrier success criterion (``robofactory/tasks/lift_barrier.py:117-122``)
is INSTANTANEOUS: ``barrier.pose.p[...,2] > agents[0].robot.pose.p[0,2] + 0.15``.
There is no persistence and no grasp requirement, so a barrier that is momentarily
flicked above the threshold (a physics bobble, a brief lift that immediately drops)
counts as a success even if the arms never actually hold it up.

Every eval driver historically *broke the episode loop at the first success*, so the
headline number is ``success_first`` — "was the criterion ever met for one frame".
This module adds the second number the audit needs: ``success_sustained_K`` — after
the first frame the criterion is met, run K MORE env steps with a HOLD-QPOS action
(command the arms to stay exactly where they are) and re-check the criterion at +K.
If the barrier was genuinely held it stays up under a hold command; if the success
was a transient bobble it falls and ``success_sustained_K`` is False.

We NEVER silently redefine the headline. Drivers report BOTH ``success_first`` and
``success_sustained_K``; the PR7 decision gate compares them on the validation wave.

Design notes
------------
- The probe is driven entirely through two caller-supplied callables so it is
  agnostic to single-env vs vectorised, dict-action vs box-action, DP vs pi0.5:
    * ``hold_action_fn()`` -> the action that keeps the robot at its current qpos
      (the caller knows its own action format; e.g. for pi0.5 the absolute joint
      target is the current qpos, for DP the same).
    * ``step_fn(action)`` -> ``(success_bool, terminated_bool, truncated_bool)``;
      the caller wraps ``env.step`` and reads ``info["success"]`` exactly as it does
      in the main loop (same instantaneous criterion).
- The probe consumes part of the SAME 400-env-step budget — it does not get a fresh
  budget. The caller passes ``budget_left`` (env steps remaining before the cap) and
  the probe runs ``min(K, budget_left)`` extra steps. If fewer than K steps are left,
  ``sustained`` is reported over however many ran and ``sustained_steps`` records it
  so the JSON never claims a full-K hold it did not measure.
- A hold-qpos action is the honest persistence test: it asks "if the policy stopped
  acting RIGHT NOW, would the barrier stay up?" A bobble fails; a real grasp+lift
  holds. (Re-querying the policy for K more steps would instead test "can the policy
  keep holding", a different and weaker question that re-introduces policy variance.)
"""
from __future__ import annotations

from typing import Callable, Tuple

# Default persistence horizon (env steps) — the K in success_sustained_K.
SUSTAIN_K = 10


def probe_sustained(
    hold_action_fn: Callable[[], object],
    step_fn: Callable[[object], Tuple[bool, bool, bool]],
    k: int = SUSTAIN_K,
    budget_left: int = None,
) -> dict:
    """Run up to ``k`` hold-qpos env steps after a first success; report persistence.

    Parameters
    ----------
    hold_action_fn : () -> action
        Returns the hold-current-qpos action in the caller's own action format,
        re-evaluated each call (qpos may drift slightly under the hold controller).
    step_fn : action -> (success, terminated, truncated)
        Steps the env once and returns the instantaneous success flag plus the
        env's terminated/truncated flags (so the probe stops if the env ends early).
    k : int
        Persistence horizon. Defaults to ``SUSTAIN_K`` (10).
    budget_left : int | None
        Env steps remaining before the shared cap. ``None`` = no cap (run full ``k``).
        The probe runs ``min(k, budget_left)`` steps so it never overruns the budget.

    Returns
    -------
    dict with:
      - ``sustained`` (bool): criterion still met after the LAST probe step.
      - ``sustained_full`` (bool): True iff the full ``k`` steps ran AND criterion held
        at every probe step (a strict hold — never dipped below threshold).
      - ``sustained_steps`` (int): how many probe steps actually ran (<= k, <= budget).
      - ``min_success_during`` (bool): False if the criterion dropped at ANY probe step
        (i.e. the success was not continuously held).
    """
    if k <= 0:
        return {
            "sustained": True,
            "sustained_full": True,
            "sustained_steps": 0,
            "min_success_during": True,
        }
    n_steps = k if budget_left is None else max(0, min(k, int(budget_left)))
    if n_steps == 0:
        # No budget to probe — report the entry success conservatively as "unknown
        # held": sustained mirrors the entry (caller only calls this on a success),
        # but sustained_full is False because we could not run the full horizon.
        return {
            "sustained": True,
            "sustained_full": False,
            "sustained_steps": 0,
            "min_success_during": True,
        }

    last_success = True
    all_held = True
    ran = 0
    for _ in range(n_steps):
        succ, term, trunc = step_fn(hold_action_fn())
        ran += 1
        last_success = bool(succ)
        if not last_success:
            all_held = False
        if term or trunc:
            break
    return {
        "sustained": bool(last_success),
        "sustained_full": bool(all_held and ran == k),
        "sustained_steps": int(ran),
        "min_success_during": bool(all_held),
    }
