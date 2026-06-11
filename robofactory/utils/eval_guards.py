"""Shared hard-failing eval fidelity guards (solution_E6.md §PR3).

Three guards that every pi0.5/DP eval driver runs at startup. Before PR3 these
lived in three different states across drivers: the shader_pack assert and the
login-node refusal were hard-fail but ONLY in the hierarchical LB driver
(``eval_hierarchical_lift_barrier.py``); the rendered-frame background check was
a silent ``warnings.warn`` in ``eval_decent_pi05.py`` and absent everywhere else.
A wrong shader / wrong node / black-sky render does not crash — it silently
corrupts eval numbers (see memory/feedback_sapien_shader_pack_eval_mismatch.md),
so all three are now HARD failures, lifted here so every driver shares one
implementation and they can be unit-tested on CPU.

Guards:
  - ``assert_shader_pack_default(gym_make_kwargs)`` — fail unless shader_pack=="default"
    on all three camera-config groups (Gate G2). Always hard.
  - ``shader_bg_guard(global_img)`` — fail if the upper region of the first
    head_camera_global frame is almost entirely black (the SAPIEN shader_pack="minimal"
    / login-node black-sky symptom). PROMOTED warn -> hard fail; set
    ``RF_ALLOW_SHADER_MISMATCH=1`` to downgrade to a warning for deliberate debugging.
  - ``assert_not_login_node()`` — refuse to run on the iris login node (renders ~32%
    darker / blacks out head_camera_global). Always hard.
"""
from __future__ import annotations

import os
import socket
import warnings

import numpy as np

# ----------------------------------------------------------------- shader_pack assert

# The data-gen renderer (robofactory.planner.run) uses shader_pack="default" on all three
# camera-config groups. ManiSkill's fallback is "minimal", whose Color shader clears the
# framebuffer to pure black instead of "default"'s gray clear — silently blacking out the
# upper ~third of head_camera_global at eval (base_0_rgb_raw train-vs-eval RMSE jumps to
# ~0.53 vs ~0.005 for the wrist cams). See memory/feedback_sapien_shader_pack_eval_mismatch.md.
REQUIRED_SHADER_PACK = "default"
_SHADER_CONFIG_KEYS = ("sensor_configs", "human_render_camera_configs", "viewer_camera_configs")

# Env override: downgrade shader_bg_guard from a hard fail back to a warning for
# deliberate debugging. Recorded in the result JSON by callers (see
# shader_mismatch_override_active) so a downgraded run is never mistaken for a clean one.
SHADER_MISMATCH_OVERRIDE_ENV = "RF_ALLOW_SHADER_MISMATCH"


def assert_shader_pack_default(gym_make_kwargs: dict) -> None:
    """Fail loudly unless shader_pack=="default" on ALL three camera-config groups.

    Gate G2: a wrong/absent shader_pack does not crash — it silently corrupts the global
    camera background, so we MUST assert rather than warn. Call this on the exact kwargs
    dict passed to gym.make (single source of truth)."""
    bad = []
    for key in _SHADER_CONFIG_KEYS:
        cfg = gym_make_kwargs.get(key)
        sp = cfg.get("shader_pack") if isinstance(cfg, dict) else None
        if sp != REQUIRED_SHADER_PACK:
            bad.append(f"{key}.shader_pack={sp!r}")
    if bad:
        raise RuntimeError(
            "SAPIEN shader_pack guard FAILED (Gate G2): expected shader_pack="
            f"{REQUIRED_SHADER_PACK!r} on {list(_SHADER_CONFIG_KEYS)}, but got: "
            f"{', '.join(bad)}. ManiSkill's 'minimal' fallback blacks out the upper region of "
            "head_camera_global vs the gray clear used at data-gen, silently corrupting eval. "
            "See memory/feedback_sapien_shader_pack_eval_mismatch.md."
        )


# ------------------------------------------------------------- rendered-frame bg guard

# One-shot guard for the SAPIEN shader_pack regression at the pixel level. Even with the
# correct gym.make kwargs, a login-node render or a stale env can still produce a black
# sky. We check the upper sixth of the first head_camera_global frame; if it's almost
# entirely near-black this is the shader_pack="minimal" / login-node symptom.
# PROMOTED from warn -> hard fail (PR3). See memory/feedback_sapien_shader_pack_eval_mismatch.md.
_SHADER_BG_CHECK_DONE = False


def shader_mismatch_override_active() -> bool:
    """True iff RF_ALLOW_SHADER_MISMATCH=1 — callers record this in the result JSON so a
    deliberately-downgraded (warn-only) run is never confused with a clean one."""
    return os.environ.get(SHADER_MISMATCH_OVERRIDE_ENV) == "1"


def shader_bg_guard(global_img: "np.ndarray | None", _reset: bool = False) -> None:
    """Hard-fail (RuntimeError) if the first head_camera_global frame is black-skied.

    Checks the upper sixth of the frame on the very first call; if >90% near-black this
    is the shader_pack="minimal" / login-node symptom that silently desyncs eval from
    training. Runs once per process (the symptom is constant across a run). Set
    RF_ALLOW_SHADER_MISMATCH=1 to downgrade to a warning. ``_reset=True`` re-arms the
    one-shot latch (test-only)."""
    global _SHADER_BG_CHECK_DONE
    if _reset:
        _SHADER_BG_CHECK_DONE = False
        return
    if _SHADER_BG_CHECK_DONE:
        return
    _SHADER_BG_CHECK_DONE = True
    if global_img is None or getattr(global_img, "ndim", 0) < 3:
        return
    upper = global_img[: max(1, global_img.shape[0] // 6)]
    near_black_frac = float((upper.sum(axis=-1) < 6).mean())
    if near_black_frac <= 0.9:
        return
    msg = (
        f"[shader_pack guard] head_camera_global upper region is {100*near_black_frac:.0f}% "
        "near-black. This is the SAPIEN shader_pack=minimal / login-node black-sky symptom, "
        "which silently desyncs eval from training. Pass "
        'sensor_configs=dict(shader_pack="default") to gym.make and run on a compute node. '
        "See ~/.claude/projects/-iris-u-mikulrai/memory/feedback_sapien_shader_pack_eval_mismatch.md"
    )
    if shader_mismatch_override_active():
        warnings.warn(
            msg + f"  [{SHADER_MISMATCH_OVERRIDE_ENV}=1 set: downgraded to warning]",
            stacklevel=2,
        )
        return
    raise RuntimeError(
        msg + f"  (Set {SHADER_MISMATCH_OVERRIDE_ENV}=1 to downgrade to a warning for "
        "deliberate debugging; that override is recorded in the result JSON.)"
    )


# ----------------------------------------------------------------- login-node refusal


def _is_compute_node() -> bool:
    """True iff we are clearly on a compute node (not the login node).

    Primary green signal: a SLURM allocation (SLURM_JOB_ID set) — a batch/interactive job
    is by construction on a compute node. Secondary: an explicit override env var for
    interactive runs on workstation nodes (iris-ws-*), since those render correctly but
    carry no SLURM_JOB_ID. The iris LOGIN node has neither and is refused."""
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"):
        return True
    if os.environ.get("ROBOFACTORY_ALLOW_NON_SLURM_NODE") == "1":
        return True
    host = socket.gethostname().split(".")[0]
    # Workstation nodes (iris-ws-*) render correctly; allow interactive use on them.
    if host.startswith("iris-ws"):
        return True
    return False


def assert_not_login_node() -> None:
    """HARD refusal to run on the iris login node (renders ~32% darker / black framebuffer).

    Gate G2: eval numbers from the login node are untrustworthy, so refuse rather than warn.
    Green iff inside a SLURM allocation OR on a workstation node OR explicit override."""
    if _is_compute_node():
        return
    host = socket.gethostname()
    raise RuntimeError(
        f"REFUSING to run SAPIEN eval on host {host!r}: no SLURM_JOB_ID and not a workstation "
        "node. The iris login node renders ~32% darker / blacks out head_camera_global, which "
        "silently corrupts eval. Run inside a SLURM allocation on a COMPUTE node (sbatch / "
        "srun / interactive). If you are knowingly on a correctly-rendering interactive node, "
        "set ROBOFACTORY_ALLOW_NON_SLURM_NODE=1. See "
        "memory/feedback_sapien_shader_pack_eval_mismatch.md."
    )
