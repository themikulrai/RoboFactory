"""Tests for the shared eval fidelity guards + DP runner env_kwargs (solution_E6.md §PR3).

Run with the RoboFactory env:
    cd /iris/u/mikulrai/projects/RoboFactory
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python -m pytest \
        robofactory/utils/test_eval_guards.py -q

Covers:
  - shader_bg_guard: hard-fail on a synthetic black-top frame, pass on a realistic frame,
    RF_ALLOW_SHADER_MISMATCH=1 downgrades to a warning, one-shot latch.
  - assert_shader_pack_default: raise when any cam cfg has shader_pack != 'default',
    pass when all 'default' / fail when missing.
  - assert_not_login_node: green inside SLURM / override / iris-ws, refuse otherwise.
  - Both DP runners build env_kwargs with enable_shadow is False and shader_pack=='default'
    (gym.make mocked to capture kwargs — no SAPIEN sim needed).
  - The multi-agent runner raises ValueError when config_path is None (silent robocasa
    default removed).
"""
import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RF_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DP_ROOT = os.path.join(_RF_ROOT, "robofactory", "policy", "Diffusion-Policy")
for p in (_RF_ROOT, _DP_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from robofactory.utils import eval_guards as G  # noqa: E402

ROBOCASA_CFG = os.path.join(
    _RF_ROOT, "robofactory", "configs", "robocasa", "three_robots_stack_cube.yaml"
)


@pytest.fixture(autouse=True)
def _reset_latch_and_env():
    """Re-arm the shader_bg_guard one-shot latch and clear the override env before each test."""
    G.shader_bg_guard(None, _reset=True)
    saved = os.environ.pop(G.SHADER_MISMATCH_OVERRIDE_ENV, None)
    yield
    if saved is not None:
        os.environ[G.SHADER_MISMATCH_OVERRIDE_ENV] = saved
    else:
        os.environ.pop(G.SHADER_MISMATCH_OVERRIDE_ENV, None)


# ----------------------------------------------------------------- shader_bg_guard

def _black_top_frame(h=96, w=128):
    """Frame whose upper sixth is pure black (the shader_pack=minimal symptom)."""
    f = np.full((h, w, 3), 120, dtype=np.uint8)  # gray body
    f[: h // 6] = 0  # black sky
    return f


def _realistic_frame(h=96, w=128):
    """Realistic non-black frame: a gray-ish sky over a brighter body."""
    f = np.full((h, w, 3), 180, dtype=np.uint8)
    f[: h // 6] = 90  # gray clear, well above the near-black threshold
    return f


def test_shader_bg_guard_fires_on_black_top():
    with pytest.raises(RuntimeError, match="black"):
        G.shader_bg_guard(_black_top_frame())


def test_shader_bg_guard_passes_on_realistic_frame():
    # Must NOT raise.
    G.shader_bg_guard(_realistic_frame())


def test_shader_bg_guard_override_downgrades_to_warning():
    os.environ[G.SHADER_MISMATCH_OVERRIDE_ENV] = "1"
    assert G.shader_mismatch_override_active() is True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        G.shader_bg_guard(_black_top_frame())  # must NOT raise
    assert any("near-black" in str(x.message) for x in w)


def test_shader_bg_guard_one_shot_latch():
    # First (black) frame fires; after that the latch is set, so a second black frame
    # is a no-op until reset. Re-arm, fire, then confirm the second call is silent.
    with pytest.raises(RuntimeError):
        G.shader_bg_guard(_black_top_frame())
    # latch is now set -> a fresh black frame must NOT raise.
    G.shader_bg_guard(_black_top_frame())


def test_shader_bg_guard_ignores_none_and_2d():
    G.shader_bg_guard(None)
    G.shader_bg_guard(None, _reset=True)
    G.shader_bg_guard(np.zeros((10, 10), dtype=np.uint8))  # ndim < 3 -> no-op


def test_override_inactive_by_default():
    assert G.shader_mismatch_override_active() is False


# --------------------------------------------------------- assert_shader_pack_default

def _kwargs(sp_sensor="default", sp_human="default", sp_viewer="default"):
    return dict(
        sensor_configs=dict(shader_pack=sp_sensor),
        human_render_camera_configs=dict(shader_pack=sp_human),
        viewer_camera_configs=dict(shader_pack=sp_viewer),
    )


def test_assert_shader_pack_default_passes_all_default():
    G.assert_shader_pack_default(_kwargs())  # must NOT raise


@pytest.mark.parametrize("bad", ["minimal", None, "rt", ""])
def test_assert_shader_pack_default_raises_on_bad_sensor(bad):
    with pytest.raises(RuntimeError, match="shader_pack guard FAILED"):
        G.assert_shader_pack_default(_kwargs(sp_sensor=bad))


def test_assert_shader_pack_default_raises_when_group_missing():
    kw = _kwargs()
    del kw["viewer_camera_configs"]
    with pytest.raises(RuntimeError, match="viewer_camera_configs"):
        G.assert_shader_pack_default(kw)


def test_assert_shader_pack_default_raises_on_minimal_human():
    with pytest.raises(RuntimeError, match="human_render_camera_configs"):
        G.assert_shader_pack_default(_kwargs(sp_human="minimal"))


# ----------------------------------------------------------------- assert_not_login_node

def test_assert_not_login_node_green_in_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.delenv("ROBOFACTORY_ALLOW_NON_SLURM_NODE", raising=False)
    G.assert_not_login_node()  # must NOT raise


def test_assert_not_login_node_green_with_override(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.setenv("ROBOFACTORY_ALLOW_NON_SLURM_NODE", "1")
    G.assert_not_login_node()  # must NOT raise


def test_assert_not_login_node_green_on_workstation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("ROBOFACTORY_ALLOW_NON_SLURM_NODE", raising=False)
    monkeypatch.setattr(G.socket, "gethostname", lambda: "iris-ws-12.stanford.edu")
    G.assert_not_login_node()  # must NOT raise


def test_assert_not_login_node_refuses_login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOBID", raising=False)
    monkeypatch.delenv("ROBOFACTORY_ALLOW_NON_SLURM_NODE", raising=False)
    monkeypatch.setattr(G.socket, "gethostname", lambda: "iris.stanford.edu")
    with pytest.raises(RuntimeError, match="REFUSING to run"):
        G.assert_not_login_node()


# ---------------------------------------------------- DP runner env_kwargs (mock gym.make)

def _capture_env_kwargs_joint(config_path):
    import gymnasium
    from diffusion_policy.env_runner.robot_joint_image_runner import RobotJointImageRunner

    captured = {}

    def _fake_make(env_id, **kwargs):
        captured.update(kwargs)
        return object()  # not used; _make_env returns it

    orig = gymnasium.make
    gymnasium.make = _fake_make
    try:
        r = RobotJointImageRunner(output_dir=None, config_path=config_path)
        r._make_env()
    finally:
        gymnasium.make = orig
    return captured


def _capture_env_kwargs_multi(config_path):
    import gymnasium
    from diffusion_policy.env_runner.robot_multi_agent_image_runner import (
        RobotMultiAgentImageRunner,
    )

    captured = {}

    def _fake_make(env_id, **kwargs):
        captured.update(kwargs)
        return object()

    orig = gymnasium.make
    gymnasium.make = _fake_make
    try:
        r = RobotMultiAgentImageRunner(output_dir=None, config_path=config_path)
        r._make_env()
    finally:
        gymnasium.make = orig
    return captured


def test_joint_runner_enable_shadow_false_and_default_shader():
    kw = _capture_env_kwargs_joint(ROBOCASA_CFG)
    assert kw["enable_shadow"] is False
    for key in ("sensor_configs", "human_render_camera_configs", "viewer_camera_configs"):
        assert kw[key]["shader_pack"] == "default"


def test_multi_runner_enable_shadow_false_and_default_shader():
    kw = _capture_env_kwargs_multi(ROBOCASA_CFG)
    assert kw["enable_shadow"] is False
    for key in ("sensor_configs", "human_render_camera_configs", "viewer_camera_configs"):
        assert kw[key]["shader_pack"] == "default"


def test_multi_runner_requires_explicit_scene_config():
    """Silent robocasa default removed: config_path=None must hard-fail with ValueError."""
    from diffusion_policy.env_runner.robot_multi_agent_image_runner import (
        RobotMultiAgentImageRunner,
    )

    r = RobotMultiAgentImageRunner(output_dir=None, config_path=None)
    with pytest.raises(ValueError, match="scene config .* is required"):
        r._make_env()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
