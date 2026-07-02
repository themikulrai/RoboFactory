"""Unit tests for --per-arm-extra-obs=gripper_force state interleaving.

Force must land at index 8 WITHIN each arm's contiguous block ([q7, grip, force]),
so the server's per-arm slice [i*9:(i+1)*9] holds one arm. Tests both with and
without the flag against a fake obs dict.
"""
from __future__ import annotations

import numpy as np

from robofactory.policy.openpi_pi05.eval_decent_pi05 import _build_state, _gripper_from_qpos

ROBOT = "panda_wristcam_multi"


def _fake_obs():
    q0 = np.arange(9, dtype=np.float32)          # arm0 qpos
    q1 = np.arange(9, dtype=np.float32) + 100.0   # arm1 qpos
    return {
        "agent": {f"{ROBOT}-0": {"qpos": q0}, f"{ROBOT}-1": {"qpos": q1}},
        "extra": {"gripper_force_arm0": 1.5, "gripper_force_arm1": 2.5},
    }, q0, q1


def test_no_flag_is_8_per_arm():
    obs, q0, q1 = _fake_obs()
    s = _build_state(obs, 2, ROBOT)  # default per_arm_extra_obs=""
    assert s.shape == (16,)
    # arm0 block [0:8] = [q0[:7], grip0]
    np.testing.assert_array_equal(s[0:7], q0[:7])
    assert s[7] == _gripper_from_qpos(q0)
    # arm1 block [8:16] = [q1[:7], grip1]
    np.testing.assert_array_equal(s[8:15], q1[:7])
    assert s[15] == _gripper_from_qpos(q1)


def test_gripper_force_interleaved_per_arm():
    obs, q0, q1 = _fake_obs()
    s = _build_state(obs, 2, ROBOT, "gripper_force")
    assert s.shape == (18,)  # 2 * 9
    # arm0 block [0:9]
    np.testing.assert_array_equal(s[0:7], q0[:7])
    assert s[7] == _gripper_from_qpos(q0)
    assert s[8] == np.float32(1.5)          # force interleaved at index 8 of arm0
    # arm1 block [9:18]
    np.testing.assert_array_equal(s[9:16], q1[:7])
    assert s[16] == _gripper_from_qpos(q1)
    assert s[17] == np.float32(2.5)         # force interleaved at index 8 of arm1


def test_force_blocks_stay_contiguous():
    # the first 8 dims of the force state must equal the no-flag arm0 block
    obs, _, _ = _fake_obs()
    s_no = _build_state(obs, 2, ROBOT)
    s_f = _build_state(obs, 2, ROBOT, "gripper_force")
    np.testing.assert_array_equal(s_f[0:8], s_no[0:8])   # arm0 q7+grip unchanged
    np.testing.assert_array_equal(s_f[9:17], s_no[8:16])  # arm1 q7+grip unchanged


def test_dtype_float32():
    obs, _, _ = _fake_obs()
    assert _build_state(obs, 2, ROBOT, "gripper_force").dtype == np.float32


def test_unknown_extra_obs_ignored():
    # any value other than "gripper_force" is treated as off (8-dim per arm)
    obs, _, _ = _fake_obs()
    assert _build_state(obs, 2, ROBOT, "something_else").shape == (16,)
