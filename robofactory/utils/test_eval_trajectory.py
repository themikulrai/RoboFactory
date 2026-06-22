"""PR9 tests: --save-trajectory (env_states + actions + qpos, absolute, no RGB).

GPU-free synthetic tests required by solution_E6.md §PR9. A real 2-seed DP sim is too
heavy on the CPU test host (needs torch/sapien + a trained ckpt), so per the spec we test
the trajectory writer + the h5 -> zarr round-trip on a small SYNTHETIC h5 that reproduces
the exact RecordEpisodeMA datagen schema (confirmed against LiftBarrier.h5):

  traj_N/
    obs/agent/<uid>-<i>/qpos     (T+1, 9) float32   proprio rides along
    obs/sensor_data/<cam>/rgb    (T+1, H, W, 3) u8   (present in a re-rendered traj)
    actions/panda-<i>            (T, 8)   float64    ABSOLUTE joint targets stepped
    env_states/.../...           (T+1, D) float32   re-renderable / success-recomputable
    success / terminated / truncated (T,) bool

Invariants asserted:
  A. round-trip h5 -> parse_h5_to_zarr_unified.py --state-source qpos:
       * zarr action count == total env-step count (sum of per-episode T)
       * data/state is derived from QPOS (== qpos[:T,:7]+qpos[:T,7]), NOT a copy of action
         (the converter default 'action' would copy action -> state; the qpos flag MUST be
          passed; digest B live-bug 3)
       * success recomputable from env_states (the synthetic env_states carry the barrier
         z height; recomputing the lift criterion reproduces the recorded `success` array)
  B. record_rgb=False drops obs/sensor_data while keeping obs/agent/.../qpos
     (drop_image_streams), and wrap_record_trajectory wires the storage discipline.
  C. pi0.5 absolute-decode-before-step: the action env.step RECEIVES (and the wrapper would
     therefore RECORD) equals cur_qpos + delta — i.e. recorded == stepped, no converter-side
     delta reconstruction (the capture-trap digest A flags).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]
CONVERTER = REPO / "robofactory" / "script" / "parse_h5_to_zarr_unified.py"
PYTHON = sys.executable


# --------------------------------------------------------------------- synthetic h5 builder
def _make_synthetic_h5(
    path: Path,
    n_agents: int = 2,
    ep_lengths=(5, 7),
    uid: str = "panda_wristcam_multi",
    H: int = 8,
    W: int = 8,
    with_rgb: bool = True,
    barrier_lift_z: float = 0.30,
    success_threshold: float = 0.25,
):
    """Write a RecordEpisodeMA-schema h5 with `len(ep_lengths)` episodes.

    Per episode of T env steps:
      * qpos has T+1 rows (initial + one per step); a small per-step ramp so qpos != action.
      * actions are ABSOLUTE targets (distinct from qpos so the qpos vs action state-source
        is distinguishable).
      * env_states/actors/barrier carries a 13-d pose whose z (index 2) ramps up; success
        is True exactly on the steps where barrier_z >= success_threshold.
    """
    rng = np.random.default_rng(0)
    seed_info = []
    with h5py.File(path, "w") as f:
        for ep_idx, T in enumerate(ep_lengths):
            g = f.create_group(f"traj_{ep_idx}", track_order=True)
            # ---- proprio: qpos (T+1, 9) per agent ----
            obs = g.create_group("obs", track_order=True)
            agent = obs.create_group("agent", track_order=True)
            for i in range(n_agents):
                ag = agent.create_group(f"{uid}-{i}", track_order=True)
                qpos = np.zeros((T + 1, 9), dtype=np.float32)
                # arm joints ramp; finger widths in a plausible [0.018, 0.04] band
                qpos[:, :7] = (np.arange(T + 1)[:, None] * 0.01 + i * 0.1).astype(np.float32)
                qpos[:, 7] = 0.02 + i * 0.001
                qpos[:, 8] = 0.02 + i * 0.001
                ag.create_dataset("qpos", data=qpos, dtype=np.float32)
                ag.create_dataset("qvel", data=np.zeros((T + 1, 9), np.float32), dtype=np.float32)
            # ---- optional RGB (a re-rendered traj has these; raw eval traj does not) ----
            if with_rgb:
                sd = obs.create_group("sensor_data", track_order=True)
                cams = [f"head_camera_agent{i}" for i in range(n_agents)] + ["head_camera_global"]
                cams += [f"hand_camera_{i}" for i in range(n_agents)]
                for cam in cams:
                    cg = sd.create_group(cam, track_order=True)
                    cg.create_dataset(
                        "rgb",
                        data=rng.integers(0, 255, size=(T + 1, H, W, 3), dtype=np.uint8),
                    )
            # ---- ABSOLUTE actions (T, 8) per agent (deliberately != qpos) ----
            acts = g.create_group("actions", track_order=True)
            for i in range(n_agents):
                a = np.zeros((T, 8), dtype=np.float64)
                a[:, :7] = (np.arange(T)[:, None] * 0.05 + 1.0 + i).astype(np.float64)
                a[:, 7] = 1.0 if i == 0 else -1.0  # gripper command in {-1,+1}
                acts.create_dataset(f"panda-{i}", data=a, dtype=np.float64)
            # ---- env_states: barrier z ramps up; success recomputable from it ----
            es = g.create_group("env_states", track_order=True)
            actors = es.create_group("actors", track_order=True)
            barrier = np.zeros((T + 1, 13), dtype=np.float32)
            barrier[:, 0] = 0.0  # x
            barrier[:, 2] = np.linspace(0.0, barrier_lift_z, T + 1).astype(np.float32)  # z lift
            actors.create_dataset("barrier", data=barrier, dtype=np.float32)
            # ---- success / terminated / truncated (T,) ----
            # success True on steps (1..T) where the post-step barrier z >= threshold.
            barrier_z_post = barrier[1:, 2]  # the env state AFTER each of the T steps
            success = (barrier_z_post >= success_threshold)
            g.create_dataset("success", data=success, dtype=bool)
            g.create_dataset("terminated", data=success, dtype=bool)  # term on success
            g.create_dataset("truncated", data=np.zeros(T, dtype=bool), dtype=bool)
            seed_info.append(dict(ep=ep_idx, T=T, n_success=int(success.sum())))
    return seed_info


# ------------------------------------------------------------------------ A. round-trip test
class TestConverterRoundTrip:
    def _run_converter(self, h5_path, out_zarr, agent_id, state_source):
        cmd = [
            PYTHON, str(CONVERTER),
            "--h5-path", str(h5_path),
            "--out-zarr", str(out_zarr),
            "--mode", "per_agent",
            "--agent-id", str(agent_id),
            "--camera-family", "workspace",
            "--state-source", state_source,
            "--resize", "8",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"converter failed:\nSTDOUT{proc.stdout}\nSTDERR{proc.stderr}"

    def test_action_count_equals_env_step_count(self, tmp_path):
        ep_lengths = (5, 7)
        h5 = tmp_path / "traj.h5"
        _make_synthetic_h5(h5, n_agents=2, ep_lengths=ep_lengths)
        out = tmp_path / "out.zarr"
        self._run_converter(h5, out, agent_id=0, state_source="qpos")

        root = zarr.open(str(out), mode="r")
        total_T = sum(ep_lengths)
        # action count == env-step count (one absolute action per env step).
        assert root["data/action"].shape[0] == total_T
        assert root["data/state"].shape[0] == total_T
        # episode_ends mark per-episode env-step boundaries.
        assert list(root["meta/episode_ends"][:]) == [5, 12]

    def test_state_source_qpos_differs_from_action(self, tmp_path):
        """The qpos flag MUST be passed: with --state-source qpos the state is the proprio
        projection (qpos[:T,:7] + qpos[:T,7]), NOT a copy of the action stream (which the
        converter default 'action' would produce). digest B live-bug 3."""
        ep_lengths = (5, 7)
        h5 = tmp_path / "traj.h5"
        _make_synthetic_h5(h5, n_agents=2, ep_lengths=ep_lengths)

        out_q = tmp_path / "out_qpos.zarr"
        out_a = tmp_path / "out_action.zarr"
        self._run_converter(h5, out_q, agent_id=0, state_source="qpos")
        self._run_converter(h5, out_a, agent_id=0, state_source="action")

        rq = zarr.open(str(out_q), mode="r")
        ra = zarr.open(str(out_a), mode="r")
        state_q = rq["data/state"][:]
        state_a = ra["data/state"][:]
        action = rq["data/action"][:]
        # action state-source copies the action stream...
        np.testing.assert_allclose(state_a, action)
        # ...qpos state-source does NOT (proprio is distinct from the absolute command).
        assert not np.allclose(state_q, action)

        # And the qpos state is exactly the proprio projection of agent-0's qpos.
        with h5py.File(h5, "r") as f:
            rows = []
            for ep_idx, T in enumerate(ep_lengths):
                q = f[f"traj_{ep_idx}/obs/agent/panda_wristcam_multi-0/qpos"][:]
                proj = np.concatenate([q[:T, :7], q[:T, 7:8]], axis=1)
                rows.append(proj)
            expected = np.concatenate(rows, axis=0).astype(np.float32)
        np.testing.assert_allclose(state_q, expected, rtol=0, atol=1e-6)

    def test_success_recomputable_from_env_states(self, tmp_path):
        """The recorded `success` array is reproducible from env_states alone — i.e. a
        self-training harvester can recompute success offline without RGB. We mirror the
        synthetic criterion (barrier z >= threshold) on the recorded env_states/barrier."""
        ep_lengths = (5, 7)
        thr = 0.25
        h5 = tmp_path / "traj.h5"
        _make_synthetic_h5(h5, ep_lengths=ep_lengths, success_threshold=thr)
        with h5py.File(h5, "r") as f:
            for ep_idx, T in enumerate(ep_lengths):
                barrier = f[f"traj_{ep_idx}/env_states/actors/barrier"][:]  # (T+1, 13)
                recorded = f[f"traj_{ep_idx}/success"][:]                   # (T,)
                # recompute from env_states post-step z height
                recomputed = barrier[1:, 2] >= thr
                np.testing.assert_array_equal(recorded, recomputed)
                # sanity: at least the criterion is exercised (some True near the end)
                assert recorded[-1] == np.True_


# ----------------------------------------------------------- B. record_rgb=False filter test
class TestRgbDrop:
    def test_drop_image_streams_keeps_qpos_drops_sensor_data(self):
        from robofactory.utils.wrappers.record import drop_image_streams

        obs = {
            "agent": {"panda_wristcam_multi-0": {"qpos": np.zeros((3, 9), np.float32)}},
            "sensor_data": {"head_camera_global": {"rgb": np.zeros((3, 8, 8, 3), np.uint8)}},
            "extra": {"foo": np.zeros(3)},
        }
        filtered = drop_image_streams(obs)
        assert "sensor_data" not in filtered          # heavy image streams dropped
        assert "agent" in filtered                     # proprio kept
        assert "extra" in filtered                     # other top-level keys kept
        assert "qpos" in filtered["agent"]["panda_wristcam_multi-0"]
        # input not mutated
        assert "sensor_data" in obs

    def test_no_rgb_h5_still_carries_qpos_for_qpos_state_source(self, tmp_path):
        """A raw eval trajectory (record_rgb=False) has NO sensor_data but DOES carry qpos.
        The converter's --state-source qpos reads qpos (not sensor_data), so the proprio
        state is recoverable from a no-RGB trajectory once RGB is re-rendered. We assert the
        no-RGB h5 lacks sensor_data but the qpos projection is intact."""
        ep_lengths = (4,)
        h5 = tmp_path / "norgb.h5"
        _make_synthetic_h5(h5, n_agents=2, ep_lengths=ep_lengths, with_rgb=False)
        with h5py.File(h5, "r") as f:
            tr = f["traj_0"]
            assert "sensor_data" not in tr["obs"]                 # no RGB recorded
            assert "panda_wristcam_multi-0" in tr["obs/agent"]    # proprio present
            q = tr["obs/agent/panda_wristcam_multi-0/qpos"][:]
            assert q.shape == (ep_lengths[0] + 1, 9)              # T+1 qpos rows

    def test_wrap_record_trajectory_config(self):
        """wrap_record_trajectory wires the storage discipline (no instantiation of the
        sapien env): inspect that the helper requests save_trajectory + record_rgb=False +
        save_video=False via a stub env that captures the RecordEpisodeMA kwargs."""
        import robofactory.utils.eval_trajectory as et

        captured = {}

        class _StubRecord:
            def __init__(self, env, output_dir, **kw):
                captured.update(kw)
                captured["output_dir"] = output_dir

        orig = et.RecordEpisodeMA
        et.RecordEpisodeMA = _StubRecord
        try:
            _, h5_path = et.wrap_record_trajectory(object(), "/tmp/_rf_pr9_test", "trajectory")
        finally:
            et.RecordEpisodeMA = orig
        assert captured["save_trajectory"] is True
        assert captured["save_video"] is False
        assert captured["record_rgb"] is False
        assert captured["record_env_state"] is True
        assert captured["record_observation"] is True
        assert captured["record_reward"] is False
        assert h5_path.endswith("/trajectory.h5")


# --------------------------------------------------- C. pi0.5 absolute-decode-before-step
class TestPi05AbsoluteDecode:
    def test_recorded_equals_stepped_absolute_target(self):
        """The pi0.5 client decodes delta->absolute (cur_qpos + delta) and passes that
        ABSOLUTE target to env.step. RecordEpisodeMA records whatever env.step receives, so
        recorded action == cur_qpos + delta for every step of a mocked episode. This is the
        capture-trap fix: no converter-side delta reconstruction is needed."""
        # import the real decoder used by the centralised pi0.5 client
        sys.path.insert(0, str(REPO / "robofactory" / "policy" / "openpi_pi05"))
        import importlib
        mod = importlib.import_module("eval_pi05")

        num_arms = 2
        rng = np.random.default_rng(1)
        action_prefix = "panda"
        for _ in range(20):  # many mocked steps
            cur_qpos = [rng.normal(size=7).astype(np.float32) for _ in range(num_arms)]
            # one chunk step = per-arm [delta_joints(7), gripper(1)]
            chunk_step = rng.normal(size=num_arms * 8).astype(np.float32)
            stepped = mod._delta_to_absolute_action(chunk_step, cur_qpos, num_arms, action_prefix)
            # what the env (and therefore RecordEpisodeMA) gets == absolute target
            for i in range(num_arms):
                s = i * 8
                expected_arm = cur_qpos[i] + chunk_step[s:s + 7]
                expected_grip = chunk_step[s + 7]
                got = stepped[f"{action_prefix}-{i}"]
                np.testing.assert_allclose(got[:7], expected_arm, rtol=0, atol=1e-6)
                np.testing.assert_allclose(got[7], expected_grip, rtol=0, atol=1e-6)
                # crucially: the stepped value is ABSOLUTE (= qpos + delta), not the raw delta
                assert not np.allclose(got[:7], chunk_step[s:s + 7])
