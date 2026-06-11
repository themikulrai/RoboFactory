"""Shared fixtures + synthetic-H5 builders for the DART tests.

These build tiny fake trajectory H5/JSON pairs that mimic the RecordEpisodeMA
on-disk layout closely enough for merge_h5 to operate on, WITHOUT any sim
import. Used by the PURE tests; the SIM tests build their data from a real
rollout instead.
"""
import json
import os.path as osp

import h5py
import numpy as np
import pytest


def make_fake_h5(h5_path, n_traj=2, T=4, n_agents=1, seed_base=100,
                 successes=None, write_json=True):
    """Create a fake multi-agent trajectory H5 + sibling JSON.

    Each traj group has:
      actions/panda-{a}  shape (T, 8)     (7 joints + gripper)
      obs/agent/panda-{a}/qpos shape (T+1, 9)
      success shape (T,)  (all True unless overridden)

    Returns the list of episode dicts written to the json.
    """
    if successes is None:
        successes = [True] * n_traj
    episodes = []
    with h5py.File(h5_path, "w") as f:
        for i in range(n_traj):
            g = f.create_group(f"traj_{i}", track_order=True)
            act = g.create_group("actions", track_order=True)
            obs = g.create_group("obs", track_order=True)
            agent = obs.create_group("agent", track_order=True)
            for a in range(n_agents):
                rng = np.random.default_rng(1000 * i + a)
                act.create_dataset(f"panda-{a}",
                                   data=rng.standard_normal((T, 8)).astype(np.float32))
                ag = agent.create_group(f"panda-{a}", track_order=True)
                ag.create_dataset("qpos",
                                  data=rng.standard_normal((T + 1, 9)).astype(np.float32))
            g.create_dataset("success",
                             data=np.full(T, successes[i], dtype=bool))
            episodes.append({
                "episode_id": i,
                "episode_seed": seed_base + i,
                "success": bool(successes[i]),
                "reset_kwargs": {"seed": seed_base + i},
                "elapsed_steps": T - 1,
            })
    if write_json:
        base, _ = osp.splitext(h5_path)
        with open(base + ".json", "w") as jf:
            json.dump({"env_info": {"env_id": "Fake-rf"}, "episodes": episodes}, jf,
                      indent=2)
    return episodes


@pytest.fixture
def fake_h5_builder():
    """Expose make_fake_h5 to tests (conftest itself is not importable)."""
    return make_fake_h5


@pytest.fixture
def scripted_pair(tmp_path):
    """A fake 'scripted' H5/JSON pair (2 trajs, single agent)."""
    h5 = tmp_path / "scripted" / "Task.h5"
    h5.parent.mkdir(parents=True, exist_ok=True)
    eps = make_fake_h5(str(h5), n_traj=2, T=4, n_agents=1, seed_base=100,
                       successes=[True, True])
    return str(h5), eps


@pytest.fixture
def dart_pair(tmp_path):
    """A fake 'dart' H5/JSON pair (2 trajs, single agent), distinct seeds."""
    h5 = tmp_path / "dart" / "Task.h5"
    h5.parent.mkdir(parents=True, exist_ok=True)
    eps = make_fake_h5(str(h5), n_traj=2, T=5, n_agents=1, seed_base=200,
                       successes=[True, True])
    return str(h5), eps
