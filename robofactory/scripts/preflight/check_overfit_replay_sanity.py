"""Heavy preflight check: overfit-replay sanity on the eval-side pipeline.

Plan v2 Phase C2 stage S5 step 8 (heavy half).  The cheap consistency
suite (`scripts/preflight/train_eval_consistency.py`) covers byte-equal
config parity; the init-pose Wasserstein sibling
(`check_init_pose_wasserstein.py`) covers the *distributional* mismatch
between train and eval start states.  This script covers the most direct
end-to-end question:

    *If we take a single training-set episode, deterministically reset
    the eval env to its recorded init pose, open-loop replay the recorded
    action sequence, render observations from the env, push them through
    the eval-side preprocessing, and ask the policy "what action next?",
    do the predictions match the recorded actions?*

If a freshly-overfit policy can't replay its own training episode at
eval time, the train- and eval-side pipelines are not compatible — and
no amount of training-time loss tracking will reveal it.

Two modes, run side-by-side:

* ``chunk[0]`` — at each replay step ``t`` we feed the env-rendered
  ``obs[t]`` to the policy, take **only the first action** of the
  predicted chunk, MSE against ``actions[t]``.  This is the "horizon = 1"
  view of the pipeline.
* ``full chunk`` — at each replay step ``t`` we MSE the full predicted
  chunk ``(K, A)`` against ``actions[t : t+K]`` (clipped on episode end).

Divergence between the two is itself the diagnostic: if chunk[0] passes
but the full chunk fails, the bug is in the chunking/horizon (e.g. a
stale action-cache, a wrong-by-one step index, a horizon mismatch
between train and eval).

Format auto-detection (path suffix), mirrored from
``check_init_pose_wasserstein.py``:

* ``*.zarr`` -> our DP zarr dataset.  State at ``data/state``, actions at
  ``data/action``, episode boundaries at ``meta/episode_ends``.  The init
  pose for env reset is ``state[episode_start]``.
* anything else -> LeRobot HuggingFace ``datasets`` path.  We pull
  ``observation.state`` (or ``state``), ``action``, and ``episode_index``.

Output: a JSON report keyed for the SLURM launcher
(``passed``/``chunk0_passed``/``full_chunk_passed``) and an exit code
that is non-zero on any failure.

Standalone-script note: like ``check_init_pose_wasserstein.py``, this
file does NOT plug into the ``Check``/``CheckResult`` machinery in
``train_eval_consistency.py`` (per that module's docstring) — it lives
behind its own SLURM driver and emits its own JSON shape.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OverfitReplayReport:
    """JSON-serialisable shape returned by :func:`run_check`.

    ``chunk0_*`` lists are length ``n_steps_replayed``; ``full_chunk_*``
    lists are also length ``n_steps_replayed`` (one MSE per step over the
    chunk-suffix from that step, clipped on episode end).  In
    ``mode='first-only'`` the ``full_chunk_*`` fields are ``None``.
    """

    ckpt_path: str
    dataset_path: str
    episode_idx: int
    n_steps_replayed: int
    chunk0_mse_per_step: list[float]
    chunk0_mse_mean: float
    full_chunk_mse_per_step: Optional[list[float]]
    full_chunk_mse_mean: Optional[float]
    mse_tolerance: float
    chunk0_passed: bool
    full_chunk_passed: bool
    passed: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Dataset loading (format auto-detect, mirrored from the W1 sibling)
# ---------------------------------------------------------------------------


def load_episode(
    path: Path, episode_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(init_pose (D,), observations (T, ...), actions (T, A))``.

    Auto-detects format by suffix:
        - ``*.zarr`` -> our DP zarr dataset
        - else -> LeRobot HuggingFace ``datasets`` path
    """
    if path.suffix == ".zarr" or (
        path.is_dir()
        and (path / ".zarray").exists() is False
        and (path / "data").exists()
        and (path / "meta").exists()
    ):
        return _load_zarr_episode(path, episode_idx)
    return _load_lerobot_episode(path, episode_idx)


def _load_zarr_episode(
    path: Path, episode_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import zarr  # local import: keeps the test file importable without zarr.

    z = zarr.open(str(path), mode="r")
    state = z["data"]["state"]            # (T_total, D)
    action = z["data"]["action"]          # (T_total, A)
    episode_ends = np.asarray(z["meta"]["episode_ends"][:]).astype(int)
    starts = np.concatenate(([0], episode_ends[:-1])).astype(int)
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(
            f"episode_idx={episode_idx} out of range (n_episodes={len(episode_ends)})"
        )
    s, e = int(starts[episode_idx]), int(episode_ends[episode_idx])
    states = np.asarray(state[s:e]).astype(np.float64)
    actions = np.asarray(action[s:e]).astype(np.float64)
    init_pose = states[0].copy()
    return init_pose, states, actions


def _load_lerobot_episode(
    path: Path, episode_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from datasets import load_from_disk  # local import: heavy dep.

    ds = load_from_disk(str(path))
    if hasattr(ds, "keys") and "train" in ds.keys():
        ds = ds["train"]

    state_col = (
        "observation.state" if "observation.state" in ds.column_names else "state"
    )
    if "action" not in ds.column_names:
        raise ValueError(f"LeRobot dataset at {path} missing column 'action'")
    if "episode_index" not in ds.column_names:
        raise ValueError(f"LeRobot dataset at {path} missing column 'episode_index'")

    ep_idx = np.asarray(ds["episode_index"])
    mask = ep_idx == episode_idx
    if not mask.any():
        raise IndexError(f"episode_idx={episode_idx} not present in dataset")
    rows = np.where(mask)[0]
    states = np.asarray([ds[int(i)][state_col] for i in rows], dtype=np.float64)
    actions = np.asarray([ds[int(i)]["action"] for i in rows], dtype=np.float64)
    init_pose = states[0].copy()
    return init_pose, states, actions


# ---------------------------------------------------------------------------
# Default env / policy factories (lazy, never imported at module load)
# ---------------------------------------------------------------------------


def _default_env_factory(scene_config: Path, episode_idx: int):
    """Production env factory.  Imported lazily so login-node tests don't
    pull in SAPIEN.  ``episode_idx`` is forwarded for any per-episode
    deterministic reset hook the env may want to honour.
    """
    import gymnasium as gym
    import yaml

    cfg = yaml.safe_load(Path(scene_config).read_text())
    env_id = cfg["task_name"] + "-rf"
    env = gym.make(
        env_id,
        config=str(scene_config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
    )
    env._preflight_episode_idx = episode_idx  # informational only
    return env


def _default_policy_loader(ckpt_path: Path):
    """Production policy loader.  Imported lazily so module-level import
    works in a bare numpy env (and on the login node).

    Returns a callable ``policy(obs) -> action_chunk: np.ndarray`` where
    ``action_chunk`` is shape ``(K, A)``.  The eval-side preprocessing
    is the policy's responsibility, not this script's — the whole point
    is to detect mismatches between *that* preprocessing and what the
    policy was trained on.
    """
    # NOTE: we deliberately don't import torch / openpi at module level.
    from robofactory.evaluation.policy_loader import (  # type: ignore
        load_policy_for_eval,
    )

    return load_policy_for_eval(ckpt_path)


# ---------------------------------------------------------------------------
# Replay core
# ---------------------------------------------------------------------------


def _ensure_chunked(predicted: np.ndarray, action_dim: int) -> np.ndarray:
    """Coerce a policy output to a ``(K, A)`` chunk.

    Some policies emit ``(A,)`` (single action), others ``(K, A)``.  We
    treat the single-action case as ``K=1``.  ``A`` is checked against
    the recorded action width.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
    if pred.ndim != 2:
        raise ValueError(
            f"policy output must be (A,) or (K, A); got shape {pred.shape}"
        )
    if pred.shape[1] != action_dim:
        raise ValueError(
            f"policy action dim {pred.shape[1]} != recorded action dim {action_dim}"
        )
    if not np.all(np.isfinite(pred)):
        raise ValueError("policy produced non-finite (NaN/inf) action prediction")
    return pred


def replay_episode(
    env,
    policy: Callable[[Any], np.ndarray],
    init_pose: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    max_steps: int,
    mode: str,
) -> tuple[list[float], Optional[list[float]], int]:
    """Run the open-loop replay and collect per-step MSE in both modes.

    Algorithm (matching the master plan):
        1. ``env.reset(init_pose=init_pose)`` -> ``obs0``.
        2. For ``t`` in ``[0, n_steps)``:
             a. Feed the env-rendered ``obs_t`` to the policy.
             b. Predicted chunk ``(K, A)``.
             c. chunk0 MSE  := mean((pred[0] - actions[t]) ** 2)
             d. full  MSE   := mean((pred[:k] - actions[t:t+k]) ** 2)
                where k = min(K, T - t).
             e. ``env.step(actions[t])`` -> ``obs_{t+1}``.

    The recorded ``observations`` array is *not* used as policy input —
    that would defeat the whole point of testing the eval-side obs
    pipeline.  We only use it to validate length and (in tests) as a
    sanity reference.

    Returns ``(chunk0_mse_per_step, full_chunk_mse_per_step_or_None,
    n_steps_replayed)``.
    """
    if mode not in ("full", "first-only"):
        raise ValueError(f"mode must be 'full' or 'first-only'; got {mode!r}")

    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            f"observations/actions length mismatch: "
            f"obs={observations.shape[0]}, actions={actions.shape[0]}"
        )
    T, A = int(actions.shape[0]), int(actions.shape[1])
    n_steps = min(max_steps, T)

    # Reset the env to the recorded init pose.  The env contract is
    # ``reset(init_pose=...) -> (obs, info)``; tests inject a fake env
    # that honours this exactly.
    obs, _info = env.reset(init_pose=init_pose)

    chunk0: list[float] = []
    full: list[float] = [] if mode == "full" else None  # type: ignore[assignment]

    for t in range(n_steps):
        pred = _ensure_chunked(policy(obs), action_dim=A)
        K = pred.shape[0]

        # chunk[0] MSE: scalar over the action vector at step t.
        diff0 = pred[0] - actions[t]
        chunk0.append(float(np.mean(diff0 * diff0)))

        if mode == "full":
            k = min(K, T - t)
            diffk = pred[:k] - actions[t : t + k]
            full.append(float(np.mean(diffk * diffk)))  # type: ignore[union-attr]

        # Open-loop step with the *recorded* action.
        obs, _r, _term, _trunc, _info = env.step(actions[t])

    return chunk0, full, n_steps


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_check(
    ckpt_path: Path,
    dataset_path: Path,
    scene_config: Path,
    *,
    episode_idx: int = 0,
    max_steps: int = 50,
    mse_tolerance: float = 0.01,
    mode: str = "full",
    env_factory: Optional[Callable[[Path, int], Any]] = None,
    policy_loader: Optional[Callable[[Path], Callable[[Any], np.ndarray]]] = None,
    dataset_loader: Optional[
        Callable[[Path, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = None,
) -> dict:
    """Full pipeline used by ``__main__`` and unit tests.

    Dependency-injection seams (``env_factory``, ``policy_loader``,
    ``dataset_loader``) let tests run without SAPIEN/torch/zarr/datasets
    on the path.  In production all three default to lazy real loaders.
    """
    if env_factory is None:
        env_factory = _default_env_factory
    if policy_loader is None:
        policy_loader = _default_policy_loader
    if dataset_loader is None:
        dataset_loader = load_episode

    init_pose, observations, actions = dataset_loader(dataset_path, episode_idx)
    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            f"observations/actions length mismatch: "
            f"obs={observations.shape[0]}, actions={actions.shape[0]}"
        )

    policy = policy_loader(ckpt_path)
    env = env_factory(scene_config, episode_idx)
    try:
        chunk0_per_step, full_per_step, n_steps = replay_episode(
            env, policy, init_pose, observations, actions,
            max_steps=max_steps, mode=mode,
        )
    finally:
        if hasattr(env, "close"):
            env.close()

    chunk0_mean = float(np.mean(chunk0_per_step)) if chunk0_per_step else 0.0
    chunk0_passed = bool(chunk0_mean <= mse_tolerance)

    if mode == "full":
        full_mean = float(np.mean(full_per_step)) if full_per_step else 0.0
        full_passed = bool(full_mean <= mse_tolerance)
        full_per_step_out: Optional[list[float]] = list(full_per_step)
        full_mean_out: Optional[float] = full_mean
    else:
        full_per_step_out = None
        full_mean_out = None
        # In first-only mode we don't evaluate the full chunk; declare it
        # "passed" so the overall verdict reduces to chunk0_passed.
        full_passed = True

    overall_passed = bool(chunk0_passed and full_passed)

    report = OverfitReplayReport(
        ckpt_path=str(ckpt_path),
        dataset_path=str(dataset_path),
        episode_idx=int(episode_idx),
        n_steps_replayed=int(n_steps),
        chunk0_mse_per_step=list(chunk0_per_step),
        chunk0_mse_mean=chunk0_mean,
        full_chunk_mse_per_step=full_per_step_out,
        full_chunk_mse_mean=full_mean_out,
        mse_tolerance=float(mse_tolerance),
        chunk0_passed=chunk0_passed,
        full_chunk_passed=full_passed,
        passed=overall_passed,
    )
    return report.to_dict()


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-path", type=Path, required=True,
                    help="Path to the freshly-overfit policy checkpoint.")
    ap.add_argument("--dataset-path", type=Path, required=True,
                    help="Zarr dataset or LeRobot HF dataset (auto-detected by suffix).")
    ap.add_argument("--episode-idx", type=int, default=0,
                    help="Index of the training episode to replay.")
    ap.add_argument("--scene-config", type=Path, required=True,
                    help="Path to eval scene yaml, "
                         "e.g. configs/table/three_robots_stack_cube.yaml")
    ap.add_argument("--max-steps", type=int, default=50,
                    help="Cap on replay steps (the cheap-fast knob).")
    ap.add_argument("--mse-tolerance", type=float, default=0.01,
                    help="Per-mode mean-MSE pass threshold in normalized action space.")
    ap.add_argument("--mode", choices=("full", "first-only"), default="full",
                    help="'full' runs both chunk[0] and full-chunk; "
                         "'first-only' skips the full-chunk check.")
    ap.add_argument("--output-json", type=Path, required=True,
                    help="Where to write the JSON report.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = run_check(
        args.ckpt_path,
        args.dataset_path,
        args.scene_config,
        episode_idx=args.episode_idx,
        max_steps=args.max_steps,
        mse_tolerance=args.mse_tolerance,
        mode=args.mode,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    full_mean = report["full_chunk_mse_mean"]
    full_str = "-" if full_mean is None else f"{full_mean:.6f}"
    chunk0_verdict = "PASS" if report["chunk0_passed"] else "FAIL"
    full_verdict = "PASS" if report["full_chunk_passed"] else "FAIL"
    print(
        f"overfit_replay_sanity: "
        f"chunk0={chunk0_verdict} (mean MSE {report['chunk0_mse_mean']:.6f}), "
        f"full={full_verdict} (mean MSE {full_str}) "
        f"-> {args.output_json}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
