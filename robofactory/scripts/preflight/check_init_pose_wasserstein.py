"""Heavy preflight check: per-dim Wasserstein-1 on init-pose distributions.

Plan v2 Phase C2 stage S5 (heavy half).  The cheap consistency suite
(`scripts/preflight/train_eval_consistency.py`) covers byte-equal config
parity; this script covers the *distributional* mismatch that those checks
can't see: a 2-cm or 0.5-deg shift between train and eval init poses
silently breaks rollouts ("model never saw this start state").

Why per-dim Wasserstein-1 and not joint KL:
- KL on a 7-D continuous joint state is noise-dominated at N=50 — you'd
  need O(10^d) samples for a tight estimate.
- Wasserstein-1 has a closed-form 1-D estimator (`scipy.stats.wasserstein_distance`),
  is robust at small N, and gives a *physically interpretable* number
  (mean transport distance in radians or metres) we can directly threshold
  against the plan's 2-cm / 0.5-deg tolerance.
- We bootstrap a 95% CI per-dim so a borderline pass-with-noise doesn't
  silently approve a real shift.

Format auto-detection (path suffix):
- ``*.zarr`` -> zarr DP dataset at /iris/u/mikulrai/data/RoboFactory/zarr_data/.
  State at ``data/state`` (per-step joint vector), episode boundaries at
  ``meta/episode_ends`` (int array).
- anything else -> LeRobot HuggingFace ``datasets`` path. We pull the
  per-step state column (``observation.state`` or ``state``) and the
  per-step ``episode_index`` column to slice out first-of-episode rows.

Output: a JSON report with per-dim Wasserstein-1 distance, bootstrap 95%
CI, the cm/deg-converted tolerance verdict, and an exit code that is
non-zero iff *any* per-dim distance exceeds its tolerance — so a SLURM
launcher can fail the job before training starts.

Standalone-script note: this file does NOT plug into the
``Check`` / ``CheckResult`` machinery in ``train_eval_consistency.py``.
The cheap suite is in-process and shares train+eval YAMLs; this heavy
check needs a sim env reset and a dataset on disk, so it lives behind
its own SLURM launcher (per the consistency-suite docstring) and emits
its own JSON report shape.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Convention: the joint init-pose vector layout this script assumes is
#     [<rot dims, in radians>, <pos dims, in metres>]
# with the last ``N_POS_DIMS`` entries being the (x, y, z) end-effector
# position.  This matches the Panda 7-DoF + xyz layout used by the canonical
# ManiSkill loggers.  If your dataset stores qpos only (no xyz), pass
# ``--n-pos-dims 0`` so every dim is treated as rotational.
DEFAULT_N_POS_DIMS = 3


@dataclasses.dataclass
class DimReport:
    dim: int
    kind: str                # "pos" or "rot"
    wasserstein_native: float  # native units: m for pos, rad for rot
    wasserstein_human: float   # cm for pos, deg for rot
    ci_low_human: float
    ci_high_human: float
    tolerance_human: float
    passed: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_train_init_poses(path: Path, n_samples: int) -> np.ndarray:
    """Return the first state of each episode, up to ``n_samples`` episodes.

    Auto-detects format by suffix:
        - ``*.zarr`` -> our DP zarr dataset
        - else -> LeRobot HuggingFace ``datasets`` path
    """
    if path.suffix == ".zarr" or (path.is_dir() and (path / ".zarray").exists() is False
                                   and (path / "data").exists() and (path / "meta").exists()):
        return _load_zarr_init_poses(path, n_samples)
    return _load_lerobot_init_poses(path, n_samples)


def _load_zarr_init_poses(path: Path, n_samples: int) -> np.ndarray:
    import zarr  # local import: keeps the test file importable without zarr.

    z = zarr.open(str(path), mode="r")
    state = z["data"]["state"][:]            # (T, D)
    episode_ends = z["meta"]["episode_ends"][:]  # (E,)
    # First state of episode i is at index 0 (i==0) or episode_ends[i-1] (i>0).
    starts = np.concatenate(([0], episode_ends[:-1])).astype(int)
    starts = starts[:n_samples]
    return state[starts].astype(np.float64)


def _load_lerobot_init_poses(path: Path, n_samples: int) -> np.ndarray:
    """Pull first-of-episode state from a HuggingFace LeRobot dataset.

    We try ``observation.state`` then ``state`` as the column name; LeRobot
    ships both conventions across versions.
    """
    from datasets import load_from_disk  # local import: heavy dep.

    ds = load_from_disk(str(path))
    # Some LeRobot exports nest a split; flatten if so.
    if hasattr(ds, "keys") and "train" in ds.keys():
        ds = ds["train"]

    state_col = "observation.state" if "observation.state" in ds.column_names else "state"
    ep_col = "episode_index"
    if ep_col not in ds.column_names:
        raise ValueError(f"LeRobot dataset at {path} missing column {ep_col!r}")

    ep_idx = np.asarray(ds[ep_col])
    # Index of first appearance of each episode id, in original order.
    _, first_idx = np.unique(ep_idx, return_index=True)
    first_idx = np.sort(first_idx)[:n_samples]
    states = np.asarray([ds[int(i)][state_col] for i in first_idx], dtype=np.float64)
    return states


# ---------------------------------------------------------------------------
# Eval-env sampling
# ---------------------------------------------------------------------------


def sample_eval_init_poses(
    eval_config: Path,
    n_samples: int,
    seed_start: int = 100,
    *,
    env_factory: Optional[Any] = None,
    proprio_extractor: Optional[Any] = None,
) -> np.ndarray:
    """Reset the eval env ``n_samples`` times with seeds [seed_start, seed_start+n).

    ``env_factory`` and ``proprio_extractor`` are injection points used by
    tests so we don't import gymnasium / SAPIEN on the login node.  In
    production both default to the real ``gym.make`` and the standard
    ``obs['agent']['qpos']`` extractor.
    """
    if env_factory is None:
        env_factory = _default_env_factory
    if proprio_extractor is None:
        proprio_extractor = _default_proprio_extractor

    env = env_factory(eval_config)
    try:
        out = []
        for k in range(n_samples):
            obs, _info = env.reset(seed=seed_start + k)
            out.append(np.asarray(proprio_extractor(obs), dtype=np.float64))
    finally:
        if hasattr(env, "close"):
            env.close()
    return np.stack(out, axis=0)


def _default_env_factory(eval_config: Path):
    """Production env factory.  Imported lazily so login-node tests don't
    pull in SAPIEN."""
    import gymnasium as gym
    import yaml

    cfg = yaml.safe_load(Path(eval_config).read_text())
    env_id = cfg["task_name"] + "-rf"
    return gym.make(
        env_id,
        config=str(eval_config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
    )


def _default_proprio_extractor(obs: dict) -> np.ndarray:
    """Best-effort extractor for the qpos vector from a SAPIEN reset obs.

    Tries ``obs['agent']['qpos']`` (single-arm) and ``obs['agent']['qpos_*']``
    (multi-arm) before giving up.
    """
    agent = obs.get("agent") if isinstance(obs, dict) else None
    if agent is None:
        raise KeyError("reset obs has no 'agent' key; cannot extract qpos")
    if "qpos" in agent:
        q = agent["qpos"]
    else:
        # multi-arm: concatenate qpos_0, qpos_1, ...
        keys = sorted(k for k in agent.keys() if k.startswith("qpos"))
        if not keys:
            raise KeyError(f"no qpos field in agent dict: keys={list(agent.keys())}")
        q = np.concatenate([np.asarray(agent[k]).reshape(-1) for k in keys])
    q = np.asarray(q).reshape(-1)
    return q


# ---------------------------------------------------------------------------
# Wasserstein-1 + bootstrap CI
# ---------------------------------------------------------------------------


def per_dim_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W1 per dim, CI-low per dim, CI-high per dim) in *native* units.

    a, b: (N, D) and (M, D) — N and M may differ.  Bootstrap resamples
    *with replacement* from each side independently and recomputes W1.
    """
    from scipy.stats import wasserstein_distance

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}")
    n, d = a.shape
    m = b.shape[0]

    point = np.array([wasserstein_distance(a[:, k], b[:, k]) for k in range(d)])

    # Bootstrap: resample both sides, recompute per-dim W1 each iteration.
    boot = np.empty((bootstrap_iters, d), dtype=np.float64)
    for it in range(bootstrap_iters):
        ai = rng.integers(0, n, size=n)
        bi = rng.integers(0, m, size=m)
        ar = a[ai]
        br = b[bi]
        for k in range(d):
            boot[it, k] = wasserstein_distance(ar[:, k], br[:, k])

    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    return point, ci_low, ci_high


# ---------------------------------------------------------------------------
# Tolerance verdict
# ---------------------------------------------------------------------------


def make_dim_reports(
    point: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    *,
    n_pos_dims: int,
    pos_tolerance_cm: float,
    rot_tolerance_deg: float,
) -> list[DimReport]:
    """Convert native-unit W1 to cm/deg, apply per-dim tolerance.

    Convention: the *last* ``n_pos_dims`` entries of the joint state vector
    are translation (xyz) in metres; everything before is rotation in
    radians.  Pass ``n_pos_dims=0`` if your state is qpos-only.
    """
    d = point.shape[0]
    out: list[DimReport] = []
    for k in range(d):
        is_pos = k >= d - n_pos_dims and n_pos_dims > 0
        if is_pos:
            scale = 100.0  # m -> cm
            tol = pos_tolerance_cm
            kind = "pos"
        else:
            scale = 180.0 / math.pi  # rad -> deg
            tol = rot_tolerance_deg
            kind = "rot"
        w_human = float(point[k] * scale)
        out.append(DimReport(
            dim=k,
            kind=kind,
            wasserstein_native=float(point[k]),
            wasserstein_human=w_human,
            ci_low_human=float(ci_low[k] * scale),
            ci_high_human=float(ci_high[k] * scale),
            tolerance_human=float(tol),
            passed=w_human <= tol,
        ))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_check(
    train_data_path: Path,
    eval_config: Path,
    *,
    n_samples: int,
    bootstrap_iters: int,
    pos_tolerance_cm: float,
    rot_tolerance_deg: float,
    n_pos_dims: int = DEFAULT_N_POS_DIMS,
    seed_start: int = 100,
    rng_seed: int = 0,
    env_factory: Optional[Any] = None,
    proprio_extractor: Optional[Any] = None,
) -> dict:
    """Full pipeline used by ``__main__`` and unit tests.  Returns the
    JSON-serialisable report dict."""
    train = load_train_init_poses(train_data_path, n_samples)
    eval_ = sample_eval_init_poses(
        eval_config, n_samples,
        seed_start=seed_start,
        env_factory=env_factory,
        proprio_extractor=proprio_extractor,
    )
    if train.shape[1] != eval_.shape[1]:
        raise ValueError(
            f"train/eval state-dim mismatch: train={train.shape[1]}, eval={eval_.shape[1]}"
        )

    rng = np.random.default_rng(rng_seed)
    point, ci_low, ci_high = per_dim_wasserstein(
        train, eval_, bootstrap_iters=bootstrap_iters, rng=rng,
    )
    dims = make_dim_reports(
        point, ci_low, ci_high,
        n_pos_dims=n_pos_dims,
        pos_tolerance_cm=pos_tolerance_cm,
        rot_tolerance_deg=rot_tolerance_deg,
    )
    n_fail = sum(1 for d in dims if not d.passed)
    return {
        "train_data_path": str(train_data_path),
        "eval_config": str(eval_config),
        "n_train_samples": int(train.shape[0]),
        "n_eval_samples": int(eval_.shape[0]),
        "state_dim": int(train.shape[1]),
        "n_pos_dims": int(n_pos_dims),
        "pos_tolerance_cm": float(pos_tolerance_cm),
        "rot_tolerance_deg": float(rot_tolerance_deg),
        "bootstrap_iters": int(bootstrap_iters),
        "seed_start": int(seed_start),
        "per_dim": [d.to_dict() for d in dims],
        "n_fail": int(n_fail),
        "passed_overall": bool(n_fail == 0),
    }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data-path", type=Path, required=True,
                    help="LeRobot dataset path or zarr path (auto-detected by suffix).")
    ap.add_argument("--eval-config", type=Path, required=True,
                    help="Path to scene yaml, e.g. configs/table/three_robots_stack_cube.yaml")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--pos-tolerance-cm", type=float, default=0.5)
    ap.add_argument("--rot-tolerance-deg", type=float, default=0.5)
    ap.add_argument("--n-pos-dims", type=int, default=DEFAULT_N_POS_DIMS,
                    help="Number of trailing dims that are xyz translation (m). "
                         "Set to 0 if state is qpos-only.")
    ap.add_argument("--seed-start", type=int, default=100,
                    help="Eval reset seed start; matches /iris/u/mikulrai/runs/eval_seeds_60.txt.")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="Bootstrap RNG seed (for reproducible CI bounds).")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="Where to write the JSON report.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = run_check(
        args.train_data_path,
        args.eval_config,
        n_samples=args.n_samples,
        bootstrap_iters=args.bootstrap_iters,
        pos_tolerance_cm=args.pos_tolerance_cm,
        rot_tolerance_deg=args.rot_tolerance_deg,
        n_pos_dims=args.n_pos_dims,
        seed_start=args.seed_start,
        rng_seed=args.rng_seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(
        f"init_pose_wasserstein: "
        f"{report['state_dim'] - report['n_fail']}/{report['state_dim']} "
        f"dims within tolerance -> {args.out_json}"
    )
    return 0 if report["passed_overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
