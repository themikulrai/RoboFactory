"""Check whether action chunks are linearly predictable from qpos history.

For each per-arm zarr, fit Y = W @ X where:
  X = state[t-To:t].flatten()              shape (To*8,)
  Y = action[t:t+Ta].flatten()             shape (Ta*8,)

Split BY EPISODE (not by step) so episodes don't leak across train/test, then
report R^2 on the held-out test episodes (overall + per-action-dim) plus MSE.

If R^2 > 0.95 for a given arm, the dataset itself is essentially templated by
qpos, so no encoder-side fix can rescue eval -- demos would need to be
recollected.

CPU-only. Outputs a single small JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import zarr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def build_windows(
    state: np.ndarray,
    action: np.ndarray,
    episode_ends: np.ndarray,
    n_obs_steps: int,
    n_action_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, Y, episode_id) arrays respecting episode boundaries.

    For each episode of length L and each t in [To, L - Ta]:
      X = state[t-To:t].flatten()
      Y = action[t:t+Ta].flatten()
    Episodes too short for a single (To+Ta) window are skipped silently.
    """
    To = n_obs_steps
    Ta = n_action_steps

    # Episode boundaries: episode i spans [start, end) where
    # start = episode_ends[i-1] (or 0 for i=0), end = episode_ends[i].
    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    ends = episode_ends.astype(np.int64)

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    ep_ids: list[np.ndarray] = []

    for ep_idx, (s, e) in enumerate(zip(starts, ends)):
        L = int(e - s)
        # Need at least To steps of history and Ta steps of future.
        # t ranges over [To, L - Ta] inclusive -> first valid t is To,
        # last valid t is L - Ta. Number of windows = L - Ta - To + 1.
        n_windows = L - Ta - To + 1
        if n_windows <= 0:
            continue

        ep_state = state[s:e]
        ep_action = action[s:e]

        # Vectorized window construction.
        # t values: To, To+1, ..., L-Ta (inclusive)
        t_values = np.arange(To, L - Ta + 1, dtype=np.int64)

        # X: stack state[t-To:t] for each t -> (n_windows, To, 8)
        # Use broadcasting: indices = t_values[:, None] + (-To .. -1)
        x_idx = t_values[:, None] + np.arange(-To, 0)[None, :]
        X_ep = ep_state[x_idx].reshape(n_windows, -1)  # (n_windows, To*8)

        # Y: stack action[t:t+Ta] for each t -> (n_windows, Ta, 8)
        y_idx = t_values[:, None] + np.arange(0, Ta)[None, :]
        Y_ep = ep_action[y_idx].reshape(n_windows, -1)  # (n_windows, Ta*8)

        X_list.append(X_ep)
        Y_list.append(Y_ep)
        ep_ids.append(np.full(n_windows, ep_idx, dtype=np.int64))

    if not X_list:
        return (
            np.zeros((0, To * state.shape[1]), dtype=np.float32),
            np.zeros((0, Ta * action.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(Y_list, axis=0),
        np.concatenate(ep_ids, axis=0),
    )


def episode_split(
    n_episodes: int, test_frac: float, rng_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle episode indices with a fixed seed, take last test_frac as test."""
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(n_episodes)
    n_test = int(round(test_frac * n_episodes))
    n_test = max(1, min(n_episodes - 1, n_test))  # guarantee non-empty splits
    train_eps = np.sort(perm[:-n_test])
    test_eps = np.sort(perm[-n_test:])
    return train_eps, test_eps


def fit_one_arm(
    zarr_path: str,
    n_obs_steps: int,
    n_action_steps: int,
    test_frac: float,
    rng_seed: int,
    max_episodes: int | None = None,
) -> dict:
    """Fit LinearRegression on one zarr and return a result dict."""
    print(f"[{Path(zarr_path).stem}] opening zarr ...", flush=True)
    z = zarr.open(zarr_path, mode="r")
    state = z["data/state"][:]
    action = z["data/action"][:]
    episode_ends = z["meta/episode_ends"][:]

    if max_episodes is not None and max_episodes < len(episode_ends):
        episode_ends = episode_ends[:max_episodes]
        cutoff = int(episode_ends[-1])
        state = state[:cutoff]
        action = action[:cutoff]

    n_episodes = int(len(episode_ends))
    print(
        f"[{Path(zarr_path).stem}] state {state.shape} action {action.shape} "
        f"n_episodes={n_episodes}",
        flush=True,
    )

    X, Y, ep_ids = build_windows(
        state, action, episode_ends, n_obs_steps, n_action_steps
    )
    if X.shape[0] == 0:
        raise RuntimeError(
            f"No valid windows for {zarr_path}; episodes too short for To+Ta."
        )

    train_eps, test_eps = episode_split(n_episodes, test_frac, rng_seed)
    train_mask = np.isin(ep_ids, train_eps)
    test_mask = np.isin(ep_ids, test_eps)

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    print(
        f"[{Path(zarr_path).stem}] fitting LR on "
        f"X_train {X_train.shape} -> Y_train {Y_train.shape}",
        flush=True,
    )
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    r2_overall = float(
        r2_score(Y_test, Y_pred, multioutput="uniform_average")
    )
    r2_per_dim = r2_score(Y_test, Y_pred, multioutput="raw_values")
    mse_test = float(mean_squared_error(Y_test, Y_pred))

    return {
        "n_episodes": n_episodes,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_train_episodes": int(len(train_eps)),
        "n_test_episodes": int(len(test_eps)),
        "R2_overall": r2_overall,
        "R2_per_action_dim": [float(v) for v in r2_per_dim],
        "MSE_test": mse_test,
        "zarr_path": str(zarr_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zarr-paths",
        nargs="+",
        required=True,
        help="Per-arm zarr paths, e.g. --zarr-paths a.zarr b.zarr c.zarr",
    )
    p.add_argument("--n-obs-steps", type=int, default=3)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="(smoke test) cap episodes per zarr; default None = use all",
    )
    p.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arms_results: dict[str, dict] = {}
    for zp in args.zarr_paths:
        key = Path(zp).stem
        arms_results[key] = fit_one_arm(
            zarr_path=zp,
            n_obs_steps=args.n_obs_steps,
            n_action_steps=args.n_action_steps,
            test_frac=args.test_frac,
            rng_seed=args.rng_seed,
            max_episodes=args.max_episodes,
        )

    out = {
        "summary": {
            "To": args.n_obs_steps,
            "Ta": args.n_action_steps,
            "test_frac": args.test_frac,
            "rng_seed": args.rng_seed,
            "n_arms": len(args.zarr_paths),
        },
        "arms": arms_results,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Per-arm R2_overall ===", flush=True)
    for key, res in arms_results.items():
        print(
            f"  {key}: R2_overall = {res['R2_overall']:.4f}  "
            f"(MSE={res['MSE_test']:.6f}, "
            f"n_train={res['n_train_samples']}, n_test={res['n_test_samples']})",
            flush=True,
        )
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
