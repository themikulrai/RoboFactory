"""Inspect arm2's action distribution across training demos.

Reads all 150 LeRobot parquet episodes for the TSC wristcam dataset, extracts
per-arm gripper and joint-delta-magnitude time series, and produces summary
plots/statistics that should reveal whether arm2's training signal is
qualitatively different from arm0/1 (which would explain why arm2 specifically
fails at eval).

Output: PNGs + a JSON summary to --output_dir (default: debug_output/<script>_<date>/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/iris/u/mikulrai/data/RoboFactory/lerobot/robofactory_three_arm_stack_wristcam/data/chunk-000")
# Module-level placeholder; rebound in __main__ from --output_dir.
OUT_DIR: Path = Path("/tmp/_uninitialized_replace_me")

NUM_ARMS = 3


def per_arm_slice(action_or_state: np.ndarray, arm_i: int) -> np.ndarray:
    """Slice the 24-dim action/state into per-arm 8-dim block.

    Layout (matches converter and eval): per-arm [7 joints, 1 gripper].
    """
    s = arm_i * 8
    return action_or_state[..., s : s + 8]


def main() -> None:
    files = sorted(DATA_DIR.glob("episode_*.parquet"))
    print(f"Found {len(files)} episodes")

    # Per-arm aggregates
    grippers: list[list[np.ndarray]] = [[] for _ in range(NUM_ARMS)]  # arm -> list of (T,) arrays
    joint_delta_mags: list[list[np.ndarray]] = [[] for _ in range(NUM_ARMS)]
    active_step_fracs: list[list[float]] = [[] for _ in range(NUM_ARMS)]
    first_motion_steps: list[list[int]] = [[] for _ in range(NUM_ARMS)]
    ep_lengths: list[int] = []

    for fpath in files:
        df = pd.read_parquet(fpath)
        # action column is a list of 24 floats per row; stack to (T, 24)
        actions = np.stack(df["actions"].to_numpy())
        T = actions.shape[0]
        ep_lengths.append(T)
        for arm in range(NUM_ARMS):
            arm_act = per_arm_slice(actions, arm)  # (T, 8)
            joint_deltas = arm_act[:, :7]  # already deltas per converter
            gripper = arm_act[:, 7]
            grippers[arm].append(gripper)
            mag = np.linalg.norm(joint_deltas, axis=1)  # (T,)
            joint_delta_mags[arm].append(mag)
            # "Active" step = joint delta magnitude > 1e-3
            active = mag > 1e-3
            active_step_fracs[arm].append(active.mean())
            first_motion = int(np.argmax(active)) if active.any() else T
            first_motion_steps[arm].append(first_motion)

    print(f"Episode lengths: min={min(ep_lengths)} max={max(ep_lengths)} median={int(np.median(ep_lengths))}")

    # ---- Summary stats per arm ----
    summary: dict = {"num_episodes": len(files), "median_episode_length": int(np.median(ep_lengths))}
    for arm in range(NUM_ARMS):
        all_grippers = np.concatenate(grippers[arm])
        all_mags = np.concatenate(joint_delta_mags[arm])
        # Bimodality measure: how often is the gripper near the extremes vs middle
        near_open = (all_grippers > 0.7).mean()
        near_close = (all_grippers < -0.7).mean()
        in_middle = ((all_grippers >= -0.7) & (all_grippers <= 0.7)).mean()
        summary[f"arm{arm}"] = {
            "gripper_mean": float(all_grippers.mean()),
            "gripper_std": float(all_grippers.std()),
            "gripper_min": float(all_grippers.min()),
            "gripper_max": float(all_grippers.max()),
            "gripper_frac_near_open": float(near_open),
            "gripper_frac_near_close": float(near_close),
            "gripper_frac_middle": float(in_middle),
            "joint_delta_mag_mean": float(all_mags.mean()),
            "joint_delta_mag_std": float(all_mags.std()),
            "joint_delta_mag_p99": float(np.percentile(all_mags, 99)),
            "active_step_frac_mean": float(np.mean(active_step_fracs[arm])),
            "active_step_frac_std": float(np.std(active_step_fracs[arm])),
            "first_motion_step_mean": float(np.mean(first_motion_steps[arm])),
            "first_motion_step_median": float(np.median(first_motion_steps[arm])),
            "first_motion_step_p25": float(np.percentile(first_motion_steps[arm], 25)),
            "first_motion_step_p75": float(np.percentile(first_motion_steps[arm], 75)),
        }

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")
    print(json.dumps(summary, indent=2))

    # ---- Plots ----
    arm_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    arm_labels = ["arm0 (blue/cubeA)", "arm1 (green/cubeB)", "arm2 (red/cubeC)"]

    # 1. Histograms of gripper command across all timesteps, per arm
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    for arm in range(NUM_ARMS):
        all_grippers = np.concatenate(grippers[arm])
        axes[arm].hist(all_grippers, bins=80, color=arm_colors[arm], alpha=0.85)
        axes[arm].set_title(f"{arm_labels[arm]} — gripper action across all 150 demos")
        axes[arm].set_xlabel("gripper command")
        axes[arm].set_ylabel("count")
        axes[arm].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gripper_histograms.png")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'gripper_histograms.png'}")

    # 2. Heatmap of gripper command over time per episode (rows: episodes, cols: time)
    # Crop or pad to median episode length so we can stack
    target_T = int(np.median(ep_lengths))
    fig, axes = plt.subplots(NUM_ARMS, 1, figsize=(14, 9), dpi=120)
    for arm in range(NUM_ARMS):
        # Pad/crop each episode to target_T
        rows = []
        for g in grippers[arm]:
            if len(g) >= target_T:
                rows.append(g[:target_T])
            else:
                rows.append(np.concatenate([g, np.full(target_T - len(g), g[-1])]))
        mat = np.stack(rows)
        im = axes[arm].imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axes[arm].set_title(f"{arm_labels[arm]} — gripper command per episode (rows) over time (cols)")
        axes[arm].set_ylabel("episode index")
        axes[arm].set_xlabel("step")
        plt.colorbar(im, ax=axes[arm], shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gripper_heatmap_per_arm.png")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'gripper_heatmap_per_arm.png'}")

    # 3. Joint-delta-magnitude over time, mean ± std across episodes per arm
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    for arm in range(NUM_ARMS):
        rows = []
        for m in joint_delta_mags[arm]:
            if len(m) >= target_T:
                rows.append(m[:target_T])
            else:
                rows.append(np.concatenate([m, np.full(target_T - len(m), m[-1])]))
        mat = np.stack(rows)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        t = np.arange(target_T)
        axes[arm].plot(t, mean, color=arm_colors[arm], lw=1.5, label="mean")
        axes[arm].fill_between(t, mean - std, mean + std, color=arm_colors[arm], alpha=0.25, label="±1 std")
        axes[arm].set_title(f"{arm_labels[arm]} — joint delta magnitude")
        axes[arm].set_xlabel("step")
        axes[arm].set_ylabel("‖δq‖")
        axes[arm].grid(alpha=0.3)
        axes[arm].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "joint_delta_mag_per_arm.png")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'joint_delta_mag_per_arm.png'}")

    # 4. Distribution of first-motion-step per arm (when does each arm START moving?)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    for arm in range(NUM_ARMS):
        ax.hist(first_motion_steps[arm], bins=30, alpha=0.55, color=arm_colors[arm], label=arm_labels[arm])
    ax.set_title("First-motion step per arm across 150 demos\n(when does each arm first take a non-trivial joint action?)")
    ax.set_xlabel("step")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "first_motion_step_per_arm.png")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'first_motion_step_per_arm.png'}")


if __name__ == "__main__":
    import argparse
    from robofactory.utils.paths import debug_output_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=debug_output_dir(__file__),
        help="Directory for outputs (default: debug_output/<script>_<YYYYMMDD>/).",
    )
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
