"""Publication-quality figures for the RoboFactory pi0.5 TSC debug study.

Outputs (all in --output_dir, default: debug_output/<script>_<date>/):
  fig1_sr_comparison.png      — SR bar chart across all experiments
  fig2_arm_imbalance.png      — Per-arm activity / gripper bias
  fig3_cube_scatter.png       — cubeC initial positions, success vs failure
  fig4_gripper_heatmap.png    — Gripper command heatmap per arm (all 150 demos)

Note: this script still reads sister-script outputs from hardcoded legacy paths
under /iris/u/mikulrai/logs/tsc_debug/{arm2_demos,seed_poses}/. To consume
freshly-regenerated inputs from sister scripts run with their new --output_dir
defaults, copy or symlink them into the legacy paths until those reads get
their own --input_dir flags.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# Module-level placeholder; rebound in __main__ from --output_dir.
OUT_DIR: Path = Path("/tmp/_uninitialized_replace_me")

ARM_LABELS = ["Arm 0\n(picks cubeA)", "Arm 1\n(picks cubeB)", "Arm 2\n(picks cubeC)"]
ARM_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

SUCCESS_SEEDS = {10014, 10023, 10028, 10029, 10030, 10034, 10035, 10036, 10038}

# ── Figure 1: Success-rate comparison ─────────────────────────────────────────

def fig1_sr_comparison() -> None:
    experiments = [
        ("Cent\nbaseline\n(0 / 30)", 0 / 30),
        ("Cent\nmask extra_0\n(0 / 5)*", 0 / 5),
        ("Cent\nreplan=1\n(0 / 5)", 0 / 5),
        ("Decent\nbaseline\n(9 / 30)", 9 / 30),
        ("Decent\nrerun\n(9 / 30)", 9 / 30),
    ]
    labels, srs = zip(*experiments)
    colors = ["#c44e52", "#c44e52", "#c44e52", "#4C72B0", "#4C72B0"]
    hatches = ["", "//", "", "", "//"]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    bars = ax.bar(range(len(labels)), [s * 100 for s in srs],
                  color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for i, (bar, sr) in enumerate(zip(bars, srs)):
        n = [30, 5, 5, 30, 30][i]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{sr * 100:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 42)
    ax.set_title("Eval success rates across experiments (pi0.5 LoRA, TSC task)", fontweight="bold")

    note = ax.text(0.99, 0.97, "* confounded test (train/eval mismatch)",
                   transform=ax.transAxes, ha="right", va="top",
                   fontsize=8, color="#888888", style="italic")

    red_patch  = mpatches.Patch(color="#c44e52", label="Centralized (cent)")
    blue_patch = mpatches.Patch(color="#4C72B0", label="Decentralized (decent)")
    ax.legend(handles=[red_patch, blue_patch], loc="upper right", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig1_sr_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Arm imbalance ────────────────────────────────────────────────────

def fig2_arm_imbalance() -> None:
    summary_path = Path("/iris/u/mikulrai/logs/tsc_debug/arm2_demos/summary.json")
    summary = json.loads(summary_path.read_text())

    active_frac   = [summary[f"arm{i}"]["active_step_frac_mean"] * 100 for i in range(3)]
    closed_frac   = [summary[f"arm{i}"]["gripper_frac_near_close"] * 100 for i in range(3)]
    open_frac     = [summary[f"arm{i}"]["gripper_frac_near_open"] * 100 for i in range(3)]
    delta_mean    = [summary[f"arm{i}"]["joint_delta_mag_mean"] for i in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # Panel A: Active step fraction
    ax = axes[0]
    bars = ax.bar(ARM_LABELS, active_frac, color=ARM_COLORS, width=0.5, edgecolor="white")
    for bar, v in zip(bars, active_frac):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("% of steps actively moving\n(joint Δ > 1e-3)")
    ax.set_title("A. Active motion fraction", fontweight="bold")
    ax.set_ylim(0, 55)

    # Panel B: Gripper state breakdown (stacked)
    ax = axes[1]
    x = np.arange(3)
    middle_frac = [100 - closed_frac[i] - open_frac[i] for i in range(3)]
    p1 = ax.bar(x, open_frac,   color="#4C72B0", width=0.5, label="Near open (>0.7)")
    p2 = ax.bar(x, middle_frac, color="#cccccc", width=0.5, bottom=open_frac, label="Mid range")
    p3 = ax.bar(x, closed_frac, color="#c44e52", width=0.5,
                bottom=[open_frac[i] + middle_frac[i] for i in range(3)], label="Near closed (<−0.7)")
    ax.set_xticks(x)
    ax.set_xticklabels(ARM_LABELS)
    ax.set_ylabel("% of training timesteps")
    ax.set_title("B. Gripper state distribution", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", fontsize=8.5)
    for i in range(3):
        ax.text(x[i], closed_frac[i] + open_frac[i] + middle_frac[i] + 1,
                f"{closed_frac[i]:.0f}%\nclosed", ha="center", va="bottom", fontsize=8.5,
                color="#c44e52", fontweight="bold")

    # Panel C: Mean joint delta magnitude
    ax = axes[2]
    bars = ax.bar(ARM_LABELS, delta_mean, color=ARM_COLORS, width=0.5, edgecolor="white")
    for bar, v in zip(bars, delta_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean joint delta magnitude ‖δq‖")
    ax.set_title("C. Motion magnitude", fontweight="bold")
    ax.set_ylim(0, 0.040)

    fig.suptitle(
        "Arm 2 (cubeC) has dramatically less training signal than Arms 0 & 1\n"
        "→ policy learns 'stay idle with gripper closed' as the arm-2 default",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig2_arm_imbalance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Cube scatter ─────────────────────────────────────────────────────

def fig3_cube_scatter() -> None:
    records = json.loads(Path("/iris/u/mikulrai/logs/tsc_debug/seed_poses/seed_poses.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    cubes = ["cubeA", "cubeB", "cubeC"]
    titles = [
        "A. cubeA (arm 0 target)",
        "B. cubeB (arm 1 target)",
        "C. cubeC (arm 2 target)",
    ]
    c_succ = "#2ca02c"
    c_fail = "#d62728"

    for ax, cname, title in zip(axes, cubes, titles):
        for success, col, lbl in [(False, c_fail, f"Failure (n=21)"), (True, c_succ, f"Success (n=9)")]:
            rs = [r for r in records if r["success"] == success]
            ax.scatter([r[cname][0] for r in rs], [r[cname][1] for r in rs],
                       c=col, label=lbl, s=90, edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8.5)
        ax.set_aspect("equal")

    fig.suptitle(
        "Initial cube positions do NOT separate success from failure\n"
        "→ failure is not caused by arm 2 facing harder geometry",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "fig3_cube_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Gripper heatmap (reuse computed arrays) ─────────────────────────

def fig4_gripper_heatmap() -> None:
    src_dir = Path("/iris/u/mikulrai/logs/tsc_debug/arm2_demos")
    # Load parquet data to recompute heatmap
    data_dir = Path("/iris/u/mikulrai/data/RoboFactory/lerobot/robofactory_three_arm_stack_wristcam/data/chunk-000")
    import pandas as pd

    files = sorted(data_dir.glob("episode_*.parquet"))
    if not files:
        print("WARNING: No parquet files found, skipping fig4")
        return

    NUM_ARMS = 3
    grippers: list[list[np.ndarray]] = [[] for _ in range(NUM_ARMS)]
    ep_lengths = []
    for fpath in files:
        df = pd.read_parquet(fpath)
        actions = np.stack(df["actions"].to_numpy())
        T = actions.shape[0]
        ep_lengths.append(T)
        for arm in range(NUM_ARMS):
            gripper = actions[:, arm * 8 + 7]
            grippers[arm].append(gripper)

    target_T = int(np.median(ep_lengths))
    arm_labels_short = ["Arm 0 (cubeA)", "Arm 1 (cubeB)", "Arm 2 (cubeC)"]

    fig, axes = plt.subplots(NUM_ARMS, 1, figsize=(14, 8), dpi=150)
    for arm in range(NUM_ARMS):
        rows = []
        for g in grippers[arm]:
            if len(g) >= target_T:
                rows.append(g[:target_T])
            else:
                rows.append(np.concatenate([g, np.full(target_T - len(g), g[-1])]))
        mat = np.stack(rows)
        im = axes[arm].imshow(mat, aspect="auto", cmap="RdBu_r",
                              vmin=-1, vmax=1, interpolation="nearest")
        axes[arm].set_title(f"{arm_labels_short[arm]} — gripper command (all 150 demos)", fontsize=10)
        axes[arm].set_ylabel("Episode")
        axes[arm].set_xlabel("Timestep")
        plt.colorbar(im, ax=axes[arm], shrink=0.7, label="Gripper cmd\n(+1=open, −1=closed)")

    fig.suptitle(
        "Gripper command heatmap across 150 training demos\n"
        "Arm 2 is almost always closed (red) — the policy learns 'hold still'",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "fig4_gripper_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    import argparse
    from robofactory.utils.paths import debug_output_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=debug_output_dir(__file__),
        help="Directory for figures (default: debug_output/<script>_<YYYYMMDD>/).",
    )
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_sr_comparison()
    fig2_arm_imbalance()
    fig3_cube_scatter()
    fig4_gripper_heatmap()
    print(f"\nAll figures saved to {OUT_DIR}")
