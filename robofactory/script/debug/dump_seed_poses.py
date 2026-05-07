"""Analytically compute cube initial positions for the 30 eval seeds.

No SAPIEN required — reproduces the RFSceneBuilder.initialize() RNG chain using
only numpy:
  _episode_rng = np.random.RandomState(seed * 100_000)
  rand(3) × goal_region  (scale [0,0,0] → no-op but draws happen)
  rand(3) × cubeA        (scale [0.05, -0.05, 0])
  rand(3) × cubeB        (scale [0.05,  0.05, 0])
  rand(3) × cubeC        (scale [0.05,  0.05, 0])

Outputs: JSON + scatter PNG to --output_dir (default: debug_output/<script>_<date>/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Module-level placeholder; rebound in __main__ from --output_dir.
OUT_DIR: Path = Path("/tmp/_uninitialized_replace_me")

SEEDS = list(range(10010, 10040))  # 30 eval seeds

SUCCESS_SEEDS = {10014, 10023, 10028, 10029, 10030, 10034, 10035, 10036, 10038}

# Base positions and randp_scales from configs/table/three_robots_stack_cube.yaml
PRIMITIVES = [
    ("goal_region", [-0.05,  0.,    0.001], [0.,   0.,   0.  ]),
    ("cubeA",       [-0.025, -0.115, 0.02 ], [0.05, -0.05, 0.  ]),
    ("cubeB",       [-0.025,  0.115, 0.02 ], [0.05,  0.05, 0.  ]),
    ("cubeC",       [ 0.185, -0.05,  0.02 ], [0.05,  0.05, 0.  ]),
]


def compute_poses(seed: int) -> dict[str, list[float]]:
    ep_seed = seed * 100_000
    rng = np.random.RandomState(ep_seed)
    poses: dict[str, list[float]] = {}
    for name, base, scale in PRIMITIVES:
        r = rng.rand(3)
        p = [base[i] + scale[i] * r[i] for i in range(3)]
        poses[name] = p
    return poses


def main() -> None:
    records = []
    for seed in SEEDS:
        poses = compute_poses(seed)
        records.append({
            "seed": seed,
            "success": seed in SUCCESS_SEEDS,
            "cubeA": poses["cubeA"],
            "cubeB": poses["cubeB"],
            "cubeC": poses["cubeC"],
            "goal_region": poses["goal_region"],
        })

    out_json = OUT_DIR / "seed_poses.json"
    out_json.write_text(json.dumps(records, indent=2))
    print(f"Saved {out_json}")

    # ── Scatter plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    cube_names = ["cubeA", "cubeB", "cubeC"]
    titles = [
        "cubeA init XY (arm0 picks up)",
        "cubeB init XY (arm1 picks up)",
        "cubeC init XY (arm2 picks up)",
    ]
    colors = {True: "#2ca02c", False: "#d62728"}
    labels = {True: "success", False: "failure"}

    for ax, cname, title in zip(axes, cube_names, titles):
        for success in [False, True]:
            xs = [r[cname][0] for r in records if r["success"] == success]
            ys = [r[cname][1] for r in records if r["success"] == success]
            ax.scatter(xs, ys, c=colors[success], label=labels[success],
                       s=80, edgecolors="white", linewidths=0.5, zorder=3)
            for r in records:
                if r["success"] == success:
                    ax.annotate(str(r["seed"] % 100),
                                (r[cname][0], r[cname][1]),
                                fontsize=6, ha="center", va="center", color="white",
                                fontweight="bold", zorder=4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        "Initial cube positions for 30 eval seeds (green = decent success, red = failure)\n"
        "Seed labels show last 2 digits",
        fontsize=11,
    )
    fig.tight_layout()
    out_fig = OUT_DIR / "cube_pose_scatter.png"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    # ── cubeC only, annotated ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for success in [False, True]:
        rs = [r for r in records if r["success"] == success]
        ax.scatter([r["cubeC"][0] for r in rs], [r["cubeC"][1] for r in rs],
                   c=colors[success], label=f"{labels[success]} ({len(rs)})",
                   s=120, edgecolors="white", linewidths=0.6, zorder=3)
        for r in rs:
            ax.annotate(str(r["seed"]),
                        (r["cubeC"][0], r["cubeC"][1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=colors[success])
    ax.set_title("cubeC initial XY — does geometry predict outcome?\n(arm2 must pick up cubeC)", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_fig2 = OUT_DIR / "cubeC_pose_scatter.png"
    fig.savefig(out_fig2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig2}")

    # Print summary
    print("\n=== cubeC position ranges ===")
    success_c = [r["cubeC"] for r in records if r["success"]]
    fail_c    = [r["cubeC"] for r in records if not r["success"]]
    for label, grp in [("success", success_c), ("failure", fail_c)]:
        xs = [p[0] for p in grp]
        ys = [p[1] for p in grp]
        print(f"  {label:8s}  x: [{min(xs):.3f}, {max(xs):.3f}]  y: [{min(ys):.3f}, {max(ys):.3f}]")


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
