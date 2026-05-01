"""Plot per-step trajectory data emitted by eval_pi05.py / eval_decent_pi05.py.

Reads JSONL files (one record per env step) and renders a 4x3 panel grid per file:
    Rows:
        (1) per-arm gripper command over time
        (2) per-arm sum-of-abs joint deltas (action_chunk[chunk_idx, 7*i:7*i+7]) over time
        (3) cube xyz over time (cubeA in col0, cubeB col1, cubeC col2)
        (4) per-arm joint-6 (wrist tilt) over time as a proxy for end-effector z
    Columns:
        arm0, arm1, arm2

Vertical dashed lines every 8 steps (replan boundaries) on the first row.

Usage:
    python plot_trajectories.py \\
        --cent-jsonl /iris/u/mikulrai/logs/tsc_debug/phase_a/cent_seed10010.jsonl \\
        --cent-jsonl /iris/u/mikulrai/logs/tsc_debug/phase_a/cent_seed10011.jsonl \\
        --decent-jsonl /iris/u/mikulrai/logs/tsc_debug/phase_a/decent_seed10014.jsonl \\
        --decent-jsonl /iris/u/mikulrai/logs/tsc_debug/phase_a/decent_seed10028.jsonl \\
        --out-dir /iris/u/mikulrai/logs/tsc_debug/phase_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green
CUBE_COLORS = {"x": "#d62728", "y": "#9467bd", "z": "#17becf"}


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _seed_from_rows(rows: list[dict], fallback: str) -> str:
    if rows and "seed" in rows[0]:
        return str(rows[0]["seed"])
    return fallback


def _gather_series(rows: list[dict]) -> dict:
    """Build numpy arrays for plotting from a list of step records."""
    n = len(rows)
    steps = np.array([r["step"] for r in rows])
    chunk_idxs = np.array([r["chunk_idx"] for r in rows])

    # applied_action_per_arm: shape (n, 3, 8) — first 7 = joint targets, last = gripper
    applied = np.array([r["applied_action_per_arm"] for r in rows])  # (n, 3, 8)

    # state_24: (n, 24) = per-arm [7 qpos, 1 gripper]
    state = np.array([r["state_24"] for r in rows])  # (n, 24)

    # cube xyz
    cubeA = np.array([r["cube_xyz"]["cubeA"] for r in rows])  # (n, 3)
    cubeB = np.array([r["cube_xyz"]["cubeB"] for r in rows])
    cubeC = np.array([r["cube_xyz"]["cubeC"] for r in rows])

    # Joint deltas: action_chunk is per-step (centralised) or per-arm (decent).
    # We compute "applied delta" = action_target - current_qpos on each step.
    # state_24 has per-arm slot starting at 8*i for [7 qpos, 1 gripper].
    # applied[:, i, :7] = target joints for arm i.
    deltas_abs_sum = np.zeros((n, 3))
    for i in range(3):
        target = applied[:, i, :7]
        cur = state[:, i * 8 : i * 8 + 7]
        d = target - cur
        deltas_abs_sum[:, i] = np.abs(d).sum(axis=1)

    # Per-arm gripper command: applied[:, i, 7]
    gripper = applied[:, :, 7]  # (n, 3)

    # Joint 6 (wrist tilt) of state per arm: state[:, i*8 + 6]
    joint6 = np.stack([state[:, i * 8 + 6] for i in range(3)], axis=1)  # (n, 3)

    return {
        "steps": steps,
        "chunk_idxs": chunk_idxs,
        "gripper": gripper,
        "delta_abs_sum": deltas_abs_sum,
        "cubeA": cubeA,
        "cubeB": cubeB,
        "cubeC": cubeC,
        "joint6": joint6,
        "n": n,
    }


def _add_replan_lines(ax, n: int, every: int = 8) -> None:
    for k in range(every, n + 1, every):
        ax.axvline(k, color="#555555", linestyle="--", linewidth=0.6, alpha=0.6)


def _plot_one(rows: list[dict], title: str, out_path: str) -> None:
    s = _gather_series(rows)
    n = s["n"]
    if n == 0:
        return
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=150)
    cubes = [("cubeA", s["cubeA"]), ("cubeB", s["cubeB"]), ("cubeC", s["cubeC"])]

    for col in range(3):
        # Row 0: gripper command for arm `col`
        ax = axes[0, col]
        ax.plot(s["steps"], s["gripper"][:, col], color=ARM_COLORS[col], linewidth=1.2)
        ax.set_title(f"arm{col} gripper cmd")
        ax.set_ylabel("gripper")
        ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)
        _add_replan_lines(ax, n, every=8)

        # Row 1: sum-abs joint deltas for arm `col`
        ax = axes[1, col]
        ax.plot(s["steps"], s["delta_abs_sum"][:, col], color=ARM_COLORS[col], linewidth=1.2)
        ax.set_title(f"arm{col} sum|delta joints|")
        ax.set_ylabel("rad (sum)")
        ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)

        # Row 2: cube xyz over time (one cube per col)
        ax = axes[2, col]
        cube_name, cube_arr = cubes[col]
        for j, axis in enumerate(("x", "y", "z")):
            ax.plot(s["steps"], cube_arr[:, j], label=axis, color=CUBE_COLORS[axis], linewidth=1.0)
        ax.set_title(f"{cube_name} xyz")
        ax.set_ylabel("m")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)

        # Row 3: joint 6 (wrist tilt) per arm
        ax = axes[3, col]
        ax.plot(s["steps"], s["joint6"][:, col], color=ARM_COLORS[col], linewidth=1.2)
        ax.set_title(f"arm{col} joint6 (wrist tilt)")
        ax.set_xlabel("step")
        ax.set_ylabel("rad")
        ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_comparison(cent_rows: list[dict], decent_rows: list[dict], cent_seed: str, decent_seed: str, out_path: str) -> None:
    """Side-by-side: 4 rows x 6 cols (3 cent arms + 3 decent arms)."""
    sc = _gather_series(cent_rows)
    sd = _gather_series(decent_rows)
    if sc["n"] == 0 or sd["n"] == 0:
        return

    fig, axes = plt.subplots(4, 6, figsize=(28, 12), dpi=150)
    cubes_cent = [("cubeA", sc["cubeA"]), ("cubeB", sc["cubeB"]), ("cubeC", sc["cubeC"])]
    cubes_dec = [("cubeA", sd["cubeA"]), ("cubeB", sd["cubeB"]), ("cubeC", sd["cubeC"])]

    for src_idx, (label, s, cubes) in enumerate([
        (f"CENT seed={cent_seed}", sc, cubes_cent),
        (f"DECENT seed={decent_seed}", sd, cubes_dec),
    ]):
        col_offset = src_idx * 3
        for col in range(3):
            c = col_offset + col
            ax = axes[0, c]
            ax.plot(s["steps"], s["gripper"][:, col], color=ARM_COLORS[col], linewidth=1.2)
            ax.set_title(f"{label} arm{col} gripper")
            ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)
            _add_replan_lines(ax, s["n"], every=8)

            ax = axes[1, c]
            ax.plot(s["steps"], s["delta_abs_sum"][:, col], color=ARM_COLORS[col], linewidth=1.2)
            ax.set_title(f"arm{col} sum|delta|")
            ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)

            ax = axes[2, c]
            cube_name, cube_arr = cubes[col]
            for j, axis in enumerate(("x", "y", "z")):
                ax.plot(s["steps"], cube_arr[:, j], label=axis, color=CUBE_COLORS[axis], linewidth=1.0)
            ax.set_title(f"{cube_name} xyz")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)

            ax = axes[3, c]
            ax.plot(s["steps"], s["joint6"][:, col], color=ARM_COLORS[col], linewidth=1.2)
            ax.set_title(f"arm{col} joint6")
            ax.set_xlabel("step")
            ax.grid(True, color="#bbbbbb", linestyle="-", linewidth=0.4, alpha=0.5)

    fig.suptitle(f"Phase A: cent vs decent — cent_seed={cent_seed} decent_seed={decent_seed}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cent-jsonl", action="append", default=[], help="cent JSONL path (repeatable)")
    ap.add_argument("--decent-jsonl", action="append", default=[], help="decent JSONL path (repeatable)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cent_loaded: list[tuple[str, list[dict]]] = []
    for p in args.cent_jsonl:
        rows = _load_jsonl(p)
        seed = _seed_from_rows(rows, Path(p).stem)
        cent_loaded.append((seed, rows))
        _plot_one(rows, title=f"CENT seed={seed} ({Path(p).name})",
                  out_path=str(out_dir / f"cent_{seed}_trajectory.png"))

    decent_loaded: list[tuple[str, list[dict]]] = []
    for p in args.decent_jsonl:
        rows = _load_jsonl(p)
        seed = _seed_from_rows(rows, Path(p).stem)
        decent_loaded.append((seed, rows))
        _plot_one(rows, title=f"DECENT seed={seed} ({Path(p).name})",
                  out_path=str(out_dir / f"decent_{seed}_trajectory.png"))

    # Cross comparisons: every cent x every decent
    for c_seed, c_rows in cent_loaded:
        for d_seed, d_rows in decent_loaded:
            _plot_comparison(c_rows, d_rows, c_seed, d_seed,
                             str(out_dir / f"comparison_{c_seed}_vs_{d_seed}.png"))


if __name__ == "__main__":
    main()
