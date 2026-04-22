"""Per-task diagnostic statistics.

Dumps one JSON per task containing aggregate numbers + Plotly figure dicts that
the single-page app renders via Plotly.react(). Figures are designed to surface
dataset-quality signals: success rate, episode-length distribution, per-joint
action saturation, action-jerk distribution across episodes, and seed diversity
(mean pairwise L2 between downsampled first frames).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

DATA_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/hf_download")
SITE_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/site")
SAT_THRESHOLD = 0.99
DIVERSITY_SIZE = 64  # downsample first frame to this (HxH) grayscale for pairwise L2


def _load_actions(grp: h5py.Group) -> np.ndarray:
    """(T, D) for single-agent; (T, D*n_agents) concatenated for multi-agent."""
    node = grp["actions"]
    if isinstance(node, h5py.Dataset):
        return node[:]
    return np.concatenate([node[k][:] for k in sorted(node.keys())], axis=-1)


_PANDA_JOINT_NAMES = [
    "shoulder yaw",    # j0
    "shoulder pitch",  # j1
    "upper arm",       # j2
    "elbow",           # j3
    "forearm",         # j4
    "wrist pitch",     # j5
    "wrist roll",      # j6
    "gripper L",       # j7
    "gripper R",       # j8
]


def _joint_name(idx: int) -> str:
    return _PANDA_JOINT_NAMES[idx] if 0 <= idx < len(_PANDA_JOINT_NAMES) else f"j{idx}"


def _load_qpos(grp: h5py.Group) -> tuple[np.ndarray, list[str]]:
    """Return (T, D) qpos and per-dim labels. Multi-agent tasks store per-arm
    qpos under obs/agent/panda-N/; we concatenate along the feature axis.

    Labels use Franka Panda joint names. Solo tasks: "shoulder yaw", "elbow", ...
    Multi-agent: "0 - shoulder yaw", "1 - elbow", ... with the arm index up front.
    """
    node = grp["obs/agent"]
    if "qpos" in node and isinstance(node["qpos"], h5py.Dataset):
        qpos = node["qpos"][:]
        labels = [_joint_name(i) for i in range(qpos.shape[1])]
        return qpos, labels
    arms = sorted(k for k in node.keys() if isinstance(node[k], h5py.Group) and "qpos" in node[k])
    parts = [node[a]["qpos"][:] for a in arms]
    qpos = np.concatenate(parts, axis=-1)
    labels = []
    for a, p in zip(arms, parts):
        # e.g. "panda-0" -> "0"
        arm_num = a.split("-")[-1] if "-" in a else a
        for i in range(p.shape[1]):
            labels.append(f"{arm_num} - {_joint_name(i)}")
    return qpos, labels


def _resample_traj(traj: np.ndarray, n: int = 100) -> np.ndarray:
    """Linear interpolation along time axis to a fixed length n. (T, D) -> (n, D)."""
    T, D = traj.shape
    if T == n:
        return traj.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, traj[:, d])
    return out


def _first_frame_small(grp: h5py.Group) -> np.ndarray:
    """Downsampled grayscale first frame of the first camera (alphabetical)."""
    sensor = grp["obs/sensor_data"]
    cam_name = sorted(sensor.keys())[0]
    rgb0 = sensor[cam_name]["rgb"][0]  # (H, W, 3) uint8
    img = Image.fromarray(rgb0).convert("L").resize(
        (DIVERSITY_SIZE, DIVERSITY_SIZE), Image.LANCZOS
    )
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def _pairwise_l2_mean(X: np.ndarray) -> float:
    """Mean upper-triangular pairwise L2 distance."""
    if len(X) < 2:
        return 0.0
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), k=1)
    return float(d[iu].mean())


def _fig_length_hist(lengths: np.ndarray) -> dict:
    return {
        "data": [{
            "type": "histogram", "x": lengths.tolist(),
            "marker": {"color": "#4f8aff"}, "name": "length",
        }],
        "layout": {
            "title": {"text": "Episode length (timesteps)"},
            "xaxis": {"title": "timesteps"},
            "yaxis": {"title": "episodes"},
            "bargap": 0.05, "height": 220,
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        },
    }


def _fig_jerk_hist(jerks: np.ndarray) -> dict:
    return {
        "data": [{
            "type": "histogram", "x": jerks.tolist(),
            "marker": {"color": "#f4a261"}, "name": "mean jerk",
        }],
        "layout": {
            "title": {"text": "Per-episode mean action jerk  ‖Δa‖₂"},
            "xaxis": {"title": "mean ‖Δa‖₂"},
            "yaxis": {"title": "episodes"},
            "bargap": 0.05, "height": 220,
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        },
    }


_JOINT_PALETTE = [
    "#4f8aff", "#f4a261", "#2a9d8f", "#e63946",
    "#b28dff", "#ffd166", "#06b6d4", "#94a3b8",
]


def _fig_path_band_plot(Q: np.ndarray, labels: list[str]) -> dict:
    """Mean ± 1σ band per joint across episodes, vs normalized trajectory time.

    Q: (n_eps, 100, D). For each joint we emit a shaded polygon (mean-std to
    mean+std) and a solid mean line, legend-grouped so clicking the joint name
    toggles both. We show the first arm's 7 arm joints (j0–j6) by default,
    ignoring gripper dims; for multi-agent tasks we pick arm 0. The rest are
    included but hidden by default (click legend to reveal).
    """
    n_eps, T, D = Q.shape
    mean = Q.mean(axis=0)  # (T, D)
    std = Q.std(axis=0)    # (T, D)
    x = np.linspace(0, 1, T).round(4)


    data = []
    for d, lbl in enumerate(labels):
        color = _JOINT_PALETTE[d % len(_JOINT_PALETTE)]
        rgba = color.replace("#", "")
        r, g, b = int(rgba[0:2], 16), int(rgba[2:4], 16), int(rgba[4:6], 16)
        fill_rgba = f"rgba({r},{g},{b},0.18)"
        vis = True
        upper = (mean[:, d] + std[:, d]).round(5)
        lower = (mean[:, d] - std[:, d]).round(5)
        data.append({
            "type": "scatter", "mode": "lines",
            "x": np.concatenate([x, x[::-1]]).tolist(),
            "y": np.concatenate([upper, lower[::-1]]).tolist(),
            "fill": "toself", "fillcolor": fill_rgba,
            "line": {"width": 0, "color": color},
            "name": lbl + " band", "legendgroup": lbl,
            "showlegend": False, "hoverinfo": "skip",
            "visible": vis,
        })
        data.append({
            "type": "scatter", "mode": "lines",
            "x": x.tolist(), "y": mean[:, d].round(5).tolist(),
            "line": {"color": color, "width": 2},
            "name": lbl, "legendgroup": lbl,
            "visible": vis,
        })

    return {
        "data": data,
        "layout": {
            "title": {"text": f"Per-joint qpos across {n_eps} episodes — mean ± 1σ band (click legend to toggle)"},
            "xaxis": {"title": "normalized trajectory time (0=start, 1=end)", "range": [0, 1]},
            "yaxis": {"title": "qpos (rad)"},
            "height": 420,
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"orientation": "v", "x": 1.02, "y": 1, "font": {"size": 10}},
            "hovermode": "x unified",
        },
    }


def _fig_saturation_bars(sat: np.ndarray) -> dict:
    return {
        "data": [{
            "type": "bar",
            "x": [f"a{i}" for i in range(len(sat))],
            "y": sat.tolist(),
            "marker": {"color": ["#e63946" if s > 0.05 else "#2a9d8f" for s in sat]},
        }],
        "layout": {
            "title": {"text": f"Action saturation (fraction of timesteps at |a|≥{SAT_THRESHOLD})"},
            "xaxis": {"title": "action dim"},
            "yaxis": {"title": "fraction saturated", "range": [0, max(0.1, float(sat.max()) * 1.1)]},
            "height": 220,
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        },
    }


def compute_task_stats(task: str) -> dict:
    h5_path = DATA_ROOT / task / f"{task}.h5"
    with h5py.File(h5_path, "r") as f:
        ep_keys = sorted(
            (k for k in f.keys() if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[1]),
        )
        lengths = []
        successes = []
        jerks = []
        all_actions = []
        first_frames = []
        qpos_resampled = []
        qpos_labels: list[str] | None = None
        n_agents = None
        for k in ep_keys:
            g = f[k]
            act = _load_actions(g)  # (T, D)
            if n_agents is None:
                n_agents = len(g["actions"].keys()) if isinstance(g["actions"], h5py.Group) else 1
            lengths.append(act.shape[0])
            successes.append(bool(g["success"][:].any()) if "success" in g else False)
            jerks.append(
                float(np.linalg.norm(np.diff(act, axis=0), axis=1).mean())
                if act.shape[0] > 1 else 0.0
            )
            all_actions.append(act)
            first_frames.append(_first_frame_small(g))
            qpos, qlabels = _load_qpos(g)
            if qpos_labels is None:
                qpos_labels = qlabels
            qpos_resampled.append(_resample_traj(qpos, 100))

    lengths = np.array(lengths)
    jerks = np.array(jerks)
    # per-dim saturation across the whole dataset
    flat = np.concatenate(all_actions, axis=0)  # (sum_T, D)
    max_abs = np.abs(flat).max(axis=0)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)
    saturated = (np.abs(flat) >= SAT_THRESHOLD * max_abs).mean(axis=0)
    diversity = _pairwise_l2_mean(np.stack(first_frames))

    # Joint-space path diversity: stack resampled trajectories -> (n_eps, 100, D)
    Q = np.stack(qpos_resampled)  # dtype float32
    path_std_map = Q.std(axis=0)  # (100, D)
    # Scalar: mean pairwise L2 between flattened trajectories. This is the
    # average "how different are two demos' joint-space paths".
    Qf = Q.reshape(len(Q), -1)
    path_diversity_l2 = _pairwise_l2_mean(Qf)

    return {
        "task": task,
        "n_agents": int(n_agents),
        "n_episodes": int(len(lengths)),
        "n_action_dim": int(flat.shape[1]),
        "n_qpos_dim": int(Q.shape[-1]),
        "success_rate": float(np.mean(successes)),
        "length_mean": float(lengths.mean()),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "jerk_mean": float(jerks.mean()),
        "jerk_median": float(np.median(jerks)),
        "diversity_l2": round(diversity, 4),
        "path_diversity_l2": round(path_diversity_l2, 4),
        "action_saturation": [round(float(x), 4) for x in saturated],
        "figures": {
            "length": _fig_length_hist(lengths),
            "jerk": _fig_jerk_hist(jerks),
            "saturation": _fig_saturation_bars(saturated),
            "path_std": _fig_path_band_plot(Q, qpos_labels or []),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    out_dir = SITE_ROOT / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        tasks = sorted(
            d.name for d in DATA_ROOT.iterdir()
            if d.is_dir() and (d / f"{d.name}.h5").exists()
        )
    else:
        assert args.task, "pass --task TASK or --all"
        tasks = [args.task]

    for task in tasks:
        stats = compute_task_stats(task)
        (out_dir / f"{task}.json").write_text(json.dumps(stats, indent=2))
        print(
            f"[{task}] n={stats['n_episodes']} succ={stats['success_rate']:.2%} "
            f"len_mean={stats['length_mean']:.0f} jerk={stats['jerk_mean']:.3f} "
            f"sat_max={max(stats['action_saturation']):.3f}"
        )


if __name__ == "__main__":
    main()
