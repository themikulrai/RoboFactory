"""T-1b: Teacher-forced replay for cent on 5 training episodes.

For each frame in 5 lerobot training episodes, feeds the recorded obs
(state + images) to the cent server and compares the first predicted action
step to the recorded action. Reports per-arm MAE and plots per-dim error.

Diagnoses:
- Match across all arms → bug is env-side (controller, camera, gripper format)
- Per-arm divergence with arm-2 worst → H1 (extra_0_rgb slot noise)
- Global divergence → ckpt-loading or LoRA-merge bug

Run after booting serve_policy.py on port 8000 (cent, pi05_wristcam_cent_v1/19999).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from PIL import Image

DATASET_DIR = Path("/iris/u/mikulrai/data/RoboFactory/lerobot/robofactory_three_arm_stack_wristcam")
PROMPT = "stack the three cubes using three robot arms"
OUT_DIR = Path("/iris/u/mikulrai/logs/tsc_debug")
PORT = 8000
N_EPISODES = 5
# Maximum frames per episode to avoid OOM
MAX_FRAMES_PER_EP = 100

IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")


def _decode_image(cell: dict) -> np.ndarray:
    img_bytes = cell["bytes"]
    return np.array(Image.open(io.BytesIO(img_bytes))).astype(np.uint8)


def _build_obs_dict_from_row(row: dict) -> dict:
    out: dict = {
        "state": np.asarray(row["state"], dtype=np.float32),
        "prompt": PROMPT,
    }
    for slot in IMAGE_SLOTS:
        out[f"{slot}"] = _decode_image(row[slot])
    return out


def main() -> None:
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = WebsocketClientPolicy(host="127.0.0.1", port=PORT)
    print(f"Server metadata: {policy.get_server_metadata()}", flush=True)

    all_errors: list[np.ndarray] = []  # (frame, 24) absolute errors
    ep_results = []

    chunk_dir = DATASET_DIR / "data" / "chunk-000"
    ep_files = sorted(chunk_dir.glob("episode_*.parquet"))[:N_EPISODES]
    print(f"Loading {len(ep_files)} episodes from {chunk_dir}", flush=True)

    for ep_path in ep_files:
        df = pd.read_parquet(ep_path)
        n_frames = min(len(df) - 1, MAX_FRAMES_PER_EP)  # -1: need next-step action
        print(f"\n  Episode {ep_path.name}: {len(df)} frames, using {n_frames}", flush=True)

        ep_errors: list[np.ndarray] = []
        for t in range(n_frames):
            row = df.iloc[t].to_dict()
            obs_dict = _build_obs_dict_from_row(row)
            result = policy.infer(obs_dict)
            pred_chunk = np.asarray(result["actions"])[:, :24]  # (16, 24)
            pred_step0 = pred_chunk[0]  # first predicted action

            recorded = np.asarray(row["actions"], dtype=np.float32)  # (24,)
            err = np.abs(pred_step0 - recorded)
            ep_errors.append(err)

            if t % 20 == 0:
                per_arm_mae = [err[i*8:(i+1)*8].mean() for i in range(3)]
                print(f"    t={t:3d} per-arm MAE: arm0={per_arm_mae[0]:.4f} arm1={per_arm_mae[1]:.4f} arm2={per_arm_mae[2]:.4f}", flush=True)

        ep_err_arr = np.stack(ep_errors)  # (n_frames, 24)
        all_errors.append(ep_err_arr)

        ep_mae = [ep_err_arr[:, i*8:(i+1)*8].mean() for i in range(3)]
        print(f"  Episode MAE: arm0={ep_mae[0]:.4f} arm1={ep_mae[1]:.4f} arm2={ep_mae[2]:.4f}")
        ep_results.append({"episode": ep_path.name, "arm_mae": ep_mae})

    all_err = np.concatenate(all_errors, axis=0)  # (total_frames, 24)
    np.save(OUT_DIR / "t1b_errors.npy", all_err)

    print("\n=== T-1b Summary ===")
    overall_mae = [all_err[:, i*8:(i+1)*8].mean() for i in range(3)]
    gripper_mae = [float(all_err[:, i*8+7].mean()) for i in range(3)]
    joint_mae = [float(all_err[:, i*8:i*8+7].mean()) for i in range(3)]
    print(f"Overall per-arm MAE (all dims): arm0={overall_mae[0]:.4f} arm1={overall_mae[1]:.4f} arm2={overall_mae[2]:.4f}")
    print(f"Gripper dim MAE:                arm0={gripper_mae[0]:.4f} arm1={gripper_mae[1]:.4f} arm2={gripper_mae[2]:.4f}")
    print(f"Joint dims MAE (7 dims avg):    arm0={joint_mae[0]:.4f} arm1={joint_mae[1]:.4f} arm2={joint_mae[2]:.4f}")

    if max(overall_mae) < 0.05:
        verdict = "MATCH: errors <5% — bug is env-side (controller, camera, gripper format at eval)"
    elif overall_mae[2] > 2 * max(overall_mae[0], overall_mae[1]):
        verdict = "H1 CONFIRMED: arm2 worst — extra_0_rgb slot poisoning arm-2 features"
    elif all(m > 0.2 for m in overall_mae):
        verdict = "GLOBAL DIVERGENCE: ckpt loading or LoRA-merge bug (all arms bad)"
    else:
        verdict = f"MIXED: inspect per-arm breakdown. arm0={overall_mae[0]:.3f} arm1={overall_mae[1]:.3f} arm2={overall_mae[2]:.3f}"
    print(f"\nVERDICT: {verdict}")

    summary = {
        "overall_arm_mae": overall_mae,
        "gripper_mae": gripper_mae,
        "joint_mae": joint_mae,
        "verdict": verdict,
        "per_episode": ep_results,
    }
    (OUT_DIR / "t1b_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_errors(all_err, OUT_DIR)
    print(f"\nSaved to {OUT_DIR}/t1b_*")


def _plot_errors(all_err: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        arm_labels = ["arm0 (left / blue)", "arm1 (right / green)", "arm2 (mid / red)"]

        for arm_i, ax in enumerate(axes):
            s = arm_i * 8
            arm_err = all_err[:, s:s+8]  # (frames, 8)
            for dim_j in range(7):
                ax.plot(arm_err[:, dim_j], alpha=0.4, linewidth=0.8, label=f"joint{dim_j}")
            ax.plot(arm_err[:, 7], linewidth=1.5, color="black", label="gripper")
            ax.set_ylabel("|pred - recorded|")
            ax.set_title(f"{arm_labels[arm_i]}")
            ax.legend(loc="upper right", fontsize=7, ncol=4)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Frame (concatenated over 5 episodes)")
        plt.suptitle("T-1b: Teacher-forced replay — |predicted - recorded| per dim\nAll arms bad → ckpt bug; arm2 worst → H1 (extra_0_rgb collapse)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "t1b_error_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved t1b_error_timeseries.png")

        # Per-arm MAE bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        per_dim_mae = all_err.mean(axis=0)  # (24,)
        x = np.arange(24)
        colors = ["#2196F3"] * 8 + ["#4CAF50"] * 8 + ["#F44336"] * 8
        ax2.bar(x, per_dim_mae, color=colors)
        for arm_i in range(3):
            ax2.axvspan(arm_i*8 - 0.5, arm_i*8 + 7.5, alpha=0.05, color=["blue","green","red"][arm_i])
        ax2.set_xlabel("Action dimension (per arm: 0-6=joints, 7=gripper)")
        ax2.set_ylabel("Mean |error|")
        ax2.set_title("T-1b: Per-dim MAE — uniform = ckpt bug; arm2-spike = H1")
        ax2.set_xticks([3.5, 11.5, 19.5])
        ax2.set_xticklabels(["arm0", "arm1", "arm2"])
        plt.tight_layout()
        plt.savefig(out_dir / "t1b_per_dim_mae.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved t1b_per_dim_mae.png")
    except Exception as exc:
        print(f"Plot failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
