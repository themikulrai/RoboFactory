"""T-1c: Cross-seed action-distribution probe for cent at step 0.

Boots a running cent server, resets the TSC env for each of 30 seeds at
step 0 (ep=0), queries the policy, and dumps the 16×24 action chunk.
Computes per-dim std across seeds and saves a gripper heatmap.

Run after booting serve_policy.py on port 8000 (cent, pi05_wristcam_cent_v1/19999).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien  # noqa
from mani_skill.envs.sapien_env import BaseEnv  # noqa
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# register tasks + panda_wristcam_multi
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "policy/openpi_pi05"))
import robofactory.tasks  # noqa
import robofactory.agents  # noqa
from eval_pi05 import _build_obs_dict, _resolve_camera_mapping

CAM_MAPPING = "/iris/u/mikulrai/projects/openpi/examples/robofactory/camera_mappings/three_robots_stack_cube_wristcam.json"
TASK_CONFIG = "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/three_robots_stack_cube.yaml"
SEEDS = list(range(10010, 10040))
PROMPT = "stack the three cubes using three robot arms"
OUT_DIR = Path("/iris/u/mikulrai/logs/tsc_debug")
PORT = 8000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam_map = _resolve_camera_mapping(CAM_MAPPING)
    policy = WebsocketClientPolicy(host="127.0.0.1", port=PORT)
    print(f"Server metadata: {policy.get_server_metadata()}", flush=True)

    env = gym.make(
        "ThreeRobotsStackCube-rf",
        config=TASK_CONFIG,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
        robot_uids=("panda_wristcam_multi", "panda_wristcam_multi", "panda_wristcam_multi"),
    )

    all_chunks: list[np.ndarray] = []
    for seed in SEEDS:
        ep_seed = seed * 100_000  # episode 0 (matches eval_pi05.py convention)
        obs, _ = env.reset(seed=ep_seed)
        obs_dict = _build_obs_dict(obs, PROMPT, 3, cam_map, "panda_wristcam_multi")
        result = policy.infer(obs_dict)
        chunk = np.asarray(result["actions"])[:, :24]  # (16, 24)
        all_chunks.append(chunk)
        g0, g1, g2 = chunk[0, 7], chunk[0, 15], chunk[0, 23]
        print(f"  seed {seed}: gripper[0] arm0={g0:.3f} arm1={g1:.3f} arm2={g2:.3f}", flush=True)

    env.close()

    chunks = np.stack(all_chunks)  # (30, 16, 24)
    np.save(OUT_DIR / "t1c_action_chunks_step0.npy", chunks)

    std = chunks.std(axis=0)  # (16, 24) — variance across seeds
    mean = chunks.mean(axis=0)
    np.save(OUT_DIR / "t1c_std.npy", std)

    summary: dict = {}
    print("\n=== Per-arm gripper std across 30 seeds (horizon steps 0..15) ===")
    for arm_i in range(3):
        gdim = arm_i * 8 + 7
        gripper_std = std[:, gdim].tolist()
        print(f"  arm{arm_i} gripper std: {[f'{v:.4f}' for v in gripper_std]}")
        summary[f"arm{arm_i}_gripper_std"] = gripper_std
        summary[f"arm{arm_i}_gripper_mean_at_h0"] = float(mean[0, gdim])

    (OUT_DIR / "t1c_summary.json").write_text(json.dumps(summary, indent=2))
    _plot_heatmap(chunks, std, OUT_DIR)
    print(f"\nSaved to {OUT_DIR}/t1c_*")


def _plot_heatmap(chunks: np.ndarray, std: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        n_seeds, n_horizon, _ = chunks.shape
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        arm_labels = ["arm0 (left / blue)", "arm1 (right / green)", "arm2 (mid / red)"]

        for arm_i in range(3):
            gdim = arm_i * 8 + 7
            data = chunks[:, :, gdim]  # (30, 16) seeds × horizon
            ax = axes[arm_i]
            norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
            # gripper open/close threshold — typically 0 = open, positive = close
            ax.axvline(x=7.5, color="k", linestyle="--", linewidth=0.8, alpha=0.6, label="replan_after=8")
            ax.set_xlabel("Horizon step (within chunk)")
            ax.set_ylabel("Seed index (10010→10039)")
            ax.set_title(f"{arm_labels[arm_i]}\ngripper (step-0 chunk, 30 seeds)")
            plt.colorbar(im, ax=ax, fraction=0.046, label="gripper action")

        plt.suptitle(
            "T-1c: Cent step-0 gripper chunk across 30 seeds\n"
            "Uniform horizontal stripe → policy ignores scene (collapse); scattered → scene-conditioned",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "t1c_gripper_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved t1c_gripper_heatmap.png")

        # also plot per-arm gripper std across horizon
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for arm_i in range(3):
            gdim = arm_i * 8 + 7
            ax2.plot(std[:, gdim], label=arm_labels[arm_i], marker="o", markersize=4)
        ax2.set_xlabel("Horizon step")
        ax2.set_ylabel("Std across 30 seeds")
        ax2.set_title("T-1c: Gripper std across seeds — low = collapse")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "t1c_gripper_std.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved t1c_gripper_std.png")
    except Exception as exc:
        print(f"Plot failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
