"""Closed-loop HIERARCHICAL eval for the vanilla Lift-Barrier policy.

Forked from eval_decent_pi05_diag.py. Adds a high-level (HL) Qwen orchestrator on top of
the two decentralized (LL) pi0.5 arm policies:

    HL (one Qwen server, memer env, HTTP)  ── sees [left_wrist_0, right_wrist_0, base_0]
        │  every K env steps -> {subtask_left, subtask_right}
        ▼
    LL arm0 (pi0.5 websocket) prompt = subtask_left   ── sees own wrist + base_0
    LL arm1 (pi0.5 websocket) prompt = subtask_right  ── sees own wrist + base_0

Three-process / three-env design (see run_hierarchical_lift_barrier_eval.sh):
  - HL Qwen server runs in the MEMER env (numpy 2.x) behind a stdlib-HTTP endpoint.
  - The 2 LL pi0.5 servers run in the OPENPI env (numpy<2) behind openpi websockets.
  - THIS eval runs in the RoboFactory env (numpy<2, SAPIEN). It is an HTTP client to HL and
    a websocket client (openpi_client) to the 2 LL servers.

Modes
-----
  default            query HL every K steps, route per-arm subtasks into the LL prompts.
  --flat-baseline    skip HL entirely; use a fixed prompt (--flat-prompt) for BOTH arms.
                     Sanity baseline to confirm the LL policies work without HL.
  --mock-ll          do not contact LL servers; return zero-action chunks. For plumbing
                     tests with no trained pi0.5 checkpoints. (HL still queried unless
                     --flat-baseline; pair with the HL server's --mock for full mock.)

Outputs eval_LiftBarrier-rf_<timestamp>.json with per-episode {seed, success, steps,
n_hl_queries} and an aggregate success_rate.
"""

from __future__ import annotations

import base64
import dataclasses
import http.client
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien  # noqa: F401
import tyro
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401
from robofactory.tasks import *  # noqa: F401, F403
import robofactory.agents  # noqa: F401

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from policy._shared.multiview_video import ordered_unique, tile_views, subsample  # noqa: E402

IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")

# Camera families: slot name -> SAPIEN sensor name. BOTH the LL helpers and the HL Boss must
# see the SAME cameras they trained on, else eval is a silent camera mismatch.
#   wristcam  : egocentric hand cameras (used by the throwaway sanity run).
#   workspace : fixed head cameras (head_camera_agent0/agent1 + global) -- matches
#               eval_decent_pi05.py / examples/.../camera_mappings/lift_barrier.json, i.e. the
#               cams the WORKSPACE decent pi0.5 helpers AND the dual-agent HL were trained on.
CAMERA_FAMILIES = {
    "wristcam": {
        "base_0_rgb_raw": "head_camera_global",
        "left_wrist_0_rgb_raw": "hand_camera_0",
        "right_wrist_0_rgb_raw": "hand_camera_1",
        "extra_0_rgb_raw": None,
    },
    "workspace": {
        "base_0_rgb_raw": "head_camera_global",
        "left_wrist_0_rgb_raw": "head_camera_agent0",
        "right_wrist_0_rgb_raw": "head_camera_agent1",
        "extra_0_rgb_raw": None,
    },
}

# Cameras the HL policy consumes, in the canonical dual-agent order [left, right, base]. The
# values are filled per --camera-family at runtime (see _hl_cam_map).
HL_CAM_ORDER = ("left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "base_0_rgb_raw")


def _hl_cam_map(camera_family: str) -> dict:
    """HL slot->sensor map for a camera family, in [left, right, base] order."""
    fam = CAMERA_FAMILIES[camera_family]
    return {slot: fam[slot] for slot in HL_CAM_ORDER}


@dataclasses.dataclass
class Args:
    task: str = "LiftBarrier-rf"
    config: str = "/iris/u/mikulrai/projects/RoboFactory/robofactory/configs/table/lift_barrier.yaml"
    # ---- LL pi0.5 servers ----
    host: str = "127.0.0.1"
    ports: str = "8000,8001"  # CSV: one port per arm
    replan_after: int = 8
    num_arms: int = 2
    robot_uid: str = "panda_wristcam_multi"
    robot_uids_csv: str = "panda_wristcam_multi,panda_wristcam_multi"
    # ---- HL Qwen server ----
    hl_host: str = "127.0.0.1"
    hl_port: int = 8200
    hl_query_interval: int = 25  # K: query HL every K env steps
    hl_instruction: str = "lift the steel barrier using two robot arms"
    # ---- rollout / eval control ----
    seed: int = 1000  # base seed; episode i uses base = seed + i
    # Lift-Barrier pi0.5 project convention: env reset seed = base * seed_stride (+0). With
    # seed_stride=100000, bases 100..159 map to env seeds 10_000_000.. ; the recorded/live
    # "seed" field stays the BASE (100..159). seed_stride=1 (default) = consecutive bases.
    seed_stride: int = 1
    max_episodes: int = 20
    max_env_steps: int = 500  # LiftBarrier-rf max_episode_steps is 500
    sim_backend: str = "auto"
    results_dir: str = "/iris/u/mikulrai/projects/RoboFactory/eval_results"
    video_dir: str = ""  # video output dir; defaults to <results_dir>/videos (always records)
    video_frame_stride: int = 2  # subsample factor before writing mp4 (1 = every frame)
    # ---- cameras ----
    camera_family: str = "wristcam"  # {wristcam, workspace}; workspace = head cams (trained-on)
    # ---- live dashboard ----
    live_json: str = ""  # if set, rewrite this JSON after every episode (live progress)
    hl_ckpt: str = ""    # recorded into live_json for the dashboard (informational)
    arm0_ckpt: str = ""
    arm1_ckpt: str = ""
    # ---- modes ----
    flat_baseline: bool = False  # skip HL; fixed prompt for both arms
    flat_prompt: str = "lift the steel barrier using two robot arms"
    mock_ll: bool = False  # zero-action chunks instead of contacting LL servers
    # PR2 server-identity handshake for the LL pi0.5 servers. If set, hard-fail (exit 3)
    # before episode 1 unless each LL arm's metadata config_name matches AND action_dim==8.
    # Accepts a single name (broadcast to all arms) OR a comma-separated name per arm,
    # aligned with --ports. Ignored under --mock-ll (no real server). Empty => no check.
    expect_config: str = ""


# ----------------------------------------------------------------------------- obs helpers


def _gripper(q):
    return float((q[7] + q[8]) / 2.0)


def _build_state(obs, num_arms, robot_uid):
    parts = []
    for i in range(num_arms):
        q = np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()
        parts.append(q[:7].astype(np.float32))
        parts.append(np.array([_gripper(q)], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _extract_image(obs, cam):
    img = obs["sensor_data"][cam]["rgb"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    return img.astype(np.uint8)


def _cur_qpos(obs, num_arms, robot_uid):
    return [
        np.asarray(obs["agent"][f"{robot_uid}-{i}"]["qpos"]).squeeze()[:7].astype(np.float32)
        for i in range(num_arms)
    ]


def _build_ll_obs(obs, args, cam_map, prompt):
    """Build one LL observation dict (4 image slots + state + prompt)."""
    out = {"state": _build_state(obs, args.num_arms, args.robot_uid), "prompt": prompt}
    ref = None
    for slot, cam in cam_map.items():
        if cam is None:
            continue
        img = _extract_image(obs, cam)
        out[slot] = img
        if ref is None:
            ref = img.shape
    if ref is None:
        ref = (224, 224, 3)
    for slot in IMAGE_SLOTS:
        if slot not in out:
            out[slot] = np.zeros(ref, dtype=np.uint8)
    return out


def _write_mp4(path, frames):
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def _overlay_subtasks(img, step, total, left, right):
    """Burn the active subtask (per arm) + step counter onto a frame.

    White text on a black bar is colour-swap invariant (the writer does an
    RGB->BGR convert), so it renders identically regardless of channel order.
    """
    import cv2

    img = np.ascontiguousarray(img)
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.putText(img, f"step {step + 1}/{total}", (5, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"L: {left or '-'}", (5, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if right is not None:
        cv2.putText(img, f"R: {right}", (5, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------- HL client


class HLClient:
    """Stdlib-HTTP client to the HL Qwen subtask server (memer env, separate process)."""

    def __init__(self, host: str, port: int, instruction: str, hl_cam_map: dict, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.instruction = instruction
        self.hl_cam_map = hl_cam_map  # slot -> sensor, in [left, right, base] order
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HL {path} -> {resp.status}: {data.decode('utf-8', 'replace')}")
            return json.loads(data.decode("utf-8"))
        finally:
            conn.close()

    def wait_healthy(self, retries: int = 120, delay: float = 2.0) -> None:
        for _ in range(retries):
            try:
                conn = http.client.HTTPConnection(self.host, self.port, timeout=5.0)
                conn.request("GET", "/healthz")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    return
            except (ConnectionRefusedError, OSError):
                pass
            time.sleep(delay)
        raise RuntimeError(f"HL server at {self.host}:{self.port} never became healthy")

    def reset(self) -> None:
        self._post("/reset", {"instruction": self.instruction})

    def query(self, obs, cam_map=None) -> dict:
        if cam_map is None:
            cam_map = self.hl_cam_map
        images = {}
        for slot, cam in cam_map.items():
            arr = _extract_image(obs, cam)
            images[slot] = {
                "b64": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii"),
                "shape": list(arr.shape),
                "dtype": "uint8",
            }
        return self._post("/query", {"images": images, "instruction": self.instruction})


# ------------------------------------------------------------------------------- LL clients


class _MockLLPolicy:
    """Zero-action LL stand-in for --mock-ll plumbing tests (no openpi server needed)."""

    def __init__(self, horizon: int = 16, dim: int = 8):
        self.horizon = horizon
        self.dim = dim

    def infer(self, obs):  # noqa: D401 - matches WebsocketClientPolicy.infer
        return {"actions": np.zeros((self.horizon, self.dim), dtype=np.float32)}


def _make_ll_policies(args):
    if args.mock_ll:
        return [_MockLLPolicy() for _ in range(args.num_arms)]
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    ports = [int(p) for p in args.ports.split(",")]
    if len(ports) < args.num_arms:
        raise ValueError(f"need {args.num_arms} LL ports, got {ports}")
    return [WebsocketClientPolicy(host=args.host, port=ports[i]) for i in range(args.num_arms)]


# ----------------------------------------------------------------------------------- guards

# The data-gen renderer (robofactory.planner.run) uses shader_pack="default" on all three
# camera-config groups. ManiSkill's fallback is "minimal", whose Color shader clears the
# framebuffer to pure black instead of "default"'s gray clear — silently blacking out the
# upper ~third of head_camera_global at eval (base_0_rgb_raw train-vs-eval RMSE jumps to
# ~0.53 vs ~0.005 for the wrist cams). See memory/feedback_sapien_shader_pack_eval_mismatch.md.
REQUIRED_SHADER_PACK = "default"
_SHADER_CONFIG_KEYS = ("sensor_configs", "human_render_camera_configs", "viewer_camera_configs")


def _assert_shader_pack_default(gym_make_kwargs: dict) -> None:
    """Fail loudly unless shader_pack=="default" on ALL three camera-config groups.

    Gate G2: a wrong/absent shader_pack does not crash — it silently corrupts the global
    camera background, so we MUST assert rather than warn. Call this on the exact kwargs
    dict passed to gym.make (single source of truth)."""
    bad = []
    for key in _SHADER_CONFIG_KEYS:
        cfg = gym_make_kwargs.get(key)
        sp = cfg.get("shader_pack") if isinstance(cfg, dict) else None
        if sp != REQUIRED_SHADER_PACK:
            bad.append(f"{key}.shader_pack={sp!r}")
    if bad:
        raise RuntimeError(
            "SAPIEN shader_pack guard FAILED (Gate G2): expected shader_pack="
            f"{REQUIRED_SHADER_PACK!r} on {list(_SHADER_CONFIG_KEYS)}, but got: "
            f"{', '.join(bad)}. ManiSkill's 'minimal' fallback blacks out the upper region of "
            "head_camera_global vs the gray clear used at data-gen, silently corrupting eval. "
            "See memory/feedback_sapien_shader_pack_eval_mismatch.md."
        )


def _is_compute_node() -> bool:
    """True iff we are clearly on a compute node (not the login node).

    Primary green signal: a SLURM allocation (SLURM_JOB_ID set) — a batch/interactive job
    is by construction on a compute node. Secondary: an explicit override env var for
    interactive runs on workstation nodes (iris-ws-*), since those render correctly but
    carry no SLURM_JOB_ID. The iris LOGIN node has neither and is refused."""
    import os

    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"):
        return True
    if os.environ.get("ROBOFACTORY_ALLOW_NON_SLURM_NODE") == "1":
        return True
    host = socket.gethostname().split(".")[0]
    # Workstation nodes (iris-ws-*) render correctly; allow interactive use on them.
    if host.startswith("iris-ws"):
        return True
    return False


def _assert_not_login_node():
    """HARD refusal to run on the iris login node (renders ~32% darker / black framebuffer).

    Gate G2: eval numbers from the login node are untrustworthy, so refuse rather than warn.
    Green iff inside a SLURM allocation OR on a workstation node OR explicit override."""
    if _is_compute_node():
        return
    host = socket.gethostname()
    raise RuntimeError(
        f"REFUSING to run SAPIEN eval on host {host!r}: no SLURM_JOB_ID and not a workstation "
        "node. The iris login node renders ~32% darker / blacks out head_camera_global, which "
        "silently corrupts eval. Run inside a SLURM allocation on a COMPUTE node (sbatch / "
        "srun / interactive). If you are knowingly on a correctly-rendering interactive node, "
        "set ROBOFACTORY_ALLOW_NON_SLURM_NODE=1. See "
        "memory/feedback_sapien_shader_pack_eval_mismatch.md."
    )


# ------------------------------------------------------------------- Gate G2 fidelity check


def _normalize_img(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 in [-1, 1] (the space the shader-mismatch note's RMSE uses)."""
    return np.asarray(img, dtype=np.float32) / 127.5 - 1.0


def _rmse_norm(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two images in normalized [-1, 1] space."""
    x = _normalize_img(a)
    y = _normalize_img(b)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _load_training_base0_frame(index: int, v21_dir: str) -> np.ndarray:
    """Read base_0_rgb_raw for a global frame `index` from the v2.1 dataset parquet
    (read-only, version-agnostic). v2.1 layout: data/chunk-*/episode_*.parquet, one
    episode per file; the global `index` column is the frame id we match on."""
    import glob
    import io

    import pandas as pd
    from PIL import Image

    files = sorted(glob.glob(str(Path(v21_dir) / "data" / "chunk-*" / "episode_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no v2.1 parquet under {v21_dir}/data/chunk-*/")
    for f in files:
        df = pd.read_parquet(f, columns=["index", "base_0_rgb_raw"])
        hit = df[df["index"] == index]
        if len(hit):
            cell = hit.iloc[0]["base_0_rgb_raw"]
            # LeRobot stores images as {"bytes": <png/jpeg>, "path": ...}
            if isinstance(cell, dict) and cell.get("bytes") is not None:
                img = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            else:
                img = np.asarray(cell)
            return img.astype(np.uint8)
    raise IndexError(f"global index {index} not found in v2.1 dataset {v21_dir}")


def fidelity_check(
    episode_index: int = 0,
    config: str = Args.config,
    v21_dir: str = "/iris/u/mikulrai/data/RoboFactory/lerobot/robofactory_lift_barrier_workspace_seedfix_v1",
    lb_json: str = "/iris/u/mikulrai/data/RoboFactory/hf_download_post_seedfix/LiftBarrier/LiftBarrier.json",
    rmse_threshold: float = 0.1,
) -> dict:
    """Gate G2: render one head_camera_global frame at the training episode's reset
    (shader_pack="default") and compare it to that episode's frame-0 base_0_rgb_raw via
    RMSE in [-1,1] space. The wrong-shader divergent case is ~0.53; we want << 0.1.

    The training episode's reset seed is read from LiftBarrier.json (episode_seed), because
    the planner skips MP failures so episode_index != seed in general.
    """
    _assert_not_login_node()

    # Recover this episode's exact reset seed (episode_seed) and its global frame-0 index.
    meta = json.loads(Path(lb_json).read_text())
    episodes = meta["episodes"]
    ep = next(e for e in episodes if e["episode_id"] == episode_index)
    reset_seed = int(ep["reset_kwargs"].get("seed", ep["episode_seed"]))
    # Global frame index of this episode's frame 0 = sum of all prior episodes' lengths.
    global_index0 = sum(int(e["elapsed_steps"]) for e in episodes if e["episode_id"] < episode_index)

    train_img = _load_training_base0_frame(global_index0, v21_dir)

    gym_make_kwargs = dict(
        config=config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",  # data-gen used cpu backend (LiftBarrier.json env_kwargs)
        robot_uids=("panda_wristcam_multi", "panda_wristcam_multi"),
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    _assert_shader_pack_default(gym_make_kwargs)
    env = gym.make("LiftBarrier-rf", **gym_make_kwargs)
    try:
        obs, _ = env.reset(seed=reset_seed)
        eval_img = _extract_image(obs, "head_camera_global")
    finally:
        env.close()

    if eval_img.shape != train_img.shape:
        raise RuntimeError(
            f"shape mismatch: eval {eval_img.shape} vs train {train_img.shape} "
            "(cannot compute RMSE)"
        )

    rmse = _rmse_norm(eval_img, train_img)

    # Top-sixth near-black fraction is the shader-bug fingerprint (note: >0.8 when broken).
    top = eval_img[: eval_img.shape[0] // 6]
    frac_near_black = float(np.mean(np.all(top < 10, axis=-1)))

    passed = rmse < rmse_threshold
    result = {
        "episode_index": episode_index,
        "reset_seed": reset_seed,
        "global_frame_index": global_index0,
        "image_shape": list(eval_img.shape),
        "rmse_norm": rmse,
        "rmse_threshold": rmse_threshold,
        "top_sixth_frac_near_black": frac_near_black,
        "gate_g2_pass": passed,
    }
    print(f"[G2-fidelity] ep={episode_index} seed={reset_seed} idx={global_index0} "
          f"RMSE={rmse:.4f} (thr={rmse_threshold}) top_black={frac_near_black:.3f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return result


# ------------------------------------------------------------------------------ episode loop


def run_episode(env, ll_policies, hl_client, args, cam_map, action_prefix, seed, video_path,
                view_sensors, env_seed=None, record_seed=None):
    obs, _ = env.reset(seed=(env_seed if env_seed is not None else seed))
    if hl_client is not None:
        hl_client.reset()

    chunks = [None] * args.num_arms
    chunk_idxs = [args.replan_after] * args.num_arms
    # Per-arm prompts. Start with the flat/instruction prompt until the first HL query.
    prompts = [args.flat_prompt if args.flat_baseline else args.hl_instruction] * args.num_arms
    frames = []
    n_hl_queries = 0
    last_hl = None

    success = False
    step = 0
    subtasks_l, subtasks_r = [], []   # per-step active subtask (the one driving this step)
    # ---- per-arm grasp-contact instrumentation (is each gripper actually holding the barrier?) ----
    try:
        _barrier = env.unwrapped.barrier
        _subagents = env.unwrapped.agent.agents
    except Exception:
        _barrier, _subagents = None, None
    def _scal(x):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return a.reshape(-1)
    grasp_trace, barz_trace = [], []   # per step: [grasp_left, grasp_right], barrier z
    for step in range(args.max_env_steps):
        # ---- HL query every K steps (skipped in flat-baseline) ----
        if hl_client is not None and (step % args.hl_query_interval == 0):
            last_hl = hl_client.query(obs)
            n_hl_queries += 1
            sl = last_hl.get("subtask_left")
            sr = last_hl.get("subtask_right")
            if sl:
                prompts[0] = sl
            if args.num_arms > 1 and sr:
                prompts[1] = sr
            right_p = prompts[1] if args.num_arms > 1 else None
            print(f"[seed {seed} step {step}] HL -> left={prompts[0]!r} right={right_p!r} "
                  f"(src={last_hl.get('dual_source')})")

        # ---- record the subtask active for THIS step's action (held between HL queries) ----
        cur_l = prompts[0]
        cur_r = prompts[1] if args.num_arms > 1 else None
        subtasks_l.append(cur_l)
        subtasks_r.append(cur_r)

        # ---- record the views the LL policy actually saw (tiled), with subtask overlay ----
        frame = tile_views([_extract_image(obs, s) for s in view_sensors])
        frame = _overlay_subtasks(frame, step, args.max_env_steps, cur_l, cur_r)
        frames.append(frame)

        # ---- per-arm LL replanning with its own prompt ----
        replanned = [False] * args.num_arms
        for i in range(args.num_arms):
            if chunks[i] is None or chunk_idxs[i] >= args.replan_after:
                obs_i = _build_ll_obs(obs, args, cam_map, prompts[i])
                chunks[i] = np.asarray(ll_policies[i].infer(obs_i)["actions"])
                chunk_idxs[i] = 0
                replanned[i] = True

        # ---- assemble per-arm absolute joint targets (delta-from-qpos decode) ----
        cur_q = _cur_qpos(obs, args.num_arms, args.robot_uid)
        action_dict = {}
        for i in range(args.num_arms):
            step_i = chunks[i][chunk_idxs[i]]
            delta = step_i[:7]
            grip = step_i[7]
            target = np.concatenate([cur_q[i] + delta, np.array([grip], dtype=np.float32)])
            action_dict[f"{action_prefix}-{i}"] = target.astype(np.float32)
            chunk_idxs[i] += 1

        obs, _, term, trunc, info = env.step(action_dict)
        s = info.get("success", False)
        success = bool(s.item() if hasattr(s, "item") else s)
        # record per-arm grasp + barrier height for THIS step (incl. the final/break step)
        if _barrier is not None:
            try:
                barz_trace.append(float(_scal(_barrier.pose.p)[2]))
                grasp_trace.append([bool(_scal(_subagents[i].is_grasping(_barrier))[0])
                                    for i in range(args.num_arms)])
            except Exception:
                pass
        if success or term or trunc:
            break

    if frames:
        _write_mp4(video_path, subsample(frames, args.video_frame_stride))

    return {
        "seed": int(record_seed if record_seed is not None else seed),
        "env_seed": int(env_seed if env_seed is not None else seed),
        "success": bool(success),
        "steps": int(step + 1),
        "n_hl_queries": int(n_hl_queries),
        "last_subtask_left": (last_hl or {}).get("subtask_left") if last_hl else None,
        "last_subtask_right": (last_hl or {}).get("subtask_right") if last_hl else None,
        "subtask_trace_left": subtasks_l,
        "subtask_trace_right": subtasks_r,
        "grasp_trace": grasp_trace,   # per step [grasp_left, grasp_right]
        "barz_trace": barz_trace,     # per step barrier z height
    }


def main(args: Args):
    _assert_not_login_node()

    if args.camera_family not in CAMERA_FAMILIES:
        raise ValueError(f"--camera-family must be one of {sorted(CAMERA_FAMILIES)}, got {args.camera_family!r}")
    cam_map = dict(CAMERA_FAMILIES[args.camera_family])      # LL helpers' cameras
    hl_cam_map = _hl_cam_map(args.camera_family)             # HL Boss's cameras (same family)
    # The views the LL policy physically saw = the non-masked sensors in its cam_map
    # (ordered, deduped). These — NOT just head_camera_global — are what we tile into video.
    view_sensors = ordered_unique(list(cam_map.values()))
    print(f"[camera-family] {args.camera_family}: LL={cam_map} HL={hl_cam_map} "
          f"video_views={view_sensors}")

    # ALWAYS record: default the video dir to <results_dir>/videos when unset. There is no
    # way to disable recording; --video-dir only chooses the location.
    if not args.video_dir:
        args.video_dir = str(Path(args.results_dir) / "videos")

    # shader_pack="default" matches the data-gen renderer; the assertion below refuses to
    # proceed if any camera-config group is missing/!="default" (Gate G2). Build the kwargs
    # once so the guard checks exactly what gym.make receives.
    gym_make_kwargs = dict(
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids=tuple(args.robot_uids_csv.split(",")),
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    _assert_shader_pack_default(gym_make_kwargs)
    env = gym.make(args.task, **gym_make_kwargs)
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]

    ll_policies = _make_ll_policies(args)

    # PR2 server-identity handshake for the LL pi0.5 servers (skipped under --mock-ll,
    # which has no real metadata). Per-arm decent dim is 8. A wrong served config or
    # action dim hard-fails before episode 1 instead of silently rolling out.
    ll_server_metadata: list[dict] = []
    for i, p in enumerate(ll_policies):
        meta = dict(p.get_server_metadata() or {}) if hasattr(p, "get_server_metadata") else {}
        ll_server_metadata.append(meta)
        print(f"[arm{i}] server metadata: {meta}")
    if args.expect_config and not args.mock_ll:
        expect_configs = [c.strip() for c in args.expect_config.split(",")]
        if len(expect_configs) == 1:
            expect_configs = expect_configs * args.num_arms  # broadcast single name
        if len(expect_configs) != args.num_arms:
            print(
                f"SERVER IDENTITY MISMATCH: --expect-config has {len(expect_configs)} entries "
                f"but --num-arms={args.num_arms} (single name or one per arm); refusing to run",
                file=_sys.stderr, flush=True,
            )
            _sys.exit(3)
        from robofactory.utils.server_identity import assert_server_identity_or_exit
        for i in range(args.num_arms):
            assert_server_identity_or_exit(
                ll_server_metadata[i],
                expect_configs[i],
                expect_action_dim=8,
                label=f"arm{i}",
            )

    hl_client = None
    if not args.flat_baseline:
        hl_client = HLClient(args.hl_host, args.hl_port, args.hl_instruction, hl_cam_map)
        hl_client.wait_healthy()

    print(f"[config] mode={'flat-baseline' if args.flat_baseline else 'hierarchical'} "
          f"mock_ll={args.mock_ll} action_prefix={action_prefix} ports={args.ports} "
          f"hl={args.hl_host}:{args.hl_port} K={args.hl_query_interval} "
          f"episodes={args.max_episodes} base_seed={args.seed}")

    def _write_live(status: str, eps: list):
        """Rewrite the live dashboard JSON after each episode (atomic via tmp+rename)."""
        if not args.live_json:
            return
        n_ok = sum(1 for e in eps if e["success"])
        payload = {
            "status": status,
            "camera_family": args.camera_family,
            "checkpoints": {"hl": args.hl_ckpt, "arm0": args.arm0_ckpt, "arm1": args.arm1_ckpt},
            "seeds_total": int(args.max_episodes),
            "episodes": [
                {
                    "seed": e["seed"],
                    "success": e["success"],
                    "steps": e["steps"],
                    "l": e.get("last_subtask_left"),
                    "r": e.get("last_subtask_right"),
                }
                for e in eps
            ],
            "sr": f"{n_ok}/{len(eps)}",
        }
        p = Path(args.live_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)

    episodes = []
    _write_live("running", episodes)  # seed an empty live file so a reader sees us start
    for ep in range(args.max_episodes):
        base = args.seed + ep                 # recorded/live seed (e.g. 100..159)
        env_seed = base * args.seed_stride     # actual env reset seed (project convention)
        video_path = str(Path(args.video_dir) / f"ep{ep:03d}_seed{base}.mp4")
        rec = run_episode(env, ll_policies, hl_client, args, cam_map, action_prefix, base,
                          video_path, view_sensors, env_seed=env_seed, record_seed=base)
        episodes.append(rec)
        print(f"[episode {ep}] {rec}")
        _write_live("running", episodes)

    n = len(episodes)
    n_success = sum(1 for e in episodes if e["success"])
    success_rate = (n_success / n) if n else 0.0

    results = {
        "task": args.task,
        "args": dataclasses.asdict(args),
        "server_metadata": {f"arm{i}": ll_server_metadata[i] for i in range(len(ll_server_metadata))},
        "mode": "flat-baseline" if args.flat_baseline else "hierarchical",
        "mock_ll": args.mock_ll,
        "hl_query_interval": args.hl_query_interval,
        "num_episodes": n,
        "num_success": n_success,
        "success_rate": success_rate,
        "episodes": episodes,
    }

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.results_dir) / f"eval_{args.task}_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    _write_live("done", episodes)
    print(f"[result] success_rate={success_rate:.3f} ({n_success}/{n}) -> {out_path}")
    return results


if __name__ == "__main__":
    main(tyro.cli(Args))
