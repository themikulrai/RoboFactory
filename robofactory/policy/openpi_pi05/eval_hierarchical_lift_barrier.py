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

IMAGE_SLOTS = ("base_0_rgb_raw", "left_wrist_0_rgb_raw", "right_wrist_0_rgb_raw", "extra_0_rgb_raw")

# Cameras the HL policy consumes, in the canonical dual-agent order. Keys are the slot names
# the HL server expects; values are the SAPIEN sensor names in the LiftBarrier env.
HL_CAM_MAP = {
    "left_wrist_0_rgb_raw": "hand_camera_0",
    "right_wrist_0_rgb_raw": "hand_camera_1",
    "base_0_rgb_raw": "head_camera_global",
}


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
    seed: int = 1000  # base seed; episode i uses seed + i
    max_episodes: int = 20
    max_env_steps: int = 500  # LiftBarrier-rf max_episode_steps is 500
    sim_backend: str = "auto"
    results_dir: str = "/iris/u/mikulrai/projects/RoboFactory/eval_results"
    video_dir: str = ""  # if set, write one global-cam mp4 per episode
    # ---- modes ----
    flat_baseline: bool = False  # skip HL; fixed prompt for both arms
    flat_prompt: str = "lift the steel barrier using two robot arms"
    mock_ll: bool = False  # zero-action chunks instead of contacting LL servers


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


# --------------------------------------------------------------------------------- HL client


class HLClient:
    """Stdlib-HTTP client to the HL Qwen subtask server (memer env, separate process)."""

    def __init__(self, host: str, port: int, instruction: str, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.instruction = instruction
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

    def query(self, obs, cam_map=HL_CAM_MAP) -> dict:
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


def _assert_not_login_node():
    # GUARDIAN: shader/login-node guards here. The SAPIEN renderer produces ~32% darker /
    # black framebuffers on the iris login node (see memory/feedback_sapien_shader_pack_eval_mismatch.md).
    # The Data & Sim-Fidelity Guardian owns hardening this (login-node hostname refusal +
    # rendered-vs-training-frame RMSE sign-off gate G2). This stub keeps the structure and a
    # loud warning so eval is never silently trusted on the login node. Leave for the Guardian.
    host = socket.gethostname()
    if host.startswith("iris") and "ws" not in host and not any(c.isdigit() for c in host.split(".")[0]):
        print(f"[GUARDIAN-WARN] hostname={host!r} looks like a login node; SAPIEN may render "
              f"~32% darker. Run eval on a COMPUTE node. (Hard refusal to be added by Guardian.)")


# ------------------------------------------------------------------------------ episode loop


def run_episode(env, ll_policies, hl_client, args, cam_map, action_prefix, seed, video_path):
    obs, _ = env.reset(seed=seed)
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
    for step in range(args.max_env_steps):
        if video_path:
            frames.append(_extract_image(obs, "head_camera_global"))

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
        if success or term or trunc:
            break

    if video_path and frames:
        _write_mp4(video_path, frames)

    return {
        "seed": int(seed),
        "success": bool(success),
        "steps": int(step + 1),
        "n_hl_queries": int(n_hl_queries),
        "last_subtask_left": (last_hl or {}).get("subtask_left") if last_hl else None,
        "last_subtask_right": (last_hl or {}).get("subtask_right") if last_hl else None,
    }


def main(args: Args):
    _assert_not_login_node()

    cam_map = {
        "base_0_rgb_raw": "head_camera_global",
        "left_wrist_0_rgb_raw": "hand_camera_0",
        "right_wrist_0_rgb_raw": "hand_camera_1",
        "extra_0_rgb_raw": None,
    }

    env = gym.make(
        args.task,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        robot_uids=tuple(args.robot_uids_csv.split(",")),
        # GUARDIAN: shader/login-node guards here. shader_pack="default" matches the data-gen
        # renderer; do NOT change without the Guardian's sign-off (G2). See
        # memory/feedback_sapien_shader_pack_eval_mismatch.md.
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]

    ll_policies = _make_ll_policies(args)

    hl_client = None
    if not args.flat_baseline:
        hl_client = HLClient(args.hl_host, args.hl_port, args.hl_instruction)
        hl_client.wait_healthy()

    print(f"[config] mode={'flat-baseline' if args.flat_baseline else 'hierarchical'} "
          f"mock_ll={args.mock_ll} action_prefix={action_prefix} ports={args.ports} "
          f"hl={args.hl_host}:{args.hl_port} K={args.hl_query_interval} "
          f"episodes={args.max_episodes} base_seed={args.seed}")

    episodes = []
    for ep in range(args.max_episodes):
        seed = args.seed + ep
        video_path = ""
        if args.video_dir:
            video_path = str(Path(args.video_dir) / f"ep{ep:03d}_seed{seed}.mp4")
        rec = run_episode(env, ll_policies, hl_client, args, cam_map, action_prefix, seed, video_path)
        episodes.append(rec)
        print(f"[episode {ep}] {rec}")

    n = len(episodes)
    n_success = sum(1 for e in episodes if e["success"])
    success_rate = (n_success / n) if n else 0.0

    results = {
        "task": args.task,
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
    print(f"[result] success_rate={success_rate:.3f} ({n_success}/{n}) -> {out_path}")
    return results


if __name__ == "__main__":
    main(tyro.cli(Args))
