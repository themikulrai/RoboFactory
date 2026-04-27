"""
Standalone eval script for the centralised (joint) diffusion policy.

One policy observes all N arms and emits a joint action chunk (8*N dims).
Wraps RobotJointImageRunner directly — no TOPP, direct env.step execution.

Example (D1 workspace):
  python policy/Diffusion-Policy/eval_joint_dp.py \
      --ckpt-path checkpoints/ThreeRobotsStackCube-rf_joint_d1_workspace_150/95.ckpt \
      --camera-family workspace \
      --seed 10010 10011 10012 \
      --max-steps 200 \
      --wandb

Example (D2 wristcam):
  python policy/Diffusion-Policy/eval_joint_dp.py \
      --ckpt-path checkpoints/ThreeRobotsStackCube-rf_joint_d2_wristcam_150/95.ckpt \
      --camera-family wristcam \
      --seed 10010 10011 10012 \
      --max-steps 200 \
      --wandb
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import os, json, socket, subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union, Annotated

import dill
import hydra
import numpy as np
import torch
import tyro
import cv2 as _cv2

import robofactory.agents  # noqa: register panda_wristcam_multi
from robofactory.tasks import *  # noqa: register envs
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.robot_joint_image_runner import RobotJointImageRunner


@dataclass
class Args:
    ckpt_path: str
    """Path to joint DP checkpoint, e.g. checkpoints/.../95.ckpt"""

    config: str = "configs/table/three_robots_stack_cube.yaml"
    """Scene config passed to the env runner."""

    camera_family: str = "workspace"
    """'workspace' for D1 (head_camera_agent{i}) or 'wristcam' for D2 (hand_camera_{i})."""

    robot_uids: Optional[str] = None
    """Comma-separated robot UIDs to override env defaults. D2 wristcam requires 'panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi'."""

    seed: Annotated[Union[int, List[int]], tyro.conf.arg(aliases=["-s"])] = 10010
    """Single seed or list of seeds."""

    max_steps: int = 200
    """Max env steps per episode."""

    n_action_exec: int = 6
    """How many steps of the action chunk to execute per policy call."""

    record_dir: str = "./eval_video/joint/{env_id}"
    """Directory to save MP4 and GIF videos."""

    quiet: bool = False
    """Suppress per-step output."""

    wandb: bool = False
    """Log to W&B project 'diffusion-robofactory'."""

    wandb_tags: str = "eval,centralised-dp"
    """Comma-separated W&B tags."""

    jsonl_path: Optional[str] = None
    """Path for per-episode JSONL log; auto-created if None."""


def load_policy(ckpt_path: str, device: str = "cuda:0"):
    payload = torch.load(open('./' + ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: RobotWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 20):
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = _cv2.VideoWriter(path, _cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(_cv2.cvtColor(f, _cv2.COLOR_RGB2BGR))
    vw.release()


def save_gif(mp4_path: str):
    gif_path = mp4_path.replace('.mp4', '.gif')
    os.system(
        f'ffmpeg -i "{mp4_path}" '
        f'-vf "fps=15,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
        f'-loop 0 "{gif_path}" -y 2>/dev/null'
    )
    return gif_path


def main(args: Args):
    seeds = [args.seed] if isinstance(args.seed, int) else list(args.seed)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    policy = load_policy(args.ckpt_path)
    print(f"Loaded policy from {args.ckpt_path}. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    robot_uids_tuple = tuple(args.robot_uids.split(",")) if args.robot_uids else None
    runner = RobotJointImageRunner(
        output_dir=None,
        env_id="ThreeRobotsStackCube-rf",
        config_path=args.config,
        n_agents=3,
        include_global=True,
        camera_family=args.camera_family,
        resize=224,
        n_action_exec=args.n_action_exec,
        max_episode_steps=args.max_steps,
        device="cuda:0",
        robot_uids=robot_uids_tuple,
    )

    env = runner._make_env()

    dataset_tag = "d1" if args.camera_family == "workspace" else "d2"
    env_id = "ThreeRobotsStackCube-rf"
    record_root = args.record_dir.format(env_id=env_id) + f"/eval_{ts}_{dataset_tag}_ckpt{os.path.basename(args.ckpt_path).replace('.ckpt','')}"

    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd='/iris/u/mikulrai/projects/RoboFactory').decode().strip()
    except Exception:
        git_sha = 'unknown'

    jsonl_path = args.jsonl_path or f'/iris/u/mikulrai/logs/eval_joint_{env_id}_{dataset_tag}_{ts}.jsonl'
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    manifest = dict(
        task=env_id, scene_config=args.config,
        ckpt_path=args.ckpt_path, camera_family=args.camera_family,
        max_steps=args.max_steps, n_seeds=len(seeds), seeds=seeds,
        git_sha=git_sha, host=socket.gethostname(),
        start_utc=ts, record_root=record_root, jsonl_path=jsonl_path,
    )
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({'kind': 'manifest', **manifest}) + '\n')
    print('MANIFEST:', json.dumps(manifest, indent=2), flush=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project='diffusion-robofactory', job_type='eval',
            name=f'eval_joint_{dataset_tag}_{ts}',
            group=f'eval_joint_{dataset_tag}',
            tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
            config=manifest,
        )

    results = []
    for idx, seed in enumerate(seeds):
        if not args.quiet:
            print(f"[seed {seed}] running...", flush=True)
        torch.cuda.reset_peak_memory_stats()
        result = runner._rollout_single_episode(env, policy, seed=seed, record_frames=True)
        vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

        video_path = os.path.join(record_root, f'seed{seed:07d}_global.mp4')
        save_mp4(result['frames'], video_path)
        gif_path = save_gif(video_path) if result['frames'] else None

        metrics = dict(
            seed=int(seed), success=int(result['success']),
            steps=int(result['length']), vram_peak_mb=vram_mb, episode_idx=idx,
        )
        results.append(metrics)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps({'kind': 'episode', **metrics}) + '\n')

        n_succ = sum(r['success'] for r in results)
        print(
            f"[seed {seed}] success={metrics['success']} steps={metrics['steps']} "
            f"vram_mb={vram_mb} video={video_path} | SR {n_succ}/{len(results)}={100.*n_succ/len(results):.1f}%",
            flush=True,
        )
        if wandb_run is not None:
            import wandb
            log = {f'episode/{k}': v for k, v in metrics.items()}
            if gif_path and os.path.exists(gif_path):
                log['episode/video'] = wandb.Video(video_path, fps=20, format='mp4')
            wandb.log(log)

    env.close()

    n_total = len(results)
    n_succ = sum(r['success'] for r in results)
    sr = n_succ / n_total if n_total else 0.0
    from math import sqrt
    ci = 1.96 * sqrt(max(sr * (1 - sr), 1e-9) / max(n_total, 1))
    summary = dict(n_total=n_total, n_success=n_succ, success_rate=sr, ci95=ci)
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps({'kind': 'summary', **summary}) + '\n')
    print('SUMMARY:', json.dumps(summary, indent=2), flush=True)
    if wandb_run is not None:
        import wandb
        wandb.log({f'summary/{k}': v for k, v in summary.items()})
        wandb_run.finish()
    print('success' if sr > 0 else 'failed')


if __name__ == '__main__':
    main(tyro.cli(Args))
