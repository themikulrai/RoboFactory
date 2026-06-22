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

    env_id: str = "ThreeRobotsStackCube-rf"
    """Registered env id; e.g. 'ThreeRobotsStackCube-rf' (3 arms) or 'LongPipelineDelivery-rf' (4 arms)."""

    n_agents: int = 3
    """Number of arms; 3 for TSC, 4 for LongPipelineDelivery."""

    robot_uids: Optional[str] = None
    """Comma-separated robot UIDs to override env defaults. D2 wristcam requires 'panda_wristcam_multi,panda_wristcam_multi,panda_wristcam_multi'."""

    seed: Annotated[Union[int, List[int]], tyro.conf.arg(aliases=["-s"])] = 10010
    """Final env seed(s) passed straight to env.reset. DEPRECATED alias for --env-seeds;
    prefer --seed-pool (PR4)."""

    seed_pool: str = ""
    """PR4: name of a frozen seed pool (robofactory.utils.eval_seeds), e.g. canonical_env_60.
    When set, overrides --seed/--env-seeds. DP and pi0.5 resolve the SAME final env seeds
    from a pool name -> true seed pairing."""

    env_seeds: str = ""
    """PR4: ad-hoc comma/space list of FINAL env seeds (recorded as pool 'adhoc'). Overrides --seed."""

    allow_train_seeds: bool = False
    """PR4: permit datagen seeds 0..182 (recorded in the result manifest)."""

    max_steps: int = 200
    """Max env steps per episode."""

    n_action_exec: int = 6
    """How many steps of the action chunk to execute per policy call."""

    record_dir: str = "./eval_video/joint/{env_id}"
    """Directory to save MP4 and GIF videos. ALWAYS recorded; this only chooses the location."""

    video_frame_stride: int = 2
    """Subsample recorded frames by this stride before writing the mp4 (1 = every frame)."""

    video_max: int = 3
    """DEPRECATED/ignored: recording is now always-on for every seed. Kept for launcher compat."""

    video_all: bool = False
    """DEPRECATED/ignored: every seed always records. Kept so --video-all still parses."""

    quiet: bool = False
    """Suppress per-step output."""

    wandb: bool = False
    """Log to W&B project 'diffusion-robofactory'."""

    wandb_tags: str = "eval,centralised-dp"
    """Comma-separated W&B tags."""

    jsonl_path: Optional[str] = None
    """Path for per-episode JSONL log; auto-created if None."""

    save_trajectory: bool = False
    """PR9: capture per-episode env_states + actions + proprio (qpos) to an h5 under
    --trajectory-root for future self-training. RGB is NOT recorded (re-renderable from
    env_states). The joint runner steps ABSOLUTE joint targets, so the recorded actions are
    absolute — directly consumable by parse_h5_to_zarr_unified.py --state-source qpos."""

    trajectory_root: Optional[str] = None
    """PR9: root dir for --save-trajectory h5s. Defaults to /iris/u/mikulrai/data/eval_trajs/
    (symlinked, never the project tree) or $RF_EVAL_TRAJ_ROOT."""


def _import_multiview():
    """Mirror the eval_context import: add the policy/ parent to sys.path then import."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from policy._shared.multiview_video import tile_views, ordered_unique, subsample
    return tile_views, ordered_unique, subsample


def _rgb_hwc_uint8(raw_obs: dict, key: str) -> np.ndarray:
    """Extract sensor `key` RGB from raw_obs as HWC uint8 (same access pattern as the runner)."""
    rgb = raw_obs["sensor_data"][key]["rgb"]
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    else:
        rgb = np.asarray(rgb)
    while rgb.ndim > 3:
        rgb = rgb[0]
    return rgb.astype(np.uint8)


def make_multiview_frame_fn(view_sensors: List[str], tile_views):
    """Return a drop-in replacement for runner._frame_from_obs that tiles all policy-input views."""
    def _frame_from_obs(raw_obs: dict):
        try:
            frames = [_rgb_hwc_uint8(raw_obs, s) for s in view_sensors]
            return tile_views(frames)
        except Exception:
            return None
    return _frame_from_obs


def load_policy(ckpt_path: str, device: str = "cuda:0"):
    ckpt_full = ckpt_path if os.path.isabs(ckpt_path) else './' + ckpt_path
    payload = torch.load(open(ckpt_full, 'rb'), pickle_module=dill)
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
    # Shared hard-fail eval fidelity guards (PR3): refuse the login node up front. The
    # shader_pack assert + black-sky bg guard run inside RobotJointImageRunner (_make_env /
    # _rollout_single_episode), which this driver delegates env construction/rollout to.
    from robofactory.utils.eval_guards import assert_not_login_node
    assert_not_login_node()

    # PR4: resolve FINAL env seeds via the single source of truth (same as the pi0.5
    # drivers) so a pool name yields identical env seeds across methods -> true pairing.
    from robofactory.utils.eval_seeds import resolve_seeds
    if args.seed_pool or args.env_seeds:
        seeds, seed_provenance = resolve_seeds(
            pool=args.seed_pool or None,
            env_seeds=args.env_seeds or None,
            allow_train=args.allow_train_seeds,
        )
    else:
        _legacy = [args.seed] if isinstance(args.seed, int) else list(args.seed)
        seeds, seed_provenance = resolve_seeds(
            env_seeds=",".join(str(s) for s in _legacy),
            allow_train=args.allow_train_seeds,
        )
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    policy = load_policy(args.ckpt_path)
    print(f"Loaded policy from {args.ckpt_path}. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    robot_uids_tuple = tuple(args.robot_uids.split(",")) if args.robot_uids else None
    runner = RobotJointImageRunner(
        output_dir=None,
        env_id=args.env_id,
        config_path=args.config,
        n_agents=args.n_agents,
        include_global=True,
        camera_family=args.camera_family,
        resize=224,
        n_action_exec=args.n_action_exec,
        max_episode_steps=args.max_steps,
        device="cuda:0",
        robot_uids=robot_uids_tuple,
    )

    env = runner._make_env()

    # PR9: optional self-training trajectory capture. The joint runner steps ABSOLUTE joint
    # targets, so RecordEpisodeMA buffers absolute actions (no converter-side fix). RGB is
    # dropped (re-renderable from env_states); qpos rides along via the recorded obs. The
    # runner resets/steps THIS wrapped env, so episodes flush in order on each reset.
    traj_h5_path = None
    if args.save_trajectory:
        from robofactory.utils.eval_trajectory import trajectory_output_dir, wrap_record_trajectory
        dataset_tag = "d1" if args.camera_family == "workspace" else "d2"
        _traj_label = f"eval_joint_dp_{args.env_id}_{dataset_tag}_{ts}"
        _traj_dir = trajectory_output_dir(args.trajectory_root, _traj_label)
        env, traj_h5_path = wrap_record_trajectory(env, _traj_dir, trajectory_name="trajectory")
        print(f"[eval_joint_dp] PR9 --save-trajectory -> {traj_h5_path}", flush=True)

    # --- Multiview recording: tile EVERY camera the policy actually consumes. ---
    # The joint policy's image inputs (see RobotJointImageRunner._build_obs_dict) are
    # one camera per agent from the camera-family template, plus the global overhead
    # when include_global=True. Derive that exact ordered/deduped list and record it.
    tile_views, ordered_unique, subsample = _import_multiview()
    from diffusion_policy.env_runner.robot_joint_image_runner import CAM_KEY_TPL
    _tpl = CAM_KEY_TPL[args.camera_family]
    view_sensors = ordered_unique(
        [_tpl.format(i=i) for i in range(args.n_agents)]
        + (["head_camera_global"] if runner.include_global else [])
    )
    print(f"Recording multiview frames from sensors: {view_sensors}", flush=True)
    runner._frame_from_obs = make_multiview_frame_fn(view_sensors, tile_views)

    dataset_tag = "d1" if args.camera_family == "workspace" else "d2"
    env_id = args.env_id
    record_root = args.record_dir.format(env_id=env_id) + f"/eval_{ts}_{dataset_tag}_ckpt{os.path.basename(args.ckpt_path).replace('.ckpt','')}"

    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd='/iris/u/mikulrai/projects/RoboFactory').decode().strip()
    except Exception:
        git_sha = 'unknown'

    jsonl_path = args.jsonl_path or f'/iris/u/mikulrai/logs/eval_joint_{env_id}_{dataset_tag}_{ts}.jsonl'
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    from robofactory.utils.eval_guards import shader_mismatch_override_active
    # PR6: full provenance (eval_protocol v2) — both-repo git sha+dirty, GPU, shader/shadow,
    # seed pool+sha, ckpt path+md5, chunk config. DP has no --prompt (no language input).
    from robofactory.utils.eval_validity import build_provenance
    provenance = build_provenance(
        shader_pack="default",
        enable_shadow=False,
        sim_backend="auto",
        seed_provenance=seed_provenance,
        prompts=None,  # DP is not language-conditioned
        max_env_steps=args.max_steps,
        chunk_config={"n_action_exec": args.n_action_exec, "camera_family": args.camera_family},
        ckpt_paths=[args.ckpt_path],
    )
    manifest = dict(
        task=env_id, scene_config=args.config,
        ckpt_path=args.ckpt_path, camera_family=args.camera_family,
        max_steps=args.max_steps, n_seeds=len(seeds), seeds=seeds,
        seed_provenance=seed_provenance,  # PR4: pool name + sha + allow_train
        git_sha=git_sha, host=socket.gethostname(),
        shader_mismatch_override=shader_mismatch_override_active(),
        start_utc=ts, record_root=record_root, jsonl_path=jsonl_path,
        save_trajectory=args.save_trajectory, trajectory_h5=traj_h5_path,  # PR9
        provenance=provenance,  # PR6: eval_protocol v2
    )
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({'kind': 'manifest', **manifest}) + '\n')
    print('MANIFEST:', json.dumps(manifest, indent=2), flush=True)

    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from policy._shared.eval_context import WandbRun, VideoRecorder

    with WandbRun(
        enabled=args.wandb,
        project=os.environ.get('WANDB_PROJECT', 'diffusion-robofactory'),
        job_type='eval',
        name=f'eval_joint_{dataset_tag}_{ts}',
        group=f'eval_joint_{dataset_tag}',
        tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
        config=manifest,
    ) as wandb_run, VideoRecorder(
        record_root, all_seeds=True,  # ALWAYS-ON: record every seed, no caps.
    ) as videos:
        results = []
        for idx, seed in enumerate(seeds):
            if not args.quiet:
                print(f"[seed {seed}] running...", flush=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                result = runner._rollout_single_episode(env, policy, seed=seed, record_frames=True)
            except Exception as _e:  # PR6: per-episode crash -> steps=-1+error so the >5%
                # validity guard can fire instead of taking the whole run down silently.
                vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                metrics = dict(seed=int(seed), success=0, steps=-1, vram_peak_mb=vram_mb,
                               episode_idx=idx, error=repr(_e))
                results.append(metrics)
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps({'kind': 'episode', **metrics}) + '\n')
                print(f"[seed {seed}] ERROR: {metrics['error']}", flush=True)
                continue
            vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

            # ALWAYS-ON: all_seeds=True + real record_root => video_path is never None.
            video_path = videos.video_path_for(idx, seed, suffix='_multiview')
            rec_frames = subsample(result['frames'], args.video_frame_stride)
            save_mp4(rec_frames, video_path)
            gif_path = save_gif(video_path) if rec_frames else None

            metrics = dict(
                seed=int(seed), success=int(result['success']),
                steps=int(result['length']), vram_peak_mb=vram_mb, episode_idx=idx,
            )
            metrics.update(result.get('partial', {}))  # CL3 partial-credit stage flags
            if traj_h5_path is not None:
                # PR9: episodes flush in order; episode idx -> traj_{idx} in the h5.
                metrics['trajectory_path'] = traj_h5_path
                metrics['trajectory_group'] = f'traj_{idx}'
            results.append(metrics)
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps({'kind': 'episode', **metrics}) + '\n')

            n_succ = sum(r['success'] for r in results)
            print(
                f"[seed {seed}] success={metrics['success']} steps={metrics['steps']} "
                f"vram_mb={vram_mb} video={video_path} | SR {n_succ}/{len(results)}={100.*n_succ/len(results):.1f}%",
                flush=True,
            )
            wandb_run.log_episode(idx, **{k: v for k, v in metrics.items() if k != 'episode_idx'})
            if wandb_run.run is not None and gif_path and video_path and os.path.exists(gif_path):
                import wandb as _wandb
                wandb_run.log_raw({'episode/video': _wandb.Video(video_path, fps=20, format='mp4')})

        env.close()

        n_total = len(results)
        n_succ = sum(r['success'] for r in results)
        sr = n_succ / n_total if n_total else 0.0
        from math import sqrt
        ci = 1.96 * sqrt(max(sr * (1 - sr), 1e-9) / max(n_total, 1))
        # PR6: validity guard — if >5% episodes invalid (steps==-1 or error) the run
        # is invalid; JSONL renamed *_INVALID.jsonl + exit 2.
        from robofactory.utils.eval_validity import classify_validity, finalize_validity
        validity = classify_validity(results)
        summary = dict(n_total=n_total, n_success=n_succ, success_rate=sr, ci95=ci,
                       valid=validity['valid'], invalid_reason=validity['invalid_reason'])
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps({'kind': 'summary', **summary}) + '\n')
            f.write(json.dumps({'kind': 'validity', **validity}) + '\n')
        print('SUMMARY:', json.dumps(summary, indent=2), flush=True)
        wandb_run.log_summary(**summary)
        print('success' if sr > 0 else 'failed')
        finalize_validity(validity, jsonl_path, wandb_run=wandb_run)


if __name__ == '__main__':
    main(tyro.cli(Args))
