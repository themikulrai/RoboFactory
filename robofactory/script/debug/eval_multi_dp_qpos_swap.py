"""A5: closed-loop eval with in-distribution qpos swap.

Mirrors `policy/Diffusion-Policy/eval_multi_dp.py`'s rollout, but on every
obs-construction site replaces `agent_pos` with qpos sampled from a different
training demo (per arm, from that arm's training zarr). Tests whether the
policy is over-relying on proprioception by checking if SR rises when the
qpos channel is decoupled from the current image.

Two swap modes:
  - per-step (default): each step picks a fresh random episode + random offset.
  - per-episode: at episode start, pick one random training episode per arm,
                 then walk through it deterministically (clamped to its length).

NOTE: rollout structure is copied from eval_multi_dp.run_episode because the
TOPP / interleaved per-arm action assembly is too tightly coupled to subclass
cleanly. Only the qpos-swap intercept and JSON aggregation differ.
"""
import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import argparse
import json
import os
import socket
import subprocess
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml
import zarr
import gymnasium as gym

from robofactory.tasks import *  # registers env ids
import robofactory.agents  # noqa: F401
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

# Import the existing DP wrapper + obs builder so we stay in sync with eval_multi_dp.
from eval_multi_dp import DP, get_model_input


def load_zarr_swap_source(zarr_path: str):
    """Return (state_array, episode_ends) loaded eagerly into memory."""
    root = zarr.open(zarr_path, mode='r')
    state = np.asarray(root['data']['state'])  # (T, qpos_dim)
    ep_ends = np.asarray(root['meta']['episode_ends']).astype(np.int64)  # (n_eps,)
    if state.ndim != 2:
        raise ValueError(f"{zarr_path}: state shape {state.shape} not 2-D")
    if ep_ends.ndim != 1 or ep_ends.size == 0:
        raise ValueError(f"{zarr_path}: bad episode_ends shape {ep_ends.shape}")
    return state, ep_ends


def episode_bounds(ep_ends: np.ndarray, ep_idx: int):
    """Return (start, end_exclusive) for episode ep_idx in a zarr with ep_ends."""
    start = int(ep_ends[ep_idx - 1]) if ep_idx > 0 else 0
    end = int(ep_ends[ep_idx])
    return start, end


def sample_swap_qpos_per_step(rng: np.random.Generator, state: np.ndarray, ep_ends: np.ndarray):
    """per-step mode: pick random ep, random offset within it; return state[start+offset]."""
    n_eps = ep_ends.shape[0]
    ep_idx = int(rng.integers(0, n_eps))
    s, e = episode_bounds(ep_ends, ep_idx)
    ep_len = e - s
    if ep_len <= 0:
        # Degenerate guard — fall back to global random index.
        idx = int(rng.integers(0, state.shape[0]))
    else:
        idx = s + int(rng.integers(0, ep_len))
    return state[idx]


def pick_episode_for_per_episode_mode(rng: np.random.Generator, ep_ends: np.ndarray):
    """Choose one random episode index for per-episode mode."""
    return int(rng.integers(0, ep_ends.shape[0]))


def sample_swap_qpos_per_episode(state: np.ndarray, ep_ends: np.ndarray, ep_idx: int, t: int):
    """per-episode mode: deterministic walk through ep_idx; clamp t at ep length."""
    s, e = episode_bounds(ep_ends, ep_idx)
    ep_len = e - s
    offset = min(t, max(ep_len - 1, 0))
    return state[s + offset]


def run_episode_qpos_swap(
    env, planner, dp_models, agent_num, seed, args,
    swap_states, swap_ep_ends, parent_rng,
    agent_prefix='panda', action_prefix='panda',
):
    """Closed-loop rollout with qpos swap. Returns metrics dict."""
    raw_obs, _ = env.reset(seed=int(seed))
    if env.action_space is not None:
        env.action_space.seed(int(seed))

    # Reset DP runners + planner gripper state.
    for m in dp_models:
        m.runner.reset_obs()
    try:
        from robofactory.planner.motionplanner import OPEN
        planner.gripper_state = [OPEN] * agent_num
    except Exception:
        pass

    # Per-rollout RNGs (one per arm) seeded from parent for reproducibility.
    rngs = [np.random.default_rng(parent_rng.integers(0, 2**63 - 1)) for _ in range(agent_num)]

    # For per-episode mode, pre-pick one source episode per arm.
    fixed_ep_per_arm = None
    if args.swap_mode == 'per-episode':
        fixed_ep_per_arm = [pick_episode_for_per_episode_mode(rngs[aid], swap_ep_ends[aid])
                            for aid in range(agent_num)]

    # Track per-step counter per arm so per-episode mode can advance independently.
    arm_t = [0] * agent_num

    def make_swap_qpos(aid: int):
        if args.swap_mode == 'per-step':
            return sample_swap_qpos_per_step(rngs[aid], swap_states[aid], swap_ep_ends[aid])
        else:
            q = sample_swap_qpos_per_episode(
                swap_states[aid], swap_ep_ends[aid], fixed_ep_per_arm[aid], arm_t[aid]
            )
            arm_t[aid] += 1
            return q

    # Seed planner state / DP runner with FIRST obs (with qpos already swapped).
    for aid in range(agent_num):
        # The "real" qpos (used as planner book-keeping for current_qpos extraction below)
        # is left in raw_obs untouched; we only swap the value we feed to the policy.
        swapped_qpos = make_swap_qpos(aid).astype(np.float32, copy=False)
        obs_dict = get_model_input(
            raw_obs, swapped_qpos, aid,
            include_global=args.include_global,
            cam_family=args.obs_cam_family,
            img_h=args.img_height, img_w=args.img_width,
        )
        dp_models[aid].update_obs(obs_dict)

    # Track action-norm stats to surface in JSON (a smoke check on the policy output).
    action_norm_sum = [0.0 for _ in range(agent_num)]
    action_norm_count = [0 for _ in range(agent_num)]

    cnt = 0
    success = False
    info = {}
    while True:
        cnt += 1
        if cnt > args.max_steps:
            break

        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)
        for aid in range(agent_num):
            action_list = dp_models[aid].get_action()
            action_norm_sum[aid] += float(np.linalg.norm(action_list))
            action_norm_count[aid] += 1
            for i in range(6):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = raw_obs['agent'][f'{agent_prefix}-{aid}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[aid].TOPP(path, 0.05, verbose=True)
                except Exception:
                    action_dict[f'{action_prefix}-{aid}'].append(now_action)
                    action_step_dict[f'{action_prefix}-{aid}'].append(1)
                    continue
                n_step = position.shape[0]
                action_step_dict[f'{action_prefix}-{aid}'].append(n_step)
                gripper_state = now_action[-1]
                if n_step == 0:
                    action_dict[f'{action_prefix}-{aid}'].append(now_action)
                for j in range(n_step):
                    true_action = np.hstack([position[j], gripper_state])
                    action_dict[f'{action_prefix}-{aid}'].append(true_action)

        start_idx = [0 for _ in range(agent_num)]
        observation = None
        for i in range(6):
            max_step = 0
            for aid in range(agent_num):
                max_step = max(max_step, action_step_dict[f'{action_prefix}-{aid}'][i])
            for j in range(max_step):
                true_action = dict()
                for aid in range(agent_num):
                    now_step = min(j, action_step_dict[f'{action_prefix}-{aid}'][i] - 1)
                    true_action[f'{action_prefix}-{aid}'] = action_dict[f'{action_prefix}-{aid}'][start_idx[aid] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
            for aid in range(agent_num):
                start_idx[aid] += action_step_dict[f'{action_prefix}-{aid}'][i]
                if action_step_dict[f'{action_prefix}-{aid}'][i] == 0:
                    continue
                # SWAP: replace agent_pos in obs we feed to the policy.
                swapped_qpos = make_swap_qpos(aid).astype(np.float32, copy=False)
                obs_dict = get_model_input(
                    observation, swapped_qpos, aid,
                    include_global=args.include_global,
                    cam_family=args.obs_cam_family,
                    img_h=args.img_height, img_w=args.img_width,
                )
                dp_models[aid].update_obs(obs_dict)
        if info.get('success', False) is True:
            success = True
            break

    action_norm_mean = [
        (action_norm_sum[aid] / action_norm_count[aid]) if action_norm_count[aid] else 0.0
        for aid in range(agent_num)
    ]
    return dict(
        seed=int(seed),
        success=bool(success),
        steps_to_success=int(cnt - 1) if success else None,
        steps=int(cnt - 1),
        per_arm_action_norm_mean=[round(x, 5) for x in action_norm_mean],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-num', type=int, default=150)
    ap.add_argument('--checkpoint-num', type=int, default=300)
    ap.add_argument('--ckpt-suffix', default="")
    ap.add_argument('--obs-cam-family', default='workspace')
    ap.add_argument('--include-global', dest='include_global', action='store_true')
    ap.add_argument('--no-include-global', dest='include_global', action='store_false')
    ap.set_defaults(include_global=True)
    ap.add_argument('--img-height', type=int, default=240)
    ap.add_argument('--img-width', type=int, default=320)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--max-steps', type=int, default=800)
    ap.add_argument('--out', required=True)
    ap.add_argument('--zarr-paths', nargs='+', required=True,
                    help='Per-arm training zarrs (in arm order, e.g. Agent0 Agent1 Agent2).')
    ap.add_argument('--rng-seed', type=int, default=12345)
    ap.add_argument('--swap-mode', choices=('per-step', 'per-episode'), default='per-step')
    args = ap.parse_args()

    np.set_printoptions(suppress=True, precision=5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    # Load env_id from config
    with open(args.config, 'r') as f:
        env_id = yaml.safe_load(f)['task_name'] + '-rf'

    # Load swap sources up-front (one per arm)
    swap_states = []
    swap_ep_ends = []
    for zp in args.zarr_paths:
        s, e = load_zarr_swap_source(zp)
        swap_states.append(s)
        swap_ep_ends.append(e)
        print(f"loaded swap source: {zp}  state.shape={s.shape}  n_eps={e.shape[0]}", flush=True)

    # Build env (same kwargs as eval_multi_dp default for D1 TSC eval)
    env_kwargs = dict(
        config=args.config,
        obs_mode='rgb',
        control_mode='pd_joint_pos',
        render_mode='sensors',
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
        num_envs=1,
        sim_backend='cpu',
        enable_shadow=False,
        parallel_in_single_scene=False,
    )
    env = gym.make(env_id, **env_kwargs)
    raw_obs, _ = env.reset(seed=int(args.seeds[0]))

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=[agent.robot.pose for agent in env.unwrapped.agent.agents],
        visualize_target_grasp_pose=False,
        print_env_info=False,
        is_multi_agent=True,
    )
    agent_num = planner.agent_num
    if len(args.zarr_paths) != agent_num:
        env.close()
        raise ValueError(
            f"--zarr-paths gave {len(args.zarr_paths)} entries, but env has {agent_num} arms."
        )

    try:
        agent_prefix = env.unwrapped.agent.agents[0].uid
    except Exception:
        agent_prefix = 'panda'
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    print(f"env_id={env_id} agent_prefix='{agent_prefix}' action_prefix='{action_prefix}' n_arms={agent_num}",
          flush=True)

    dp_models = [DP(env_id, args.checkpoint_num, args.data_num, id=i, ckpt_suffix=args.ckpt_suffix)
                 for i in range(agent_num)]
    print(f"Loaded {agent_num} ckpts.", flush=True)

    parent_rng = np.random.default_rng(args.rng_seed)

    per_seed = []
    per_arm_success = [0 for _ in range(agent_num)]  # task SR is binary, but track arm action stats
    per_arm_action_norm_running = [[] for _ in range(agent_num)]
    n_succ = 0
    for seed in args.seeds:
        # Use a child RNG per seed so re-running a single seed is reproducible.
        seed_rng = np.random.default_rng(parent_rng.integers(0, 2**63 - 1))
        metrics = run_episode_qpos_swap(
            env, planner, dp_models, agent_num, seed, args,
            swap_states, swap_ep_ends, seed_rng,
            agent_prefix=agent_prefix, action_prefix=action_prefix,
        )
        per_seed.append(metrics)
        if metrics['success']:
            n_succ += 1
        for aid, v in enumerate(metrics['per_arm_action_norm_mean']):
            per_arm_action_norm_running[aid].append(v)
        print(f"[seed {seed}] success={metrics['success']} steps={metrics['steps']} | "
              f"running SR {n_succ}/{len(per_seed)} = {100.0 * n_succ / len(per_seed):.2f}%",
              flush=True)

    env.close()

    task_sr = n_succ / len(per_seed) if per_seed else 0.0
    per_arm_action_norm_mean = [
        float(np.mean(v)) if v else 0.0 for v in per_arm_action_norm_running
    ]
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd='/iris/u/mikulrai/projects/RoboFactory'
        ).decode().strip()
    except Exception:
        git_sha = 'unknown'

    out = dict(
        config=args.config,
        checkpoint_num=args.checkpoint_num,
        data_num=args.data_num,
        ckpt_suffix=args.ckpt_suffix,
        ckpt_paths=[m.ckpt_path for m in dp_models],
        zarr_paths=list(args.zarr_paths),
        swap_mode=args.swap_mode,
        rng_seed=args.rng_seed,
        n_seeds=len(args.seeds),
        seeds=[int(s) for s in args.seeds],
        max_steps=args.max_steps,
        n_arms=agent_num,
        per_seed=per_seed,
        task_SR=task_sr,
        per_arm_action_norm_mean=per_arm_action_norm_mean,
        git_sha=git_sha,
        host=socket.gethostname(),
        utc=datetime.utcnow().isoformat(),
    )
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}", flush=True)
    print(f"SUMMARY: task_SR={task_sr:.4f}  ({n_succ}/{len(per_seed)})", flush=True)


if __name__ == '__main__':
    main()
