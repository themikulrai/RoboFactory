"""Open-loop replay of an H5 demo's recorded actions in the eval-side env.

Diagnostic: if the demo's verbatim actions reproduce success in env(seed=demo_seed),
then the eval env CAN execute the trajectory and the policy is the only gap.
If replay fails, there is a train/eval env mismatch (control_mode, integration,
action scaling, etc).
"""
import os
import sys
import argparse
import h5py
import yaml
import numpy as np

sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory')
os.chdir('/iris/u/mikulrai/projects/RoboFactory/robofactory')

import gymnasium as gym
import robofactory  # noqa
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='/iris/u/mikulrai/datasets/multi_robot/RoboFactory/hf_download_post_seedfix/TwoRobotsStackCube/TwoRobotsStackCube.h5')
    p.add_argument('--demo-idx', type=int, default=13)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--config', default='configs/table/two_robots_stack_cube.yaml')
    p.add_argument('--robot-uids', default='panda_wristcam_multi,panda_wristcam_multi')
    p.add_argument('--control-mode', default=None, help="e.g., 'pd_joint_pos' to match data-gen explicit setting")
    p.add_argument('--video-out', default='/iris/u/mikulrai/projects/RoboFactory/robofactory/eval_video/TwoRobotsStackCube-rf/openloop_replay_demo13_seed17/seed0000017_replay.mp4')
    args = p.parse_args()

    # Load demo actions
    with h5py.File(args.h5, 'r') as f:
        d = f[f'traj_{args.demo_idx}']
        actions_p0 = np.asarray(d['actions']['panda-0'], dtype=np.float32)
        actions_p1 = np.asarray(d['actions']['panda-1'], dtype=np.float32)
        demo_success = bool(d['success'][-1])
    n_steps = actions_p0.shape[0]
    print(f"demo traj_{args.demo_idx}: {n_steps} steps, demo final success={demo_success}")

    # Build env exactly like eval
    with open(args.config, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    env_id = cfg_yaml['task_name'] + '-rf'
    robot_uids = tuple(args.robot_uids.split(','))
    shader = 'default'
    env = gym.make(
        env_id,
        config=args.config,
        obs_mode='rgb',
        reward_mode='dense',
        control_mode=args.control_mode,
        render_mode='sensors',
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
        viewer_camera_configs=dict(shader_pack=shader),
        sim_backend='gpu',
        enable_shadow=False,
        parallel_in_single_scene=False,
        robot_uids=robot_uids,
        num_envs=1,
    )

    raw_obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    print(f"agent keys: {list(raw_obs['agent'].keys())}")

    # Determine action key prefix
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit('-', 1)[0]
    print(f"action_prefix='{action_prefix}'")

    # Video
    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    vw = None

    success_any = False
    for t in range(n_steps):
        action = {f'{action_prefix}-0': actions_p0[t], f'{action_prefix}-1': actions_p1[t]}
        obs, reward, terminated, truncated, info = env.step(action)
        # Grab global cam frame for video
        sd = obs['sensor_data']
        if 'head_camera_global' in sd:
            frame = sd['head_camera_global']['rgb']
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            frame = frame[0] if frame.ndim == 4 else frame
            frame_bgr = frame[..., ::-1].astype(np.uint8)
            if vw is None:
                h, w = frame_bgr.shape[:2]
                vw = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
            vw.write(frame_bgr)
        s = info.get('success')
        if hasattr(s, 'item'):
            s = s.item()
        elif hasattr(s, '__iter__'):
            s = bool(s[0])
        if s:
            success_any = True
            print(f"  step {t}: SUCCESS")
            break
        if t % 25 == 0:
            print(f"  step {t}/{n_steps}: success={s}")

    if vw is not None:
        vw.release()
    print(f"\nOpenloop replay result: success={success_any} (demo recorded success={demo_success})")
    print(f"Video: {args.video_out}")
    env.close()


if __name__ == '__main__':
    main()
