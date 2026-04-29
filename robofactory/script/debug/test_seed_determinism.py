"""Tests that env.reset(seed=k) is fully deterministic after the
RFSceneBuilder fix that routes all spawn randomness through self.env._episode_rng.

Verifies four properties for both PickMeat-rf and ThreeRobotsStackCube-rf:
  1. Same-seed determinism within a process (5x reset(seed=0) all match).
  2. Cross-seed variance (seed=0 vs seed=1 differ by >= 1e-3).
  3. Cross-process determinism (seed=0 spawn matches in a fresh subprocess).
  4. Global numpy RNG isolation (spinning np.random.rand before reset has no effect).

Run:
  python script/debug/test_seed_determinism.py

Exit 0 = all pass, 1 = any failure. Prints PASS/FAIL per check.
"""
import sys
sys.path.append('./')

import argparse
import os
import subprocess
import numpy as np
import gymnasium as gym

from robofactory.tasks import *  # noqa: F401,F403  registers env ids


# Map task -> (env id, config path, list of object names whose xy we track).
TASKS = {
    'PickMeat-rf': (
        'PickMeat-rf',
        'configs/table/pick_meat.yaml',
        ['meat'],
    ),
    'ThreeRobotsStackCube-rf': (
        'ThreeRobotsStackCube-rf',
        'configs/table/three_robots_stack_cube.yaml',
        ['cubeA', 'cubeB', 'cubeC'],
    ),
}


def _to_np(x):
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return np.asarray(x)


def make_env(env_id, config_path):
    return gym.make(
        env_id,
        config=config_path,
        obs_mode='none',
        control_mode='pd_joint_pos',
        render_mode='sensors',
        num_envs=1,
        sim_backend='cpu',
        enable_shadow=False,
    )


def get_xy_vec(env, object_names):
    """Walk env.unwrapped.scene.actors, return concatenated xy for tracked objects."""
    actors = env.unwrapped.scene.actors
    out = []
    for name in object_names:
        actor = actors.get(name)
        if actor is None:
            raise KeyError(f"actor {name!r} not in scene.actors keys={list(actors.keys())}")
        p = _to_np(actor.pose.p).squeeze()
        out.extend([float(p[0]), float(p[1])])
    return np.asarray(out, dtype=np.float64)


def reset_and_get(env, seed, object_names):
    env.reset(seed=seed)
    return get_xy_vec(env, object_names)


# ---------------------- single-shot mode for cross-process test ----------------------

def single_shot(task, seed):
    env_id, cfg_path, obj_names = TASKS[task]
    env = make_env(env_id, cfg_path)
    xy = reset_and_get(env, seed, obj_names)
    # Print one line: "seed: x0 y0 x1 y1 ..."
    print(f"{seed}: " + " ".join(f"{v:.10f}" for v in xy))
    env.close()


# ---------------------- main test mode ----------------------

def run_task(task):
    env_id, cfg_path, obj_names = TASKS[task]
    print(f"\n--- {task} ---")
    results = {}

    env = make_env(env_id, cfg_path)

    # Property 1: same-seed determinism within process
    seed0_runs = [reset_and_get(env, 0, obj_names) for _ in range(5)]
    diffs = [np.max(np.abs(seed0_runs[0] - seed0_runs[i])) for i in range(1, 5)]
    max_diff = max(diffs)
    p1_pass = max_diff < 1e-5
    print(f"[1] same-seed determinism (5x reset(seed=0)): max|diff|={max_diff:.3e} -> "
          f"{'PASS' if p1_pass else 'FAIL'}")
    results['p1'] = p1_pass

    # Property 2: cross-seed variance
    s0 = reset_and_get(env, 0, obj_names)
    s1 = reset_and_get(env, 1, obj_names)
    cross_diff = np.max(np.abs(s0 - s1))
    p2_pass = cross_diff >= 1e-3
    print(f"[2] cross-seed variance (seed=0 vs seed=1): max|diff|={cross_diff:.3e} -> "
          f"{'PASS' if p2_pass else 'FAIL'}")
    results['p2'] = p2_pass

    # Property 4: global numpy RNG isolation
    # Reset to seed=0 to capture the canonical xy first.
    canonical = reset_and_get(env, 0, obj_names)
    # Now thrash the global RNG, then reset(seed=0) again.
    np.random.rand(100)
    after_thrash = reset_and_get(env, 0, obj_names)
    iso_diff = np.max(np.abs(canonical - after_thrash))
    p4_pass = iso_diff < 1e-5
    print(f"[4] global-RNG isolation (np.random.rand(100) then reset(seed=0)): "
          f"max|diff|={iso_diff:.3e} -> {'PASS' if p4_pass else 'FAIL'}")
    results['p4'] = p4_pass

    canonical_seed0 = canonical
    env.close()

    # Property 3: cross-process determinism
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         '--single-shot', f'--task={task}', '--seed=0'],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"[3] cross-process subprocess failed (rc={proc.returncode})")
        print(f"    stdout: {proc.stdout!r}")
        print(f"    stderr (tail): {proc.stderr[-500:]!r}")
        results['p3'] = False
    else:
        # Find the line "0: x y ..."
        line = None
        for ln in proc.stdout.strip().splitlines():
            if ln.startswith('0:'):
                line = ln
                break
        if line is None:
            print(f"[3] cross-process: no '0:' line in output. stdout={proc.stdout!r}")
            results['p3'] = False
        else:
            payload = line.split(':', 1)[1].strip().split()
            sub_xy = np.asarray([float(v) for v in payload], dtype=np.float64)
            if sub_xy.shape != canonical_seed0.shape:
                print(f"[3] cross-process: shape mismatch {sub_xy.shape} vs {canonical_seed0.shape}")
                results['p3'] = False
            else:
                xp_diff = np.max(np.abs(sub_xy - canonical_seed0))
                p3_pass = xp_diff < 1e-5
                print(f"[3] cross-process determinism (seed=0): max|diff|={xp_diff:.3e} -> "
                      f"{'PASS' if p3_pass else 'FAIL'}")
                results['p3'] = p3_pass

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-shot', action='store_true',
                    help='print one xy line for the given seed/task and exit (used by sub-process check).')
    ap.add_argument('--task', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.single_shot:
        assert args.task in TASKS, f"--task must be one of {list(TASKS)}"
        single_shot(args.task, args.seed)
        return 0

    all_results = {}
    for task in TASKS:
        all_results[task] = run_task(task)

    # Summary
    print("\n=== SUMMARY ===")
    overall_pass = True
    for task, res in all_results.items():
        flags = " ".join(f"{k}={'P' if v else 'F'}" for k, v in res.items())
        line_pass = all(res.values())
        overall_pass = overall_pass and line_pass
        print(f"  {task}: {flags}  -> {'PASS' if line_pass else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
