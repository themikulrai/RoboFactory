#!/usr/bin/env python
"""PR7 success-criterion audit probes for LiftBarrier (solution_E6.md §PR7.1a/1b).

The LB success criterion (robofactory/tasks/lift_barrier.py:117-122) is INSTANTANEOUS
and grasp-free:  ``barrier.pose.p[...,2] > agents[0].robot.pose.p[0,2] + 0.15``.
These two CPU-sim probes (NO GPU, NO model) measure the floor that the criterion gives
away on its own — independent of any policy:

  --policy hold    HOLD-QPOS: every step commands each arm to stay at its current qpos
                   (robot does NOT move). Any success here == a spawn-state / physics
                   false positive (the barrier already satisfies the criterion at rest,
                   or settles into it without being lifted). EXPECT 0/60. >0 is a finding
                   that taints every LB number (GATE-6 input).

  --policy random  RANDOM-ACTION: uniform random joint targets each step. The floor the
                   criterion gives to flailing. Reported as context for the hold result.

Runs the LB env directly through gym.make on the SAME canonical_env_60 seed pool the real
evals use (robofactory.utils.eval_seeds), on the TABLE scene (memory:
project_robofactory_tsc_scene — the d2_wristcam LB ckpts trained on TABLE, not robocasa).
Uses obs_mode='state' so no camera/Vulkan rendering is needed → safe on CPU, fast.

Usage:
  python -m robofactory.tools.probe_noop_lb --policy hold   --seed-pool canonical_env_60
  python -m robofactory.tools.probe_noop_lb --policy random --seed-pool canonical_env_60
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import gymnasium as gym
import numpy as np

from robofactory.tasks import *  # noqa: F401,F403  registers LiftBarrier-rf with gym
import robofactory.agents  # noqa: F401  registers panda agents
from robofactory.utils.eval_seeds import resolve_seeds


def _qpos_per_arm(env, num_arms: int):
    """Return list of (7,) arm-joint qpos per arm, read straight from the agent robots.

    Reading from env.unwrapped.agent.agents[i].robot.get_qpos() is obs-mode-agnostic
    (works with obs_mode='state', which returns a flat tensor, not a dict)."""
    agents = env.unwrapped.agent.agents
    out = []
    for i in range(num_arms):
        q = np.asarray(agents[i].robot.get_qpos().cpu()).squeeze()
        out.append(q[:7].astype(np.float32))
    return out


def _hold_action(env, num_arms, action_prefix):
    """Action = stay at current qpos, gripper open (0). Robot does not move."""
    qs = _qpos_per_arm(env, num_arms)
    return {
        f"{action_prefix}-{i}": np.concatenate([qs[i], np.zeros(1, dtype=np.float32)]).astype(np.float32)
        for i in range(num_arms)
    }


def _random_action(action_space, rng):
    """Uniform random action sampled from the env's action space (per arm)."""
    out = {}
    for k, sp in action_space.spaces.items():
        lo = np.asarray(sp.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(sp.high, dtype=np.float32).reshape(-1)
        # action_space bounds may be +-inf for joint targets; clip to a sane joint range.
        lo = np.where(np.isfinite(lo), lo, -3.14)
        hi = np.where(np.isfinite(hi), hi, 3.14)
        out[k] = rng.uniform(lo, hi).astype(np.float32)
    return out


def _is_success(info) -> bool:
    s = info.get("success", False)
    if hasattr(s, "item"):
        s = s.item()
    return bool(s)


def run_probe(policy: str, seeds, scene: str, max_env_steps: int, num_arms: int, verbose: bool):
    env = gym.make(
        "LiftBarrier-rf",
        scene=scene,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
        num_envs=1,
        enable_shadow=False,  # match datagen / PR3
    )
    action_prefix = list(env.action_space.spaces.keys())[0].rsplit("-", 1)[0]
    try:
        robot_uid = env.unwrapped.agent.agents[0].uid
    except Exception:
        robot_uid = "panda"

    per_seed = []
    n_success = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=int(seed))
        if env.action_space is not None:
            env.action_space.seed(int(seed))
        succ = False
        steps_done = 0
        for step in range(max_env_steps):
            if policy == "hold":
                action = _hold_action(env, num_arms, action_prefix)
            elif policy == "random":
                action = _random_action(env.action_space, rng)
            else:
                raise ValueError(f"unknown policy {policy!r}")
            obs, _r, terminated, truncated, info = env.step(action)
            steps_done = step + 1
            if _is_success(info):
                succ = True
                break
            if bool(np.asarray(truncated).reshape(-1)[0]):
                break
        n_success += int(succ)
        per_seed.append({"seed": int(seed), "success": int(succ), "steps": int(steps_done)})
        if verbose:
            print(f"[{policy}] seed={seed} success={int(succ)} steps={steps_done} "
                  f"| running {n_success}/{len(per_seed)}", flush=True)
    env.close()
    return n_success, per_seed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", choices=["hold", "random"], required=True,
                   help="hold = hold-qpos (no-op); random = uniform random actions")
    p.add_argument("--seed-pool", default="canonical_env_60",
                   help="eval_seeds pool name (default canonical_env_60)")
    p.add_argument("--scene", default="table", choices=["table", "robocasa"],
                   help="LB scene config (default table; d2_wristcam LB trained on table)")
    p.add_argument("--max-env-steps", type=int, default=400,
                   help="per-episode env-step budget (PR7 unified cap = 400)")
    p.add_argument("--num-arms", type=int, default=2)
    p.add_argument("--json-out", default="", help="optional path to write the full report JSON")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    seeds, prov = resolve_seeds(pool=a.seed_pool)
    print(f"[probe] policy={a.policy} scene={a.scene} pool={a.seed_pool} "
          f"n_seeds={len(seeds)} sha={prov['seed_pool_sha'][:12]} max_env_steps={a.max_env_steps}",
          flush=True)
    t0 = time.time()
    n_success, per_seed = run_probe(
        a.policy, seeds, a.scene, a.max_env_steps, a.num_arms, verbose=not a.quiet
    )
    dt = time.time() - t0
    report = {
        "policy": a.policy,
        "scene": a.scene,
        "seed_pool": a.seed_pool,
        "seed_pool_sha": prov["seed_pool_sha"],
        "n_seeds": len(seeds),
        "max_env_steps": a.max_env_steps,
        "n_success": n_success,
        "success_rate": n_success / len(seeds) if seeds else 0.0,
        "wall_s": round(dt, 1),
        "per_seed": per_seed,
    }
    print("\n=== PROBE RESULT ===")
    print(f"{a.policy} on {a.scene} / {a.seed_pool}: {n_success}/{len(seeds)} successes "
          f"({100.0 * report['success_rate']:.1f}%) in {dt:.1f}s")
    if a.policy == "hold":
        if n_success == 0:
            print("GATE-6: hold-qpos 0/60 -> NO spawn-state/physics false positive at rest. PASS.")
        else:
            print(f"GATE-6 FINDING: hold-qpos {n_success}/{len(seeds)} > 0 -> spawn-state/physics "
                  f"FALSE POSITIVE; every LB number gets a false-positive asterisk.")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
