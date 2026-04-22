"""Go/no-go feasibility check: replay RoboFactory trajectories and measure success rate.

Usage:
    python replay_check.py --task TakePhoto --robot panda --count 10
    python replay_check.py --task TakePhoto --robot panda_wristcam --count 10
    python replay_check.py --task LongPipelineDelivery --robot panda --count 10
    python replay_check.py --task LongPipelineDelivery --robot panda_wristcam --count 10

Measures:
  - replay success rate (final step success flag)
  - L2 drift in qpos at last step vs. stored
  - how many trajs timeout (truncated)
"""
import os
os.environ.setdefault("SAPIEN_HEADLESS", "1")

import argparse
import json
import time
import numpy as np
import h5py
import gymnasium as gym
import robofactory  # registers envs

# Map dataset-name -> (registered env_id, h5/json base path, #agents)
TASK_TO_ENV = {
    "TakePhoto": ("TakePhoto-rf", "/iris/u/mikulrai/data/RoboFactory/hf_download/TakePhoto/TakePhoto", 4),
    "LongPipelineDelivery": ("LongPipelineDelivery-rf", "/iris/u/mikulrai/data/RoboFactory/hf_download/LongPipelineDelivery/LongPipelineDelivery", 4),
}


def run(task, robot, count):
    env_id, base, n_agents = TASK_TO_ENV[task]
    h5_path = base + ".h5"
    json_path = base + ".json"
    print(f"[{task}] env_id={env_id}  robot={robot}  count={count}")

    with open(json_path) as f:
        meta = json.load(f)
    env_kwargs = dict(meta["env_info"]["env_kwargs"])
    # Override problematic defaults for a clean headless CPU replay
    env_kwargs["render_mode"] = "sensors"
    env_kwargs["obs_mode"] = "state"         # skip rendering for speed during go/no-go
    env_kwargs["reward_mode"] = "sparse"
    env_kwargs["sim_backend"] = "cpu"
    env_kwargs["robot_uids"] = tuple([robot] * n_agents)

    env = gym.make(env_id, **env_kwargs)
    agent_keys = [f"panda-{i}" for i in range(n_agents)]

    results = []
    h5f = h5py.File(h5_path, "r")
    episodes = meta["episodes"][:count]
    t0 = time.time()
    for ep in episodes:
        ep_id = ep["episode_id"]
        seed = ep["episode_seed"]
        reset_kwargs = dict(ep.get("reset_kwargs", {}))
        reset_kwargs["seed"] = seed

        traj = h5f[f"traj_{ep_id}"]
        actions_per_agent = {k: np.asarray(traj["actions"][k][:]) for k in agent_keys}
        T = actions_per_agent[agent_keys[0]].shape[0]

        env.reset(**reset_kwargs)
        truncated = False
        success_any = False
        last_info = {}
        for t in range(T):
            act = {k: actions_per_agent[k][t].astype(np.float32) for k in agent_keys}
            _obs, _rew, _term, truncated, info = env.step(act)
            last_info = info
            if bool(np.asarray(info.get("success", False)).any()):
                success_any = True
            if truncated:
                break

        final_success = bool(np.asarray(last_info.get("success", False)).any())
        stored_final_success = bool(traj["success"][-1])

        # qpos drift between replayed agent and stored (best-effort; skip if shapes mismatch)
        try:
            obs_now = env.unwrapped.get_obs()
            drifts = []
            for k in agent_keys:
                qn = obs_now["agent"][k]["qpos"]
                qn = qn.cpu().numpy() if hasattr(qn, "cpu") else np.asarray(qn)
                qn = qn.reshape(-1)[:9]
                qs = np.asarray(traj["obs"]["agent"][k]["qpos"][-1]).reshape(-1)[:9]
                drifts.append(float(np.linalg.norm(qn - qs)))
            max_drift = max(drifts) if drifts else float("nan")
        except Exception as e:
            max_drift = float("nan")

        print(f"  ep{ep_id:3d} seed={seed:3d} T={T:4d} final_success={final_success}  "
              f"any_success={success_any}  stored_final={stored_final_success}  "
              f"truncated={truncated}  max_qpos_drift={max_drift:.4f}")
        results.append({
            "episode_id": ep_id,
            "seed": seed,
            "T": T,
            "final_success": final_success,
            "success_any": success_any,
            "stored_final_success": stored_final_success,
            "truncated": bool(truncated),
            "max_qpos_drift": max_drift,
        })

    h5f.close()
    env.close()

    elapsed = time.time() - t0
    n = len(results)
    n_success = sum(r["final_success"] for r in results)
    n_any_success = sum(r["success_any"] for r in results)
    mean_drift = float(np.mean([r["max_qpos_drift"] for r in results])) if results else 0.0
    max_drift_all = float(max((r["max_qpos_drift"] for r in results), default=0.0))
    print()
    print(f"=== SUMMARY [{task} | {robot}] ===")
    print(f"final_success: {n_success}/{n}    any-step success: {n_any_success}/{n}")
    print(f"mean max-qpos-drift: {mean_drift:.4f}   worst: {max_drift_all:.4f}")
    print(f"elapsed: {elapsed:.1f}s ({elapsed/max(n,1):.1f}s/ep)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_TO_ENV), required=True)
    ap.add_argument("--robot", choices=["panda", "panda_wristcam"], required=True)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()
    run(args.task, args.robot, args.count)
