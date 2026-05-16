"""Tier A — Quantify the train/eval proprio gap on the overfit demo (traj_13).

At training: state[t] = action[t]  (zarr converter writes the command, not qpos)
At eval:     state[t] = qpos[t]    (env.get_obs returns actual joint state)

Compute, per step, what the model gets at training vs eval. The smoking gun
candidate is dim 7 (gripper): commanded in {-1, +1} at training, but a finger
width in [0.018, 0.04] m at eval.
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='/iris/u/mikulrai/data/RoboFactory/h5_data/TwoRobotsStackCube-rf.h5')
    ap.add_argument('--traj', type=int, default=13)
    ap.add_argument('--agent-key', default='panda_wristcam_multi')
    ap.add_argument('--action-key', default='panda')
    ap.add_argument('--out-json', default='/iris/u/mikulrai/runs/diagnostics/overfit_h3_tierA/summary.json')
    args = ap.parse_args()

    with h5py.File(args.h5, 'r') as f:
        tr = f[f'traj_{args.traj}']
        arms = {}
        for i in (0, 1):
            a = np.asarray(tr[f'actions/{args.action_key}-{i}'], dtype=np.float64)         # (T, 8)
            q = np.asarray(tr[f'obs/agent/{args.agent_key}-{i}/qpos'], dtype=np.float64)   # (T+1, 9)
            T = a.shape[0]
            # Align: action[t] is the command issued *at* step t; qpos[t] is the joint state
            # observed *before* applying action[t]. The most natural eval-time comparison is
            # action[t] vs qpos[t] (what the model would see at the same conditioning step).
            arm_diff = a[:, :7] - q[:T, :7]                             # (T, 7) rad
            # Compose an eval-time "agent_pos" the way eval_multi_dp.py does: qpos[:-2] + planner_gripper.
            # But the gripper at eval comes from a planner state machine; here we proxy it as
            # "the model is asked to look at qpos[7] (finger width) where it trained on action[7] (command)".
            grip_train = a[:, 7]                                        # (T,) in [-1, +1]
            grip_eval_finger = q[:T, 7]                                 # (T,) in [0.018, 0.04]
            arms[i] = dict(
                arm_diff=arm_diff,
                grip_train=grip_train,
                grip_eval_finger=grip_eval_finger,
                action=a,
                qpos=q,
                T=T,
            )

    summary = dict(h5=args.h5, traj=args.traj)
    for i, d in arms.items():
        arm_diff = d['arm_diff']                                        # rad
        per_joint_max_abs = np.max(np.abs(arm_diff), axis=0).tolist()   # (7,) rad
        per_joint_rms = np.sqrt(np.mean(arm_diff ** 2, axis=0)).tolist()
        per_step_l2 = np.linalg.norm(arm_diff, axis=1)                  # (T,) rad
        # Where the gap is worst:
        worst_t = int(np.argmax(per_step_l2))
        # Gripper analysis:
        grip_train = d['grip_train']
        grip_eval = d['grip_eval_finger']
        # If a normalizer fit on grip_train (range [-1,1]) is applied to grip_eval (range [0.02,0.04]),
        # the normalized eval value lands far outside the trained interval.
        train_min, train_max = float(grip_train.min()), float(grip_train.max())
        # Linear-normalize to [-1, +1] using the training range:
        def linnorm(x, lo, hi):
            scale = max(hi - lo, 1e-9)
            return 2 * (x - lo) / scale - 1
        norm_eval_using_train_stats = linnorm(grip_eval, train_min, train_max)
        # If train range covers eval values, this stays in [-1, +1]. If not, it overshoots.
        norm_eval_min = float(norm_eval_using_train_stats.min())
        norm_eval_max = float(norm_eval_using_train_stats.max())

        arm_key = f'arm{i}'
        summary[arm_key] = dict(
            T=d['T'],
            arm_dims_rad=dict(
                per_joint_max_abs=per_joint_max_abs,
                per_joint_rms=per_joint_rms,
                worst_step=worst_t,
                worst_step_l2_rad=float(per_step_l2[worst_t]),
                mean_step_l2_rad=float(per_step_l2.mean()),
                p95_step_l2_rad=float(np.percentile(per_step_l2, 95)),
            ),
            gripper=dict(
                action_train_range=[train_min, train_max],
                qpos_eval_range=[float(grip_eval.min()), float(grip_eval.max())],
                normalized_eval_signal_using_train_stats=[norm_eval_min, norm_eval_max],
                comment=(
                    "If norm range stays within [-1, +1], gripper signal is in-distribution. "
                    "If norm range is e.g. [-0.96, -0.94], the model sees a near-constant signal "
                    "where it trained on ~{-1, +1}; almost certainly an OOD input."
                ),
            ),
        )

    # Verdict
    max_arm_max = max(
        max(summary[f'arm{i}']['arm_dims_rad']['per_joint_max_abs']) for i in (0, 1)
    )
    worst_grip_overshoot = max(
        max(abs(summary[f'arm{i}']['gripper']['normalized_eval_signal_using_train_stats'][0] + 1),
            abs(summary[f'arm{i}']['gripper']['normalized_eval_signal_using_train_stats'][1] - 1))
        for i in (0, 1)
    )
    arm_verdict = 'large' if max_arm_max > 0.03 else 'small'    # 30 mrad
    grip_verdict = 'OOD' if worst_grip_overshoot > 0.1 else 'in_distribution'
    summary['verdict'] = dict(
        arm_dims=arm_verdict,
        gripper=grip_verdict,
        max_arm_per_joint_max_abs_rad=max_arm_max,
        gripper_worst_normalized_signal_distance_from_train_range=worst_grip_overshoot,
        keeps_hypothesis_alive=(arm_verdict == 'large' or grip_verdict == 'OOD'),
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    # Also print a brief summary
    print(json.dumps({'verdict': summary['verdict']}, indent=2))
    print(f"Full summary written to {args.out_json}")


if __name__ == '__main__':
    main()
