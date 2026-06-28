# Why flat-baseline LB pi0.5 lifts-then-lets-go — training root-cause (2026-06-25)

**Q:** the policy does a brief genuine two-arm co-lift then releases into a one-arm dangle (sustained-hold SR 2/60). What in the
last training caused it, what to change. **Method:** 8-agent forensic workflow (data/horizon/DART/encoding/recipe/structure/best-practice + adversarial synthesis).

## Primary cause (HIGH confidence): the DATA has no sustained-hold target
- Trained on `robofactory_lift_barrier_{ws,wc}_subtaskdart_v1` (200 ep, from `subtask_combined_lb_v1/full200_merged/LiftBarrier/`), ckpt 14999.
- The data-gen keep-gate (`check_barrier_ends_held`, instantaneous z-check — same false-positive family as LB `success()`) **auto-terminates every demo ~8-16 frames after the bar clears threshold**:
  - **0/200** episodes have any ≥16-frame sustained hold after success; success fires on the *final* frame (run=1).
  - Final-frame barrier speed **median 0.255 m/s** (192/200 still moving >0.1); the **terminal delta-action is the LARGEST upward command of the whole trajectory (54× mid-traj).**
- => the BC/flow target distribution has **zero mass on a low-velocity two-arm equilibrium hold.** The policy faithfully reproduces "lift hard, then episode over": a real ~8-20 frame co-lift, then no hold target → it extrapolates the lift → one arm's trajectory carries its (still-closed) gripper off the bar → dangles ~400 frames.

## Amplifier (MEDIUM, unresolved): decentralized desync
- Release is **asymmetric** — arm1 releases first 9-14× vs arm0 2-3× despite symmetric labels → inference-time desync, not a data artifact. Two independent per-arm policies, no coupling beyond a shared global cam + static prompt, nothing represents "the bar is jointly held, keep gripping." NOT isolated from the data cause; a centralized re-eval resolves it.

## Refuted (don't chase these)
- **A2 OOD-horizon:** let-go at eval step ~53 (range 43-90) is **before the shortest demo ends (69)** → in-distribution, NOT run-off-the-end.
- **A3 DART-shoves-teach-release:** 0/200 gripper re-open labels; shoves were σ=0.1 frozen-grip, never recorded as a release.
- **A4 action-encoding/gripper-drift:** gripper stays -1.0, fingers clamped 0.0, joint err ~0.02 rad; release = contact→0 while the empty arm's end-z KEEPS RISING (trajectory/coordination failure, not drift); train/eval encodings match; norm-stats benign.
- **A5 recipe:** real but secondary — ran 15k not the 20k spec, `lr_schedule.decay_steps` left at the global 30000 so LR was still 57% of peak at cutoff; but loss is low/flat and 50k-memory shows steps don't buy execution.

## Fix plan — cheapest-high-impact first
- **STEP 0 (free, eval-only):** centralized + replan-sweep re-eval of existing ckpts under the sustained-hold criterion + contact traces → resolve the structure question (A6) before spending on training/data.
- **STEP 1 (cheap, NO sim):** append a **velocity-zeroed terminal static-hold tail** (~30 const-pose frames, delta→0, gripper -1.0) to every parquet episode, recompute norm stats, retrain one arm/cam. Config deltas: **action_horizon 16→32** (openpi pi05 default is 50; 16 barely spans the co-lift), **num_train_steps 20000**, **lr decay_steps=20000**, **batch 16→32**, keep **lora_only** (capacity not saturated). Eval under sustained-hold + contact traces. Directly tests the primary cause without sim re-gen.
- **STEP 2 (only if STEP 1 underperforms):** full sim re-data-gen with a **sustained-hold keep-gate** (z held + both-arms contact ≥20 consec settled frames) + **recorded terminal static-hold tail** (settle_steps≥30), reconvert, retrain with the STEP 1 config.
- **DON'T** add gripper-close supervision (LB is a joint-position cradle-hold; gripper sits at -1.0 by design).

## Open risks
- Synthetic freeze-tail assumes the freeze pose is a true equilibrium, but the demo end is mid-motion at peak → velocity-zero the pose; treat STEP 1 as directional, not the final dataset.
- 50k-memory warns the data fix may be **necessary but not sufficient** (policy may still fail to execute) → budget cent/full-finetune fallback.
- Decentralization (A6) MEDIUM confidence, not isolated from the data cause — STEP 0 closes it.
- Sustained-hold keep-gate may drop episodes (3-arm planner may not settle) → could shrink the 200-ep set.
