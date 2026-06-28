# Subtask-obedience GATE (LiftBarrier, decentralized pi0.5 LL)

Does the trained per-arm LL actually FOLLOW its subtask language channel, or
ignore it and run on proprioception? This is the GO/NO-GO before training a
downstream VLM. Scripted, open-loop, no VLM.

Files: `eval_subtask_obedience.py` (eval), `run_subtask_obedience_eval.sh`
(launcher), `scripts/dart/tests/test_subtask_obedience.py` (pure off-GPU tests).

## VALID vs LEGACY verdict (decentralized ckpts)

The per-arm DECENT ckpts only ever trained on ONE side (arm0 = left end). So the
**left-vs-right** probe is INVALID for them — "grasp the right end" is OOD and its 0%
is an artifact, not disobedience. For decent ckpts use **`--mode valid`**, which combines
only the two IN-DISTRIBUTION probes below (wait-vs-act + grasp-vs-lift) into a `verdict`.
`--mode gate` (left-vs-right + waitact) is kept for the legacy/centralized case only.

## Probes (from a fixed env reset, sweep N seeds)

1. **left-vs-right** (`--mode probe`): same reset, probed arm gets "grasp the
   left end" (variant L) vs "grasp the right end" (variant R); other arm "wait".
   Obedient(seed) := variant L ends strictly nearer the LEFT contact end AND
   variant R nearer the RIGHT. `obedience_rate` = fraction of seeds obedient.
   The prompt-ignoring baseline is chance / always-nearest (~0.5 if it always
   goes to one fixed end -> 0 by the strict both-variants rule).

2. **wait-vs-act** (`--mode waitact`): probed arm gets "wait" vs "grasp the left
   end"; other arm "wait" in both. Metric: probed-arm TCP displacement. Obedient
   => wait-motion ~ 0, act-motion clearly larger. Per-seed pass = ratio
   `act/wait >= 3` AND `wait_motion <= 0.05 m` (the arm holds still on "wait").

3. **grasp-vs-lift** (`--mode grasplift`): same side, only the VERB differs — "grasp
   the left end" vs "lift the left end". Both variants are PRIMED with an identical
   approach->close grasp prefix first (the LL only ever saw "lift" after a grasp, so a
   cold lift is OOD), then we measure signed vertical TCP displacement `dz` from the
   post-grasp pose. Obedient => `dz_lift - dz_grasp >= 0.03 m`. Also logs `tcp_sep`
   (3D separation of the two verbs' final TCP) as an axis-agnostic insurance metric.

`--mode valid` runs wait-vs-act + grasp-vs-lift + emits a single `verdict` (the decent
go/no-go). `--mode gate` runs left-vs-right + wait-vs-act (legacy).

## Classifier-free guidance (CFG) sweep

The openpi pi0.5 sampler takes an optional guidance scale `w` (env var `OPENPI_CFG_W` on
the served policy; `--cfg-w W` on the launcher). `w=1` is the default (byte-identical to
the original sampler); `w>1` sharpens prompt-conditioning via `v = v_uncond + w*(v_cond -
v_uncond)`, with the null = prompt masked off. CAVEAT: these ckpts train with NO prompt
dropout, so the empty-prompt null is OUT-OF-DISTRIBUTION and CFG is an approximation.
Sweep over w with `scripts/run_cfg_sweep_obedience.sh`.

## PASS threshold (the verdict)

PASS iff BOTH:
- left-right `obedience_rate >= 0.7`, AND
- mean `act/wait` motion ratio `>= 3.0` AND mean `wait_motion <= 0.05 m`.

Rationale: 0.7 is well above the 0.5 chance of a prompt-ignorer on the binary
left/right choice (and above an always-one-end policy's 0.0 under the strict
both-variants rule), while tolerant of a few noisy seeds. The motion gate
separates "moves on command" from "drifts regardless": a policy that ignores
"wait" and creeps fails the `wait_motion` cap even with a high ratio; a frozen
policy fails the ratio. Both must hold => the language channel is causally live.

## Output JSON (the deliverable)

`probe.{obedience_rate,num_obedient,num_seeds,per_seed[...]}`,
`waitact.{pass_rate,mean_wait_motion,mean_act_motion,mean_ratio,per_seed[...]}`,
`verdict.{verdict,left_right_obedience_pass,wait_act_pass,...}`. Optional
`probe_baseline`/`waitact_baseline`/`verdict_baseline` when a prompt-ignoring
(flatbaseline) ckpt is served via `--baseline-ckpt-arm{0,1}`.

Tiled multiview rollout videos (the views the LL saw, with subtask overlay) are
written per variant unless `--no-record-video`.

## Run it (checkpoints land at step 15000)

Compute/workstation node only (SAPIEN; login node renders dark). Probe arm0:

```bash
CK=/iris/u/mikulrai/checkpoints/openpi
bash run_subtask_obedience_eval.sh \
  --cam ws --arm 0 --n-seeds 15 --mode gate \
  --ckpt-arm0 $CK/pi05_robofactory_lb_ws_subtaskdart_decent_arm0/lb_ws_subtaskdart_decent_arm0_15k_v1/15000 \
  --ckpt-arm1 $CK/pi05_robofactory_lb_ws_subtaskdart_decent_arm1/lb_ws_subtaskdart_decent_arm1_15k_v1/15000
```

Swap `--cam ws` -> `--cam wc` and the `ws`->`wc` ckpt paths for wristcam.
`--arm 1` probes arm1 (ckpts already served; documents the second arm).
The launcher derives config names `pi05_robofactory_lb_{cam}_subtaskdart_decent_arm{i}`
and runs the PR2 server-identity handshake (`--no-expect-config` to skip).
Ports are job-unique free ports (no 8000/8001 collision).
