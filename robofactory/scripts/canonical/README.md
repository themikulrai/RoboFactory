# scripts/canonical/

Production SLURM launchers for the openpi-robofactory **keep-set** runs — the
ones that produce the primary table cells (PM workspace baseline, TSC
d2_wristcam decentralised, TSC d1 decentralised, etc.). Tracked in git so the
exact recipe that produced each canonical result is reproducible.

Sister directory: [`scripts/ablations/`](../ablations/) for image-encoder
ablations and similar studies.

## Launchers

### Pick Meat (workspace cam) — DP

| Script | What it does |
|---|---|
| `retrain_dp_pm_d1_ep300_in1k.sh` | DP train PM workspace 150 demos, ImageNet encoder, 300 epochs. Produces the **58.3% baseline** ckpt at `/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/backup/300_in1k.ckpt`. |
| `pm_eval_in1k_60seeds.sh` | DP eval the in1k PM ckpt on 60 seeds (the run that confirmed 35/60 = 58.3%). |

### Three-Robot Stack Cube d2_wristcam decentralised — DP

| Script | What it does |
|---|---|
| `retrain_dp_tsc_d2_ep300_in1k_a0.sh` | DP train TSC d2_wristcam decent **arm0** (orion partition with `--requeue`). |
| `retrain_dp_tsc_d2_ep300_in1k_a1.sh` | Same for **arm1**. |
| `retrain_dp_tsc_d2_ep300_in1k_a2.sh` | Same for **arm2**. |
| `resume_dp_tsc_d2_in1k_a0_from285.sh` | Resume arm0 from epoch 285 (used 2026-05-07 to push past the orion preempt boundary). |
| `resume_dp_tsc_d2_in1k_a1_from285.sh` | Same for arm1. |
| `resume_dp_tsc_d2_in1k_a2_from285.sh` | Same for arm2. |
| `tsc_d2_wristcam_table_60seeds_reeval.sh` | Re-eval the three d2_wristcam decent ckpts on the **TABLE** scene (`configs/table/three_robots_stack_cube.yaml`). **Verdict (2026-05-07, job 15359012):** 0/29 seeds at the time of writing, fully consistent — the scene-mismatch hypothesis is **rejected** and these ckpts are confirmed encoder-collapse (mirroring the D1 finding). Per plan v2 Phase B "re-eval-then-delete" rule, the d2 decent ckpts move to deletable pending the user's call. |

### Three-Robot Stack Cube d1 decentralised — DP

| Script | What it does |
|---|---|
| `tsc_d1_eval_decent_in1k_60seeds.sh` | DP eval d1 decent ckpts (arm0/1/2) on 60 seeds. **Note:** d1 ckpts are subject to the encoder-collapse failure mode documented in `project_d1_tsc_encoder_collapse` — `val_action_mse` looks fine but rollouts fail; verify with the Stage-2 encoder-collapse probe before trusting any non-zero SR. |

## Cross-run checkpoint clobber — read before launching anything

Both DP and Pi0.5 have a checkpoint-path collision footgun (Phase C2 will fix
the underlying code). Until then:

### DP — shared zarr stem = shared ckpt dir

`policy/Diffusion-Policy/diffusion_policy/workspace/robotworkspace.py:381`
saves to `checkpoints/{zarr_stem}/{epoch+1}.ckpt`. Path is keyed only on zarr
stem and epoch — **not** on run id, encoder, or git sha. Concrete consequence:

- Two DP runs that share a zarr stem (e.g. `_agent0_d2_wristcam_150.zarr`)
  both write into the same dir.
- Each save can silently overwrite an earlier run's same-epoch ckpt.
- A later resume can load encoder-incompatible weights without warning.

**Operational rule until C2 fix lands:** never run two DP jobs on the same
zarr stem concurrently or with different encoder configs.

### Pi0.5 — shared `(config, exp_name)` = shared ckpt dir + `--overwrite` rmtree

openpi `scripts/train.py` keys `checkpoint_dir` only on `(config, exp_name)`.
`--overwrite` calls `shutil.rmtree(checkpoint_dir)` at startup, **deleting
any prior run's ckpts that happen to share the same exp_name**. Real loss:
~14 GPU-hours of `fuktr1yw`'s 18000-step ckpt vanished when `v7fldf8d` ran
with `--overwrite`.

**Operational rule until C2 fix lands:** never reuse an `exp_name` across
distinct training attempts on Pi0.5.
