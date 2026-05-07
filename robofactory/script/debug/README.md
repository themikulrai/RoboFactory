# script/debug/

Diagnostic, replay, and analysis scripts used during the openpi-robofactory
debugging cycles. Kept on purpose — these are the tools that produced the
encoder-collapse, scene-mismatch, and pipeline-parity findings, and will be
re-run when similar bugs recur.

Output paths: scripts that write artifacts accept `--output_dir`; default is
`/iris/u/mikulrai/debug_output/<script-stem>_<YYYYMMDD>/` via
`robofactory.utils.paths.debug_output_dir(__file__)`. That root is swept
nightly: files older than 30 days are removed (cron installed via
`robofactory/utils/install_debug_output_cron.sh`). Pin anything you want to
keep by passing `--output_dir` to a path outside `debug_output/`.

## Python scripts

| Script | What it does |
|---|---|
| `analyze_decent_dp_features.py` | Analyze .npz features from probe script: cosine-sim stats, PCA scatter, action overlays, heatmaps. |
| `check_action_qpos_R2.py` | Fit linear models predicting action chunks from qpos history; per-dim R² split by episode. |
| `check_diffusion_stochasticity.py` | Test whether DDPM sampling is stochastic or deterministic via seed/noise sweeps. |
| `dump_seed_poses.py` | Analytically compute cube initial positions for 30 eval seeds by reproducing the RNG chain. |
| `eval_multi_dp_qpos_swap.py` | Closed-loop eval with in-distribution qpos swap to test proprioceptive over-reliance. |
| `forward_one.py` | Feed one observation through a loaded policy; inspect every layer with raw/normalized stats. |
| `inspect_arm2_demos.py` | Inspect arm2 action distribution: gripper + joint-delta time series across 150 demos. |
| `inspect_ckpt.py` | Checkpoint contract audit: shape_meta, n_obs/horizon/action_steps, normalizer state. |
| `plot_atlas_figures.py` | Publication figures for pi0.5 TSC debug: SR bars, arm imbalance, cube scatter, gripper heatmap. |
| `plot_trajectories.py` | Plot per-step trajectory data (gripper, joint deltas, cube xyz) from JSONL eval logs. |
| `probe_decent_dp_features.py` | Probe whether DP encoder features vary with cube positions; save .npz per seed/agent. |
| `probe_decent_dp_features_bn.py` | BN-mode × epoch sweep: encoder in eval vs train mode across multiple checkpoint epochs. |
| `probe_decent_dp_features_joint.py` | Joint (centralised) DP encoder probe: dump global features + joint action chunk. |
| `probe_decent_dp_features_qposswap.py` | In-distribution qpos-swap probe: swap agent_pos with training qpos from a different seed. |
| `probe_encoder_dino.py` | Encoder feature-discrimination: DINOv2 vs ResNet18 variants; H2 metric across seeds. |
| `probe_env.py` | Dump env contract: sensor keys, image stats, per-agent qpos, actor positions over seeds. |
| `probe_pi05_action_dist_tsc.py` | Cross-seed action-distribution probe for pi0.5 cent at step 0: 16×24 chunks + std. |
| `probe_spawn_distribution.py` | Quick scan: print meat xy + whether within training spawn range for a seed range. |
| `probe_train_data.py` | Dump training-data contract: camera stats, per-dim action/state stats, qpos, env_state. |
| `probe_tsc_spawn.py` | Probe TSC cube positions for train episodes + eval seeds; reconstruct via colour detection. |
| `reconstruct_train_spawns_pm.py` | Reconstruct training meat xy from zarr images via pixel-variance + back-projection. |
| `reconstruct_train_spawns_pm_v2.py` | Reconstruct training meat xy from h5 tcp_pose at grasp time. |
| `replay_h5.py` | Replay h5 actions on eval env: separates env-drift vs obs-conditioning failure modes. |
| `teacher_force_pi05_tsc.py` | Teacher-forced replay on 5 cent training episodes; compare predicted vs recorded action. |
| `test_dinov2_factories.py` | Sanity test for DINOv2 encoder factories: shapes, deepcopy, LoRA/patch-attn trainability. |
| `test_render_brightness.py` | Test rendering brightness matches training data across shader packs. |
| `test_seed_determinism.py` | Verify `env.reset(seed)` is fully deterministic across same-seed, cross-seed, cross-process. |
| `test_step_n_conditioning.py` | Test policy conditioning on image at step N after the arm has moved (step-0 padding check). |

## Shell wrappers

| Script | What it does |
|---|---|
| `compare_predictions.sh` | Compare policy predictions for success vs failure seeds; runs `forward_one` for 3 seeds. |
| `run_brightness_test.sh` | Brightness test across shader packs (default, rt-fast, rt) with env-var manipulation. |
| `run_compute_diag.sh` | Compute diagnostics: `forward_one` train/env + `replay_h5` on PickMeat. |
| `run_post_vulkan_diag.sh` | Post-Vulkan diagnostics: brightness + `forward_one` train/env + `replay_h5` with ICD checks. |
| `run_shadow_fix_verify.sh` | Verify shadow=False closes the train-vs-env image gap; compares `forward_one` outputs. |
| `run_shadow_fix_verify2.sh` | Verify env obs mean ~0.484 with shader_pack=default + shadow=False. |
| `run_shadow_fix_verify3.sh` | Confirm env obs mean ~0.484 with shader_pack=default + enable_shadow=False. |
| `run_spawn_and_stochasticity_diagnostics.sh` | Run spawn reconstruction + TSC probe + DDPM stochasticity test in sequence. |
| `watch_ckpt.sh` | Watch checkpoint directory for new ckpts and trigger downstream eval. |

## Footguns — remaining hardcoded paths

Five scripts (`dump_seed_poses.py`, `inspect_arm2_demos.py`,
`plot_atlas_figures.py`, `probe_pi05_action_dist_tsc.py`,
`teacher_force_pi05_tsc.py`) now take `--output_dir` and default to
`debug_output/<script-stem>_<YYYYMMDD>/`. Two paths still hardcoded
intentionally; one residual carry-over.

### Pinned checkpoints (intentional — leave alone)

These reference a specific canonical PickMeat ckpt. Pass a different ckpt by
editing the constant if you need to.

| Script | Line | Hardcoded path |
|---|---|---|
| `check_diffusion_stochasticity.py` | L31 | `/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt` |
| `test_step_n_conditioning.py` | L34 | `/iris/u/mikulrai/checkpoints/RoboFactory/PickMeat-rf_150/300.ckpt` |

### Cross-script input-read carry-over (will be parameterized later)

`plot_atlas_figures.py` reads outputs from `inspect_arm2_demos.py` and
`dump_seed_poses.py` from their *legacy* paths under
`/iris/u/mikulrai/logs/tsc_debug/`. After the `--output_dir` refactor those
sister scripts default elsewhere — so to consume freshly-regenerated inputs,
copy or symlink them into the legacy paths, or wait for `plot_atlas_figures.py`
to grow `--seed_poses_dir` and `--arm2_demos_dir` flags.

| Script | Line | Hardcoded read |
|---|---|---|
| `plot_atlas_figures.py` | L93 | `/iris/u/mikulrai/logs/tsc_debug/arm2_demos/summary.json` |
| `plot_atlas_figures.py` | L157 | `/iris/u/mikulrai/logs/tsc_debug/seed_poses/seed_poses.json` |
| `plot_atlas_figures.py` | L195 | `/iris/u/mikulrai/logs/tsc_debug/arm2_demos` |
