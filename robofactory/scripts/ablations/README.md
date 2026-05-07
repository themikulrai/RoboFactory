# scripts/ablations/

SLURM launchers for the **image-encoder ablation study** on PickMeat /
workspace cameras (the study that produced the publication site at
`docs/ablations/`). Tracked in git so each ablation cell is reproducible.

The encoder ranking confirmed by these runs:

> ImageNet 58.3% > DinoS-LoRA 53.3% > ImageNet-crop 43.3% > R3M 35% > DinoS-SPatch 16.7%

Sister directory: [`scripts/canonical/`](../canonical/) for production
keep-set launchers (incl. the in1k baseline that anchors this ranking).

## Train launchers

| Script | Encoder | What it does |
|---|---|---|
| `retrain_dp_pm_d1_ep300_crop.sh` | ImageNet (legacy crop)| DP train PM workspace 150 demos with the older crop preprocessing. Superseded by `_in1k_crop` below. |
| `retrain_dp_pm_d1_ep300_in1k_crop.sh` | ImageNet + crop aug | DP train PM workspace 150 demos with explicit ImageNet + crop augmentation pipeline. |
| `retrain_dp_pm_d1_ep300_dinov2_blora.sh` | DINOv2-S + LoRA | DP train PM workspace 150 demos with DINOv2-Small encoder, LoRA-tuned. |
| `retrain_dp_pm_d1_ep300_dinov2_spatch.sh` | DINOv2-S + SPatch | DP train PM workspace 150 demos with DINOv2-Small encoder, SPatch attention. |
| `retrain_dp_pm_d1_ep300_r3m.sh` | R3M | DP train PM workspace 150 demos with R3M encoder. |

## Eval launchers (60 seeds each)

| Script | Encoder | What it does |
|---|---|---|
| `pm_eval_in1k_crop_60seeds.sh` | ImageNet-crop | Eval the in1k+crop ckpt on the canonical 60-seed set. |
| `pm_eval_dino_blora_60seeds.sh` | DINOv2-S+LoRA | Eval the DinoS-LoRA ckpt on 60 seeds. |
| `pm_eval_dino_spatch_60seeds.sh` | DINOv2-S+SPatch | Eval the DinoS-SPatch ckpt on 60 seeds. |
| `pm_eval_r3m_60seeds.sh` | R3M | Eval the R3M ckpt on 60 seeds. |

The ImageNet (no-crop) baseline lives in [`scripts/canonical/`](../canonical/)
because it's the production keep-set ckpt, not just the winning ablation cell.

## Pi0.5 LIBERO ablation

The Pi0.5 LIBERO-spatial-LoRA phase-script ablation
(`pi05_libero_spatial_lora_*` ckpts + `slurm_phase{1..8}*.sh` launchers) lives
in the openpi repo at `/iris/u/mikulrai/projects/openpi/scripts/`, not here.
Those launchers are tightly coupled to openpi paths and stay with their host
repo.
