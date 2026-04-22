"""Fast sanity check: instantiate dataset + policy, run one forward pass.

Args:
    --variant  (decent | joint3 | joint4)
    --zarr     zarr path
"""
import argparse
import os
import sys
import time

import torch
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _cfg(variant: str, zarr_path: str):
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "diffusion_policy", "config")
    if variant == "decent":
        config = OmegaConf.load(os.path.join(base, "robot_dp.yaml"))
        task = OmegaConf.load(os.path.join(base, "task", "default_task_wristcam.yaml"))
    elif variant == "joint3":
        config = OmegaConf.load(os.path.join(base, "joint_dp.yaml"))
        task = OmegaConf.load(os.path.join(base, "task", "joint_task_3arm.yaml"))
    elif variant == "joint4":
        config = OmegaConf.load(os.path.join(base, "joint_dp.yaml"))
        task = OmegaConf.load(os.path.join(base, "task", "joint_task_4arm.yaml"))
    else:
        raise ValueError(variant)
    config = OmegaConf.merge(config, OmegaConf.create({"task": task}))
    config.task.dataset.zarr_path = zarr_path
    config.task.dataset.max_train_episodes = 4
    config.dataloader.batch_size = 2
    config.val_dataloader.batch_size = 2
    config.training.device = "cpu"
    OmegaConf.resolve(config)
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--zarr", required=True)
    args = ap.parse_args()

    t0 = time.time()
    cfg = _cfg(args.variant, args.zarr)
    print(f"[dry] loaded cfg in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"[dry] dataset len={len(dataset)} ({time.time()-t0:.1f}s)", flush=True)

    sample = dataset[0]
    print("[dry] sample obs keys:", list(sample["obs"].keys()))
    for k, v in sample["obs"].items():
        print(f"  obs/{k}: {tuple(v.shape)} {v.dtype}")
    print(f"  action: {tuple(sample['action'].shape)} {sample['action'].dtype}")

    t0 = time.time()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    print(f"[dry] policy instantiated ({time.time()-t0:.1f}s)", flush=True)

    batch = {
        "obs": {k: v.unsqueeze(0) for k, v in sample["obs"].items()},
        "action": sample["action"].unsqueeze(0),
    }
    t0 = time.time()
    loss = policy.compute_loss(batch)
    print(f"[dry] forward loss={float(loss):.4f} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
