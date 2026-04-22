"""
Dataset for the centralised (joint) diffusion policy.

Mirrors robot_image_dataset.RobotImageDataset exactly, except it loads N camera
streams (`head_camera_0 ... head_camera_{N-1}`) and emits an obs dict with keys
`head_cam_0 ... head_cam_{N-1}` plus a single `agent_pos` of shape (8*N,).

The zarr schema is produced by scripts/parse_h5_to_zarr_joint.py.
"""
from typing import Dict
import copy
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset


class RobotJointImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 n_agents: int,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 batch_size: int = 64,
                 max_train_episodes=None):
        super().__init__()
        self.n_agents = n_agents
        cam_keys = [f"head_camera_{i}" for i in range(n_agents)]
        keys = cam_keys + ["state", "action"]
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size
        self.cam_keys = cam_keys

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for i in range(self.n_agents):
            normalizer[f"head_cam_{i}"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        obs = {"agent_pos": agent_pos}
        for i, ck in enumerate(self.cam_keys):
            # (T, 3, H, W) uint8 -> float32 [0, 1]
            obs[f"head_cam_{i}"] = sample[ck].astype(np.float32) / 255.0
        return {"obs": obs, "action": sample["action"].astype(np.float32)}

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            data = self._sample_to_data(sample)
            # to torch
            data["obs"] = {k: torch.from_numpy(v) for k, v in data["obs"].items()}
            data["action"] = torch.from_numpy(data["action"])
            return data
        raise NotImplementedError(f"unsupported idx type {type(idx)}")
