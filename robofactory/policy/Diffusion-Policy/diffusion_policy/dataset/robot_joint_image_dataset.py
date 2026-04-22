"""
Dataset for the centralised (joint) diffusion policy.

Loads N per-arm camera streams (`head_camera_0 ... head_camera_{N-1}`) and an
optional global camera (`head_camera_global`) plus a single concatenated
`agent_pos` of shape (8*N,).

The zarr schema is produced by scripts/parse_h5_to_zarr_unified.py
(--mode joint). All images are stored at 3x224x224 uint8.

Implements the same two `__getitem__` paths as the per-arm RobotImageDataset:
  * int  -> single sequence (for ad-hoc inspection / sanity checks)
  * ndarray -> full batch via pre-allocated buffers (the hot path used by
    the custom BatchSampler in workspace.create_dataloader).

`postprocess` renames zarr key `state` to the policy-facing `agent_pos` and
normalises the uint8 RGB streams to float32 in [0,1]. Image key names match
shape_meta exactly (head_camera_0..N-1, head_camera_global), so no rename.
"""
from typing import Dict
import copy
import numba
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
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
                 include_global: bool = False,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 batch_size: int = 64,
                 max_train_episodes=None):
        super().__init__()
        self.n_agents = n_agents
        self.include_global = include_global
        cam_keys = [f"head_camera_{i}" for i in range(n_agents)]
        if include_global:
            cam_keys.append("head_camera_global")
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

        # pre-allocate per-key batch buffers (same pattern as RobotImageDataset)
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

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
        for ck in self.cam_keys:
            normalizer[ck] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        if isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size, f"batch size mismatch: {len(idx)} vs {self.batch_size}"
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices,
                                       idx, self.sampler.sequence_length)
            return self.buffers_torch
        raise TypeError(f"unsupported idx type {type(idx)}")

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        obs = {"agent_pos": agent_pos}
        for ck in self.cam_keys:
            obs[ck] = samples[ck].to(device, non_blocking=True) / 255.0
        return {"obs": obs, "action": action}


def _batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray,
                           indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray,
                          indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2 ** 16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
