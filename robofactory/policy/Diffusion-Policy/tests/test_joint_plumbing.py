"""
Plumbing test for the joint (centralised) diffusion policy.

Builds a synthetic joint zarr with N=4 agents (matches TakePhoto-rf /
LongPipelineDelivery-rf), runs it through RobotJointImageDataset, instantiates a
DiffusionUnetImagePolicy from the joint shape_meta, and asserts that a forward
pass produces an action tensor of shape (B, horizon, 8*N=32).

Run from the Diffusion-Policy package root:
    cd robofactory/policy/Diffusion-Policy
    python -m pytest tests/test_joint_plumbing.py -s
or directly:
    python tests/test_joint_plumbing.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import zarr

# make diffusion_policy importable when run directly
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from diffusion_policy.dataset.robot_joint_image_dataset import RobotJointImageDataset  # noqa: E402


N_AGENTS = 4
ACTION_DIM = 8 * N_AGENTS
H, W = 240, 320


def build_synthetic_zarr(path: str, n_eps: int = 3, ep_len: int = 20):
    if os.path.exists(path):
        shutil.rmtree(path)
    root = zarr.group(path)
    data = root.create_group("data")
    meta = root.create_group("meta")

    total = n_eps * ep_len
    rng = np.random.RandomState(0)
    for i in range(N_AGENTS):
        data.create_dataset(
            f"head_camera_{i}",
            data=rng.randint(0, 255, size=(total, 3, H, W), dtype=np.uint8),
            chunks=(50, 3, H, W),
            overwrite=True,
        )
    data.create_dataset(
        "action",
        data=rng.randn(total, ACTION_DIM).astype(np.float32),
        chunks=(100, ACTION_DIM),
        overwrite=True,
    )
    data.create_dataset(
        "state",
        data=rng.randn(total, ACTION_DIM).astype(np.float32),
        chunks=(100, ACTION_DIM),
        overwrite=True,
    )
    episode_ends = np.arange(1, n_eps + 1, dtype=np.int64) * ep_len
    meta.create_dataset("episode_ends", data=episode_ends, overwrite=True)
    return path


def _build_shape_meta():
    return {
        "obs": {
            **{
                f"head_cam_{i}": {"shape": [3, H, W], "type": "rgb"}
                for i in range(N_AGENTS)
            },
            "agent_pos": {"shape": [ACTION_DIM], "type": "low_dim"},
        },
        "action": {"shape": [ACTION_DIM]},
    }


def test_dataset_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "synth_joint.zarr")
        build_synthetic_zarr(zpath, n_eps=3, ep_len=20)
        ds = RobotJointImageDataset(
            zarr_path=zpath, n_agents=N_AGENTS, horizon=8,
            pad_before=2, pad_after=7, val_ratio=0.0, batch_size=2,
        )
        assert len(ds) > 0, "dataset should have at least one sample"
        sample = ds[0]
        # shapes
        assert sample["action"].shape == (8, ACTION_DIM), sample["action"].shape
        assert sample["obs"]["agent_pos"].shape == (8, ACTION_DIM)
        for i in range(N_AGENTS):
            assert sample["obs"][f"head_cam_{i}"].shape == (8, 3, H, W)
        # dtypes
        assert sample["action"].dtype == torch.float32
        for i in range(N_AGENTS):
            assert sample["obs"][f"head_cam_{i}"].dtype == torch.float32
        # normalized image range
        for i in range(N_AGENTS):
            cam = sample["obs"][f"head_cam_{i}"]
            assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0
        print("[test] dataset shapes OK, n_samples =", len(ds))


def test_policy_forward():
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.model.vision.model_getter import get_resnet
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

    # fit a normalizer against a synthetic dataset so predict_action works end-to-end
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "synth_joint.zarr")
        build_synthetic_zarr(zpath, n_eps=3, ep_len=20)
        ds = RobotJointImageDataset(
            zarr_path=zpath, n_agents=N_AGENTS, horizon=8,
            pad_before=2, pad_after=7, val_ratio=0.0, batch_size=2,
        )
        normalizer = ds.get_normalizer()

    shape_meta = _build_shape_meta()
    scheduler = DDIMScheduler(
        num_train_timesteps=100, beta_start=1e-4, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", clip_sample=True,
        prediction_type="epsilon", set_alpha_to_one=True, steps_offset=0,
    )
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet(name="resnet18", weights=None),
        resize_shape=None, crop_shape=None, random_crop=False,
        use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
    )
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=obs_encoder,
        horizon=8, n_action_steps=8, n_obs_steps=3,
        num_inference_steps=4,  # fast for test
        obs_as_global_cond=True,
        diffusion_step_embed_dim=64,
        down_dims=[128, 256, 512],
        kernel_size=5, n_groups=8, cond_predict_scale=True,
    )
    policy.set_normalizer(normalizer)
    policy.eval()

    B = 2
    obs = {
        f"head_cam_{i}": torch.rand(B, 3, 3, H, W)  # (B, n_obs, C, H, W)
        for i in range(N_AGENTS)
    }
    obs["agent_pos"] = torch.randn(B, 3, ACTION_DIM)
    with torch.no_grad():
        out = policy.predict_action(obs)
    assert "action" in out, out.keys()
    # DiffusionUnetImagePolicy returns actions over horizon - n_obs_steps + 1 = 6 steps
    # (standard DP behaviour; same as per-arm config).
    expected_T = 8 - 3 + 1
    assert out["action"].shape == (B, expected_T, ACTION_DIM), out["action"].shape
    print("[test] policy forward OK, action shape =", tuple(out["action"].shape))


if __name__ == "__main__":
    test_dataset_shapes()
    test_policy_forward()
    print("ALL TESTS PASSED")
