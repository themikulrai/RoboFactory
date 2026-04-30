"""Sanity test for the two new DINOv2 encoder factories.

Verifies:
  1. get_dinov2_lora returns a model that takes (B,3,224,224) and returns (B, 768).
  2. get_dinov2_patchattn returns a model that takes (B,3,224,224) and returns (B, output_dim).
  3. Both are deepcopy-able (MultiImageObsEncoder calls copy.deepcopy).
  4. LoRA model has only LoRA params trainable.
  5. Patch-attn model has backbone frozen, head trainable.
"""
import sys
import copy
sys.path.insert(0, '/iris/u/mikulrai/projects/RoboFactory/robofactory/policy/Diffusion-Policy')

import torch
from diffusion_policy.model.vision.model_getter import (
    get_dinov2_lora, get_dinov2_patchattn,
)


def count_params(m, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())


def test_lora():
    print("\n=== get_dinov2_lora (ViT-B/14) ===")
    m = get_dinov2_lora(name="vit_base_patch14_dinov2", lora_rank=16, img_size=224, weights=None)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f"  input  : {tuple(x.shape)}")
    print(f"  output : {tuple(y.shape)}  (expect (2, 768))")
    assert y.shape == (2, 768), f"unexpected shape {y.shape}"
    total = count_params(m)
    train = count_params(m, only_trainable=True)
    print(f"  total params      : {total/1e6:.2f}M")
    print(f"  trainable (LoRA)  : {train/1e6:.4f}M  ({100*train/total:.2f}% of total)")
    assert train < 0.1 * total, "too many trainable params for LoRA"
    # deepcopy
    m2 = copy.deepcopy(m)
    y2 = m2(x)
    assert y2.shape == (2, 768)
    print("  deepcopy: OK")


def test_patchattn():
    print("\n=== get_dinov2_patchattn (ViT-S/14) ===")
    m = get_dinov2_patchattn(
        name="vit_small_patch14_dinov2", weights=None,
        num_latents=64, n_heads=6, depth=2, output_dim=256, img_size=224,
    )
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f"  input  : {tuple(x.shape)}")
    print(f"  output : {tuple(y.shape)}  (expect (2, 256))")
    assert y.shape == (2, 256), f"unexpected shape {y.shape}"
    total = count_params(m)
    train = count_params(m, only_trainable=True)
    print(f"  total params       : {total/1e6:.2f}M")
    print(f"  trainable (head)   : {train/1e6:.4f}M  ({100*train/total:.2f}% of total)")
    # backbone should be frozen
    backbone_train = sum(p.numel() for p in m.backbone.parameters() if p.requires_grad)
    assert backbone_train == 0, "backbone is not frozen"
    # gradient flows through head
    y.sum().backward()
    assert m.latents.grad is not None and m.latents.grad.abs().sum() > 0, "no grad on latents"
    assert m.out_proj.weight.grad is not None, "no grad on out_proj"
    print("  backbone frozen, head trainable: OK")
    # deepcopy
    m2 = copy.deepcopy(m)
    y2 = m2(torch.randn(2, 3, 224, 224))
    assert y2.shape == (2, 256)
    print("  deepcopy: OK")


if __name__ == "__main__":
    test_lora()
    test_patchattn()
    print("\nAll factory tests passed.")
