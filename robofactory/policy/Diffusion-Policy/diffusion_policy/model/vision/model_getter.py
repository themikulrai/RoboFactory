import torch
import torch.nn as nn
import torchvision


def get_dinov2_lora(name="vit_base_patch14_dinov2", weights=None,
                    lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                    img_size=224, **kwargs):
    """DINOv2 ViT (timm) with frozen backbone + LoRA on attention projections.

    `weights` is accepted but ignored (DINOv2 is always pretrained).
    forward(x:[B,3,H,W]) -> CLS token [B, D].  Trainable params: only LoRA.
    """
    import timm
    from peft import LoraConfig, get_peft_model
    backbone = timm.create_model(
        name, pretrained=True, num_classes=0, img_size=img_size,
    )
    for p in backbone.parameters():
        p.requires_grad = False
    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    return get_peft_model(backbone, cfg)


class DINOv2PatchAttn(nn.Module):
    """Frozen DINOv2 backbone + Perceiver-style cross-attention over patch tokens.

    Reads spatial patch tokens (skipping CLS) and compresses them into N learnable
    latents via stacked cross-attention. Final linear projects flattened latents to
    output_dim. Trainable params: latents + cross-attn + projection (~1-3M).

    forward(x:[B,3,H,W]) -> [B, output_dim].
    """
    def __init__(self, name="vit_small_patch14_dinov2", weights=None,
                 num_latents=64, n_heads=6, depth=2, output_dim=256,
                 img_size=224, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            name, pretrained=True, num_classes=0, img_size=img_size,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        D = self.backbone.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, D) * 0.02)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(D, n_heads, batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        self.norms_q = nn.ModuleList([nn.LayerNorm(D) for _ in range(depth)])
        self.norms_kv = nn.ModuleList([nn.LayerNorm(D) for _ in range(depth)])
        self.out_proj = nn.Linear(num_latents * D, output_dim)
        self._output_dim = output_dim
        self.num_prefix_tokens = self.backbone.num_prefix_tokens  # 1 (CLS) for non-reg

    def forward(self, x):
        with torch.no_grad():
            tokens = self.backbone.forward_features(x)  # (B, prefix+N, D)
        patches = tokens[:, self.num_prefix_tokens:, :]  # (B, N, D)
        B = patches.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        for attn, nq, nkv in zip(self.attn_layers, self.norms_q, self.norms_kv):
            q_norm = nq(q)
            kv_norm = nkv(patches)
            attn_out, _ = attn(q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + attn_out
        flat = q.reshape(B, -1)
        return self.out_proj(flat)


def get_dinov2_patchattn(**kwargs):
    """Factory wrapper so Hydra can instantiate DINOv2PatchAttn via _target_."""
    return DINOv2PatchAttn(**kwargs)


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    # resnet_new = torch.nn.Sequential(
    #     resnet,
    #     torch.nn.Linear(512, 128)
    # )
    # return resnet_new
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
