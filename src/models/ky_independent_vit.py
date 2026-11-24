"""Independent ViT architecture, implemented by Nguyen Ngoc Ky to support Trinh.

Uses PyTorch primitives, not torchvision ViT internally. Architecture reference:
Dosovitskiy et al. https://arxiv.org/abs/2010.11929 ; torchvision source:
https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html
No replacement of the group's factory or training models. Weight mapping is explicit.
"""

from __future__ import annotations

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) / (dim // self.heads) ** 0.5).softmax(-1)
        return self.projection((attention @ v).transpose(1, 2).reshape(batch, tokens, dim))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class IndependentViT(nn.Module):
    def __init__(self, num_classes=37, image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Image size must be divisible by patch size")
        self.image_size = image_size
        self.patch_embedding = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, dim))
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("Expected NCHW RGB images at configured image size")
        tokens = self.patch_embedding(image).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.class_token.expand(image.shape[0], -1, -1), tokens], dim=1)
        tokens = tokens + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens)[:, 0])

    def load_torchvision_state(self, state):
        """Map all learned tensors; reject unconsumed or missing source weights."""
        mapping = {
            "patch_embedding.weight": "conv_proj.weight",
            "patch_embedding.bias": "conv_proj.bias",
            "class_token": "class_token",
            "position_embedding": "encoder.pos_embedding",
            "norm.weight": "encoder.ln.weight",
            "norm.bias": "encoder.ln.bias",
            "head.weight": "heads.head.weight",
            "head.bias": "heads.head.bias",
        }
        for index in range(len(self.blocks)):
            own, tv = f"blocks.{index}.", f"encoder.layers.encoder_layer_{index}."
            for target, source in [
                ("norm1", "ln_1"),
                ("norm2", "ln_2"),
                ("attention.qkv", "self_attention.in_proj"),
                ("attention.projection", "self_attention.out_proj"),
                ("mlp.0", "mlp.0"),
                ("mlp.2", "mlp.3"),
            ]:
                for field in ["weight", "bias"]:
                    source_name = source + ("_" if source.endswith("in_proj") else ".") + field
                    mapping[own + target + "." + field] = tv + source_name
        if set(state) != set(mapping.values()):
            raise ValueError("Source checkpoint keys differ from expected torchvision architecture")
        return self.load_state_dict({target: state[source] for target, source in mapping.items()}, strict=True)
