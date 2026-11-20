"""Tao backbone pretrained ImageNet voi tang phan lop moi (slide 07).

Chien luoc hoc chuyen giao:
    linear_probe  dong bang toan bo backbone, chi huan luyen head (feature extraction)
    partial       mo bang them k block cuoi va LayerNorm cuoi
    full          huan luyen toan bo tham so (fine-tuning toan bo)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

STRATEGIES = ("linear_probe", "partial", "full")


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Tao mo hinh tu muc ``model`` cua file config."""
    name = cfg["name"]
    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", True)
    strategy = cfg.get("strategy", "full")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}', expected one of {STRATEGIES}")

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = zero_init_linear(model.heads.head.in_features, num_classes)
        head, blocks, post_blocks = model.heads, list(model.encoder.layers), [model.encoder.ln]
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head, blocks, post_blocks = model.fc, [model.layer1, model.layer2, model.layer3, model.layer4], []
    else:
        raise ValueError(f"Unknown model '{name}'")

    apply_strategy(model, head, blocks, post_blocks, strategy, cfg.get("unfreeze_last_k", 0))
    return model


def apply_strategy(
    model: nn.Module,
    head: nn.Module,
    blocks: list[nn.Module],
    post_blocks: list[nn.Module],
    strategy: str,
    unfreeze_last_k: int = 0,
) -> None:
    """Dat requires_grad theo chien luoc; head luon duoc huan luyen."""
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False
    trainable = [head]
    if strategy == "partial":
        if not 0 < unfreeze_last_k <= len(blocks):
            raise ValueError(f"unfreeze_last_k must be in [1, {len(blocks)}]")
        trainable += blocks[-unfreeze_last_k:] + post_blocks
    for module in trainable:
        for param in module.parameters():
            param.requires_grad = True


def zero_init_linear(in_features: int, out_features: int) -> nn.Linear:
    """Head khoi tao bang 0 khi fine-tune ViT (Dosovitskiy va cs., 2021, muc 3.2)."""
    head = nn.Linear(in_features, out_features)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def vit_layer_id(param_name: str, num_layers: int) -> int:
    """Do sau cua tham so ViT: 0 = embedding, 1..L = encoder block, L + 1 = LayerNorm cuoi va head."""
    if param_name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
        return 0
    if param_name.startswith("encoder.layers.encoder_layer_"):
        return int(param_name.split(".")[2].rsplit("_", 1)[1]) + 1
    return num_layers + 1
