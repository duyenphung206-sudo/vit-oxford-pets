"""Bo toi uu va lich learning rate cho fine-tuning."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from src.models.factory import vit_layer_id

NO_DECAY_NAMES = ("class_token", "pos_embedding")


def build_param_groups(model: nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Chia tham so thanh nhom: bias, norm, class token, position embedding khong weight decay;
    layer_decay < 1 (chi ViT) thi block l nhan lr * layer_decay ** (L + 1 - l)."""
    weight_decay = cfg.get("weight_decay", 0.0)
    layer_decay = cfg.get("layer_decay", 1.0)
    num_layers = len(model.encoder.layers) if hasattr(model, "encoder") else 0
    use_layer_decay = layer_decay < 1.0 and num_layers > 0

    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = param.ndim <= 1 or name.endswith(NO_DECAY_NAMES)
        layer_id = vit_layer_id(name, num_layers) if use_layer_decay else 0
        key = (layer_id, no_decay)
        if key not in groups:
            scale = layer_decay ** (num_layers + 1 - layer_id) if use_layer_decay else 1.0
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay, "lr_scale": scale}
        groups[key]["params"].append(param)
    return list(groups.values())


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """SGD momentum (mac dinh, theo bai bao ViT) hoac AdamW."""
    lr = cfg["lr"]
    groups = build_param_groups(model, cfg)
    for group in groups:
        group["lr"] = lr * group["lr_scale"]

    name = cfg.get("name", "sgd").lower()
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=cfg.get("momentum", 0.9))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.0
) -> LambdaLR:
    """Warm-up tuyen tinh roi giam theo cosine, cap nhat sau moi buoc toi uu."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
