"""Vong lap train va validation (slide 04, trang 108-110)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        backward_step(loss, model, optimizer, scaler if use_amp else None, max_grad_norm)
        if scheduler is not None:
            scheduler.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += n

    return {"loss": total_loss / count, "acc": correct / count}


def backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    max_grad_norm: float | None,
) -> None:
    """Lan truyen nguoc, cat gradient theo chuan L2 (neu co) roi cap nhat tham so."""
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    if max_grad_norm:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)

        n = labels.size(0)
        total_loss += loss_fn(logits, labels).item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += n

    return {"loss": total_loss / count, "acc": correct / count}
