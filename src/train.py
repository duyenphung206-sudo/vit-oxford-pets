"""Fine-tune mo hinh pretrained tren Oxford-IIIT Pet.

Cach chay:
    python -m src.train --config configs/e4_vit_full_ft.yaml --seed 42
    python -m src.train --config configs/e4_vit_full_ft.yaml --fake-data --epochs 1

Chon checkpoint theo val_loss nho nhat (slide 04). Tap test khong dung o day,
chi danh gia mot lan sau khi huan luyen bang src/evaluate.py.
"""

from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.engine import evaluate, train_one_epoch
from src.models.factory import build_model
from src.optim import build_optimizer, build_scheduler
from src.utils import CSVLogger, count_parameters, get_device, load_config, save_json, set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOG_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def prepare_run(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    """Doc config, ghi de seed/epochs tu dong lenh, tao thu muc ket qua."""
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    run = f"{cfg['experiment']}/seed{cfg['train']['seed']}"
    output_dir = Path(args.output_dir or f"results/{run}")
    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/{run}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg, output_dir, checkpoint_dir


def build_fake_dataloaders(cfg: dict[str, Any], generator: torch.Generator) -> dict[str, DataLoader]:
    """Anh ngau nhien de chay thu pipeline (smoke test), khong dung de bao cao."""
    size = cfg["data"].get("image_size", 224)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    num_classes, batch_size = cfg["model"]["num_classes"], cfg["data"]["batch_size"]
    train_set = datasets.FakeData(4 * batch_size, (3, size, size), num_classes, tf, random_offset=0)
    val_set = datasets.FakeData(2 * batch_size, (3, size, size), num_classes, tf, random_offset=1000)
    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
    }


def build_real_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    """Loader tu src/data.py (T12-T13): split 80/20 da luu, test chinh thuc."""
    from src.data import build_dataloaders

    data = cfg["data"]
    return build_dataloaders(
        data_dir=data["root"],
        split_dir=data["splits_dir"],
        batch_size=data["batch_size"],
        num_workers=data.get("num_workers", 0),
        seed=cfg["train"]["seed"],
        pin_memory=torch.cuda.is_available(),
        augment=data.get("augment", True),
    )


def build_training(cfg: dict[str, Any], model: nn.Module, steps_per_epoch: int, device: torch.device) -> dict:
    """Tao loss, optimizer, scheduler (warm-up + cosine) va GradScaler cho AMP."""
    train_cfg = cfg["train"]
    optimizer = build_optimizer(model, cfg["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        total_steps=train_cfg["epochs"] * steps_per_epoch,
        warmup_steps=math.ceil(train_cfg.get("warmup_epochs", 0) * steps_per_epoch),
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.0),
    )
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    return {
        "loss_fn": nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0)),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": torch.amp.GradScaler(device.type, enabled=use_amp),
        "max_grad_norm": train_cfg.get("max_grad_norm"),
    }


def fit(model, loaders, parts, cfg, device, output_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Huan luyen tung epoch, luu best.pt theo val_loss, dung som sau ``patience`` epoch khong cai thien."""
    epochs, patience = cfg["train"]["epochs"], cfg["train"].get("patience", cfg["train"]["epochs"])
    logger = CSVLogger(output_dir / "log.csv", LOG_FIELDS)
    best = {"loss": math.inf, "epoch": 0, "acc": 0.0}
    epoch_times = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        lr = max(group["lr"] for group in parts["optimizer"].param_groups)
        train_m = train_one_epoch(model, loaders["train"], parts["loss_fn"], parts["optimizer"], device,
                                  parts["scheduler"], parts["scaler"], parts["max_grad_norm"])
        val_m = evaluate(model, loaders["val"], parts["loss_fn"], device)
        epoch_times.append(time.perf_counter() - start)
        logger.log({"epoch": epoch, "lr": f"{lr:.6g}", "train_loss": f"{train_m['loss']:.4f}",
                    "train_acc": f"{train_m['acc']:.4f}", "val_loss": f"{val_m['loss']:.4f}",
                    "val_acc": f"{val_m['acc']:.4f}", "epoch_time_s": f"{epoch_times[-1]:.1f}"})
        print(f"epoch {epoch:>3}/{epochs} | train {train_m['loss']:.4f} / {train_m['acc']:.4f} | "
              f"val {val_m['loss']:.4f} / {val_m['acc']:.4f} | {epoch_times[-1]:.0f}s")
        if val_m["loss"] < best["loss"]:
            best = {"loss": val_m["loss"], "epoch": epoch, "acc": val_m["acc"]}
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        elif epoch - best["epoch"] >= patience:
            break
    if best["epoch"] == 0:
        raise RuntimeError("Validation loss khong huu han; kiem tra learning rate va du lieu.")
    return {**best, "epochs_run": epoch, "mean_epoch_time_s": sum(epoch_times) / len(epoch_times)}


def write_metrics(cfg, args, best, n_params, device, output_dir: Path, checkpoint_dir: Path) -> None:
    cuda = device.type == "cuda"
    save_json({
        "experiment": cfg["experiment"],
        "seed": cfg["train"]["seed"],
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["loss"], 6),
        "best_val_acc": round(best["acc"], 6),
        "epochs_run": best["epochs_run"],
        "parameters_total": n_params["total"],
        "parameters_trainable": n_params["trainable"],
        "mean_epoch_time_s": round(best["mean_epoch_time_s"], 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "device": torch.cuda.get_device_name() if cuda else "cpu",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": platform.python_version(),
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "fake_data": args.fake_data,
    }, output_dir / "metrics.json")


def main() -> None:
    args = parse_args()
    cfg, output_dir, checkpoint_dir = prepare_run(args)
    generator = set_seed(cfg["train"]["seed"], cfg["train"].get("deterministic", False))
    device = get_device()
    loaders = build_fake_dataloaders(cfg, generator) if args.fake_data else build_real_dataloaders(cfg)

    model = build_model(cfg["model"]).to(device)
    n_params = count_parameters(model)
    parts = build_training(cfg, model, len(loaders["train"]), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"{cfg['experiment']} | seed {cfg['train']['seed']} | {device} | "
          f"trainable {n_params['trainable']:,} / {n_params['total']:,}")

    best = fit(model, loaders, parts, cfg, device, output_dir, checkpoint_dir)
    write_metrics(cfg, args, best, n_params, device, output_dir, checkpoint_dir)
    print(f"best epoch {best['epoch']} | val loss {best['loss']:.4f} | val acc {best['acc']:.4f}")


if __name__ == "__main__":
    main()
