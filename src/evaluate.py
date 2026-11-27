"""Danh gia mo hinh da huan luyen tren tap test/val (accuracy top-1/top-5, macro P/R/F1, confusion matrix 37x37).

Cach chay:
    python -m src.evaluate --config configs/e4_vit_full_ft.yaml --checkpoint checkpoints/e4_vit_full_ft/seed44/best.pt
    python -m src.evaluate --config configs/e4_vit_full_ft.yaml --fake-data
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.models.factory import build_model  # noqa: E402
from src.utils import get_device, load_config, save_json  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Duong dan file config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Duong dan checkpoint best.pt.")
    parser.add_argument("--split", default="test", choices=("val", "test"), help="Tap du lieu danh gia.")
    parser.add_argument("--output-dir", default=None, help="Thu muc luu JSON, CSV va hinh anh.")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi de batch_size trong config.")
    parser.add_argument("--device", default=None, help="Thiet bi suy luan (vd: cuda, cpu).")
    parser.add_argument("--fake-data", action="store_true", help="Chay thu voi du lieu gia lap (smoke test).")
    return parser.parse_args()


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Nap trong so tu file checkpoint (ho tro ca raw state_dict lan dict co khoa 'model')."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)


def build_eval_loader_and_classes(
    cfg: dict[str, Any], split: str, fake_data: bool, batch_size: int
) -> tuple[DataLoader[Any], list[str]]:
    """Tao DataLoader danh gia va danh sach 37 ten lop."""
    num_classes = int(cfg["model"]["num_classes"])
    if fake_data:
        from torchvision import datasets, transforms

        size = int(cfg["data"].get("image_size", 224))
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        dataset = datasets.FakeData(4 * batch_size, (3, size, size), num_classes, tf, random_offset=2026)
        loader: DataLoader[Any] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        classes = [f"class_{i:02d}" for i in range(num_classes)]
        return loader, classes

    from src.data import build_dataloaders, class_names, read_annotations

    data_cfg = cfg["data"]
    loaders = build_dataloaders(
        data_dir=data_cfg["root"],
        split_dir=data_cfg["splits_dir"],
        batch_size=batch_size,
        num_workers=int(data_cfg.get("num_workers", 0)),
        seed=int(cfg["train"]["seed"]),
        pin_memory=torch.cuda.is_available(),
        augment=False,
    )
    classes = class_names(read_annotations(data_cfg["root"], "trainval"))
    return loaders[split], classes


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    """Chay suy luyen tren toan bo loader, tra ve logits (N, C), nhan that (N,) va cross-entropy loss."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)

        n = int(labels.size(0))
        total_loss += float(loss.item()) * n
        total_samples += n
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    if total_samples == 0:
        raise ValueError("DataLoader rong, khong co mau de danh gia.")
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0), total_loss / total_samples


def compute_metrics(logits: np.ndarray, targets: np.ndarray, class_labels: list[str]) -> dict[str, Any]:
    """Tinh top-1/top-5 accuracy, macro P/R/F1, classification report va confusion matrix."""
    num_classes = len(class_labels)
    preds = np.argmax(logits, axis=1)
    top1_acc = float(np.mean(preds == targets))

    k = min(5, num_classes)
    topk_preds = np.argsort(logits, axis=1)[:, -k:]
    top5_acc = float(np.mean(np.any(topk_preds == targets[:, None], axis=1)))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0
    )

    label_indices = list(range(num_classes))
    report = classification_report(
        targets,
        preds,
        labels=label_indices,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    assert isinstance(report, dict)
    cm = confusion_matrix(targets, preds, labels=label_indices)

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    flat_order = np.argsort(off_diag.ravel())[::-1]
    top_confused: list[dict[str, Any]] = []
    for idx in flat_order[:15]:
        true_idx, pred_idx = divmod(int(idx), num_classes)
        count = int(off_diag[true_idx, pred_idx])
        if count == 0:
            break
        top_confused.append({
            "true_class": class_labels[true_idx],
            "pred_class": class_labels[pred_idx],
            "count": count,
        })

    return {
        "num_samples": int(len(targets)),
        "num_classes": num_classes,
        "top1_acc": round(top1_acc, 6),
        "top5_acc": round(top5_acc, 6),
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_p), 6),
        "weighted_recall": round(float(weighted_r), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "top_confused_pairs": top_confused,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_classification_report_csv(report: dict[str, Any], class_labels: list[str], out_path: Path) -> None:
    """Ghi bang classification report theo tung giong va dong tong hop ra CSV."""
    fieldnames = ["class_id", "breed", "precision", "recall", "f1_score", "support"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, name in enumerate(class_labels):
            row = report[name]
            writer.writerow({
                "class_id": idx,
                "breed": name,
                "precision": f"{row['precision']:.6f}",
                "recall": f"{row['recall']:.6f}",
                "f1_score": f"{row['f1-score']:.6f}",
                "support": int(row["support"]),
            })
        for avg_key in ("macro avg", "weighted avg"):
            if avg_key in report:
                row = report[avg_key]
                writer.writerow({
                    "class_id": "-",
                    "breed": avg_key,
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1_score": f"{row['f1-score']:.6f}",
                    "support": int(row["support"]),
                })


def save_confusion_matrix_csv(cm: np.ndarray, class_labels: list[str], out_path: Path) -> None:
    """Ghi ma tran nham lan NxN ra file CSV co ten hang (true) va cot (pred)."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true / pred", *class_labels])
        for name, row in zip(class_labels, cm.tolist(), strict=True):
            writer.writerow([name, *row])


def plot_confusion_matrix(
    cm: np.ndarray, class_labels: list[str], out_path: Path, title: str
) -> None:
    """Ve heatmap ma tran nham lan 37x37 va luu ra file anh PNG."""
    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("So luong mau", fontsize=10)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_xlabel("Nhan du doan (Predicted)", fontsize=11)
    ax.set_ylabel("Nhan thuc te (Ground Truth)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)

    thresh = float(cm.max()) / 2.0 if cm.size else 0.0
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            if val > 0:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val > thresh else "black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_checkpoint(
    cfg: dict[str, Any],
    checkpoint_path: str | Path | None,
    split: str = "test",
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    device_str: str | None = None,
    fake_data: bool = False,
) -> dict[str, Any]:
    """Chay quy trinh danh gia day du va xuat JSON, CSV, PNG."""
    if not fake_data and checkpoint_path is None:
        raise ValueError("Can truyen --checkpoint khi danh gia tren du lieu that.")

    device = torch.device(device_str) if device_str else get_device()
    eval_batch_size = batch_size if batch_size is not None else int(cfg["data"]["batch_size"])

    if fake_data:
        cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_model(cfg["model"]).to(device)
    if checkpoint_path is not None:
        load_checkpoint_weights(model, checkpoint_path)

    loader, class_labels = build_eval_loader_and_classes(cfg, split, fake_data, eval_batch_size)
    logits, targets, loss = collect_predictions(model, loader, device)
    metrics = compute_metrics(logits, targets, class_labels)
    metrics["experiment"] = cfg["experiment"]
    metrics["split"] = split
    metrics["loss"] = round(loss, 6)
    metrics["checkpoint"] = str(checkpoint_path) if checkpoint_path is not None else None
    metrics["fake_data"] = fake_data

    out_dir = Path(output_dir or f"results/{cfg['experiment']}/eval_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics, out_dir / "eval_metrics.json")
    save_classification_report_csv(
        metrics["classification_report"],
        class_labels,
        out_dir / "classification_report.csv",
    )
    cm_arr = np.array(metrics["confusion_matrix"], dtype=int)
    save_confusion_matrix_csv(cm_arr, class_labels, out_dir / "confusion_matrix.csv")

    title = (
        f"Confusion Matrix ({cfg['experiment']} - {split}) | "
        f"Top-1: {metrics['top1_acc'] * 100:.2f}% | "
        f"Top-5: {metrics['top5_acc'] * 100:.2f}% | "
        f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%"
    )
    plot_confusion_matrix(cm_arr, class_labels, out_dir / "confusion_matrix.png", title)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    metrics = evaluate_checkpoint(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        fake_data=args.fake_data,
    )
    print(
        f"[{metrics['experiment']} | {metrics['split']}] "
        f"loss={metrics['loss']:.4f} | "
        f"top1={metrics['top1_acc'] * 100:.2f}% | "
        f"top5={metrics['top5_acc'] * 100:.2f}% | "
        f"macro_P={metrics['macro_precision'] * 100:.2f}% | "
        f"macro_R={metrics['macro_recall'] * 100:.2f}% | "
        f"macro_F1={metrics['macro_f1'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
