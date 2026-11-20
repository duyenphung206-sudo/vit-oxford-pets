"""Seed-42 E3/E5 orchestration; preserve E4 and reject unverified result reuse."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from src.utils import load_config

CONFIGS = {
    "E3": "e3_vit_partial.yaml", "E5-LR1": "e5_lr1.yaml", "E5-LR2": "e5_lr2.yaml",
    "E5-LR3": "e5_lr3.yaml", "E5-LR4": "e5_lr4.yaml", "E5-AUG": "e5_aug.yaml",
}


def run_dir(cfg):
    return Path("results") / cfg["experiment"] / "seed42"


def verified_run(directory, expected):
    """Require real metrics, resolved recipe and consistent per-epoch evidence.

    Logs use four decimal places, hence 5.1e-5 validation comparison tolerance.
    This verifies repository evidence, not independent training reproduction.
    """
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    cfg = load_config(directory / "config.yaml")
    if cfg != expected or metrics.get("fake_data") is not False or metrics.get("seed") != 42:
        raise ValueError("Recipe/seed mismatch or synthetic result")
    if metrics.get("experiment") != expected["experiment"]:
        raise ValueError("Experiment mismatch")
    with (directory / "log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = metrics["epochs_run"]
    if not rows or len(rows) != epochs or not 1 <= epochs <= expected["train"]["epochs"]:
        raise ValueError("Incomplete epoch evidence")
    for epoch, row in enumerate(rows, 1):
        if int(row["epoch"]) != epoch:
            raise ValueError("Epoch sequence mismatch")
        if not all(math.isfinite(float(v)) for v in row.values()):
            raise ValueError("Non-finite log")
        if not all(0 <= float(row[k]) <= 1 for k in ("train_acc", "val_acc")):
            raise ValueError("Accuracy must be a fraction")
        if min(float(row[k]) for k in ("train_loss", "val_loss", "epoch_time_s")) < 0:
            raise ValueError("Negative loss/time")
    best_epoch = metrics["best_epoch"]
    if not 1 <= best_epoch <= epochs:
        raise ValueError("Invalid best epoch")
    best = rows[best_epoch - 1]
    if float(best["val_loss"]) > min(float(r["val_loss"]) for r in rows) + 1e-4:
        raise ValueError("Checkpoint was not selected by minimum validation loss")
    for key, column in (("best_val_loss", "val_loss"), ("best_val_acc", "val_acc")):
        if not math.isfinite(metrics[key]) or abs(metrics[key] - float(best[column])) > 5.1e-5:
            raise ValueError("Metrics disagree with selected epoch")
    elapsed = sum(float(r["epoch_time_s"]) for r in rows)
    if abs(metrics["mean_epoch_time_s"] * epochs - elapsed) > epochs * 0.06:
        raise ValueError("Timing disagrees with log")
    return metrics, elapsed


def run(experiment):
    import torch

    cfg = load_config(Path("configs") / CONFIGS[experiment])
    directory = run_dir(cfg)
    if (directory / "metrics.json").exists():
        verified_run(directory, cfg)
        print(f"Reuse verified evidence: {directory}")
        return
    if experiment == "E5-LR2":
        print("PENDING: supply valid E4 seed42 log.csv, config.yaml and metrics.json; E4 will not be rerun.")
        return
    checkpoint = Path("checkpoints") / cfg["experiment"] / "seed42"
    if (directory.exists() and any(directory.iterdir())) or (checkpoint.exists() and any(checkpoint.iterdir())):
        raise FileExistsError(f"Incomplete run exists: {directory}; preserve it and select a fresh runtime.")
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError(f"BLOCKED: {experiment} requires GPU; no real CPU training is allowed")
    print("GPU:", torch.cuda.get_device_name(0))
    from torchvision.datasets import OxfordIIITPet

    OxfordIIITPet(cfg["data"]["root"], split="trainval", download=True)
    subprocess.run([sys.executable, "-m", "src.train", "--config",
                    str(Path("configs") / CONFIGS[experiment]), "--seed", "42"], check=True)
    verified_run(directory, cfg)
    from src.summarize import plot_curves

    plot_curves(directory.parent, directory / "curves.png")


def summarize():
    """Write only E5 products, with missing runs explicit and empty numeric fields."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "ky-seed42"

    output = Path("results/e5_ablation")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in CONFIGS:
        if name == "E3":
            continue
        cfg = load_config(Path("configs") / CONFIGS[name])
        row = dict(id=name, learning_rate=cfg["optimizer"]["lr"], augmentation=cfg["data"]["augment"],
                   status="PENDING", val_accuracy=None, val_loss=None, training_time_s=None,
                   epochs=None, seed=42, source=run_dir(cfg).as_posix(), reason="Awaiting GPU run")
        try:
            metrics, elapsed = verified_run(run_dir(cfg), cfg)
            row.update(status="DONE", val_accuracy=metrics["best_val_acc"], val_loss=metrics["best_val_loss"],
                       training_time_s=elapsed, epochs=metrics["epochs_run"], reason="Repository evidence verified")
        except FileNotFoundError:
            if name == "E5-LR2":
                row["reason"] = "Supply E4 seed42 evidence; three-seed mean is not eligible"
        except (ValueError, KeyError, TypeError) as exc:
            row.update(status="BLOCKED", reason=str(exc))
        rows.append(row)
    (output / "summary_seed42.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output / "summary_seed42.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, selected in (("lr_comparison", rows[:4]), ("augmentation_comparison", [rows[1], rows[4]])):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, label in zip(
            axes, ("val_accuracy", "val_loss"), ("Validation accuracy", "Validation loss"), strict=True
        ):
            for index, row in enumerate(selected):
                if row["status"] == "DONE":
                    ax.scatter(index, row[metric], color="tab:blue", zorder=3)
                else:
                    ax.text(index, 0.5, row["status"], transform=ax.get_xaxis_transform(),
                            ha="center", va="center", rotation=90, color="gray")
            labels = [f"{r['learning_rate']:g}\n{r['id']}" if filename == "lr_comparison"
                      else f"{'on' if r['augmentation'] else 'off'}\n{r['id']}" for r in selected]
            ax.set_xticks(range(len(selected)), labels)
            ax.set_xlabel("Learning rate" if filename == "lr_comparison" else "Augmentation (LR = 0.003)")
            ax.set_xlim(-0.5, len(selected) - 0.5)
            ax.set_ylabel(label)
            if metric == "val_accuracy":
                ax.set_ylim(0, 1)
            else:
                values = [row[metric] for row in selected if row["status"] == "DONE"]
                ax.set_ylim(0, max(values) * 1.1 if values else 1)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Seed 42; checkpoint selected by validation loss; missing runs have no values")
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = output / f"{filename}.{extension}"
            fig.savefig(path, dpi=300)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=CONFIGS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.experiment:
        run(args.experiment)
    if args.summarize:
        summarize()
    if not args.experiment and not args.summarize:
        parser.error("Select an experiment or --summarize")


if __name__ == "__main__":
    main()
