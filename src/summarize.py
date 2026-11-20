"""Tong hop ket qua nhieu seed cua mot thi nghiem va ve bieu do huan luyen."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import save_json  # noqa: E402

KEYS = ("best_val_acc", "best_val_loss", "best_epoch", "mean_epoch_time_s", "peak_gpu_memory_mb")


def read_runs(experiment_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Doc metrics.json cua tung thu muc seed*/."""
    runs = {}
    for path in sorted(Path(experiment_dir).glob("seed*/metrics.json")):
        runs[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise FileNotFoundError(f"Khong tim thay seed*/metrics.json trong {experiment_dir}")
    return runs


def summarize_runs(experiment_dir: str | Path) -> dict[str, Any]:
    """Tinh trung binh va do lech chuan qua cac seed, ghi summary.json."""
    runs = read_runs(experiment_dir)
    summary: dict[str, Any] = {"seeds": [r["seed"] for r in runs.values()], "n_runs": len(runs)}
    for key in KEYS:
        values = [r[key] for r in runs.values() if r.get(key) is not None]
        if values:
            summary[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    summary["parameters_trainable"] = next(iter(runs.values()))["parameters_trainable"]
    save_json(summary, Path(experiment_dir) / "summary.json")
    return summary


def read_log(path: str | Path) -> dict[str, list[float]]:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: [float(r[key]) for r in rows] for key in ("epoch", "train_loss", "val_loss", "train_acc", "val_acc")}


def plot_curves(experiment_dir: str | Path, out_path: str | Path) -> Path:
    """Ve loss va accuracy theo epoch cho moi seed (train: net lien, val: net dut)."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for log_path in sorted(Path(experiment_dir).glob("seed*/log.csv")):
        log, seed = read_log(log_path), log_path.parent.name
        line, = ax_loss.plot(log["epoch"], log["train_loss"], label=f"{seed} train")
        ax_loss.plot(log["epoch"], log["val_loss"], "--", color=line.get_color(), label=f"{seed} val")
        ax_acc.plot(log["epoch"], log["train_acc"], color=line.get_color(), label=f"{seed} train")
        ax_acc.plot(log["epoch"], log["val_acc"], "--", color=line.get_color(), label=f"{seed} val")
    for ax, ylabel in ((ax_loss, "Cross-entropy loss"), (ax_acc, "Accuracy")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    ax_acc.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
