import json

import pytest

from src.summarize import plot_curves, summarize_runs

LOG = (
    "epoch,lr,train_loss,train_acc,val_loss,val_acc,epoch_time_s\n"
    "1,0.003,1.2,0.6,0.9,0.7,30\n"
    "2,0.002,0.6,0.8,0.5,0.85,30\n"
)


def write_run(root, seed, acc):
    run = root / f"seed{seed}"
    run.mkdir(parents=True)
    (run / "log.csv").write_text(LOG, encoding="utf-8")
    metrics = {"seed": seed, "best_val_acc": acc, "best_val_loss": 0.5, "best_epoch": 2,
               "mean_epoch_time_s": 30.0, "peak_gpu_memory_mb": None, "parameters_trainable": 10}
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_summary_mean_and_std_over_seeds(tmp_path):
    for seed, acc in ((42, 0.90), (43, 0.92), (44, 0.94)):
        write_run(tmp_path, seed, acc)
    summary = summarize_runs(tmp_path)
    assert summary["n_runs"] == 3
    assert summary["best_val_acc"]["mean"] == pytest.approx(0.92)
    assert summary["best_val_acc"]["std"] == pytest.approx(0.02)
    assert "peak_gpu_memory_mb" not in summary
    assert (tmp_path / "summary.json").exists()


def test_plot_curves_writes_png(tmp_path):
    write_run(tmp_path, 42, 0.9)
    out = plot_curves(tmp_path, tmp_path / "curves.png")
    assert out.exists() and out.stat().st_size > 0
