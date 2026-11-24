"""E2 execution helper by Nguyen Ngoc Ky supporting Trinh; uses unchanged src.train.

Colab-only, GPU-only. Epoch logs and best checkpoint are written directly to Drive.
Incomplete earlier runs are preserved; they are never resumed from weights alone.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

from src.summarize import plot_curves
from src.utils import load_config


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED: E2 requires free Colab GPU, no CPU training")
    root = Path("/content/vit-oxford-pets-e2-support")
    if Path.cwd() != root or subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() == "main":
        raise RuntimeError("Use isolated E2 support checkout, not main")
    store = Path("/content/drive/MyDrive/vit-oxford-pets-ky/e2-support")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount authorized Drive before training")
    run = store / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    results = run / "results/e2_vit_linear_probe/seed42"
    checkpoints = run / "checkpoints/e2_vit_linear_probe/seed42"
    cfg = load_config("configs/e2_vit_linear_probe.yaml")
    assert cfg["model"]["strategy"] == "linear_probe" and cfg["optimizer"]["lr"] == 0.003
    assert cfg["data"]["batch_size"] == 64 and cfg["train"]["seed"] == 42 and cfg["train"]["epochs"] == 10
    metadata = {
        "author": "Nguyen Ngoc Ky, supporting Trinh; not attributed as Trinh implementation",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(0),
        "training_engine": "unchanged src.train",
        "run": str(run),
        "status": "RUNNING",
        "test_evaluation": "NOT_RUN; validation only",
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path("configs/e2_vit_linear_probe.yaml").read_bytes()).hexdigest(),
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("E2_PERSISTENT_RUN", str(run), flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            "configs/e2_vit_linear_probe.yaml",
            "--seed",
            "42",
            "--output-dir",
            str(results),
            "--checkpoint-dir",
            str(checkpoints),
        ],
        check=True,
    )
    metrics = json.loads((results / "metrics.json").read_text())
    rows = list(csv.DictReader((results / "log.csv").open()))
    assert metrics["fake_data"] is False and metrics["parameters_trainable"] == 28453
    assert len(rows) == metrics["epochs_run"] and metrics["seed"] == 42
    selected = rows[metrics["best_epoch"] - 1]
    assert float(selected["val_loss"]) <= min(float(r["val_loss"]) for r in rows) + 1e-4
    assert abs(float(selected["val_acc"]) - metrics["best_val_acc"]) < 5.1e-5
    assert abs(float(selected["val_loss"]) - metrics["best_val_loss"]) < 5.1e-5
    digest = hashlib.sha256((checkpoints / "best.pt").read_bytes()).hexdigest()
    metadata.update(
        status="DONE",
        checkpoint_sha256=digest,
        checkpoint_bytes=(checkpoints / "best.pt").stat().st_size,
        epochs_run=len(rows),
        elapsed_epoch_time_s=sum(float(r["epoch_time_s"]) for r in rows),
        gpu_memory_unit="MiB (trainer field is named peak_gpu_memory_mb)",
    )
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    figure = run / "e2_curves.png"
    plot_curves(results.parent, figure)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["config.yaml", "log.csv", "metrics.json"]:
            bundle.write(results / name, "results/e2_vit_linear_probe/seed42/" + name)
        bundle.write(run / "run_metadata.json", "docs/KY_E2_RUN_MANIFEST.json")
    print("E2_METRICS", json.dumps(metrics), flush=True)
    print("E2_MANIFEST", json.dumps(metadata), flush=True)
    print("E2_TEXT_ZIP_SHA256", hashlib.sha256(payload.getvalue()).hexdigest(), flush=True)
    print("E2_TEXT_ZIP_BASE64", base64.b64encode(payload.getvalue()).decode(), flush=True)
    print("E2_DONE", flush=True)


if __name__ == "__main__":
    main()
