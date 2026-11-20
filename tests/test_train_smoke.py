import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_train_script_runs_end_to_end_on_fake_data(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["experiment"] = "smoke"
    cfg["model"]["pretrained"] = False
    cfg["data"]["batch_size"] = 2
    cfg["data"]["image_size"] = 224
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out, ckpt = tmp_path / "out", tmp_path / "ckpt"
    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(config_path), "--fake-data",
         "--epochs", "1", "--output-dir", str(out), "--checkpoint-dir", str(ckpt)],
        cwd=ROOT, check=True,
    )

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_epoch"] == 1
    assert metrics["parameters_trainable"] == metrics["parameters_total"]
    assert (ckpt / "best.pt").exists()
    assert len((out / "log.csv").read_text(encoding="utf-8").splitlines()) == 2
