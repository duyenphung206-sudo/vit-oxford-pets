import ast
import csv
import json
from pathlib import Path

import pytest

from src.ky_experiments import CONFIGS, run, verified_run
from src.models.factory import build_model
from src.utils import count_parameters, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_recipes_change_only_requested_conditions():
    base = load_config(ROOT / "configs/e4_vit_full_ft.yaml")
    for name, filename in CONFIGS.items():
        cfg = load_config(ROOT / "configs" / filename)
        expected = load_config(ROOT / "configs/e4_vit_full_ft.yaml")
        if name == "E3":
            expected["experiment"] = "e3_vit_partial"
            expected["model"].update(strategy="partial", unfreeze_last_k=4)
        elif name != "E5-LR2":
            expected["experiment"] = "e5_" + name.split("-")[1].lower()
            expected["optimizer"]["lr"] = {"E5-LR1": 0.001, "E5-LR3": 0.01,
                                              "E5-LR4": 0.03, "E5-AUG": 0.003}[name]
            expected["data"]["augment"] = name != "E5-AUG"
        assert cfg == expected
        assert cfg["train"]["seed"] == base["train"]["seed"] == 42


def test_e3_exact_trainable_parameter_set():
    cfg = load_config(ROOT / "configs/e3_vit_partial.yaml")
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"])
    for name, parameter in model.named_parameters():
        expected = name.startswith(("heads.", "encoder.ln.")) or any(
            name.startswith(f"encoder.layers.encoder_layer_{i}.") for i in range(8, 12))
        assert parameter.requires_grad == expected, name
    assert count_parameters(model) == {"total": 85_827_109, "trainable": 28_381_477}


def test_e4_seed42_evidence_is_consistent():
    cfg = load_config(ROOT / "configs/e5_lr2.yaml")
    metrics, seconds = verified_run(ROOT / "results/e4_vit_full_ft/seed42", cfg)
    assert metrics["seed"] == 42 and seconds > 0


def test_reject_synthetic_or_inconsistent_reuse(tmp_path):
    import shutil

    source = ROOT / "results/e4_vit_full_ft/seed42"
    for filename in ("config.yaml", "log.csv", "metrics.json"):
        shutil.copyfile(source / filename, tmp_path / filename)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["fake_data"] = True
    metrics_path.write_text(json.dumps(metrics))
    cfg = load_config(ROOT / "configs/e5_lr2.yaml")
    with pytest.raises(ValueError, match="synthetic"):
        verified_run(tmp_path, cfg)
    metrics["fake_data"] = False
    metrics["best_val_acc"] = 0.5
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="disagree"):
        verified_run(tmp_path, cfg)


def test_no_real_training_without_cuda(tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("src.ky_experiments.run_dir", lambda cfg: tmp_path / "absent")
    with pytest.raises(RuntimeError, match="requires GPU"):
        run("E3")


def test_persisted_splits_unchanged_and_disjoint():
    groups = []
    for split, size in (("train", 2944), ("val", 736)):
        with (ROOT / f"data/splits/{split}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == size
        assert set(int(row["label"]) for row in rows) == set(range(37))
        groups.append({row["id"] for row in rows})
    assert len(groups[0]) == 2944 and len(groups[1]) == 736
    assert not groups[0] & groups[1]


@pytest.mark.parametrize("filename", ["07_train_e3_vit_partial.ipynb", "08_train_e5_ablation.ipynb"])
def test_colab_notebooks_are_valid_python(filename):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["gpuType"] == "T4"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not any(output["output_type"] == "error" for output in cell["outputs"])
