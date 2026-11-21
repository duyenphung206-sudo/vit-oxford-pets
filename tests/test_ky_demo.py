"""Guard label ordering and reject unsuitable demo inputs/checkpoints."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ky_demo", ROOT / "app/app.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_official_label_order():
    classes = demo.official_class_names()
    assert len(classes) == len(set(classes)) == 37
    assert classes[0] == "Abyssinian"
    assert classes[5] == "Bengal"
    assert classes[36] == "yorkshire_terrier"


@pytest.mark.parametrize("checkpoint, message", [
    ({"smoke_only": True}, "smoke-only"),
    ({"architecture": "vit_b_16", "classes": list(reversed(demo.official_class_names())),
      "model": {}}, "class order"),
    ({"unrelated": "metadata"}, "Unsupported checkpoint"),
])
def test_reject_wrong_demo_checkpoint(tmp_path, checkpoint, message):
    path = tmp_path / "rejected.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message):
        demo.Predictor(path)


def test_invalid_image_is_rejected_before_inference():
    predictor = demo.Predictor.__new__(demo.Predictor)
    with pytest.raises(ValueError, match="valid image"):
        predictor(None)
