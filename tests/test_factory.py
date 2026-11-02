import pytest
import torch
from torchvision import models

from src.models.factory import build_model, vit_layer_id
from src.utils import count_parameters

NUM_CLASSES = 37
VIT_B16_IMAGENET_PARAMS = 86_567_656
HEAD_PARAMS = 768 * NUM_CLASSES + NUM_CLASSES


def vit(strategy: str, k: int = 0) -> torch.nn.Module:
    cfg = {"name": "vit_b_16", "num_classes": NUM_CLASSES, "pretrained": False,
           "strategy": strategy, "unfreeze_last_k": k}
    return build_model(cfg)


def test_torchvision_vit_b16_matches_documented_size():
    assert count_parameters(models.vit_b_16(weights=None))["total"] == VIT_B16_IMAGENET_PARAMS


def test_full_finetuning_trains_every_parameter():
    params = count_parameters(vit("full"))
    expected = VIT_B16_IMAGENET_PARAMS - (768 * 1000 + 1000) + HEAD_PARAMS
    assert params["total"] == params["trainable"] == expected


def test_head_is_zero_initialised_and_outputs_37_logits():
    model = vit("full").eval()
    assert torch.count_nonzero(model.heads.head.weight) == 0
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, NUM_CLASSES)


def test_linear_probe_trains_only_the_head():
    assert count_parameters(vit("linear_probe"))["trainable"] == HEAD_PARAMS


def test_partial_trains_last_blocks_final_norm_and_head():
    model = vit("partial", k=4)
    block = sum(p.numel() for p in model.encoder.layers[-1].parameters())
    final_norm = sum(p.numel() for p in model.encoder.ln.parameters())
    assert count_parameters(model)["trainable"] == HEAD_PARAMS + 4 * block + final_norm
    assert not any(p.requires_grad for p in model.encoder.layers[0].parameters())


def test_partial_rejects_invalid_k():
    with pytest.raises(ValueError):
        vit("partial", k=0)


def test_resnet50_head_replaced():
    cfg = {"name": "resnet50", "num_classes": NUM_CLASSES, "pretrained": False, "strategy": "linear_probe"}
    model = build_model(cfg)
    assert model.fc.out_features == NUM_CLASSES
    assert count_parameters(model)["trainable"] == 2048 * NUM_CLASSES + NUM_CLASSES


@pytest.mark.parametrize("name, expected", [
    ("conv_proj.weight", 0),
    ("class_token", 0),
    ("encoder.pos_embedding", 0),
    ("encoder.layers.encoder_layer_0.ln_1.weight", 1),
    ("encoder.layers.encoder_layer_11.mlp.3.bias", 12),
    ("encoder.ln.weight", 13),
    ("heads.head.weight", 13),
])
def test_vit_layer_id(name, expected):
    assert vit_layer_id(name, num_layers=12) == expected
