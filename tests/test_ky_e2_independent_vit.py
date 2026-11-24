"""Verify independent attention/ViT and linear-probe optimizer isolation."""

import pytest
import torch
from torchvision.models.vision_transformer import VisionTransformer

from src.models.factory import build_model
from src.models.ky_independent_vit import IndependentViT
from src.optim import build_optimizer
from src.utils import load_config


def test_independent_logits_nonzero_head_and_weight_mapping():
    torch.manual_seed(42)
    reference = VisionTransformer(32, 16, 2, 3, 12, 24, num_classes=5).eval()
    with torch.no_grad():
        reference.heads.head.weight.normal_(std=0.1)
        reference.heads.head.bias.normal_(std=0.1)
    candidate = IndependentViT(5, 32, 16, 12, 2, 3, 24).eval()
    candidate.load_torchvision_state(reference.state_dict())
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        actual, expected = candidate(x), reference(x)
    assert actual.abs().max() > 0.01
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    bad = dict(reference.state_dict())
    bad.pop("class_token")
    with pytest.raises(ValueError, match="checkpoint keys"):
        candidate.load_torchvision_state(bad)


def test_e2_freeze_backward_and_optimizer_only_updates_head():
    cfg = load_config("configs/e2_vit_linear_probe.yaml")
    assert cfg["optimizer"]["lr"] == 0.003 and cfg["data"]["batch_size"] == 64
    assert cfg["train"]["epochs"] == 10 and cfg["train"]["seed"] == 42
    model_cfg = dict(cfg["model"], pretrained=False)
    model = build_model(model_cfg)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"heads.head.weight", "heads.head.bias"}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 28453
    optimizer = build_optimizer(model, cfg["optimizer"])
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    logits = model(torch.randn(1, 3, 224, 224))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)
    optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and changed <= trainable


def test_full_vit_b16_logits_with_identical_nonzero_weights():
    torch.manual_seed(7)
    reference = build_model({"name": "vit_b_16", "num_classes": 37, "pretrained": False, "strategy": "full"})
    reference.eval()
    with torch.no_grad():
        reference.heads.head.weight.normal_(std=0.02)
        reference.heads.head.bias.normal_(std=0.01)
    candidate = IndependentViT().eval()
    candidate.load_torchvision_state(reference.state_dict())
    assert len(candidate.blocks) == 12 and candidate.blocks[0].attention.heads == 12
    assert sum(p.numel() for p in candidate.parameters()) == 85827109
    with torch.inference_mode():
        for x in [torch.randn(1, 3, 224, 224), torch.zeros(1, 3, 224, 224)]:
            expected, actual = reference(x), candidate(x)
            assert actual.shape == (1, 37) and expected.abs().max() > 0.01
            error = (actual - expected).abs()
            print("FULL_VIT_LOGITS", float(error.max()), float(error.mean()))
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
