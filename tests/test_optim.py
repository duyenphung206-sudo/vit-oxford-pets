import math

import pytest
import torch

from src.models.factory import build_model
from src.optim import build_optimizer, build_param_groups, build_scheduler


@pytest.fixture(scope="module")
def vit():
    return build_model({"name": "vit_b_16", "num_classes": 37, "pretrained": False, "strategy": "full"})


def test_no_weight_decay_for_biases_norms_and_embeddings(vit):
    groups = build_param_groups(vit, {"weight_decay": 0.05})
    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    assert id(vit.class_token) not in decayed
    assert id(vit.encoder.pos_embedding) not in decayed
    assert id(vit.heads.head.bias) not in decayed
    assert id(vit.heads.head.weight) in decayed
    assert sum(len(g["params"]) for g in groups) == len(list(vit.parameters()))


def test_layer_decay_scales_learning_rate_by_depth(vit):
    optimizer = build_optimizer(vit, {"name": "adamw", "lr": 1e-3, "layer_decay": 0.5})
    lr_of = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    assert lr_of[id(vit.heads.head.weight)] == pytest.approx(1e-3)
    assert lr_of[id(vit.class_token)] == pytest.approx(1e-3 * 0.5 ** 13)


def test_warmup_then_cosine_schedule():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = build_scheduler(optimizer, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert lrs[55] == pytest.approx(0.5 * (1 + math.cos(math.pi * 45 / 90)))
    assert lrs[-1] < 0.01
