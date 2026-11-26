"""Compare Ky's independent ViT to torchvision with identical, nonzero trained weights.

Declared tolerances before execution: float32 CPU atol=1e-5, rtol=1e-4.
Fused and explicit attention can differ by floating-point rounding. Top1 must agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models import ViT_B_16_Weights, vit_b_16

from src.models.ky_independent_vit import IndependentViT


def verify(checkpoint, image_dir, output):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "heads.head.weight" not in state:
        raise ValueError("Expected official raw torchvision state_dict")
    if torch.count_nonzero(state["heads.head.weight"]) == 0:
        raise ValueError("A zero classifier makes logits comparison vacuous")
    classes = state["heads.head.weight"].shape[0]
    reference = vit_b_16(weights=None)
    reference.heads.head = torch.nn.Linear(768, classes)
    reference.load_state_dict(state, strict=True)
    candidate = IndependentViT(num_classes=classes)
    candidate.load_torchvision_state(state)
    reference.eval()
    candidate.eval()
    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    inputs = [("random_seed42", torch.randn(1, 3, 224, 224)), ("zeros", torch.zeros(1, 3, 224, 224))]
    for image_id in ["Abyssinian_100", "beagle_100"]:
        with Image.open(Path(image_dir) / (image_id + ".jpg")) as image:
            inputs.append((image_id, transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)))
    records = []
    with torch.inference_mode():
        for name, x in inputs:
            expected, actual = reference(x), candidate(x)
            error = (expected - actual).abs()
            record = {
                "input": name,
                "shape": list(actual.shape),
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
                "allclose": torch.allclose(actual, expected, atol=1e-5, rtol=1e-4),
                "reference_top1": expected.argmax(-1).tolist(),
                "independent_top1": actual.argmax(-1).tolist(),
                "reference_logit_max_abs": float(expected.abs().max()),
            }
            records.append(record)
    result = {
        "author": "Nguyen Ngoc Ky supporting Trinh",
        "architecture": "ViT-B/16, 12 blocks, 12 heads, dim768, MLP3072",
        "implementation": "src/models/ky_independent_vit.py; no torchvision model inside",
        "device": "cpu",
        "dtype": "float32",
        "eval_mode": True,
        "dropout": 0,
        "atol": 1e-5,
        "rtol": 1e-4,
        "weight_source": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path("src/models/ky_independent_vit.py").read_bytes()).hexdigest(),
        "cases": records,
        "status": "PASS"
        if all(r["allclose"] and r["reference_top1"] == r["independent_top1"] for r in records)
        else "FAILED",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("KY_VIT_LOGITS_RESULT", json.dumps(result), flush=True)
    if result["status"] != "PASS":
        raise AssertionError("Independent ViT does not meet declared tolerances")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.checkpoint, args.image_dir, args.output)
