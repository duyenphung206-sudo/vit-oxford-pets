"""CPU Top-5 demo for group E4 state dictionaries and metadata bundles."""

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models import ViT_B_16_Weights, vit_b_16


def official_class_names():
    """Read the group's zero-based label order, never alphabetical filename order."""
    path = Path(__file__).resolve().parents[1] / "data/splits/train.csv"
    mapping = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            label = int(row["label"])
            name = row["id"].rsplit("_", 1)[0]
            if label in mapping and mapping[label] != name:
                raise ValueError("Inconsistent class mapping in the persisted train split")
            mapping[label] = name
    if set(mapping) != set(range(37)) or len(set(mapping.values())) != 37:
        raise ValueError("Expected 37 distinct official classes")
    return [mapping[i] for i in range(37)]


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, Mapping):
            raise ValueError("Unsupported checkpoint")
        if ckpt.get("smoke_only", False):
            raise ValueError("Reject smoke-only checkpoint")
        if "model" in ckpt:
            if ckpt.get("architecture") != "vit_b_16" or ckpt.get("classes") != official_class_names():
                raise ValueError("Checkpoint architecture or class order differs from the group protocol")
            state = ckpt["model"]
        elif ckpt and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in ckpt.items()):
            state = ckpt
        else:
            raise ValueError("Unsupported checkpoint")
        self.classes = official_class_names()
        self.model = vit_b_16(weights=None)
        self.model.heads.head = torch.nn.Linear(768, 37)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    @torch.inference_mode()
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Upload a valid image")
        x = self.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(x).float().softmax(-1)[0]
        values, indices = probs.topk(5)
        return {self.classes[i]: float(v) for i, v in zip(indices.cpu().tolist(), values.cpu().tolist(), strict=True)}


def create_demo(predictor):
    import gradio as gr

    def classify(image):
        try:
            return predictor(image)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", sources=["upload"]),
        outputs=gr.Label(num_top_classes=5),
        title="Oxford-IIIT Pet — Top 5",
        description=(
            "T37 partial: attention map pending verification. "
            "Softmax scores are not calibrated correctness probabilities."
        ),
        flagging_mode="never",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    create_demo(Predictor(a.checkpoint, a.device)).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
