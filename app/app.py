"""CPU Top-5 demo for group E4 state dictionaries and metadata bundles."""

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path
from threading import RLock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import functional as TF


def projected_attention(tokens, attention):
    """Mean-head attention from the existing packed Q/K projection, without changing a forward pass."""
    dimension = attention.embed_dim
    bias = attention.in_proj_bias
    projected = F.linear(
        tokens, attention.in_proj_weight[: 2 * dimension], None if bias is None else bias[: 2 * dimension]
    )
    query, key = projected.chunk(2, dim=-1)
    batch, length, _ = query.shape
    head_size = dimension // attention.num_heads
    query = query.reshape(batch, length, attention.num_heads, head_size).transpose(1, 2)
    key = key.reshape(batch, length, attention.num_heads, head_size).transpose(1, 2)
    return ((query @ key.transpose(-2, -1)) / head_size**0.5).softmax(-1).mean(1)


def attention_rollout(matrices):
    """Residual-adjusted attention: R = A_L ... A_1, each A = (mean_heads + I) / 2."""
    if not matrices:
        raise ValueError("No attention matrices captured")
    identity = torch.eye(matrices[0].shape[-1], device=matrices[0].device)
    result = identity
    for matrix in matrices:
        if matrix.shape != identity.shape or not torch.isfinite(matrix).all():
            raise ValueError("Invalid attention matrix")
        adjusted = (matrix + identity) / 2
        result = adjusted @ result
    return result


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
        self._inference_lock = RLock()

    def _input(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Upload a valid image")
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image, self.transform(image).unsqueeze(0).to(self.device)

    def _scores(self, logits):
        values, indices = logits.float().softmax(-1)[0].topk(5)
        return {self.classes[i]: float(v) for i, v in zip(indices.cpu().tolist(), values.cpu().tolist(), strict=True)}

    @torch.inference_mode()
    def __call__(self, image):
        _, x = self._input(image)
        with self._inference_lock:
            return self._scores(self.model(x))

    @torch.inference_mode()
    def with_attention(self, image):
        """Visualize the 224px model crop; rollout is qualitative, not causal attribution."""
        from matplotlib import colormaps

        image, x = self._input(image)
        matrices, handles = [], []
        with self._inference_lock:
            try:
                for block in self.model.encoder.layers:

                    def capture(module, args, output, attention=block.self_attention):
                        matrices.append(projected_attention(output, attention)[0].detach())

                    handles.append(block.ln_1.register_forward_hook(capture))
                scores = self._scores(self.model(x))
            finally:
                for handle in handles:
                    handle.remove()
        if len(matrices) != 12:
            raise ValueError("Expected attention from all 12 ViT-B/16 blocks")
        rollout = attention_rollout(matrices)
        patch_map = rollout[0, 1:].reshape(1, 1, 14, 14)
        mask = F.interpolate(patch_map, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        mask = ((mask - mask.min()) / (mask.max() - mask.min()).clamp_min(1e-12)).cpu().numpy()
        crop = TF.center_crop(
            TF.resize(
                image,
                self.transform.resize_size,
                interpolation=self.transform.interpolation,
                antialias=self.transform.antialias,
            ),
            self.transform.crop_size,
        )
        heat = colormaps["jet"](mask)[..., :3] * 255
        overlay = Image.fromarray(np.uint8(np.clip(0.55 * np.asarray(crop) + 0.45 * heat, 0, 255)))
        self.last_rollout_stats = {
            "layers": len(matrices),
            "grid": [14, 14],
            "row_sum_max_error": float((rollout.sum(-1) - 1).abs().max()),
            "patch_min": float(patch_map.min()),
            "patch_max": float(patch_map.max()),
        }
        return scores, overlay


def create_demo(predictor, attention=False):
    import gradio as gr

    def classify(image):
        try:
            return predictor.with_attention(image) if attention else predictor(image)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", sources=["upload"]),
        outputs=(
            [gr.Label(num_top_classes=5), gr.Image(label="Attention rollout (model crop)", type="pil")]
            if attention
            else gr.Label(num_top_classes=5)
        ),
        title="Oxford-IIIT Pet — Top 5",
        description=(
            ("Attention rollout is qualitative, not a causal explanation. " if attention else "")
            + "Softmax scores are not calibrated correctness probabilities."
        ),
        flagging_mode="never",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--attention", action="store_true", help="Show attention rollout for report figures 3.10/3.11")
    a = p.parse_args()
    create_demo(Predictor(a.checkpoint, a.device), attention=a.attention).launch(
        server_name="127.0.0.1", server_port=a.port
    )


if __name__ == "__main__":
    main()
