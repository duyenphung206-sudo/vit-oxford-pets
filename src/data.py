"""Official test split; stratified train/val split persisted by image identifier."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ViT_B_16_Weights

import random

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1


def transforms_for(training=False):
    if not training:
        return WEIGHTS.transforms()
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(), T.ColorJitter(0.15, 0.15, 0.15, 0.05),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def read_annotations(root, split):
    path = Path(root) / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            rows.append({"id": fields[0], "label": int(fields[1]) - 1})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image identifiers in annotations")
    return rows


def class_names(rows):
    mapping = {}
    for row in rows:
        label, name = row["label"], row["id"].rsplit("_", 1)[0]
        if label in mapping and mapping[label] != name:
            raise ValueError("Inconsistent class mapping")
        mapping[label] = name
    if set(mapping) != set(range(37)):
        raise ValueError("Expected all 37 classes")
    return [mapping[i] for i in range(37)]


def split_rows(rows, fraction, seed):
    train, val = train_test_split(
        np.arange(len(rows)), test_size=fraction, random_state=seed,
        stratify=[r["label"] for r in rows],
    )
    return [rows[i] for i in train], [rows[i] for i in val]


def prepare(root, fraction, seed, output):
    OxfordIIITPet(root, split="trainval", download=True)
    rows = read_annotations(root, "trainval")
    test = read_annotations(root, "test")
    train, val = split_rows(rows, fraction, seed)
    groups = [set(r["id"] for r in group) for group in (train, val, test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Data leakage: split image identifiers overlap")
    names = class_names(rows)
    if class_names(test) != names:
        raise ValueError("Test class mapping mismatch")
    annotation_dir = Path(root) / "oxford-iiit-pet" / "annotations"
    hashes = {s: hashlib.sha256((annotation_dir / f"{s}.txt").read_bytes()).hexdigest()
              for s in ("trainval", "test")}
    manifest = {"seed": seed, "val_fraction": fraction, "classes": names,
                "annotation_sha256": hashes, "train": train, "val": val, "test": test}
    write_json(Path(output) / "splits.json", manifest)
    return manifest


class PetImages(Dataset):
    def __init__(self, root, rows, training=False):
        self.image_dir = Path(root) / "oxford-iiit-pet" / "images"
        self.rows = rows
        self.transform = transforms_for(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / (row["id"] + ".jpg")) as image:
            x = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return x, row["label"]


def loader(dataset, batch_size, workers, shuffle, seed, cuda):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=cuda, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=workers > 0)


def build_dataloaders(data_dir="data", split_dir="data/splits", batch_size=16,
                      num_workers=0, seed=42, pin_memory=False, augment=True):
    """Return train/val/test loaders using persisted IDs (never resplit).

    data_dir is the parent of oxford-iiit-pet; labels are zero-based.
    Requires committed train.csv/val.csv, generated once, and official test.txt.
    """
    import csv
    rows = {}
    official = read_annotations(data_dir, "trainval")
    labels = {r["id"]: r["label"] for r in official}
    for split in ("train", "val"):
        with (Path(split_dir) / f"{split}.csv").open(encoding="utf-8", newline="") as f:
            rows[split] = [{"id": r["id"], "label": int(r["label"])} for r in csv.DictReader(f)]
        if len({r["id"] for r in rows[split]}) != len(rows[split]):
            raise ValueError("Duplicate ID in persisted split")
        if any(labels.get(r["id"]) != r["label"] for r in rows[split]):
            raise ValueError("Persisted split differs from official annotations")
    train_ids = {r["id"] for r in rows["train"]}
    val_ids = {r["id"] for r in rows["val"]}
    if train_ids & val_ids or train_ids | val_ids != set(labels):
        raise ValueError("Train/val overlap or incomplete trainval coverage")
    rows["test"] = read_annotations(data_dir, "test")
    if set(labels) & {r["id"] for r in rows["test"]}:
        raise ValueError("Official test overlaps trainval")
    return {name: loader(PetImages(data_dir, group, name == "train" and augment),
                         batch_size, num_workers, name == "train", seed, pin_memory)
            for name, group in rows.items()}
