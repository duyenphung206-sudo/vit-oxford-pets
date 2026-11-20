import pytest
import torch
from PIL import Image
from src.data import transforms_for, build_dataloaders

def test_eval_matches_official_weights():
    from torchvision.models import ViT_B_16_Weights
    image=Image.new("RGB", (319, 271), (100, 50, 20))
    assert torch.equal(transforms_for(False)(image), ViT_B_16_Weights.IMAGENET1K_V1.transforms()(image))

def test_overlap_is_rejected(tmp_path):
    root=tmp_path/"data/oxford-iiit-pet/annotations"; root.mkdir(parents=True)
    (root/"trainval.txt").write_text("pet_1 1 1 1\npet_2 1 1 1\n")
    splits=tmp_path/"splits"; splits.mkdir()
    for name in ("train", "val"):
        (splits/f"{name}.csv").write_text("id,label\npet_1,0\n")
    with pytest.raises(ValueError, match="overlap"):
        build_dataloaders(tmp_path/"data",splits)
