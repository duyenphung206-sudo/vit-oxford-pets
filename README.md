# Vision Transformer (ViT-B/16) Fine-tuning on Oxford-IIIT Pet

> Understanding self-attention for images and transferring an ImageNet-pretrained **ViT-B/16** to fine-grained classification of **37 cat and dog breeds**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/duyenphung206-sudo/vit-oxford-pets/actions/workflows/tests.yml/badge.svg)

<p align="center">
  <img src="figures/e4_curves.png" width="760" alt="Training curves of full fine-tuning of ViT-B/16 (3 seeds)">
</p>

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Checkpoints](#checkpoints)
- [Demo](#demo)
- [Team](#team)
- [References](#references)
- [License](#license)

## Overview
Final-term project of the **Deep Learning** course, Faculty of Information Technology, HUFLIT (Ho Chi Minh City University of Foreign Languages – Information Technology), academic year 2026–2027. Instructor: Tôn Quang Toại.

The project answers three questions with the same data split and training protocol:
1. **Q1** – With the same ImageNet-1K pre-training, does ViT-B/16 outperform the convolutional baseline ResNet-50 [4]?
2. **Q2** – How much of ViT-B/16 must be updated: only the head, the last four encoder blocks, or the whole network?
3. **Q3** – How sensitive is the main configuration to the learning rate and to data augmentation?

## Results
All runs: Google Colab, NVIDIA T4 16 GB, batch 64, at most 10 epochs, checkpoint with the lowest validation loss. Validation set: 736 images (20 % of the official `trainval`, stratified, seed 42).

| ID | Model | Strategy | Trainable params | Val accuracy (%) | Val loss | Time / epoch (s) |
|----|-------|----------|-----------------:|-----------------:|---------:|-----------------:|
| E1 | ResNet-50 | full fine-tuning | 23,583,845 | pending | pending | pending |
| E2 | ViT-B/16 | feature extraction (linear probe) | 28,453 | pending | pending | pending |
| E3 | ViT-B/16 | fine-tune last 4 blocks | 28,381,477 | 94.97 | 0.2371 | 44.4 |
| E4 | ViT-B/16 | full fine-tuning, seeds 42/43/44 | 85,827,109 | **95.47 ± 0.08** | 0.3097 ± 0.0019 | 55.2 |

E5 changes one factor of the E4 recipe at a time (seed 42):

| Run | Learning rate | Augmentation | Val accuracy (%) | Val loss |
|-----|--------------:|:------------:|-----------------:|---------:|
| E5-LR1 | 0.001 | on | 91.30 | 1.1390 |
| E5-LR2 (= E4 seed 42) | 0.003 | on | 95.52 | 0.3112 |
| E5-LR3 | 0.01 | on | 95.65 | 0.1587 |
| E5-AUG | 0.003 | off | 95.92 | 0.2655 |

With 736 validation images one image is 0.14 percentage points and the standard error at 95 % accuracy is about 0.8 points, so differences below about 2 points between single-seed runs are not conclusive. Test-set results (official `test` split, 3,669 images) are added when the evaluation is merged. Each value comes from `results/<experiment>/`.

For context, Dosovitskiy *et al.* report 93.81 % on Oxford-IIIT Pets for ViT-B/16 pretrained on ImageNet and fine-tuned at 384 × 384 with batch 512 ([1], Table 5). Our setting differs (224 × 224, batch 64, `torchvision` weights, validation instead of test), so the numbers are not directly comparable.

## Dataset
**Oxford-IIIT Pet** [3]: 37 breeds (12 cat, 25 dog), 7,349 annotated images. 32 of the 37 classes have exactly 100 `trainval` images; the smallest class (Egyptian Mau) has 93.

| Split | Images | Usage |
|-------|-------:|-------|
| official `trainval` → train | 2,944 | training |
| official `trainval` → val | 736 | checkpoint selection, early stopping, hyper-parameters |
| official `test` | 3,669 | final evaluation, once per model |

The split is stored in `data/splits/train.csv` and `data/splits/val.csv` with SHA-256 hashes of the annotation files in `data/splits/metadata.json`; every experiment reads the same files. Images are **not** stored in this repository; they are downloaded by `torchvision.datasets.OxfordIIITPet(download=True)`. Licence: CC BY-SA 4.0 (see [`docs/data_card.md`](docs/data_card.md)).

## Method
- Backbone: `torchvision.models.vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1` (86.6 M parameters, ImageNet-1K top-1 81.07 %).
- Head: zero-initialised `Linear(768 → 37)` replacing the ImageNet head ([1], Sec. 3.2).
- Transfer strategies: feature extraction, partial fine-tuning (last 4 blocks + final LayerNorm), full fine-tuning (`src/models/factory.py`).
- Optimisation: SGD, momentum 0.9, no weight decay, learning rate 0.003 with a 1-epoch linear warm-up and per-step cosine decay, global gradient-norm clipping at 1, mixed precision ([1], App. B.1.1).
- Pre-processing: evaluation = resize 256 → center crop 224 → ImageNet mean/std; training = `RandomResizedCrop(224, scale=(0.7, 1.0))`, horizontal flip, mild `ColorJitter`.
- Full protocol and sources: [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Repository Structure
```
vit-oxford-pets/
├── app/              # Gradio demo (top-5 breeds)
├── configs/          # YAML config of every experiment (inherits base.yaml)
├── data/splits/      # train/val index files and metadata (images are NOT committed)
├── docs/             # data card, experiment protocol, references
├── figures/          # figures used in the README and the report
├── notebooks/        # environment check, training runs on Colab
├── results/          # log.csv, metrics.json, summary.json of each experiment
├── src/              # data.py, train.py, engine.py, optim.py, summarize.py, models/factory.py
└── tests/            # unit tests run by GitHub Actions (ruff + pytest)
```

## Getting Started
```bash
git clone https://github.com/duyenphung206-sudo/vit-oxford-pets.git
cd vit-oxford-pets
pip install -r requirements.txt
python -m pytest
```
Versions in `requirements.txt` are the ones verified on Google Colab (Python 3.13, PyTorch 2.14.1, torchvision 0.29.1) by `notebooks/00_setup_check.ipynb`.

## Reproducing the Experiments
On Google Colab with a T4 GPU, open `notebooks/06_train_e4_vit.ipynb` and run all cells: it clones the repository, downloads the dataset to `/content/data`, trains the three seeds and writes `results/e4_vit_full_ft/summary.json` and `figures/e4_curves.png`.

From a terminal (dataset under `<data_root>/oxford-iiit-pet`):
```bash
python -m src.train --config configs/e4_vit_full_ft.yaml --seed 42
python -m src.train --config configs/e4_vit_full_ft.yaml --fake-data --epochs 1   # pipeline check without the dataset
```
Set `data.root` in `configs/base.yaml` to `<data_root>`. Outputs go to `results/<experiment>/seed<seed>/`, checkpoints to `checkpoints/<experiment>/seed<seed>/best.pt` (ignored by git).

## Checkpoints
Checkpoints are not stored in the repository (about 330 MB each). The E4 checkpoint used for evaluation and the demo (seed 44, lowest validation loss) is attached to the GitHub Release `v1.0.0`.

## Demo
```bash
python app/app.py --checkpoint <path/to/best.pt> --device cpu --port 7860
```
Then open http://127.0.0.1:7860, upload a cat or dog photo and read the five most likely breeds.

## Team
| Member | Student ID | Main responsibilities | GitHub |
|--------|-----------|-----------------------|--------|
| Phùng Thị Hoàng Duyên (leader) | 24DH190211 | model factory, training pipeline, experiment protocol, E4, repository | [@duyenphung206-sudo](https://github.com/duyenphung206-sudo) |
| Nguyễn Ngọc Kỳ | | data pipeline and EDA, E3, E5, demo | [@Edward2106](https://github.com/Edward2106) |
| Yến Trinh | | ViT and self-attention analysis, E2, references | [@nlyen-trinh](https://github.com/nlyen-trinh) |
| Đức Tâm | | E1, evaluation, error analysis, attention visualisation | |

## References
[1] A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
[2] A. Vaswani *et al.*, "Attention is all you need," in *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017, pp. 5998–6008.
[3] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and dogs," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2012, pp. 3498–3505, doi: 10.1109/CVPR.2012.6248092.
[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
[5] S. Abnar and W. Zuidema, "Quantifying attention flow in transformers," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 4190–4197, doi: 10.18653/v1/2020.acl-main.385.

## License
Code: [MIT](LICENSE). Dataset: CC BY-SA 4.0, © original image owners — see [3].
