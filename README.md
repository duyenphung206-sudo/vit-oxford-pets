<!--
  README TEMPLATE — mọi chỗ trong dấu < > hoặc ghi "TBD" sẽ được nhóm điền dần
  khi có kết quả thật. TUYỆT ĐỐI không điền số liệu chưa chạy ra.
-->

# Vision Transformer (ViT-B/16) Fine-tuning on Oxford-IIIT Pet

> Understanding self-attention for images and transferring an ImageNet-pretrained **ViT-B/16** to fine-grained classification of **37 cat & dog breeds**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<p align="center">
  <!-- TBD: thay bằng hình attention map / demo GIF thật -->
  <img src="figures/teaser.png" width="720" alt="Attention rollout of ViT-B/16 on Oxford-IIIT Pet">
</p>

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Demo](#demo)
- [Team](#team)
- [References](#references)
- [License](#license)

## Overview
This project is the final-term project of the **Deep Learning** course at HUFLIT (Ho Chi Minh City University of Foreign Languages – Information Technology), academic year 2026–2027.

**Goals**
1. Explain the self-attention mechanism and the Vision Transformer architecture [1], [2].
2. Re-implement ViT-B/16 from scratch in PyTorch and verify it against `torchvision` (identical logits).
3. Compare three transfer-learning strategies — *feature extraction*, *partial fine-tuning*, *full fine-tuning* — against a CNN baseline (ResNet-50 [4]).
4. Visualise what the model attends to (Attention Rollout [5]) and analyse its errors.

## Results
Test set: official Oxford-IIIT Pet `test` split (3,669 images). Mean ± std over 3 seeds where indicated.

| ID | Model | Strategy | Trainable params | Top-1 Acc (%) | Macro-F1 (%) |
|----|-------|----------|-----------------:|--------------:|-------------:|
| E1 | ResNet-50 | Full fine-tuning | TBD | TBD | TBD |
| E2 | ViT-B/16 | Feature extraction (linear probe) | TBD | TBD | TBD |
| E3 | ViT-B/16 | Fine-tune last *k* blocks | TBD | TBD | TBD |
| E4 | ViT-B/16 | Full fine-tuning (3 seeds) | TBD | TBD ± TBD | TBD ± TBD |

For context, Dosovitskiy *et al.* report 93.81 % on Oxford-IIIT Pets for ViT-B/16 pretrained on ImageNet and fine-tuned at 384×384 resolution ([1], Table 5). Our setting differs (224×224, `torchvision` weights), so the numbers are not directly comparable.

## Dataset
**Oxford-IIIT Pet** [3]: 37 breeds (12 cat, 25 dog), ~200 images per class, 7,349 images in total.

| Split | Images | Usage |
|-------|-------:|-------|
| `trainval` (official) → train | TBD | training |
| `trainval` (official) → val | TBD | model selection (stratified, seed 42) |
| `test` (official) | 3,669 | final report only |

The dataset is **not** stored in this repository. It is downloaded automatically by `torchvision.datasets.OxfordIIITPet(download=True)`. Licence: CC BY-SA 4.0 (see [`docs/data_card.md`](docs/data_card.md)).

## Method
<!-- TBD: hình sơ đồ kiến trúc ViT tự vẽ -->
- Backbone: `torchvision.models.vit_b_16` with `ViT_B_16_Weights.IMAGENET1K_V1` (86.6 M parameters).
- Head: `Linear(768 → 37)` replacing the 1000-class ImageNet head.
- Pre-processing: resize 256 → center crop 224 → ImageNet mean/std normalisation.
- Training details: see [`configs/`](configs/) and Chapter 4 of the report.

## Repository Structure
```
vit-oxford-pets/
├── app/            # Gradio demo
├── configs/        # YAML config of every experiment (E1–E5)
├── data/splits/    # train/val index files (images are NOT committed)
├── docs/           # data card, model card, paper notes, meeting notes
├── figures/        # figures used in README / report
├── notebooks/      # EDA, from-scratch ViT, attention maps, error analysis
├── report/         # final report (PDF)
├── results/        # metrics.json + training logs of each experiment
├── slides/         # presentation (PDF)
├── src/            # data.py, engine.py, train.py, evaluate.py, models/
└── tests/          # unit tests (shapes, parameter count)
```

## Getting Started
```bash
git clone https://github.com/<owner>/vit-oxford-pets.git
cd vit-oxford-pets
pip install -r requirements.txt
```
Or open the notebooks directly in Google Colab (GPU runtime).

## Reproducing the Experiments
```bash
# TBD — điền lệnh thật khi src/train.py hoàn thành
python -m src.train    --config configs/e4_vit_full_ft.yaml --seed 42
python -m src.evaluate --config configs/e4_vit_full_ft.yaml --checkpoint <path/to/best.pt>
```

## Pretrained Checkpoints
| Experiment | Download |
|-----------|----------|
| E4 – ViT-B/16 full fine-tuning | TBD (GitHub Release `v1.0.0`) |

## Demo
```bash
python app/app.py
```
<!-- TBD: chèn figures/demo.gif -->

## Team
| Member | Student ID | Main responsibilities | GitHub |
|--------|-----------|-----------------------|--------|
| Phùng Thị Hoàng Duyên (leader) | 24DH190211 | Training pipeline, integration, repo management | [@duyenphung206-sudo](https://github.com/duyenphung206-sudo) |
| Nguyễn Ngọc Kỳ | 24DH112710 | Data, CNN baseline, demo | [@Edward2106](https://github.com/Edward2106) |
| TBD | TBD | Self-attention & ViT theory, from-scratch ViT | TBD |
| TBD | TBD | Evaluation, attention visualisation, error analysis | TBD |

Instructor: TBD — Faculty of Information Technology, HUFLIT.

## References
[1] A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
[2] A. Vaswani *et al.*, "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 5998–6008.
[3] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, "Cats and dogs," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2012, pp. 3498–3505, doi: 10.1109/CVPR.2012.6248092.
[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.
[5] S. Abnar and W. Zuidema, "Quantifying attention flow in transformers," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2020, pp. 4190–4197, doi: 10.18653/v1/2020.acl-main.385.

## License
Code: [MIT](LICENSE). Dataset: CC BY-SA 4.0, © original image owners — see [3].
