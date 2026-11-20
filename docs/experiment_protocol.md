# Experiment protocol

All experiments (E1–E5) follow this protocol so that their results are directly comparable.
Any deviation must be stated in the corresponding `configs/*.yaml` and in the report.

## Data

| Item | Setting |
|------|---------|
| Dataset | Oxford-IIIT Pet [3], 37 classes, loaded with `torchvision.datasets.OxfordIIITPet` |
| Training / validation | official `trainval` split (3,680 images), stratified by class into 80 % train / 20 % validation, seed 42; indices stored in `data/splits/` and shared by every experiment |
| Test | official `test` split (3,669 images), used once per trained model, after training |
| Evaluation transform | resize 256 → center crop 224 → ImageNet mean/std, identical to `ViT_B_16_Weights.IMAGENET1K_V1.transforms()` [17] |
| Training augmentation | `RandomResizedCrop(224)` + `RandomHorizontalFlip()`; E5 ablates it |

## Model selection and reporting

- The checkpoint with the lowest validation loss is kept (lecture 04). Hyper-parameters are chosen on validation only.
- The main configuration (E4) is trained with seeds 42, 43 and 44; results are reported as mean ± standard deviation.
- Reported for every run: top-1 and top-5 accuracy, macro-F1, trainable / total parameters, mean time per epoch, peak GPU memory (`results/<experiment>/seed<seed>/metrics.json`).

## Fine-tuning recipe (E4, `configs/e4_vit_full_ft.yaml`)

The defaults follow the fine-tuning protocol of the original ViT paper [1, Sec. 3.2 and App. B.1.1], adapted to a single Colab GPU.

| Setting | Value | Source |
|---------|-------|--------|
| Backbone | `vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1` (86.6 M parameters) | [17] |
| Head | ImageNet head removed, zero-initialised `Linear(768, 37)` | [1, Sec. 3.2] |
| Optimizer | SGD, momentum 0.9, no weight decay | [1, App. B.1.1, Table 4] |
| Learning rate | 0.003; E5 searches {0.001, 0.003, 0.01, 0.03}, the grid used in [1] | [1, Table 4] |
| Schedule | cosine decay to 0, updated every step [1, Table 4], [22], preceded by a 1-epoch linear warm-up (our addition, stabilises the first updates of the zero-initialised head) [23] | [1], [22], [23] |
| Gradient clipping | global norm 1.0 | [1, Table 4] |
| Batch size / epochs | 64 / at most 10, early stopping after 5 epochs without lower validation loss | deviation from [1] (batch 512), GPU memory |
| Resolution | 224 × 224 | deviation from [1] (384), compute; noted when comparing with [1, Table 5] |
| Mixed precision | `torch.autocast` + `GradScaler` on GPU | [24] |

`src/optim.py` also supports AdamW [6] with layer-wise learning-rate decay [25], [26]; it is not used by default.

## Code interface

`src/data.py` (T12–T13) exposes:

```python
build_dataloaders(data_dir, split_dir, batch_size, num_workers, seed, pin_memory, augment) -> dict[str, DataLoader]
# keys: "train", "val", "test"; data_dir is the parent folder of oxford-iiit-pet; the split is read from data/splits/*.csv
```

`src/train.py` consumes `train` and `val` (mapping `data.root`, `data.splits_dir`, `data.batch_size`, `data.num_workers`, `data.augment` and `train.seed` from the config); `src/evaluate.py` consumes `test`.

## References

Numbers [1]–[21] refer to `docs/references.md`.

[22] I. Loshchilov and F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.
[23] P. Goyal *et al.*, "Accurate, large minibatch SGD: Training ImageNet in 1 hour," 2017, *arXiv:1706.02677*.
[24] P. Micikevicius *et al.*, "Mixed precision training," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.
[25] J. Howard and S. Ruder, "Universal language model fine-tuning for text classification," in *Proc. 56th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2018, pp. 328–339.
[26] H. Bao, L. Dong, S. Piao, and F. Wei, "BEiT: BERT pre-training of image transformers," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.
