# E3 — Nguyễn Ngọc Kỳ

DONE: recipe, Colab notebook, exact trainability verification (28,381,477 / 85,827,109 parameters).

PENDING: real seed42 GPU training. No accuracy/loss is reported until a verified run exists.

Run `python -m src.ky_experiments E3` on a CUDA runtime. The wrapper will not train on CPU
or overwrite incomplete output. Outputs: `seed42/{config.yaml,log.csv,metrics.json,curves.png}`;
checkpoint: `checkpoints/e3_vit_partial/seed42/best.pt` (excluded from Git).
