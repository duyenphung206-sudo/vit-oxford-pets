# E5 — Nguyễn Ngọc Kỳ

DONE: required seed42 runs E5-LR1, E5-LR3 and E5-AUG on Tesla T4; verified config/log/metrics.
DONE: E5-LR2 reuses the committed E4 seed42 evidence. E4 was not rerun or modified.
DONE: summary_seed42.csv/json, PNG/SVG comparisons and SHA256-verified Drive checkpoint backups.
DONE: optional E5-LR4, LR=0.03, augmentation on; 10 real epochs on T4, val accuracy0.951087, val loss0.148726.
Evidence and fifth checkpoint backup: docs/KY_LR4_RUN_MANIFEST.json.

The current plan supersedes the earlier provisional LR grid in this directory.
Checkpoint selection follows minimum validation loss, not maximum validation accuracy.
Accuracy is a 0–1 fraction. Training time is the sum of rounded epoch_time_s values, excluding setup/download.
Only seed42 is compared; no multi-seed stability claim is made. Test was not used for tuning.
Regenerate: `python -m src.ky_experiments --summarize`.

## E5-LR4 — COMPLETED (verified 2026-11-27)

| Field | Verified value |
| --- | --- |
| Learning rate (configured base LR) | 0.03 |
| Seed | 42 |
| Augmentation | enabled |
| Epochs run / best epoch | 10 / 10 |
| Best validation loss | 0.148726 |
| Validation accuracy at the selected checkpoint | 0.951087 = 95.1087% |
| Training time (sum of logged epochs) | 459.5 s |
| Status | COMPLETED |

Sources: [summary_seed42.csv](summary_seed42.csv), [run config](../e5_lr4/seed42/config.yaml), [metrics.json](../e5_lr4/seed42/metrics.json), [log.csv](../e5_lr4/seed42/log.csv), and [original run manifest](../../docs/KY_LR4_RUN_MANIFEST.json).
The summary and metrics agree at their stored precision. The log has 10 epochs; epoch 10 has the minimum validation loss and records rounded values 0.1487 / 0.9511, consistent with metrics. The configured LR is 0.03; per-epoch LR varies with warmup/cosine scheduling. Epochs 8–9 have higher rounded validation accuracy, but checkpoint selection uses minimum validation loss.
This is a documentation verification of the existing T4 run; no experiment, summary, LR1/LR2/LR3/AUG evidence, or checkpoint was rerun or changed.
