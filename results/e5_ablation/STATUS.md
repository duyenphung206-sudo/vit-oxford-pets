# E5 — Nguyễn Ngọc Kỳ

DONE: required seed42 runs E5-LR1, E5-LR3 and E5-AUG on Tesla T4; verified config/log/metrics.
DONE: E5-LR2 reuses the committed E4 seed42 evidence. E4 was not rerun or modified.
DONE: summary_seed42.csv/json, PNG/SVG comparisons and SHA256-verified Drive checkpoint backups.
PENDING (optional): E5-LR4, LR=0.03, augmentation on. No value is imputed.

The current plan supersedes the earlier provisional LR grid in this directory.
Checkpoint selection follows minimum validation loss, not maximum validation accuracy.
Accuracy is a 0–1 fraction. Training time is the sum of rounded epoch_time_s values, excluding setup/download.
Only seed42 is compared; no multi-seed stability claim is made. Test was not used for tuning.
Regenerate: `python -m src.ky_experiments --summarize`.
