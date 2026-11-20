# E5 — Nguyễn Ngọc Kỳ

DONE: E4 seed42 baseline evidence validated; see summary_seed42.csv/json.
DONE: E4-inheriting SGD configurations, GPU-only runner, selectable Colab notebook and comparison plots.
PENDING: new seed42 runs E5-LR1, E5-LR3, E5-AUG; E5-LR4 is optional.

This plan supersedes the earlier provisional LR grid in this directory.
Selection follows minimum validation loss. Reuse E4 seed42, never the three-seed average.
Missing or inconsistent evidence is labelled explicitly, with no fabricated values.

Regenerate: `python -m src.ky_experiments --summarize`.
Select one run: `python -m src.ky_experiments E5-LR1` (requires CUDA).
