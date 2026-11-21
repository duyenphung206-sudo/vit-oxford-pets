# Kỳ — figure and table evidence register

No report outline or caption mapping exists in the current `report/` directory or repository
documentation. Numbers below are assignments supplied by the user, not inferred captions.
Do not assign experiment plots to report numbers until the outline is supplied.

| Kind | Assigned number | Exact caption | Status | Required evidence |
|---|---|---|---|---|
| Figure | 1.1 | unknown | BLOCKED | Report outline/caption |
| Figure | 2.1 | unknown | BLOCKED | Report outline/caption |
| Figure | 2.2 | unknown | BLOCKED | Report outline/caption |
| Figure | 2.3 | unknown | BLOCKED | Report outline/caption |
| Figure | 3.4 | unknown | BLOCKED | Report outline/caption and underlying evidence |
| Figure | 3.6 | unknown | BLOCKED | Report outline/caption and underlying evidence |
| Figure | 3.10 | unknown | BLOCKED | Report outline/caption and underlying evidence |
| Figure | 3.11 | unknown | BLOCKED | Report outline/caption and underlying evidence |
| Table | 2-3 | unknown | BLOCKED | Exact caption only; EDA size data recovered and verified (see below) |
| Table | 3-9 | unknown | BLOCKED | Report outline/caption and underlying evidence |
| Table | 3-12 | unknown | BLOCKED | Report outline/caption and underlying evidence |

Reproducible E5 comparison products live separately under `results/e5_ablation/`.
`python -m src.ky_experiments --summarize` generates CSV/JSON and 300-dpi PNG/SVG.
Pending runs have no numeric values and are explicitly labelled in each plot.
These products are not claimed to satisfy any numbered report figure/table yet.

## Recovered EDA evidence (2026-11-21)

Ky's archived PR11 contains the executed `01_eda.ipynb` and original-size statistics.
Recovered immutable commit: `a03bad2fc31f1b87e550629c7566aa5dba05dda8`.
The notebook display output matches the CSV; independently opening all 7,349 annotated
image headers reproduced every CSV statistic. No resize/crop or EXIF transpose was applied.

Prepared assets: `report/ky_assets/eda_dimension_statistics.{csv,md,png,svg}`
and `eda_dimension_provenance.json`. The figure was visually checked at 300 dpi.
These are evidence assets for table 2-3; its exact caption remains BLOCKED. No caption
has been invented, and no numbered asset is claimed complete.

Reproduce from the repository root (dataset is read-only):

```powershell
git fetch origin refs/pull/11/head
python report/ky_prepare_dimensions.py --dataset <folder-containing-images-and-annotations>
```

Only the three small original EDA references were recovered into Downloads/.ky-tools
for inspection; the 4.7 MB notebook and dataset are not added to this PR.
