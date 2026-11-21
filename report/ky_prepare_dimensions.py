"""Reproduce Ky's original-size evidence from the archived EDA and actual images.

Run from the repository root after fetching refs/pull/11/head. The output is an
unnumbered evidence asset; report captions must come from the team's outline.
"""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

EDA_REF = "a03bad2fc31f1b87e550629c7566aa5dba05dda8"
ROOT = Path(__file__).resolve().parents[1]


def archived(path):
    return subprocess.check_output(["git", "show", f"{EDA_REF}:{path}"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Directory containing images/ and annotations/")
    args = parser.parse_args()
    notebook_bytes = archived("notebooks/01_eda.ipynb")
    notebook = json.loads(notebook_bytes)
    csv_bytes = archived("results/eda/dimension_statistics.csv")
    reference = pd.read_csv(io.BytesIO(csv_bytes), index_col=0)
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code"
                and "dimension_statistics.csv" in "".join(c["source"]))
    assert cell["execution_count"] is not None
    assert not any(o["output_type"] == "error" for o in cell["outputs"])
    plain = next("".join(o.get("data", {}).get("text/plain", []))
                 for o in cell["outputs"] if "width" in "".join(
                     o.get("data", {}).get("text/plain", [])))
    rendered = pd.read_csv(io.StringIO(plain), sep=r"\s+", index_col=0)
    np.testing.assert_allclose(rendered.values, reference.values, atol=0.000001, rtol=0)
    annotation = args.dataset / "annotations/list.txt"
    ids = [line.split()[0] for line in annotation.read_text().splitlines()
           if line.strip() and not line.startswith("#")]
    assert len(ids) == len(set(ids)) == 7349
    dimensions = []
    for name in ids:
        with Image.open(args.dataset / "images" / (name + ".jpg")) as image:
            dimensions.append(image.size)
    frame = pd.DataFrame(dimensions, columns=["width", "height"])
    frame["aspect_ratio"] = frame.width / frame.height
    measured = frame.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    np.testing.assert_allclose(measured.values, reference.values, atol=1e-9, rtol=1e-12)
    output = ROOT / "report/ky_assets"
    output.mkdir(exist_ok=True)
    (output / "eda_dimension_statistics.csv").write_bytes(csv_bytes)
    headers = ["Statistic", "Width (px)", "Height (px)", "Width / height"]
    rows = [[label, f"{row.width:.2f}", f"{row.height:.2f}",
             f"{row.aspect_ratio:.4f}"] for label, row in reference.iterrows()]
    markdown = ("# Original image dimensions\n\n"
                "Source statistics for Bảng 2-3. Đặc điểm của 7.349 ảnh có nhãn.\n\n"
                "| " + " | ".join(headers) + " |\n|---|---:|---:|---:|\n"
                + "".join("| " + " | ".join(row) + " |\n" for row in rows)
                + "\n7,349 annotated images, before resize/crop or EXIF transpose. "
                "Width and height statistics are marginal summaries, not paired image sizes. "
                "`std` is the sample standard deviation (ddof=1); count is an image count. "
                "Full precision is retained in the CSV.\n\n"
                f"Source: [Ky EDA, PR11](https://github.com/duyenphung206-sudo/"
                f"vit-oxford-pets/blob/{EDA_REF}/notebooks/01_eda.ipynb). "
                "Independently remeasured all image headers and matched every statistic.\n")
    (output / "eda_dimension_statistics.md").write_text(markdown, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)
    for (row, col), item in table.get_celld().items():
        item.set_edgecolor("#d8dee5")
        if row == 0:
            item.set_facecolor("#e8edf3")
            item.set_text_props(weight="bold")
        if col == 0:
            item.set_text_props(ha="left")
    fig.suptitle("Oxford-IIIT Pet: original image dimensions", fontsize=15, y=.98)
    fig.text(.5, .025, "7,349 annotated images; before resizing. Full precision in CSV.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(output / "eda_dimension_statistics.png", dpi=300)
    fig.savefig(output / "eda_dimension_statistics.svg")
    plt.close(fig)
    svg = output / "eda_dimension_statistics.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
                   encoding="utf-8")
    evidence = {
        "source_pr": 11, "source_commit": EDA_REF,
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        "images_measured": len(dimensions), "notebook_output_matches_csv": True,
        "actual_image_headers_match_all_csv_statistics": True,
        "dimensions": "Original width and height, before EXIF transpose or resizing",
        "caption_status": "RESOLVED: Bảng 2-3. Đặc điểm của 7.349 ảnh có nhãn",
    }
    (output / "eda_dimension_provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
