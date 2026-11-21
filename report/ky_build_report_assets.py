"""Build only Ky's assigned report assets from real images and verified experiment logs.

Run from the repository root with --dataset pointing to images/ and annotations/.
The supplied group PDF is read-only; no shared code or other member's figures are edited.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from src.data import transforms_for
from src.ky_experiments import CONFIGS, verified_run
from src.summarize import plot_curves, read_log
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
ASSETS = ROOT / "report/ky_assets"
CAPTIONS = {
    "1.1": "Sáu cặp giống có ngoại hình gần nhau trong tập Oxford-IIIT Pet",
    "2.1": "Số ảnh trainval của 37 lớp",
    "2.2": "Phân bố tỉ lệ khung hình và kích thước ảnh",
    "2.3": "Ví dụ tăng cường dữ liệu trên một ảnh huấn luyện",
    "3.4": "Biểu đồ huấn luyện của thực nghiệm E3 (tinh chỉnh một phần)",
    "3.6": "Val accuracy theo learning rate, có và không có tăng cường dữ liệu",
    "3.10": "Giao diện ứng dụng demo khi chưa tải ảnh",
    "3.11": "Kết quả dự đoán và bản đồ attention trên ứng dụng demo",
    "2-3": "Đặc điểm của 7.349 ảnh có nhãn",
    "3-9": "Kết quả của thực nghiệm E3 trên tập kiểm định",
    "3-12": "Kết quả của thực nghiệm E5 trên tập kiểm định",
}


def save_figure(fig, relative):
    path = FIGURES / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    svg = path.with_suffix(".svg")
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def table(number, headers, rows, widths, heights=None, note=""):
    stem = ASSETS / ("table_" + number.replace("-", "_"))
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    markdown = f"# Bảng {number}. {CAPTIONS[number]}\n\n"
    markdown += "| " + " | ".join(s.replace("\n", " ") for s in headers) + " |\n"
    markdown += "|" + "---|" * len(headers) + "\n"
    markdown += "".join("| " + " | ".join(str(s).replace("\n", " ") for s in row) + " |\n" for row in rows)
    stem.with_suffix(".md").write_text(markdown + "\n" + note + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 5 if number == "2-3" else 3.4))
    ax.axis("off")
    cells = [
        [
            "\n".join(textwrap.wrap(str(value), 64 if number == "2-3" and col == 1 else 30))
            for col, value in enumerate(row)
        ]
        for row in rows
    ]
    item = ax.table(cellText=cells, colLabels=headers, colWidths=widths, loc="center", cellLoc="center")
    item.auto_set_font_size(False)
    item.set_fontsize(11)
    item.scale(1, 2.0)
    for (row, _col), cell in item.get_celld().items():
        cell.set_edgecolor("#b8c4d0")
        if row == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold")
            cell.set_height(0.20 if number != "2-3" else 0.12)
        elif heights:
            cell.set_height(heights[row - 1])
    fig.suptitle(f"Bảng {number}. {CAPTIONS[number]}", fontsize=13, y=0.97)
    fig.text(0.5, 0.025, note, fontsize=8.5, ha="center", wrap=True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report-pdf", required=True, type=Path)
    args = parser.parse_args()
    if (
        hashlib.sha256(args.report_pdf.read_bytes()).hexdigest()
        != "2a37280287e242873c06756a24306678b3bee276d072333cbb2360666f10a34c"
    ):
        raise ValueError("Confirm captions against the changed source PDF before rebuilding")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "ky/e3-e5-demo-report":
        raise RuntimeError("Build report assets only on Ky's work branch")
    ASSETS.mkdir(exist_ok=True)
    rows, inventory = [], []
    for line in (args.dataset / "annotations/trainval.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            identifier, label, species, *_ = line.split()
            rows.append(
                {
                    "id": identifier,
                    "label": int(label) - 1,
                    "species": int(species),
                    "name": identifier.rsplit("_", 1)[0],
                }
            )
    with (ROOT / "data/splits/train.csv").open(newline="") as stream:
        train = list(csv.DictReader(stream))
    train_ids = {row["id"] for row in train}
    ids = [
        s.split()[0]
        for s in (args.dataset / "annotations/list.txt").read_text().splitlines()
        if s.strip() and not s.startswith("#")
    ]
    for identifier in ids:
        with Image.open(args.dataset / "images" / (identifier + ".jpg")) as image:
            image.load()
            inventory.append((image.width, image.height, image.mode))
    assert len(inventory) == len(set(ids)) == 7349
    dimensions = np.array([(w, h) for w, h, _ in inventory])
    with (ASSETS / "eda_dimension_statistics.csv").open() as stream:
        eda = {row[0]: [float(value) for value in row[1:3]] for row in list(csv.reader(stream))[1:]}
    for statistic, measured in [("min", dimensions.min(0)), ("max", dimensions.max(0)), ("mean", dimensions.mean(0))]:
        np.testing.assert_allclose(measured, eda[statistic], atol=1e-9, rtol=0)
    modes = Counter(mode for _, _, mode in inventory)
    assert modes == {"RGB": 7346, "RGBA": 3}
    ratios = dimensions[:, 0] / dimensions[:, 1]
    orientation = {
        "landscape": int((ratios > 1.05).sum()),
        "portrait": int((ratios < 0.95).sum()),
        "near_square": int(((ratios >= 0.95) & (ratios <= 1.05)).sum()),
    }
    assert orientation == {"landscape": 4775, "portrait": 2263, "near_square": 311}
    counts = Counter(row["name"] for row in rows)
    species = {row["name"]: row["species"] for row in rows}
    assert len(counts) == 37 and sum(counts.values()) == 3680
    ordered = sorted(counts, key=lambda name: (-counts[name], name.lower()))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(
        range(37),
        [counts[name] for name in ordered],
        color=["#5a75b8" if species[name] == 1 else "#c87b39" for name in ordered],
    )
    ax.axhline(100, linestyle="--", color="#444444", linewidth=1)
    ax.set_xticks(range(37), ordered, rotation=90, fontsize=9)
    ax.set_ylabel("Số ảnh trainval")
    ax.set_ylim(0, 110)
    ax.plot([], [], color="#5a75b8", linewidth=8, label="Mèo")
    ax.plot([], [], color="#c87b39", linewidth=8, label="Chó")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "eda/class_distribution")
    bins = np.round(np.arange(0, np.ceil(ratios.max() / 0.05) * 0.05 + 0.05, 0.05), 2)
    histogram, _ = np.histogram(ratios, bins=bins)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    left.hist(ratios, bins=bins, color="#5279b0", edgecolor="white", linewidth=0.3)
    for limit in (0.95, 1.05):
        left.axvline(limit, linestyle="--", color="#b04c38", linewidth=1.3)
    left.set(xlabel="Tỉ lệ rộng / cao", ylabel="Số ảnh", title="(a) Tỉ lệ khung hình; độ rộng cột 0,05")
    right.scatter(dimensions[:, 0], dimensions[:, 1], s=5, alpha=0.22, color="#5279b0", rasterized=True)
    limit = max(dimensions.max(), 500)
    right.plot([0, limit], [0, limit], "--", color="#666666", linewidth=1)
    right.set(xlabel="Chiều rộng gốc (pixel)", ylabel="Chiều cao gốc (pixel)", title="(b) Mỗi điểm là một ảnh")
    for ax in (left, right):
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "eda/aspect_ratio")
    names = [
        "British_Shorthair",
        "Russian_Blue",
        "Birman",
        "Ragdoll",
        "Egyptian_Mau",
        "Bengal",
        "american_pit_bull_terrier",
        "staffordshire_bull_terrier",
        "samoyed",
        "great_pyrenees",
        "basset_hound",
        "beagle",
    ]
    rng = np.random.default_rng(42)
    selected = [
        str(rng.choice(sorted(row["id"] for row in rows if row["name"] == name and row["id"] in train_ids)))
        for name in names
    ]
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.0))
    for ax, identifier, name in zip(axes.flat, selected, names, strict=True):
        with Image.open(args.dataset / "images" / (identifier + ".jpg")) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title("\n".join(textwrap.wrap(name.replace("_", " ").title(), 20)), fontsize=10)
        ax.axis("off")
    fig.tight_layout(h_pad=2)
    save_figure(fig, "eda/example_pairs")
    augmentation_id = sorted(row["id"] for row in rows if row["name"] == "Abyssinian" and row["id"] in train_ids)[0]
    with Image.open(args.dataset / "images" / (augmentation_id + ".jpg")) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    torch.manual_seed(42)
    tensors = [transforms_for(False)(original)] + [transforms_for(True)(original) for _ in range(4)]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4))
    for index, (ax, tensor) in enumerate(zip(axes, tensors, strict=True)):
        ax.imshow(TF.to_pil_image((tensor * std + mean).clamp(0, 1)))
        ax.set_title("Đánh giá" if index == 0 else f"Huấn luyện {index}", fontsize=12)
        ax.axis("off")
    fig.text(0.5, 0.035, f"Ảnh train: {augmentation_id}.jpg | Seed 42 | Ảnh hiển thị sau bỏ chuẩn hóa", ha="center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, "augmentation_tv2")
    e3, _ = verified_run(ROOT / "results/e3_vit_partial/seed42", load_config(ROOT / "configs/e3_vit_partial.yaml"))
    plot_curves(ROOT / "results/e3_vit_partial", FIGURES / "e3_curves.png")
    runs = {}
    for name in ["E5-LR1", "E5-LR2", "E5-LR3", "E5-LR4", "E5-AUG"]:
        cfg = load_config(ROOT / "configs" / CONFIGS[name])
        metrics, _ = verified_run(ROOT / "results" / cfg["experiment"] / "seed42", cfg)
        runs[name] = metrics
    x = [0.001, 0.003, 0.01, 0.03]
    y = [runs[f"E5-LR{i}"]["best_val_acc"] * 100 for i in range(1, 5)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, "o-", color="#3567a0", label="Có tăng cường")
    off = runs["E5-AUG"]["best_val_acc"] * 100
    ax.scatter([0.003], [off], marker="^", s=85, color="#c8752f", label="Không tăng cường", zorder=5)
    for index, (rate, accuracy) in enumerate(zip(x, y, strict=True)):
        ax.annotate(
            f"{accuracy:.2f}%",
            (rate, accuracy),
            xytext=(0, -20 if index == 1 else 10),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax.annotate(f"{off:.2f}%", (0.003, off), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(x, ["0,001", "0,003", "0,01", "0,03"])
    ax.set(xlabel="Learning rate (thang log)", ylabel="Validation accuracy (%)", ylim=(90, 97))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "e5_lr_aug")
    table(
        "2-3",
        ["Đặc điểm", "Giá trị"],
        [
            ["Chế độ màu", "RGB: 7.346; RGBA: 3; ảnh xám/CMYK: 0; lỗi giải mã: 0"],
            [
                "Hướng ảnh (r = rộng/cao)",
                "Ngang (r > 1,05): 4.775 (65,0%); dọc (r < 0,95): 2.263 (30,8%); gần vuông: 311 (4,2%)",
            ],
            [
                "Chiều rộng nhỏ nhất / lớn nhất / trung bình",
                f"{int(eda['min'][0])} / {int(eda['max'][0]):,} / {eda['mean'][0]:.2f} pixel".translate(
                    str.maketrans({",": ".", ".": ","})
                ),
            ],
            [
                "Chiều cao nhỏ nhất / lớn nhất / trung bình",
                f"{int(eda['min'][1])} / {int(eda['max'][1]):,} / {eda['mean'][1]:.2f} pixel".translate(
                    str.maketrans({",": ".", ".": ","})
                ),
            ],
        ],
        [0.34, 0.66],
        [0.18, 0.24, 0.18, 0.18],
        "Kích thước gốc trước EXIF/resize/crop; thống kê khớp notebook EDA gốc.",
    )
    table(
        "3-9",
        ["Epoch tốt nhất", "Val loss", "Val accuracy\n(%)", "Thời gian / epoch\n(s)", "Bộ nhớ GPU đỉnh\n(MiB)"],
        [
            [
                e3["best_epoch"],
                f"{e3['best_val_loss']:.6f}",
                f"{e3['best_val_acc'] * 100:.4f}",
                e3["mean_epoch_time_s"],
                e3["peak_gpu_memory_mb"],
            ]
        ],
        [0.17, 0.16, 0.22, 0.23, 0.22],
        note="E3, seed 42, Tesla T4. Đơn vị bộ nhớ thực trong log là MiB; 2.140,6 MiB = 2.244,58 MB (SI).",
    )
    e5_rows = [
        [
            rate,
            "có",
            runs[name]["best_epoch"],
            f"{runs[name]['best_val_loss']:.6f}",
            f"{runs[name]['best_val_acc'] * 100:.4f}",
        ]
        for rate, name in [("0,001", "E5-LR1"), ("0,003 (E4 seed42)", "E5-LR2"), ("0,01", "E5-LR3"), ("0,03", "E5-LR4")]
    ]
    e5_rows.append(
        ["0,003", "không", runs["E5-AUG"]["best_epoch"], f"{runs['E5-AUG']['best_val_loss']:.6f}", f"{off:.4f}"]
    )
    table(
        "3-12",
        ["Learning rate", "Tăng cường dữ liệu", "Epoch tốt nhất", "Val loss", "Val accuracy (%)"],
        e5_rows,
        [0.23, 0.24, 0.18, 0.17, 0.18],
        note="Tất cả seed42, 10 epoch; chọn checkpoint theo val loss nhỏ nhất. E5-LR2 tái dùng E4.",
    )
    top_bins = sorted(range(len(histogram)), key=lambda k: -histogram[k])[:3]
    evidence = {
        "source_pdf_sha256": hashlib.sha256(args.report_pdf.read_bytes()).hexdigest(),
        "captions": CAPTIONS,
        "pdf_pages": {
            "1.1": 15,
            "2.1": 18,
            "2.2": 19,
            "2.3": 20,
            "3.4": 35,
            "3.6": 38,
            "3.10": 43,
            "3.11": 43,
            "2-3": 19,
            "3-9": 35,
            "3-12": 38,
        },
        "train_pair_ids": selected,
        "augmentation_train_id": augmentation_id,
        "seed": 42,
        "images_decoded": len(inventory),
        "image_modes": dict(modes),
        "orientation": orientation,
        "histogram_bin_width": 0.05,
        "largest_histogram_bins": [
            {"left": float(bins[k]), "right": float(bins[k + 1]), "count": int(histogram[k])} for k in top_bins
        ],
        "class_trainval_counts": {name: counts[name] for name in ordered},
        "source_annotations_sha256": {
            s: hashlib.sha256((args.dataset / "annotations" / (s + ".txt")).read_bytes()).hexdigest()
            for s in ["list", "trainval", "test"]
        },
        "e3_log": read_log(ROOT / "results/e3_vit_partial/seed42/log.csv"),
        "demo_screenshots": {
            name: {
                "path": f"figures/{name}.png",
                "sha256": hashlib.sha256((FIGURES / (name + ".png")).read_bytes()).hexdigest(),
                "capture": (
                    "Genuine browser screenshot, 1280x720, E4 seed44 CPU --attention; "
                    "JPEG pixels converted losslessly to PNG"
                ),
            }
            for name in ["demo_ui_empty", "demo_ui_result"]
            if (FIGURES / (name + ".png")).exists()
        },
    }
    (ASSETS / "report_asset_provenance.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pair_ids": selected,
                "augmentation_id": augmentation_id,
                "histogram_peaks": evidence["largest_histogram_bins"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
