"""Package Ky's 8 figures and 3 tables as a standalone review PDF; never edit the group PDF."""

import csv
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib.font_manager import findfont
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report/ky_assets"
FIGURES = {
    "1.1": "eda/example_pairs.png",
    "2.1": "eda/class_distribution.png",
    "2.2": "eda/aspect_ratio.png",
    "2.3": "augmentation_tv2.png",
    "3.4": "e3_curves.png",
    "3.6": "e5_lr_aug.png",
    "3.10": "demo_ui_empty.png",
    "3.11": "demo_ui_result.png",
}


def main():
    captions = json.loads((ASSETS / "report_asset_provenance.json").read_text(encoding="utf-8"))["captions"]
    pdfmetrics.registerFont(TTFont("KySans", findfont("DejaVu Sans")))
    pdfmetrics.registerFont(TTFont("KyBold", findfont("DejaVu Sans:weight=bold")))
    style = ParagraphStyle("body", fontName="KySans", fontSize=10, leading=15)
    bold = ParagraphStyle("heading", parent=style, fontName="KyBold", fontSize=12, leading=17, alignment=1)
    output = ROOT / "report/KY_REPORT_FIGURES_TABLES.pdf"
    width, height = landscape(A4)
    canvas = Canvas(str(output), pagesize=(width, height))
    canvas.setTitle("Hình và bảng báo cáo của Nguyễn Ngọc Kỳ - ViT Oxford-IIIT Pet")
    canvas.setAuthor("Nguyễn Ngọc Kỳ")
    entries = [("Hình", number, path) for number, path in FIGURES.items()]
    entries += [("Bảng", number, f"table_{number.replace('-', '_')}.csv") for number in ["2-3", "3-9", "3-12"]]
    for page, (kind, number, path) in enumerate(entries, 1):
        canvas.setFont("KySans", 9)
        canvas.drawString(
            30, height - 24, "Nguyễn Ngọc Kỳ | ViT Oxford-IIIT Pet | Tệp riêng để duyệt/chèn vào báo cáo nhóm"
        )
        canvas.drawRightString(width - 30, height - 24, f"{page} / {len(entries)}")
        caption = Paragraph(escape(f"{kind} {number}. {captions[number]}"), bold)
        _cw, caption_height = caption.wrap(width - 60, 60)
        if kind == "Hình":
            image = ROOT / "figures" / path
            with Image.open(image) as bitmap:
                image_width, image_height = bitmap.size
                if number in {"1.1", "2.3"}:
                    review_image = bitmap.convert("RGB")
                    review_image.thumbnail((2600, 1800), Image.Resampling.LANCZOS)
                    jpeg = io.BytesIO()
                    review_image.save(jpeg, format="JPEG", quality=94)
                    packaged_image = ImageReader(io.BytesIO(jpeg.getvalue()))
                else:
                    packaged_image = str(image)
            scale = min((width - 60) / image_width, (height - 125 - caption_height) / image_height)
            shown_width, shown_height = image_width * scale, image_height * scale
            canvas.drawImage(
                packaged_image,
                (width - shown_width) / 2,
                85 + (height - 125 - caption_height - shown_height) / 2,
                width=shown_width,
                height=shown_height,
            )
            caption.drawOn(canvas, 30, 58)
            source = f"figures/{path}"
        else:
            caption.drawOn(canvas, 30, height - 68)
            with (ASSETS / path).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            cells = [[Paragraph(escape(value).replace("\n", "<br/>"), style) for value in row] for row in rows]
            proportions = (
                [0.34, 0.66]
                if number == "2-3"
                else ([0.17, 0.16, 0.22, 0.23, 0.22] if number == "3-9" else [0.23, 0.24, 0.18, 0.17, 0.18])
            )
            table = Table(cells, colWidths=[p * (width - 70) for p in proportions], repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a8b4c0")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ]
                )
            )
            _tw, table_height = table.wrap(width - 70, height - 130)
            table.drawOn(canvas, 35, height - 100 - table_height)
            note = (
                "Bộ nhớ trong log: 2.140,6 MiB; tương đương 2.244,58 MB (SI). Không đổi số liệu nguồn."
                if number == "3-9"
                else "Số liệu thật; xem CSV và hồ sơ nguồn tương ứng trong report/ky_assets."
            )
            paragraph = Paragraph(escape(note), style)
            paragraph.wrap(width - 70, 45)
            paragraph.drawOn(canvas, 35, 75)
            source = f"report/ky_assets/{path}"
        canvas.setFont("KySans", 8)
        canvas.drawString(30, 30, source)
        canvas.showPage()
    canvas.save()
    print(f"Created {len(entries)}-page review PDF: {output}")


if __name__ == "__main__":
    main()
