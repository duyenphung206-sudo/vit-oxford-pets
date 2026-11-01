# T11 — Audit và chạy lại

Owner: Edward2106. Reviewer: Bambizled. Issue #1.

| Hạng mục | Đã có trong project cá nhân | Tái dùng | Còn thiếu / hành động |
|---|---|---|---|
| Phân bố 37 lớp | class_counts.csv | nguyên bản làm đối chứng | bảng official split và biểu đồ mới |
| Chó/mèo | chưa có bảng EDA | không | đọc species ID chính thức |
| Kích thước | audit.py chưa lưu | vòng Image.open/load | đo toàn bộ ảnh, bảng phân vị và histogram |
| Aspect ratio | chưa có | không | width/height, ngưỡng .95/1.05 |
| Lưới 37 lớp | chưa có | không | seed 42, ảnh trainval thật |
| Bất thường | audit.json, audit.py | kiểm tra mode/decode, audit.json | thêm ID bất thường, missing, warnings, metadata |
| Notebook EDA | chỉ có notebook Colab train | không | notebooks/01_eda.ipynb |
| Figures EDA | chỉ thấy figures đánh giá train | không | 6 PNG mới |

Đã kiểm tra `vit_pets/audit.py`, `runs/audit_20261025/{audit.json,class_counts.csv,splits.json}` và bản `reports/colab_final/artifacts/audit/`. Hai class_counts.csv trùng hash; audit.json và splits.json khác byte nên không coi hai bộ hoàn toàn giống nhau. Chọn bộ local audit_20261025 nhất quán làm tham chiếu, không sao chép splits cá nhân sang nhóm. Hash nguồn được lưu trong reference/provenance.json. Không tìm thấy Downloads/experiment-results tại thời điểm audit. Source-only chứa cùng chức năng audit, không có thống kê kích thước hay EDA notebook riêng. Không dùng số liệu train hoặc checkpoint.

## Chạy lại trên máy khác

```powershell
python -m pip install pillow pandas matplotlib nbformat nbclient ipykernel
$env:OXFORD_PET_ROOT = 'D:\datasets\oxford-iiit-pet'
python -c "import nbformat; from nbclient import NotebookClient; p='notebooks/01_eda.ipynb'; n=nbformat.read(p,4); NotebookClient(n,timeout=600,resources={'metadata':{'path':'.'}}).execute(); nbformat.write(n,p)"
```

Hoặc mở notebook và Restart & Run All. Không cần GPU/checkpoint. Mặc định dataset nằm ở data/oxford-iiit-pet. Notebook chỉ đọc dataset. Kết quả ghi figures/eda và results/eda. Xem summary.json để kiểm chứng số lượng, anomalies.csv để tra ảnh non-RGB/lỗi, inventory.json cho các file ngoài annotation. Kích thước đo trước EXIF, ảnh hiển thị áp dụng EXIF trên bản sao.

Lưới mẫu là sản phẩm minh họa EDA được yêu cầu, không chép từng ảnh dataset vào Git. Không phân tích near-duplicate, không đánh giá chất lượng nhãn thủ công, không train hoặc tạo split của nhóm. File Excel phân công không được cung cấp nên chưa cập nhật trạng thái trong Excel.
