# Kỳ — hình và bảng đã hoàn tất theo PDF nhóm

Nguồn đối chiếu: bản PDF người dùng cung cấp ngày 21/11/2026, 46 trang. Không sửa tài liệu gốc hoặc phần của thành viên khác.
PDF SHA256: `2a37280287e242873c06756a24306678b3bee276d072333cbb2360666f10a34c`. Caption đối chiếu ở nội dung từng trang, không dựa vào số trang trong mục lục chưa cập nhật.

| Loại | Số | Caption chính xác | Tệp | Trang PDF nguồn | Trạng thái |
|---|---|---|---|---:|---|
| Hình | 1.1 | Sáu cặp giống có ngoại hình gần nhau trong tập Oxford-IIIT Pet | `figures/eda/example_pairs.png` | 15 | DONE |
| Hình | 2.1 | Số ảnh trainval của 37 lớp | `figures/eda/class_distribution.png` | 18 | DONE |
| Hình | 2.2 | Phân bố tỉ lệ khung hình và kích thước ảnh | `figures/eda/aspect_ratio.png` | 19 | DONE |
| Hình | 2.3 | Ví dụ tăng cường dữ liệu trên một ảnh huấn luyện | `figures/augmentation_tv2.png` | 20 | DONE |
| Hình | 3.4 | Biểu đồ huấn luyện của thực nghiệm E3 (tinh chỉnh một phần) | `figures/e3_curves.png` | 35 | DONE |
| Hình | 3.6 | Val accuracy theo learning rate, có và không có tăng cường dữ liệu | `figures/e5_lr_aug.png` | 38 | DONE |
| Hình | 3.10 | Giao diện ứng dụng demo khi chưa tải ảnh | `figures/demo_ui_empty.png` | 43 | DONE |
| Hình | 3.11 | Kết quả dự đoán và bản đồ attention trên ứng dụng demo | `figures/demo_ui_result.png` | 43 | DONE |
| Bảng | 2-3 | Đặc điểm của 7.349 ảnh có nhãn | `report/ky_assets/table_2_3.md/.csv/.png/.svg` | 19 | DONE |
| Bảng | 3-9 | Kết quả của thực nghiệm E3 trên tập kiểm định | `report/ky_assets/table_3_9.md/.csv/.png/.svg` | 35 | DONE |
| Bảng | 3-12 | Kết quả của thực nghiệm E5 trên tập kiểm định | `report/ky_assets/table_3_12.md/.csv/.png/.svg` | 38 | DONE |

## Nguồn và tái lập

- `report/ky_build_report_assets.py`: các hình EDA, tăng cường, E3/E5 và ba bảng; chỉ ghi file thuộc Kỳ. Dùng ảnh gốc chỉ đọc, annotation chính thức và log/metrics đã xác minh.
- `report/ky_package_report.py`: đóng gói 11 mục vào `report/KY_REPORT_FIGURES_TABLES.pdf`; cần thêm `reportlab` nếu môi trường chưa có.
- `report/ky_assets/report_asset_provenance.json`: caption, trang PDF nguồn, ID ảnh train, seed42, histogram/counts và SHA256.
- `docs/KY_ATTENTION_VERIFICATION.json`: suy luận/attention thật E4 seed44 CPU; forward gốc không đổi, hook được gỡ sau khi chạy.
- Hai ảnh demo là screenshot trình duyệt thật 1280×720; không dựng lại giao diện bằng hình giả. Pixel JPEG được chuyển sang PNG, không tăng độ phân giải giả.
- Bảng 2-3 lấy đúng min/max/mean EDA gốc và đối chiếu với 7.349 ảnh; ba ảnh RGBA và 0 lỗi giải mã được kiểm tra lại.
- Hình 3.4 xuất bằng `src.summarize.plot_curves` hiện có, 200dpi; các biểu đồ mới khác có PNG300dpi/SVG. Ảnh demo giữ độ phân giải capture gốc.

```powershell
python report/ky_build_report_assets.py --dataset <images-and-annotations-folder> --report-pdf <provided-PDF>
python app/app.py --checkpoint <verified-E4-seed44.pt> --device cpu --port 7861 --attention
python report/ky_package_report.py
```

Để chụp lại hình 3.10, mở demo attention và chụp vùng trang khi chưa tải ảnh. Với hình 3.11, tải `Abyssinian_201.jpg` (dòng đầu annotation test, chọn trước suy luận), gửi và chụp đủ Top-5 cùng overlay. Không chọn ảnh dựa trên đúng/sai sau khi thử nhiều ảnh.

## Nội dung có thể điền vào phần của Kỳ

E3: train loss epoch1 = 3.3811; epoch10 = 0.2093. Val loss nhỏ nhất ở epoch10 = 0.237091, val accuracy94.9728%. Train/val accuracy epoch10 chênh 1.46 điểm phần trăm. Chưa đủ bằng chứng về E2 nên không tự điền so sánh hoặc phần khoảng cách E2–E4.

EDA: ba khoảng histogram đông nhất là [1,30; 1,35): 1.963 ảnh, [1,50; 1,55): 1.415 ảnh và [0,65; 0,70): 980 ảnh. Đây là số đếm theo cột rộng0,05, không phải mọi ảnh trong cột có đúng cùng một tỉ lệ.

E5: trong bốn learning rate có tăng cường, LR0,01 đạt val accuracy cao nhất95.6522%, chỉ hơn LR0,003 một ảnh validation (0.1359 điểm). LR0,03 thấp hơn LR0,01 khoảng0.5435 điểm, chưa đủ để kết luận phá vỡ đặc trưng tiền huấn luyện. Tắt tăng cường tại LR0,003 đạt95.9239%, cao hơn có tăng cường3 ảnh (0.4076 điểm). Đây là quan sát seed42; không suy rộng ý nghĩa thống kê/độ ổn định nhiều seed và không ghép LR0,01 với tắt tăng cường khi tổ hợp đó chưa chạy.

Demo: ảnh test Abyssinian_201.jpg, nhãn thật Abyssinian, Top-1 Abyssinian với softmax0.394977 (khoảng39,50%; UI làm tròn phần trăm nguyên nên hiển thị39%). Các lớp tiếp theo là Bengal, Egyptian_Mau, Russian_Blue, Maine_Coon. Overlay dùng đúng vùng model crop224×224; có vùng sáng cả trên con vật và bàn/nền, không coi rollout là phân đoạn hay bằng chứng nhân quả.

## Lưu ý cho Duyên khi ghép báo cáo

- Bảng 3-9: số liệu peak_gpu_memory_mb thực chất là MiB do trainer chia1024². Tệp bảng dùng nhãn MiB đúng với2140.6; nếu giữ MB theo SI trong Word thì dùng2244.58MB. Không đổi trainer/log.
- Hình 3.11 cần chạy demo với `--attention`; mặc định Top-5 cũ vẫn được giữ. Method khớp công thức(3.6) trong PDF: trung bình head, cộng I rồi chia2, tích12 ma trận và lấy hàngCLS.
- Bản PDF gốc dùng số trang trong danh mục khác với trang nội dung ở một số hình. Khi ghép Word, cập nhật lại các trường/danh mục hình-bảng.
- Chỉ cung cấp phần Kỳ; không tự điền E1/E2, ma trận nhầm lẫn, phân tích lỗi, hình3.7–3.9 hoặc các bảng của Tâm/Trinh.
- Duyên chèn/duyệt bộ asset và quyết định merge PR33. Agent không sửa Word/PDF chung và không tự merge.

Ảnh minh họa lấy từ Oxford-IIIT Pet (Parkhi et al., 2012); thông tin nguồn/giấy phép theo `docs/data_card.md`.
