# KY_PROGRESS — Nguyễn Ngọc Kỳ

Cập nhật: 2026-11-21 (Asia/Saigon). Branch duy nhất: `ky/e3-e5-demo-report`.

`DONE` = đã làm và kiểm chứng; `PENDING` = chưa hoàn tất; `BLOCKED` = thiếu đầu vào/quyền;
`FAILED` = đã thử nhưng không đạt. Không có số liệu huấn luyện giả trong báo cáo.

## 1. Công việc đã hoàn thành

- DONE: clone đúng repository vào Downloads, kiểm tra README/giao thức/code/config/log/split.
- DONE: 6 config E3/E5 kế thừa E4, không sửa config hay kết quả E4.
- DONE: runner riêng `src/ky_experiments.py`, chỉ seed42, chặn CPU train thật và ghi đè run dở.
- DONE: notebook Colab 07/08, schema nbformat hợp lệ, toàn bộ code cell parse được.
- DONE: kiểm chứng chính xác head + final LayerNorm + encoder blocks 8–11 trainable.
- DONE: bảng CSV/JSON E5 và biểu đồ PNG 300 dpi/SVG, nguồn tạo hình tái lập được.
- DONE: tạo branch riêng trên GitHub bằng giao diện tài khoản Edward2106.
- DONE: Colab đã cấp Tesla T4, CUDA True; đã bắt đầu E3 trong phiên thật.

## 2. Công việc chưa hoàn tất

- PENDING: E3 và các run E5-LR1/LR3/AUG đang được thực thi/theo dõi trên Colab.
- PENDING: E5-LR4 tùy chọn, không nằm trong danh sách mặc định.
- BLOCKED: demo E4 thiếu checkpoint đúng của nhóm.
- BLOCKED: 8 hình và 3 bảng thiếu đề cương/caption; bảng 2-3 cần notebook EDA gốc.
- PENDING: push nội dung và tạo PR để Duyên review; không merge.

## 3. E3

Config `configs/e3_vit_partial.yaml`: partial, last_k=4, các điều kiện còn lại kế thừa E4.
Trainable/total đã kiểm chứng: 28,381,477 / 85,827,109.
Smoke test CPU 1 epoch bằng dữ liệu giả đã chạy thành công, artifact nằm riêng ngoài repository
trong `Downloads/.ky-tools/smoke-e3-*`; không dùng các số đó làm kết quả nghiên cứu.
Kết quả thật: PENDING, chỉ công bố sau khi lấy được log/config/metrics đầy đủ và xác minh.

## 4. E5

| Run | LR | Augmentation | Val accuracy | Val loss | Thời gian train (s) | Epoch | Seed | Trạng thái |
|---|---:|---|---:|---:|---:|---:|---:|---|
| E5-LR1 | 0.001 | true | | | | | 42 | PENDING |
| E5-LR2 | 0.003 | true | 0.955163 | 0.311232 | 538.6 | 10 | 42 | DONE: tái dùng E4 seed42 |
| E5-LR3 | 0.01 | true | | | | | 42 | PENDING |
| E5-LR4 | 0.03 | true | | | | | 42 | PENDING, tùy chọn |
| E5-AUG | 0.003 | false | | | | | 42 | PENDING |

Thời gian là tổng epoch_time_s trong log (đã làm tròn), không gồm download/setup.
Accuracy là tỷ lệ 0–1. Không dùng trung bình 3 seed hoặc tập test để chọn siêu tham số.

## 5. Demo

BLOCKED: `waiting for E4 checkpoint`. Chi tiết ở `docs/KY_DEMO_CHECKPOINT_STATUS.md`.
Không sửa giao diện demo. Checkpoint cá nhân hiện có không phải E4 của nhóm.
Release `v1.0.0` trả HTTP 404; giao diện GitHub cũng hiển thị chưa có release.
Shared trainer lưu raw state_dict, demo đòi bundle metadata/classes/model: ghi nhận lỗi giao tiếp,
không sửa trainer dùng chung. CLI: `python app/app.py --checkpoint <E4-bundle.pt> --device cpu --port 7860`.

## 6. 8 hình và 3 bảng

Tất cả BLOCKED do thiếu caption/đề cương: hình 1.1, 2.1, 2.2, 2.3, 3.4, 3.6, 3.10, 3.11;
bảng 2-3, 3-9, 3-12. Theo dõi từng mục ở `report/KY_FIGURE_TABLE_STATUS.md`.
Không tự gán tên hình hoặc dùng kích thước crop 224 thay cho thống kê ảnh gốc EDA.

## 7. Kiểm thử và bảo vệ phạm vi

- DONE: baseline pytest 23 passed; sau bổ sung test Kỳ: 31 passed.
- DONE: E3 fake-data end-to-end 1 epoch, CPU; checkpoint giả chỉ dùng smoke và không commit.
- DONE: Ruff chỉ trên 2 file Python mới, nbformat schema và AST notebook.
- DONE: kiểm tra split 2944/736, nhãn 37 lớp, train/val không trùng; split không chỉnh sửa.
- DONE: E4 seed42 config/log/metrics khớp, best epoch theo minimum validation loss.
- DONE: không sửa src/train.py, engine.py, optim.py, data.py, models/factory.py,
  README chính, workflow, E1/E2/E4 hoặc sản phẩm thành viên khác.
- DONE: main và origin/main giữ commit gốc `4d5721bc486dc76930ee49fb07bb8be1917f09e6`.

## 8. Lỗi/giới hạn còn tồn tại

- FAILED: Git credential helper trong terminal không chạy được; thiếu xác thực để push.
- FAILED: connector GitHub tạo branch/blob trả 403 Resource not accessible by integration.
  Browser GitHub đã đăng nhập và tạo được branch riêng; chưa publish file.
- BLOCKED: chưa có checkpoint E4 và đề cương/EDA notebook gốc.
- Python tải vào Downloads bị Windows Application Control chặn; dùng interpreter đóng gói sẵn,
  dependency/cache/temp đặt trong Downloads. Không đổi cấu hình bảo mật Windows.
- Notebook 07/08 chưa Restart & Run all trên Colab; phiên execution riêng dùng cùng config/runner,
  điều này không được tính là đã xác minh đầy đủ hai notebook.

## 9. Commit và PR

Branch: https://github.com/duyenphung206-sudo/vit-oxford-pets/tree/ky/e3-e5-demo-report
Commit nội dung: PENDING. Pull Request URL: PENDING; chưa có PR, chưa merge.
Phiên Colab: https://colab.research.google.com/drive/1JbJbd3SKr8L5k-shFy0w7M5VWdU_8GEk

## 10. Thao tác cần người dùng

- Cung cấp checkpoint E4 đúng và thứ tự 37 lớp từ nhóm trưởng.
- Cung cấp đề cương chứa tên/nội dung 8 hình + 3 bảng, notebook EDA cho bảng 2-3.
- Hoàn tất xác thực Git cục bộ nếu cần để push branch đã review; không push main.
- Duyên review PR, quyết định merge. Không tự merge.
- Giữ phiên Colab trong lúc train; checkpoint không được đưa vào Git.
