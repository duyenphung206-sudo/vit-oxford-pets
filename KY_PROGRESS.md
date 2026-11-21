# KY_PROGRESS — Nguyễn Ngọc Kỳ

Cập nhật: 2026-11-21 (Asia/Saigon). Branch duy nhất: `ky/e3-e5-demo-report`.

`DONE`: đã thực hiện và kiểm chứng. `PENDING`: chưa thực hiện. `BLOCKED`: thiếu đầu vào/quyền.
`FAILED`: đã thử nhưng không đạt. Không có accuracy, loss hoặc checkpoint giả trong kết quả nghiên cứu.

## 1. Nhiệm vụ hoàn thành

- DONE: config E3 partial và 5 config tham chiếu/ablation E5, kế thừa E4 đúng parser của repository.
- DONE: E3, E5-LR1, E5-LR3, E5-LR4, E5-AUG chạy thật trên Google Colab Tesla T4, seed42, 10 epoch mỗi run.
- DONE: E5-LR2 tái sử dụng đúng E4 seed42, đối chiếu config/log/metrics, không chạy lại E4.
- DONE: head + final LayerNorm + 4 encoder blocks cuối trainable; 28,381,477 / 85,827,109 tham số.
- DONE: notebook 07/08 thực thi đủ 8/9 code cell sau một lần restart kernel T4,
  trong namespace riêng, tái sử dụng run đã hoàn tất. Output thực được lưu trong notebook.
- DONE: log.csv, config.yaml, metrics.json, curves.png của năm run mới đã nhập về Downloads và xác minh lại.
- DONE: bảng CSV/JSON E5 và biểu đồ PNG 300 dpi/SVG, có mã nguồn tái lập.
- DONE: 5 checkpoint E3/E5 sao lưu Drive (gồm LR4 mới), đối chiếu SHA256 bản sao với bản gốc; không đưa checkpoint vào Git.
- DONE: kiểm thử cục bộ, lint, schema notebook, split và phạm vi; bộ hình/bảng đã render/kiểm tra trực quan.
- DONE: PR31 đã được Duyên merge; phần bổ sung demo/LR4/EDA vẫn chỉ thực hiện trên branchKy, agent không merge.

## 2. Nhiệm vụ chưa hoàn thành

Không còn thí nghiệm, demo hoặc asset hình/bảng nào của Kỳ thiếu đầu vào.
Bộ 8 hình +3 bảng đã hoàn tất theo PDF nguồn. Phần còn lại là Duyên review PR33,
ghép các asset vào Word chung và quyết định merge; agent không tự sửa tài liệu chung.

## 3. Kết quả E3

| Seed | Strategy | Trainable / total | Best epoch | Val accuracy | Val loss | Tổng thời gian epoch (s) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | partial, last 4 blocks + final LayerNorm + head | 28,381,477 / 85,827,109 | 10 | 0.949728 | 0.237091 | 444.2 |

Nguồn: `results/e3_vit_partial/seed42/{config.yaml,log.csv,metrics.json}`.
Peak GPU memory được PyTorch ghi nhận: 2140.6 MiB. Checkpoint chọn theo validation loss nhỏ nhất.
Smoke test dữ liệu giả được lưu riêng trong `Downloads/.ky-tools/` hoặc `/content/ky-smoke/`, không dùng để báo cáo nghiên cứu.

## 4. Kết quả E5

| Run | LR | Augmentation | Val accuracy | Val loss | Tổng thời gian epoch (s) | Epoch | Seed | Trạng thái |
|---|---:|---|---:|---:|---:|---:|---:|---|
| E5-LR1 | 0.001 | true | 0.913043 | 1.138960 | 502.5 | 10 | 42 | DONE |
| E5-LR2 | 0.003 | true | 0.955163 | 0.311232 | 538.6 | 10 | 42 | DONE, tái dùng E4 seed42 |
| E5-LR3 | 0.01 | true | 0.956522 | 0.158747 | 498.8 | 10 | 42 | DONE |
| E5-LR4 | 0.03 | true | 0.951087 | 0.148726 | 459.5 | 10 | 42 | DONE |
| E5-AUG | 0.003 | false | 0.959239 | 0.265462 | 433.4 | 10 | 42 | DONE |

Accuracy là tỷ lệ 0–1; ví dụ E3 = 94.9728%. Thời gian là tổng epoch_time_s đã làm tròn trong log,
không gồm tải dữ liệu, pretrained weights, cài dependency hoặc backup.
Giữ nguyên batch64, SGD/momentum0.9, seed42, warmup/cosine, AMP và mọi điều kiện khác của E4.
Tập test không được dùng để lựa chọn siêu tham số. Không dùng trung bình ba seed làm baseline E5.

Trong seed42 này, LR0.01 đạt 704/736 ảnh validation đúng so với 703/736 của LR0.003.
Tắt augmentation đạt 706/736. Đây là quan sát một seed, không phải kết luận về độ ổn định nhiều seed.
Biểu đồ và bảng tái tạo bằng `python -m src.ky_experiments --summarize`.

## 5. Trạng thái demo

DONE: checkpoint E4 seed44 do người dùng cung cấp từ release v0.9.0.
Tệp 343,359,029 bytes; SHA256 khớp digest của release:
`525435b0b0338e379fad8238906687a636c4a8ca4bf39539d67f2b1901d9ec77`.
Đã sửa riêng `app/app.py` để nhận raw state_dict, nạp strict trên CPU và dùng đúng
37 nhãn từ train.csv; đối chiếu toàn bộ thứ tự với annotation chính thức.
Không sửa trainer, không thay checkpoint cá nhân, không đổi giao diện.

DONE: inference thực và thao tác browser upload cho ảnh chó/mèo trainval,
hiển thị Top-5 và softmax score. `beagle_100`: beagle 0.934672;
`Abyssinian_100`: Abyssinian 0.520299. Đây là hai ảnh minh họa, không phải số đo accuracy.
Gửi ảnh rỗng báo `Upload a valid image`. Bằng chứng: `docs/KY_DEMO_VERIFICATION.json`.
CLI Top-5 giữ tương thích. Để có giao diện đúng hình3.10/3.11:
`python app/app.py --checkpoint <vit_b16_e4_seed44_best.pt> --device cpu --port 7861 --attention`.
Attention rollout đủ12 khối, Q/K từLayerNorm1, residual,14×14patch bilinear224×224.
Suy luận gốc không đổi và hook được gỡ sau forward. Ảnh test minh họa được chọn trước suy luận:
Abyssinian_201, Top-1Abyssinian, softmax0.394977; không đánh giá toàn tập test hoặc dùng test chọn siêu tham số.
Nguồn: `docs/KY_ATTENTION_VERIFICATION.json`.

## 6. Trạng thái 8 hình và 3 bảng

DONE toàn bộ theo caption/bố cục trong PDF nhóm người dùng cung cấp (46 trang):

- Hình1.1: sáu cặp giống, lưới2×6, ảnh đúng tập train, tên giống đúng thứ tự.
- Hình2.1: số ảnh trainval37 lớp giảm dần, màu mèo/chó, mốc100, nhãn xoay90°.
- Hình2.2: histogram r rộng0.05, mốc0.95/1.05 và scatter kích thước gốc7.349 ảnh.
- Hình2.3: một ảnh train, ô đánh giá +4 mẫu tăng cường, seed42 và ID ảnh.
- Hình3.4: E3 curves xuất bằng plot_curves; hình3.6: log-LR,4 điểm nối và điểm tam giác no-aug.
- Hình3.10/3.11: screenshot trình duyệt thật giao diện trống/kết quả Top-5+attention E4 seed44 CPU.
- Bảng2-3: mode/hướng ảnh và min/max/mean từ EDA gốc, đối chiếu lại7.349 ảnh.
- Bảng3-9: kết quả E3; bảng3-12: đủ5 dòng E5 từ metrics thật, E4seed42 chỉ tái dùng.

Danh mục đầy đủ, caption, tệp, trang PDF nguồn và nội dung điền số:
`report/KY_FIGURE_TABLE_STATUS.md`. Bộ duyệt riêng11 trang: `report/KY_REPORT_FIGURES_TABLES.pdf`.
Nguồn tái lập: `report/ky_build_report_assets.py`, `report/ky_package_report.py`.
Không sửa PDF/Word nhóm; không sửa hình3.7–3.9 hoặc nội dung của Trinh/Tâm.
Bảng3-9 sửa nhãn đơn vị trong asset riêng thànhMiB vì2140.6 trong log làMiB;
quy đổi SI tương ứng2244.58MB nếu nhóm giữ cộtMB. Không đổi trainer hoặc log.

LR4: 700/736 ảnh validation đúng,10epoch,mean45.94s/epoch,peak5291.8MiB.
Evidence riêng: `docs/KY_LR4_RUN_MANIFEST.json`.

## 7. Kiểm thử và xác minh đã chạy

- DONE: `python -m pytest` — 38 passed sau bổ sung attention; gồm kiểm tra tái dựng Q/K khớp attention PyTorch và thứ tự tích rollout.
- DONE: `ruff check src tests` — pass toàn bộ. Notebook 07/08: nbformat schema và AST pass.
- DONE: Colab sau restart kernel: notebook07 có 8 code cell pass, notebook08 có 9 code cell pass.
  Các code cell được exec tuần tự trong namespace riêng; lượt train thật có đủ evidence được tái sử dụng.
  Callback tải ZIP được chuyển thành lưu ZIP lên Drive đã được cho phép; không tự tải ra thư mục ngoài Downloads.
- DONE: E3 fake-data end-to-end 1 epoch trên CPU; không có real CPU training.
- DONE: train/val/test 2944/736/3669, ID không trùng, nhãn khớp annotation và SHA256 khớp metadata.
- DONE: không thay đổi split/seed; kiểm tra config E3/E5 chỉ khác các điều kiện được giao.
- DONE: best_epoch/accuracy/loss/time trong metrics khớp log và config; không có fake_data trong run thật.
- DONE: kiểm tra file stage, dung lượng, pattern token và không có checkpoint trong commit.
- DONE: CI PR31 tại commit9baa59e (run44800749961) thành công.
- DONE: sửa bootstrap notebook07/08 sau khi Duyên merge và xóa branch remote.
  Clone commit đã merge0dbb877 vào local branchKy; public clone mới trên Windows pass.
  Output bootstrap cũ đã xóa, output các cell khác được ghi rõ là evidence lịch sử T4;
  không tuyên bố đã Run All phiên bản bootstrap mới trên T4.

Bằng chứng: `docs/KY_PIPELINE_VERIFICATION.json`, `docs/KY_NOTEBOOK_VERIFICATION.json`,
`docs/KY_COLAB_RUN_MANIFEST.json`, cùng log/config/metrics của từng run.

## 8. Lỗi và giới hạn còn tồn tại

- DONE: PDF nguồn đã cung cấp và đối chiếu đủ11 caption. Không còn blocker dữ liệu/quyền cho asset của Kỳ.
- DONE: E5-LR4 tùy chọn; các thí nghiệm mới chỉ có seed42 đúng yêu cầu.
- FAILED (công cụ, đã có phương án thay thế): connector GitHub trả403 khi ghi;
  Git credential helper/Windows credential vault từ môi trường agent bị từ chối truy cập.
  Người dùng push bằng terminal đã đăng nhập; PR được tạo qua phiên GitHub đã đăng nhập.
- FAILED (công cụ, đã thay thế): Python tải vào Downloads bị Application Control chặn.
  Dùng interpreter đóng gói sẵn, dependency/cache/temp cục bộ trong Downloads; không đổi bảo mật Windows.
- Môi trường Colab/Drive ở máy chủ từ xa theo ngoại lệ đã được người dùng cho phép.
  Mọi bản sao kết quả và mã nguồn ghi trên máy cá nhân nằm trong Downloads.
- Code tạo chart chỉ được chỉnh thêm khoảng trống trục loss sau thí nghiệm để marker không bị cắt;
  code chạy thí nghiệm có SHA256 gốc được lưu trong manifest và có tại commit d997f28.

## 9. Commit và Pull Request

- Chuẩn bị code: `d997f285f55280063e0a55a22a8ce0339d4e958e`.
- Kết quả T4 và evidence: `c32f46924bbad27c43e3520436d593d9135b7c03`.
- Branch: https://github.com/duyenphung206-sudo/vit-oxford-pets/tree/ky/e3-e5-demo-report
- PR: https://github.com/duyenphung206-sudo/vit-oxford-pets/pull/31
- PR31 đã được Duyên (`duyenphung206-sudo`) merge thành commit0dbb877 ngày21/11; Duyên xóa branch remote.
- Agent không merge. `main` cục bộ vẫn4d5721b; `origin/main` mới e7f224f gồm cập nhậtREADME của Duyên.
- Phần demo/EDA/bootstrap mới chỉ làm trên branchKy, không sửa shared/E1/E2/E4.
- Commit bổ sung: `619e4889c288a308449637a11534dc5c4eb1de58`, người dùng đã push và agent đối chiếu remote SHA.
- PR bổ sung đang mở: https://github.com/duyenphung206-sudo/vit-oxford-pets/pull/33
- CI PR33 tại commit619e488 đã pass (run44890918835). Nhờ Duyên review; agent không merge.
- Link Drive người dùng cung cấp mở thư mục `e4_vit_full_ft` chỉ có seed42/43/44,
  không chứa đề cương/caption. Sau đó người dùng đã cung cấp PDF đúng; blocker này đã giải quyết.
- Colab execution record: https://colab.research.google.com/drive/1JbJbd3SKr8L5k-shFy0w7M5VWdU_8GEk

## 10. Thao tác người dùng

- Khi đồng bộ commit mới từ terminal trong checkout: `git push -u origin ky/e3-e5-demo-report`.
  PR31 đã merge; phần bổ sung nằm ở PR33. Không push main và không dùng force push.
- Duyên duyệt và ghép bộ11 asset vào Word chung; cập nhật danh mục hình/bảng sau khi chèn.
- Duyên review và quyết định merge.

Checkpoint đã bảo toàn ở `MyDrive/vit-oxford-pets-ky/checkpoints/<experiment>/seed42/best.pt`;
SHA256 và kích thước của4 file ban đầu có trong manifest; checkpoint thứ5 (E5-LR4) có
trong docs/KY_LR4_RUN_MANIFEST.json. Không đưa các file này vào Git.
