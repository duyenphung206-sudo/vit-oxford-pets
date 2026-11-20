# TV2 — Audit tích hợp, 03/11/2026

Đọc workbook Downloads/PhanCong_DoAn_DL.xlsx ở chế độ chỉ đọc (OOXML ZIP/XML); 12 dòng TV2 khớp danh sách người dùng. Không sửa workbook, README, .gitignore hoặc file của TV1/TV3/TV4. Repository đích đúng Downloads/vit-oxford-pets, main 8b1a92d. Notebook T11 có thay đổi local chưa commit được giữ nguyên. Công việc mới ở worktree riêng, branch data/TV2-t10-t13-integration.

| Mã | Trước | Tái sử dụng | File mới | Kiểm chứng / trạng thái cuối |
|---|---|---|---|---|
| T05 | BLOCKED | clone/tài khoản có sẵn | không | PR #11 CLOSED chưa merge; README Team chưa có tên, bị cấm sửa; BLOCKED |
| T10 | PARTIALLY COMPLETED | dataset/annotation, audit cá nhân | docs/data_card.md | annotation 3680/3669/37, nguồn và giấy phép Oxford đã tra; PENDING REVIEW |
| T11 | PARTIALLY COMPLETED | notebook và 6 hình đã có | không | không ghi đè; PR #11 đóng chưa merge, Issue mới #13; PENDING REVIEW |
| T12 | PARTIALLY COMPLETED | split_rows/read_annotations, splits.json | src/data.py, train.csv, val.csv, metadata.json | seed42/20%, khớp ID và label nguồn, không overlap, chưa có split nhóm; PENDING REVIEW |
| T13 | PARTIALLY COMPLETED | transforms/PetImages/loader | build_dataloaders, tests, augmentation PNG | 2 tests PASS, batch thật train/val/test; interface đề xuất vì nhóm chưa có; PENDING REVIEW |
| T18 | PARTIALLY COMPLETED | nhận xét CNN bias trong báo cáo cá nhân | docs/T18_cnn_section_2_1.md | nội dung mới bổ sung đủ convolution/pooling/BN/ResNet và nguồn primary; PENDING REVIEW |
| T25 | NOT VERIFIED | không tìm thấy ResNet run | configs/e1_resnet50.yaml, results/e1_resnet50/STATUS.md | chưa có train API nhóm/CUDA; config đề xuất, không metrics; PENDING TRAINING |
| T30 | NOT VERIFIED | không tìm thấy ablation run | results/e5_ablation/plan.json, STATUS.md | 6 tổ hợp chỉ là kế hoạch, chưa có best config nhóm; PENDING TRAINING |
| T32 | PARTIALLY COMPLETED | báo cáo cá nhân phần4, EDA | docs/T32_data_preparation_draft.md | chưa tích hợp Word chung; PENDING REVIEW |
| T37 | PARTIALLY COMPLETED | Predictor và demo nguồn | app/app.py, JSON kiểm chứng | đã chuyển Top3 thành Top5; attention/GIF/browser upload chưa xác minh; PARTIALLY COMPLETED |
| T44 | BLOCKED | đọc README hiện tại | bản ghi này | README clone URL placeholder; src.train/src.evaluate/config E4 không có trên main; không clean install/eval nhóm, không đóng Issue; BLOCKED |
| T47 | BLOCKED | đối chiếu 12 công việc trong workbook | danh sách này để họp | chưa có tỷ lệ được nhóm thống nhất; không sửa PL B; BLOCKED |

## File nguồn và thay đổi
- src/data.py: tái dùng code từ ViT-B16-SOURCE-CODE/vit_pets/data.py, thay dependency utils cá nhân bằng worker seeding/write_json tối thiểu, thêm interface đọc CSV đã lưu. Giữ helper nguồn để truy vết; build_dataloaders không tự chia lại.
- CSV mới xuất từ thuật toán nguồn và kiểm tra khớp splits.json local, không sửa split nhóm vì main chưa có.
- T18 bổ sung tài liệu primary, T32 biên tập dữ liệu/preprocessing từ báo cáo cá nhân; không sao chép metrics train.
- app/app.py biên tập tối thiểu từ Predictor/demo/model load cá nhân, không import model factory đang do TV1 làm, không sửa factory.
- Không sao chép toàn project hoặc checkpoint; dataset đích đã tồn tại nên không chép lại.
- Mọi file tracked trong PR đều mới. Không thay đổi README/requirements/.gitignore. Không sửa các sản phẩm T11 hiện có.

## Còn thiếu
T05 cần quy trình review/merge và xử lý xung đột yêu cầu README bởi người dùng; không tự sửa. T11 cần owner/reviewer quyết định mở lại PR #11 hoặc dùng Issue mới #13; không tạo PR trùng. T25/T30 cần pipeline TV1, tài nguyên train và log/metrics thật. T37 thiếu attention map thực, GIF và kiểm thử tương tác browser. T44 cần README/pipeline/checkpoint của nhóm và môi trường sạch. T47 cần cuộc họp thống nhất tỷ lệ. Không coi metrics ViT cá nhân là metrics của nhóm.

Kiểm chứng bổ sung: notebook T11 hiện có đã chạy lại 14/14 cell không lỗi bằng dataset tại repo đích, output lưu workspace riêng; không ghi notebook local. Toàn bộ 7349 ảnh annotated tại đích khớp SHA256 nguồn. Top-5 inference thật và khởi tạo Gradio Interface thành công trong môi trường .venv nguồn, xem JSON. Python mặc định thiếu Gradio. Không coi đây là clean-environment reproduction.
