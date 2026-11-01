# Quy ước làm việc của nhóm

Áp dụng cho cả 4 thành viên. Mục tiêu: lịch sử GitHub sạch, ai làm gì nhìn vào là thấy.

## 1. Nhánh (branch)
- `main` luôn chạy được. **Không ai sửa trực tiếp trên `main`.**
- Mỗi nhiệm vụ = 1 nhánh, tạo từ `main` mới nhất, đặt tên `<loại>/<mã-nhiệm-vụ>-<mô-tả-ngắn>`:

| Loại | Dùng khi | Ví dụ |
|------|----------|-------|
| `feat` | thêm code/tính năng | `feat/T21-vit-scratch` |
| `data` | dữ liệu, EDA, split | `data/T11-eda` |
| `exp` | chạy thí nghiệm | `exp/T28-e4-full-finetune` |
| `docs` | tài liệu, README, ghi chú paper | `docs/T09-references` |
| `fix` | sửa lỗi | `fix/T20-val-loss-nan` |

## 2. Commit message
Theo chuẩn Conventional Commits (https://www.conventionalcommits.org): `<loại>: <mô tả ngắn, tiếng Anh, thì hiện tại>`

```
feat: add multi-head self-attention module
data: add stratified train/val split with seed 42
exp: add E4 full fine-tuning results (seed 42)
docs: add ViT paper summary
fix: correct normalization mean/std in val transforms
```
Một commit = một thay đổi có ý nghĩa. Không commit kiểu "update", "abc", "final final".

## 3. Pull Request (PR)
- Mỗi PR nhỏ, chỉ phục vụ **một** nhiệm vụ; mô tả theo template, ghi `Closes #<số Issue>`.
- Gán đúng người review như cột "Review bởi" trong file phân công.
- Chỉ merge khi đã có 1 Approve. Dùng **Squash and merge**, sau đó xóa nhánh.

## 4. Notebook
- Mỗi notebook có **một chủ** (người phụ trách nhiệm vụ). Người khác không sửa trực tiếp — góp ý bằng comment trong PR.
- Trước khi commit: *Restart & Run all*, giữ output (biểu đồ, bảng) nhưng không in hàng nghìn dòng log.
- Đặt tên có số thứ tự: `01_eda.ipynb`, `02_vit_from_scratch.ipynb`, ...

## 5. Tuyệt đối không commit
- Ảnh dataset, file `.pt/.pth` (GitHub chặn file > 100 MB) → checkpoint đưa lên GitHub Release.
- Token/mật khẩu (`kaggle.json`, `.env`, token Hugging Face).
- File tạm của Word/Excel (`~$...`).

## 6. Kết quả thí nghiệm
- Mỗi thí nghiệm có 1 file `configs/eX_<tên>.yaml` và 1 thư mục `results/eX_<tên>/seed<seed>/` gồm `metrics.json` + `log.csv`.
- Số liệu trong báo cáo và README phải lấy từ các file này — không chép tay số từ màn hình Colab.
