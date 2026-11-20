# T12–T13 — Interface dữ liệu đề xuất

`from src.data import build_dataloaders`

`loaders = build_dataloaders(data_dir='data', split_dir='data/splits', batch_size=16, num_workers=0, seed=42)`

Trả dict train/val/test DataLoader. Mỗi batch (images, labels), ảnh float32 N×3×224×224, nhãn int64 0–36. data_dir là thư mục cha oxford-iiit-pet. Không tự download, không tự chia lại. Gốc relative path là working directory; caller truyền đường dẫn tuyệt đối nếu cần. Interface nhóm chưa có trên main tại thời điểm audit nên đây là đề xuất cần TV1 review, không sửa pipeline TV1.

CSV đã commit chứa ID và label; split seed 42, stratified, validation 20% (kế thừa project nguồn vì nhóm chưa chốt tỉ lệ khác). Đối chiếu chính xác thứ tự/nhãn với splits.json local trước khi tạo; 2944 train, 736 val, official test 3669 giữ nguyên. metadata.json ghi hash annotation. Mọi thí nghiệm cần dùng CSV này, không gọi prepare để chia lại. Đề nghị TV1 chấp thuận 20% trước khi merge.

Tái sử dụng functions transforms_for/read_annotations/class_names/split_rows/PetImages/loader từ ViT-B16-SOURCE-CODE/vit_pets/data.py. Bổ sung worker seeding tự chứa và build_dataloaders để tránh import pipeline cá nhân. Train dùng RandomResizedCrop 224 scale .7–1, flip .5, ColorJitter .15/.15/.15/.05; eval dùng đúng weights.transforms() (bilinear resize 256, center crop 224, mean/std ImageNet). EXIF transpose rồi RGB như nguồn. RGBA được bỏ alpha theo convert RGB của nguồn; quyết định composite nền là thay đổi preprocessing cần kiểm thử riêng, không đổi âm thầm.

Các split không trùng và phủ trainval được kiểm tra mỗi lần dựng loaders. Khi phát hiện split/annotation khác nhau sẽ báo lỗi. num_workers=0 thuận tiện Windows; tăng workers phải gọi trong if __name__ == '__main__'. worker seeding và generator seed cố định; caller vẫn cần seed torch/random/numpy cho transform ở main process.

Kiểm tra: `python -m pytest tests/test_tv2_data.py -q`. Không cần GPU/checkpoint cho T12–T13. Xem figures/augmentation_tv2.png và docs/TV2_VERIFICATION.json cho kết quả chạy dữ liệu thật.
