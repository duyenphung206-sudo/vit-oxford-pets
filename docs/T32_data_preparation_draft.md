# T32 — Dự thảo Chương 3: Chuẩn bị dữ liệu

Owner TV2. Bản nháp độc lập cho nhóm trưởng tích hợp, chưa phải chương đã duyệt.

## 3.1 Dataset
Oxford-IIIT Pet gồm 37 giống với nhãn loài và giống, official trainval 3680/test 3669 mẫu. Nguồn và giấy phép xem data_card.md. Dữ liệu gốc được giữ ngoài Git. Classification đọc ID chính thức, không đọc trimap hoặc metadata ._* như ảnh gốc.

## 3.2 EDA
Kế thừa kết quả audit cá nhân và notebook T11 đã chạy: 7349 ảnh giải mã thành công, 7346 RGB và 3 RGBA; Cat 2371/Dog 4978. Theo notebook, 4775 ảnh ngang, 2263 ảnh dọc, 311 gần vuông với ngưỡng width/height .95–1.05. Khác biệt giữa tỉ lệ loài và cân bằng lớp cần được phân biệt. 41 JPG ngoài annotation không tham gia phân loại. PR T11 #11 đã đóng chưa merge; các hình chưa nằm trên main, không khẳng định đã tích hợp chương/hình.

## 3.3 Chia dữ liệu
Kế thừa phương pháp và split local đã kiểm chứng: train 2944, val 736, phân tầng theo class với seed 42, test chính thức 3669 giữ riêng. CSV có ID tương đối và label 0–36, dùng chung cho thí nghiệm. Tỉ lệ validation 20% cần TV1 phê duyệt vì Issue chưa quy định tỷ lệ cụ thể. Không chia lại tùy từng model hoặc chọn siêu tham số bằng test.

## 3.4 Tiền xử lý
EXIF transpose rồi RGB, resize cạnh ngắn 256 và center crop 224, ToTensor và chuẩn hóa mean [0.485,0.456,0.406], std [0.229,0.224,0.225] theo weights IMAGENET1K_V1. Ảnh không vuông có thể mất nội dung ở crop; cần kiểm tra mẫu. Non-RGB không tự coi là hỏng. Alpha bị bỏ theo source convert RGB; quyết định ghép nền cần đánh giá riêng.

## 3.5 Augmentation và DataLoader
Train dùng RandomResizedCrop 224, scale [.7,1], horizontal flip, ColorJitter; val/test không augmentation ngẫu nhiên. Hình augmentation_tv2.png được xuất từ ảnh thật với seed 42. DataLoader train shuffle, val/test không shuffle; generator và worker seeding dùng 42. Windows mặc định num_workers=0. Xem T12_T13_data_interface.md để tái chạy.

Tái sử dụng và biên tập phần 4 của reports/academic_report_vi.md nguồn cá nhân; không chuyển bảng metrics ViT cá nhân thành kết quả nhóm và không sửa chương các thành viên khác. Chưa có đánh giá tác động augmentation qua T30, không kết luận mức tăng accuracy.
