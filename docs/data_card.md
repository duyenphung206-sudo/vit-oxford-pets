# T10 — Oxford-IIIT Pet data card

Owner: Ngọc Kỳ (TV2). Nguồn chính thức: https://www.robots.ox.ac.uk/~vgg/data/pets/ (đối chiếu 03/11/2026).
Giấy phép dataset: Creative Commons Attribution-ShareAlike 4.0 International; quyền tác giả ảnh thuộc chủ sở hữu gốc. Không đồng nhất giấy phép MIT của code với giấy phép ảnh. Khi phân phối lại ảnh hoặc bản phái sinh phải tuân thủ CC BY-SA và ghi nguồn; không đưa dataset vào Git.

Trích dẫn: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, “Cats and Dogs”, IEEE CVPR, 2012, pp. 3498–3505, doi:10.1109/CVPR.2012.6248092.

Dữ liệu: 37 giống, 12 giống mèo và 25 giống chó; 3.680 mẫu official trainval và 3.669 mẫu test, tổng 7.349 mẫu được annotation. Các cột ID/class/species/breed đọc trực tiếp từ annotation; class ID chính thức 1–37 chuyển thành label 0–36. Species 1=Cat, 2=Dog. Không suy loài qua cách viết hoa. Trimaps và head ROI phục vụ segmentation/localization, không phải ảnh đầu vào classification.

Kiểm chứng local: 7.349 ảnh giải mã được; 7.346 RGB, 3 RGBA, không có L/CMYK, thiếu hoặc decode lỗi theo audit T11 đã chạy. Có 41 JPG ngoài official annotation, 3 .mat: không nhập vào split. T11 PR #11 hiện CLOSED, chưa merge; không coi sản phẩm đó đã nằm trên main.

Tải (nếu chưa có): `torchvision.datasets.OxfordIIITPet('data', split='trainval', download=True)`. Giữ images/ và annotations/ dưới data/oxford-iiit-pet. Không tải lại nếu dataset hợp lệ đã có.

Rủi ro: dataset giới hạn 37 giống, không đại diện mọi thú nuôi hoặc ảnh ngoài miền; softmax không chứng minh ảnh thuộc miền. Có khác biệt tư thế, ánh sáng, bối cảnh và tỉ lệ ảnh. EDA không dùng test để tune siêu tham số. Tham khảo nguồn gốc khi phân phối lưới mẫu.
