# T18 — Dự thảo mục 2.1: Cơ sở CNN

Owner TV2. Dự thảo để nhóm tích hợp vào báo cáo chung; không thay thế chương đang biên soạn.

## Convolution
Ảnh là tensor H×W×C. Bộ lọc có kích thước k×k×C trượt trên lưới, mỗi vị trí tính tổng tích với vùng ảnh và bias. Trong thư viện deep learning, phép toán thường là cross-correlation (không đảo kernel). Với dilation d, padding p và stride s, chiều đầu ra là floor((H+2p-d(k-1)-1)/s+1). Chia sẻ trọng số giảm số tham số so với lớp fully connected; receptive field tích lũy qua nhiều lớp. Lọc cục bộ phù hợp khai thác cạnh và texture nhưng không tự đảm bảo bất biến với mọi biến dạng.

## Pooling
Max pooling lấy giá trị lớn nhất trong cửa sổ; average pooling lấy trung bình. Stride giảm độ phân giải và chi phí nhưng mất chi tiết. Global average pooling tổng hợp mỗi feature map thành một số trước classifier. Pooling có thể làm biểu diễn bớt nhạy với dịch chuyển nhỏ, không đảm bảo bất biến dịch chuyển hoàn toàn.

## Batch Normalization
Với một kênh, dùng thống kê mini-batch để chuẩn hóa: x_hat=(x-mu_B)/sqrt(var_B+epsilon), y=gamma*x_hat+beta. Với ảnh, thống kê thường trên chiều batch và không gian. Training cập nhật running statistics; evaluation dùng thống kê tích lũy thay vì thống kê batch hiện tại. Batch nhỏ hoặc khác miền có thể làm thống kê kém ổn định. Khi fine-tune phải quyết định rõ có cập nhật BN hay không. Tham khảo Ioffe & Szegedy [1].

## ResNet-50
Residual block học F(x), kết quả y=F(x)+x; nếu kích thước thay đổi dùng projection shortcut. Đường tắt giúp tối ưu mạng sâu [2]. ResNet-50 dùng bottleneck 1×1 → 3×3 → 1×1; classifier ImageNet cần thay bằng Linear 2048→37 cho bài toán này. Pretraining là điểm khởi đầu, không phải bằng chứng baseline đã được train trên Pets.

## Inductive bias CNN và ViT
CNN có locality và weight sharing theo vị trí; ViT biểu diễn ảnh bằng patch token, attention cho phép tương tác giữa token, và cần vị trí để mô tả cấu trúc không gian. CNN thuận lợi khi học cấu trúc cục bộ với dữ liệu nhỏ; ViT dựa nhiều hơn vào pretraining và thiết kế huấn luyện [3]. Đây là lập luận kiến trúc, không kết luận ResNet tốt/xấu hơn trong nhóm khi E1 chưa chạy. So sánh thực nghiệm phải dùng cùng split, tiêu chí chọn checkpoint bằng val, và bộ thước đo thống nhất.

Nguồn mới bổ sung vì báo cáo cá nhân chưa có đủ convolution/pooling/BN/ResNet. Kế thừa nhận xét pretrained và ít locality bias từ phần lý thuyết báo cáo cá nhân; không sao chép chương ViT của TV3.

[1] https://arxiv.org/abs/1502.03167
[2] https://arxiv.org/abs/1512.03385
[3] https://arxiv.org/abs/2010.11929
