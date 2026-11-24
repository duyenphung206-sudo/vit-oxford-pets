# E2 và ViT độc lập — Kỳ hỗ trợ Trinh

Nguyễn Ngọc Kỳ thực hiện phần hỗ trợ này theo yêu cầu của nhóm. Không gán tác giả cho Trinh. Dựa trên main `3359b2bdd2a081faa20f9ff83a93c2ccbac12071`; giữ nguyên trainer, engine, optimizer, data, model factory, E1/E4, README và CI.

## E2 thực nghiệm thật: DONE

Chạy Tesla T4, seed 42, split train/validation/test 2944/736/3669. Config kế thừa E4: ImageNet1K V1, batch 64, SGD momentum 0.9, LR 0.003, augmentation bật, warmup 1 epoch rồi cosine, AMP, tối đa 10 epoch. Chỉ thay experiment và strategy=linear_probe; không thay giao thức chung.

| Chỉ số | Kết quả |
|---|---:|
| Tổng tham số | 85.827.109 |
| Tham số được huấn luyện | 28.453 (0,033152%) |
| Epoch tốt nhất theo validation loss | 10 |
| Validation accuracy tại checkpoint được chọn | 93,0707% |
| Validation loss | 0,356996 |
| Epoch đã chạy | 10 |
| Thời gian trung bình/epoch | 51,23 giây |
| Tổng thời gian epoch từ log đã làm tròn | 512,4 giây |
| Peak GPU memory | 861,4 MiB |
| Train loss epoch đầu | 3,3847 |
| Train accuracy epoch cuối | 95,24% |
| Validation accuracy epoch cuối | 93,07% |

Checkpoint chọn theo loss thấp nhất; epoch 8 có accuracy làm tròn cao hơn epoch 10 nhưng không phải tiêu chí chọn. Kết quả chỉ một seed và validation, chưa chạy test E2. Thời gian E2 trong lần chạy này cao hơn E3; không suy diễn linear probe luôn nhanh hơn. Tên trường memory của trainer là `peak_gpu_memory_mb` nhưng phép chia thực tế dùng MiB.

Bằng chứng: `KY_E2_RUN_MANIFEST.json`, `KY_E2_CHECKPOINT_VERIFICATION.json`, `results/e2_vit_linear_probe/seed42/{config.yaml,log.csv,metrics.json}`, `figures/e2_curves.png`, bảng `report/ky_assets/e2_results_summary.*`.

Kiểm tra checkpoint: strict load đạt; cả 150 tensor backbone bằng từng bit với ImageNet1K V1; head 37×768; đánh giá lại 736 ảnh đạt accuracy 0.9307065217391305, loss 0.3569956421852112, khớp metrics trong sai số 1e-6. Kiểm tra dùng model torchvision chính thức để chuẩn hóa tên khóa MLP cũ của weights V1.

Checkpoint 343.359.029 byte được lưu tại `MyDrive/vit-oxford-pets-ky/e2-support/20261124-151320/checkpoints/e2_vit_linear_probe/seed42/best.pt`. SHA256: `79a29b4d7ca922e81e57aa4c32bfd1f2d55de29a7fa8b753f654392db65abd91`. Không đưa checkpoint/dataset vào Git.

## Notebook và khả năng tái lập

Notebook nguồn `notebooks/09_train_e2_ky_support.ipynb` có preflight, checkout cô lập từ main đã ghim, gắn Drive, kiểm tra run hoàn thành để tránh chạy lại, training GPU-only và kiểm tra checkpoint. Output nguồn để trống, không chèn output lịch sử như thể vừa thực thi.

Bản sao đã chạy thật: https://colab.research.google.com/drive/1-Dd1ItiuniR67f71BYGCJ8SjvGNlzf2f . Notebook E3/E5 gốc được giữ nguyên. `src/ky_e2_runner.py` gọi nguyên `python -m src.train`, log và checkpoint lưu vào thư mục Drive mới có timestamp; không nối log hoặc resume không đầy đủ.

## ViT độc lập: DONE

`src/models/ky_independent_vit.py` triển khai bằng PyTorch primitives: patch embedding 16×16, CLS/position embeddings, 12 encoder blocks, 12 attention heads, QKV/scaled dot-product/softmax, pre-LayerNorm, residual, GELU MLP 3072 và head 37 lớp. Không gọi torchvision ViT bên trong; không thay factory của nhóm. Đây là cài đặt kiến trúc độc lập, không phải kết quả huấn luyện từ đầu.

Tham chiếu: [ViT paper](https://arxiv.org/abs/2010.11929), [torchvision source](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html). Mã cũ của Kỳ thực tế bọc torchvision và dùng recipe khác; không tái gán checkpoint cũ thành E2 chuẩn.

`src/ky_verify_vit.py` chuyển cùng trọng số E4 seed 44 vào hai kiến trúc; CPU float32/eval, dropout 0, cùng input. Cả random, zeros, ảnh mèo và chó đều allclose với atol=1e-5/rtol=1e-4 và Top-1 trùng. Sai số tuyệt đối lớn nhất 2,6226043701171875e-6. Chi tiết trong `KY_VIT_LOGITS_VERIFICATION.json`.

## Kiểm thử, demo và giới hạn

41 kiểm thử của toàn repo đạt trong phiên Colab T4; Ruff đạt cục bộ. Kiểm thử bổ sung xác nhận freeze/optimizer/backward, ánh xạ trọng số và logits ViT-B/16 đầy đủ. Windows hiện chặn torch_python.dll bằng Application Control; không thay chính sách hệ thống.

Demo dùng đúng checkpoint release E4 seed 44, strict load, đúng 37 nhãn và preprocess bằng tensor với pipeline nhóm. Dự đoán thật hai ảnh, attention không đổi dự đoán và hook được gỡ. Bằng chứng `KY_DEMO_CURRENT_VERIFICATION.json` phân biệt kiểm tra API với thao tác upload trình duyệt.

Báo cáo nhóm 22-11: BLOCKED. File có đuôi PDF hiện chỉ 328 byte XML báo lỗi Microsoft; Word Online lỗi kết nối. Chưa xác nhận caption/bảng/phân công của bản mới, chưa đánh số hình kiến trúc mới thay cho nhóm. Không sửa Word chung.
