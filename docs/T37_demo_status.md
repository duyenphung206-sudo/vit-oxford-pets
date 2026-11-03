# T37 — Demo Top-5 (PARTIALLY COMPLETED)

Tái dùng semantics Predictor và Gradio demo nguồn; nâng Top-3 thành Top-5. App độc lập chỉ phục vụ demo checkpoint torchvision ViT-B/16, không sửa model factory TV1.

Từ gốc checkout: `python app/app.py --checkpoint path/to/best.pt --device cpu --port 7860`.
Cần môi trường có torch, torchvision, pillow, gradio phù hợp (đã thử Gradio trong .venv nguồn). Checkpoint không đóng gói; loader từ chối smoke-only và architecture khác vit_b_16/37 lớp. Không sửa nguồn hoặc checkpoint.

Đã chạy inference ảnh thật Abyssinian_100 bằng checkpoint SHA256 4c897b64829284a09bd9e85ba2984b97683cd074cb5711086bd698c4158331cb; trả 5 lớp và điểm softmax. Đã khởi tạo Gradio Interface thành công. JSON ghi kết quả thật ở T37_INFERENCE_VERIFICATION.json. Môi trường Python mặc định thiếu Gradio, nên phép kiểm tra này dùng môi trường nguồn đã có, không phải môi trường sạch.

Còn thiếu: attention map lấy từ mô hình thật, kiểm thử upload qua browser và GIF/ảnh demo. Không tạo figures/demo.gif giả, không dùng heatmap minh họa. Không được báo T37 hoàn tất trước khi ba phần này được kiểm chứng. Softmax không phải xác suất đúng được hiệu chỉnh.
