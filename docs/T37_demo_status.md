# T37 — Demo Top-5 (COMPLETED)

Demo chính thức dùng `app/app.py` của main hiện tại, không thay đổi mã ứng dụng. Đã đối chiếu với `origin/main` tại commit `e7335a9ffa8dcaf25356759e1e5626d84d41a019` (sau PR #36 và #37); Git blob của app là `f920f39a82594b80bbf4eb8dea0c2b31292b3e18`. PR [#36](https://github.com/duyenphung206-sudo/vit-oxford-pets/pull/36) đã merge tại `66214b01cfbe6ec69bb84f2f1b44313f977c9304`.

## Checkpoint chính thức

- Release: [v1.0.0](https://github.com/duyenphung206-sudo/vit-oxford-pets/releases/tag/v1.0.0).
- Asset: [vit_b16_e4_seed44_best.pt](https://github.com/duyenphung206-sudo/vit-oxford-pets/releases/download/v1.0.0/vit_b16_e4_seed44_best.pt), 343,359,029 byte; E4 seed 44.
- SHA256: `525435b0b0338e379fad8238906687a636c4a8ca4bf39539d67f2b1901d9ec77`.
- Ngày 2026-11-27: SHA256 cục bộ khớp chính xác digest asset của Release v1.0.0. Tái dùng file hiện có, không tải lại. Đây cũng là các byte checkpoint đã kiểm chứng trước ở v0.9.0; không thay bằng mô hình cá nhân.
- Nạp raw state_dict bằng `weights_only=True`, `strict=True`, ViT-B/16 với head 37 lớp. Nhãn lấy theo label ID trong `data/splits/train.csv`; tiền xử lý dùng `IMAGENET1K_V1.transforms()`.

## Chức năng đã kiểm chứng

- Giao diện Gradio có tải ảnh, Submit/Clear, Top-5 và ảnh “Attention rollout (model crop)” khi bật `--attention`.
- Suy luận CPU trên hai ảnh thật `Abyssinian_100` và `beagle_100`: trả đúng 5 nhãn và điểm softmax. Kết quả minh họa Top-1 lần lượt Abyssinian 52.0299% và beagle 93.4672%; không coi hai ảnh là phép đo accuracy.
- Attention rollout thật từ Q/K của đủ 12 encoder: trung bình head, cộng identity và chia 2, nhân theo thứ tự lớp; lấy CLS tới 196 patch (14×14), nội suy và phủ lên crop 224×224.
- Kiểm tra lại ngày 2026-11-27: dự đoán có attention trùng hoàn toàn dự đoán thường trên cả hai ảnh; ảnh phủ 224×224, sai số tổng hàng rollout tối đa khoảng 2.39e-7; hook được tháo sau suy luận.
- Khởi chạy đúng CLI `--attention`, mở giao diện tại localhost và kiểm tra API tải file → phân loại → Top-5 + ảnh rollout thành công cho cả hai ảnh. Môi trường hiện có: Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, Gradio 6.32.0; không phải kiểm thử cài mới môi trường sạch.
- 7 kiểm thử trong `tests/test_ky_demo.py` đạt: thứ tự nhãn, từ chối checkpoint không phù hợp, đầu vào rỗng, thứ tự rollout và đối chiếu Q/K với attention gốc.
- Bằng chứng lịch sử đã có: [upload ảnh thật qua browser](KY_DEMO_VERIFICATION.json), [attention rollout](KY_ATTENTION_VERIFICATION.json), [ảnh giao diện](../figures/demo_ui_result.png), [ảnh attention](../report/ky_assets/demo_attention_overlay.png). JSON T37_INFERENCE_VERIFICATION.json là phép kiểm tra checkpoint cũ; không dùng SHA của nó để nhận diện checkpoint demo chính thức.
- Giới hạn kiểm tra mới: giao diện đã mở trực tiếp bằng trình duyệt; thao tác upload bằng chuột trong lượt 2026-11-27 bị công cụ điều khiển trình duyệt dừng vì không xác định URL đủ chắc chắn. Không tuyên bố đã lặp lại trọn luồng upload bằng chuột trong lượt này; kiểm tra upload mới dùng API Gradio cục bộ. Luồng browser trước đó được ghi trong bằng chứng lịch sử ở trên.

Softmax không phải xác suất đúng đã hiệu chỉnh; attention rollout là minh họa định tính, không phải giải thích nhân quả. Không chỉnh model factory, trainer, mô hình hay checkpoint; không chạy lại thực nghiệm.

## Chạy demo để quay video

Từ gốc repository, với Python đã cài các thư viện trong `requirements.txt`:

```bash
python app/app.py --checkpoint <đường dẫn tệp .pt> --device cpu --port 7860 --attention
```

Mở [http://127.0.0.1:7860](http://127.0.0.1:7860). Nếu port đang có phiên demo kiểm tra thì dùng phiên đang chạy, hoặc dừng phiên đó trước khi chạy lại.

Kịch bản quay: mở giao diện → tải ảnh mèo → Submit → chỉ Top-5 và attention → Clear → tải ảnh chó → Submit → nêu giới hạn softmax/attention.

**Demo phần mềm: COMPLETED. Video giới thiệu: chưa quay/chưa hoàn thành; không có GIF/video giả và không dùng trạng thái COMPLETED của T37 để tuyên bố đã quay video.**
