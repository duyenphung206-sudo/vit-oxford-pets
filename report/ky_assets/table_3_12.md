# Bảng 3-12. Kết quả của thực nghiệm E5 trên tập kiểm định

| Learning rate | Tăng cường dữ liệu | Epoch tốt nhất | Val loss | Val accuracy (%) |
|---|---|---|---|---|
| 0,001 | có | 10 | 1.138960 | 91.3043 |
| 0,003 (E4 seed42) | có | 10 | 0.311232 | 95.5163 |
| 0,01 | có | 10 | 0.158747 | 95.6522 |
| 0,03 | có | 10 | 0.148726 | 95.1087 |
| 0,003 | không | 10 | 0.265462 | 95.9239 |

Tất cả seed42, 10 epoch; chọn checkpoint theo val loss nhỏ nhất. E5-LR2 tái dùng E4.
