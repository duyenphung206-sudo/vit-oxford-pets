# Kiểm thử ngày24 tháng11

DONE:41 trường hợp kiểm thử đạt trên runtime ColabTeslaT4, pytestreturncode0. Bao gồm38test hiện có và3test mới: mapping/logits trên mô hình nhỏ cóheadkhác0;freeze/backward/optimizerisolation củaE2; đối chiếu kiến trúc đầy đủViT-B/16 với cùngtrọngsốkhác0. Outputthật: pytest_output.txt. Thời gian suite: NOT_RECORDED trong outputcaptured; không suy ra từ số liệu cũ.

DONE:Ruff src/tests/app và hai scriptasset mới đạt trên máyWindows. Kiểm tra cú pháp tất cả module và notebook mới đạt; không tự chèn outputexecution vào notebooknguồn.

DONE:ViT với checkpointE4thật và4input đạt allclose,Top1trùng; CPUinference demo mới trên ảnhchó/mèovớiattention đạt. NguồnJSON trong cùngthư mục và docs/.

Windows không chạy lại pytest được vì ApplicationControl chặntorch_python.dll (WinError4551). Đây là lỗi môi trường, không sửa bảo mậtWindows. Kiểm chứngruntimeđược thực hiện trongColab; không mặc định38test lịch sử vẫnđạt.

Browser upload mới đạt trên ảnh mèo và chó; Top-5 và ảnh attention được xác nhận qua proxy Colab đã xác thực. Đã sửa root_path khi khởi chạy phiên kiểm tra để ảnh dùng đúng proxy, không sửa app dùng chung. Ảnh demo trong gói cũ vẫn giữ rõ nguồn lịch sử. Chưa đọc được Word/PDF22-11; không báo cáo nội dung đó MATCH.
