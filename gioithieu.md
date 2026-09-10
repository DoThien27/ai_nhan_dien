# PHẦN MỞ ĐẦU

## 1. Giới thiệu tổng quan về đề tài

Đề tài **"AI Nhận Diện Trái Cây"** là một ứng dụng sử dụng trí tuệ nhân tạo, cụ thể là mạng nơ-ron tích chập (Convolutional Neural Network - CNN) để tự động nhận diện và phân loại các loại trái cây khác nhau thông qua hình ảnh. 

Ứng dụng không chỉ có khả năng nhận diện chính xác cao thông qua việc huấn luyện mô hình học sâu với các tập dữ liệu hình ảnh (dataset), mà còn được tích hợp một giao diện đồ họa thân thiện (GUI) để người dùng dễ dàng tương tác. Người dùng chỉ cần tải lên một bức ảnh trái cây, hệ thống sẽ tự động phân tích và trả về kết quả dự đoán (tên loại quả, tên tiếng Việt, độ tin cậy) chỉ trong vài giây.

Dự án bao gồm các thành phần chính:
- **Xử lý dữ liệu:** Kiểm tra, chuẩn hóa và chuẩn bị hình ảnh cho huấn luyện.
- **Xây dựng & Huấn luyện mô hình AI:** Sử dụng TensorFlow/Keras để xây dựng mô hình CNN, huấn luyện, đánh giá và lưu trữ mô hình tốt nhất.
- **Giao diện người dùng (GUI):** Xây dựng ứng dụng Desktop bằng Tkinter để người dùng trực tiếp chọn ảnh và xem kết quả một cách trực quan, dễ dàng.

---

## 2. Danh sách thành viên và phân công công việc từng thành viên trong nhóm

Dựa vào cấu trúc và các mô-đun chức năng trong mã nguồn, công việc của nhóm được phân công thành các mảng chính như sau (bạn có thể điền tên các thành viên vào đây):

| STT | Họ và Tên | Vai trò | Chi tiết công việc thực hiện (dựa trên source code) |
|:---:|:---|:---|:---|
| 1 | **[Tên Thành Viên 1]** | **Xây dựng Mô hình AI (AI Modeling)** | - Chịu trách nhiệm chính về kiến trúc mạng CNN.<br>- Viết mã cho `train.py`: Xây dựng cấu trúc model (Conv2D, MaxPooling, Dense), thiết lập các callback (EarlyStopping) và tối ưu hóa mô hình trong quá trình huấn luyện. |
| 2 | **[Tên Thành Viên 2]** | **Kỹ sư Dữ liệu (Data Engineer)** | - Chịu trách nhiệm chuẩn bị và xử lý tập dữ liệu hình ảnh (dataset).<br>- Viết mã cho `check_dataset.py`: Kiểm tra tính toàn vẹn và phân bố số lượng ảnh trong các tập train/validation.<br>- Viết mã cho `rename_coco.py` và `add_new_class.py`: Tự động hóa quá trình đổi tên file và thêm nhãn dữ liệu mới. |
| 3 | **[Tên Thành Viên 3]** | **Phân tích & Kiểm thử (Evaluation & Prediction)** | - Phụ trách luồng nhận diện và đánh giá độ chính xác của mô hình.<br>- Viết mã cho `predict.py`: Tải mô hình đã huấn luyện (`fruit_model.keras`) và xây dựng hàm dự đoán kết quả đầu ra.<br>- Đánh giá biểu đồ accuracy/loss và tài liệu hóa quá trình trong `training_history.md`. |
| 4 | **[Tên Thành Viên 4]** | **Phát triển Giao diện (GUI Development)** | - Chịu trách nhiệm thiết kế và lập trình giao diện Desktop ứng dụng.<br>- Viết mã cho `app.py` và `gui_dashboard.py`: Xây dựng Dashboard Tkinter, xử lý các sự kiện nút bấm (Chọn ảnh, Nhận diện), và hiển thị kết quả phân tích trực quan cho người dùng. |
