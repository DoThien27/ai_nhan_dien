# Lịch Sử Huấn Luyện Mô Hình Nhận Diện Trái Cây

Tài liệu này dùng để theo dõi quá trình huấn luyện AI, các thông số đã sử dụng và kết quả đạt được.

---

## [Phiên bản 1.0] - Ngày: .../.../202...

### 1. Dữ liệu huấn luyện (Dataset)
- **Nguồn dữ liệu**: fruits-360-100x100
- **Số lượng nhãn (Classes)**: 5 loại quả (Apple, Banana, Orange, Grape, Mango)
- **Tổng số ảnh**: 9.099 ảnh
- **Tỉ lệ phân chia**: 80% Train, 20% Validation
  - Train: 7.277 ảnh
  - Validation: 1.822 ảnh

### 2. Thông số huấn luyện (Hyperparameters)
- **Kiến trúc mô hình**: (Ví dụ: CNN tự xây dựng, MobileNetV2, ResNet50...)
- **Kích thước ảnh đầu vào (Image Size)**: 100 x 100
- **Số lượng Epochs**: ...
- **Batch Size**: ...
- **Learning Rate**: ...
- **Hàm Loss (Loss Function)**: Categorical Crossentropy (Ví dụ)
- **Thuật toán tối ưu (Optimizer)**: Adam / SGD (Ví dụ)

### 3. Kết quả huấn luyện (Results)
- **Độ chính xác trên tập Train (Train Accuracy)**: ... %
- **Độ chính xác trên tập Validation (Val Accuracy)**: ... %
- **Mức độ lỗi (Train Loss)**: ...
- **Mức độ lỗi (Val Loss)**: ...

### 4. Nhận xét & Đánh giá (Notes)
- Mô hình học tốt hay bị overfitting/underfitting?
- Nhận diện loại quả nào tốt nhất, loại quả nào dễ bị nhầm lẫn nhất?
- (Ví dụ: Nhầm lẫn nhiều giữa Cam và Quýt do màu sắc tương đồng).
- **Hướng cải thiện cho lần sau**: Cần thêm dữ liệu augmentation (xoay, lật ảnh) hoặc thay đổi learning rate.

---

*(Copy lại mẫu trên cho mỗi lần huấn luyện mới)*
