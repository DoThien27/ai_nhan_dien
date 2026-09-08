# 🍎 AI Nhận Diện Trái Cây

> **Đồ án / Bài tập Python + TensorFlow** - Nhận diện trái cây qua ảnh bằng mạng nơ-ron CNN

---

## 📦 Cấu trúc project

```
AI_nhan_dien_qua/
│
├── dataset/
│   ├── train/
│   │   ├── apple/        ← Ảnh táo để huấn luyện
│   │   ├── banana/       ← Ảnh chuối
│   │   ├── orange/       ← Ảnh cam
│   │   ├── grape/        ← Ảnh nho
│   │   └── mango/        ← Ảnh xoài (thêm loại mới tùy ý)
│   │
│   └── validation/
│       ├── apple/        ← Ảnh táo để kiểm tra
│       ├── banana/
│       ├── orange/
│       ├── grape/
│       └── mango/
│
├── models/
│   ├── fruit_model.keras      ← Model đã huấn luyện (tự tạo sau train)
│   ├── class_names.json       ← Danh sách tên class (tự tạo sau train)
│   └── training_history.png   ← Biểu đồ quá trình train
│
├── train.py           ← Huấn luyện mô hình AI
├── predict.py         ← Dự đoán qua dòng lệnh
├── app.py             ← Giao diện đồ họa Tkinter
├── check_dataset.py   ← Kiểm tra dataset trước khi train
├── requirements.txt   ← Danh sách thư viện cần cài
└── README.md          ← File này
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1: Cài Python

- Tải Python 3.10+ từ: https://www.python.org/downloads/
- ✅ **Quan trọng**: Tích chọn **"Add Python to PATH"** khi cài
- Kiểm tra sau khi cài:
  ```bash
  python --version
  ```

---

### Bước 2: Tạo môi trường ảo (khuyến nghị)

Môi trường ảo giúp cô lập thư viện, tránh xung đột:

```bash
# Mở terminal trong thư mục project (d:\AI_nhan_dien_qua)
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
venv\Scripts\activate

# Bạn sẽ thấy (venv) ở đầu dòng lệnh
```

---

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

> ⏳ Lần đầu cài TensorFlow sẽ mất vài phút (~500MB). Hãy kiên nhẫn.

---

### Bước 4: Chuẩn bị dataset

Đặt ảnh trái cây vào đúng thư mục:

```
dataset/train/apple/     ← Khoảng 50-100 ảnh táo
dataset/train/banana/    ← Khoảng 50-100 ảnh chuối
...

dataset/validation/apple/    ← Khoảng 10-20 ảnh táo
dataset/validation/banana/   ← Khoảng 10-20 ảnh chuối
...
```

**Định dạng hỗ trợ:** `.jpg`, `.jpeg`, `.png`, `.bmp`

**Nguồn dataset gợi ý:**
- 🏆 **Fruits-360** (tốt nhất): https://www.kaggle.com/datasets/moltean/fruits
- Google Images (tìm kiếm thủ công)
- Tự chụp bằng điện thoại

**Thêm loại quả mới:**
```bash
# Chỉ cần tạo thêm thư mục, không cần sửa code
mkdir dataset\train\strawberry
mkdir dataset\validation\strawberry
# Sau đó thêm ảnh vào 2 thư mục trên
```

---

### Bước 5: Kiểm tra dataset

Trước khi train, kiểm tra xem dataset có đủ không:

```bash
python check_dataset.py
```

Kết quả mẫu:
```
📊 KIỂM TRA DATASET - AI NHẬN DIỆN TRÁI CÂY
  Loại quả         Train  Validation  Tổng   Trạng thái
  apple               80          20   100   ✅ Đủ
  banana              75          18    93   ✅ Đủ
  orange              60          15    75   ✅ Đủ
```

---

### Bước 6: Huấn luyện AI

```bash
python train.py
```

Quá trình train sẽ:
- Hiển thị tiến độ theo từng epoch
- Tự động dừng nếu không còn cải thiện (EarlyStopping)
- Lưu model tốt nhất vào `models/fruit_model.keras`
- Vẽ biểu đồ accuracy và loss sau khi xong

> ⏳ Thời gian train phụ thuộc vào số ảnh và cấu hình máy (thường 5-30 phút)

---

### Bước 7: Kiểm tra bằng dòng lệnh

```bash
python predict.py
```

Nhập đường dẫn ảnh khi được hỏi:
```
>>> C:\Users\TenBan\Pictures\tao.jpg

KẾT QUẢ NHẬN DIỆN
  🍎 Loại quả:   APPLE
  📊 Độ tin cậy: 94.3%
```

---

### Bước 8: Chạy giao diện đồ họa

```bash
python app.py
```

Giao diện sẽ hiện lên cho phép:
- Bấm **"Chọn Ảnh"** để chọn ảnh từ máy
- Bấm **"Nhận Diện"** để AI phân tích
- Xem kết quả: tên quả, tên tiếng Việt, độ tin cậy, top-3 kết quả

---

## 💡 Giải thích AI hoạt động như thế nào

```
Ảnh đầu vào (128x128 pixel)
        ↓
┌─────────────────────────────────┐
│   Lớp Conv2D (x3)               │  ← Phát hiện đặc trưng:
│   Mỗi bộ lọc học cách nhận ra   │    cạnh, góc, màu sắc,
│   một đặc trưng trong ảnh       │    hình dạng...
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   MaxPooling + Dropout           │  ← Giảm kích thước,
│                                  │    chống overfitting
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   Dense Layer (Fully Connected) │  ← Kết hợp các đặc trưng
│                                  │    để đưa ra quyết định
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   Softmax Output                 │  ← Xác suất cho mỗi class
│   [apple: 0.92, banana: 0.05,   │    Chọn class có xác suất
│    orange: 0.03]                 │    cao nhất → Kết quả
└─────────────────────────────────┘
```

**Các khái niệm chính:**
- **CNN (Convolutional Neural Network)**: Mạng nơ-ron chuyên xử lý ảnh
- **Epoch**: Một lần model học qua toàn bộ dataset
- **Accuracy**: Tỷ lệ dự đoán đúng (càng cao càng tốt)
- **Loss**: Sai số của model (càng thấp càng tốt)
- **Overfitting**: Model học thuộc lòng train, không tổng quát hóa được
- **Dropout**: Kỹ thuật tắt ngẫu nhiên neuron để chống overfitting

---

## ❓ Câu hỏi thường gặp

**Q: Model cho kết quả không chính xác?**  
A: Cần nhiều ảnh hơn (≥ 100 ảnh/class) và ảnh đa dạng hơn.

**Q: Lỗi "No module named tensorflow"?**  
A: Chạy `pip install tensorflow` hoặc kiểm tra môi trường ảo đã kích hoạt chưa.

**Q: Thêm loại quả mới như thế nào?**  
A: Tạo thư mục mới trong `train/` và `validation/`, thêm ảnh, rồi chạy lại `train.py`.

**Q: Train xong nhưng accuracy thấp?**  
A: Thêm nhiều ảnh, tăng số epoch trong train.py (`EPOCHS = 50`), hoặc dùng dataset chất lượng cao.

---

## 📚 Tài liệu tham khảo

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Keras Image Classification](https://keras.io/guides/transfer_learning/)
- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)
- [CNN Explained](https://cs231n.github.io/convolutional-networks/)

---

*Được tạo để học tập và nghiên cứu. Chúc bạn học tốt! 🍀*
