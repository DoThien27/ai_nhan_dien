# ĐẶT ẢNH DATASET VÀO ĐÂY

Thư mục này dùng để chứa ảnh training cho AI nhận diện trái cây.

## Cấu trúc thư mục:

```
dataset/
├── train/           ← Ảnh dùng để HUẤN LUYỆN (70-80% tổng số ảnh)
│   ├── apple/       ← Ảnh táo
│   ├── banana/      ← Ảnh chuối
│   ├── orange/      ← Ảnh cam
│   ├── grape/       ← Ảnh nho
│   └── mango/       ← Ảnh xoài
│
└── validation/      ← Ảnh dùng để KIỂM TRA (20-30% tổng số ảnh)
    ├── apple/
    ├── banana/
    ├── orange/
    ├── grape/
    └── mango/
```

## Yêu cầu:
- Mỗi loại quả: tối thiểu 50 ảnh train + 10 ảnh validation
- Định dạng hỗ trợ: .jpg, .jpeg, .png, .bmp
- Ảnh phải rõ ràng, nền đơn giản sẽ cho kết quả tốt hơn

## Nguồn dataset gợi ý:
- Fruits-360: https://www.kaggle.com/datasets/moltean/fruits
- Google Images (tìm kiếm thủ công)
- Chụp ảnh thật bằng điện thoại

## Thêm loại quả mới:
Tạo thư mục mới trong cả train/ và validation/ với tên loại quả.
Ví dụ: tạo `train/strawberry/` và `validation/strawberry/` để thêm dâu tây.
