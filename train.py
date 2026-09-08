"""
train.py - Huấn luyện mô hình AI nhận diện trái cây
=====================================================
Script này sẽ:
1. Đọc dataset từ thư mục dataset/train và dataset/validation
2. Xây dựng mô hình CNN bằng TensorFlow/Keras
3. Huấn luyện mô hình và hiển thị kết quả
4. Lưu mô hình vào models/fruit_model.keras
5. Lưu danh sách class vào models/class_names.txt
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Kiểm tra TensorFlow trước khi import
# ============================================================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except ImportError:
    print("❌ Lỗi: Chưa cài TensorFlow!")
    print("   Chạy lệnh: pip install tensorflow")
    exit(1)

# ============================================================
# Cấu hình chung
# ============================================================
BASE_DIR    = Path(__file__).resolve().parent   # Thư mục gốc của project
TRAIN_DIR   = BASE_DIR / "dataset" / "train"    # Thư mục train
VAL_DIR     = BASE_DIR / "dataset" / "validation"  # Thư mục validation
MODEL_DIR   = BASE_DIR / "models"               # Thư mục lưu model
MODEL_PATH  = MODEL_DIR / "fruit_model.keras"   # Đường dẫn lưu model
CLASS_FILE  = MODEL_DIR / "class_names.json"    # File lưu tên class

IMG_SIZE    = 128           # Kích thước ảnh (128x128 pixel)
BATCH_SIZE  = 32            # Số ảnh mỗi batch
EPOCHS      = 30            # Số vòng huấn luyện tối đa
CONFIDENCE_THRESHOLD = 0.60 # Ngưỡng độ tin cậy tối thiểu


def kiem_tra_dataset():
    """
    Kiểm tra xem dataset có tồn tại và có đủ ảnh không.
    Trả về True nếu hợp lệ, False nếu không.
    """
    print("=" * 60)
    print("🔍 Kiểm tra dataset...")
    print("=" * 60)

    # Kiểm tra thư mục train tồn tại
    if not TRAIN_DIR.exists():
        print(f"❌ Lỗi: Không tìm thấy thư mục train: {TRAIN_DIR}")
        print("   Hãy tạo cấu trúc thư mục theo README.md")
        return False

    if not VAL_DIR.exists():
        print(f"❌ Lỗi: Không tìm thấy thư mục validation: {VAL_DIR}")
        print("   Hãy tạo cấu trúc thư mục theo README.md")
        return False

    # Lấy danh sách class từ thư mục con trong train
    classes = sorted([
        d.name for d in TRAIN_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if len(classes) == 0:
        print(f"❌ Lỗi: Không tìm thấy thư mục con nào trong: {TRAIN_DIR}")
        print("   Mỗi thư mục con là một loại quả (vd: apple, banana, ...)")
        return False

    if len(classes) < 2:
        print(f"❌ Lỗi: Cần ít nhất 2 loại quả để huấn luyện!")
        print(f"   Hiện chỉ có: {classes}")
        return False

    # Kiểm tra số lượng ảnh trong từng class
    print(f"\n📦 Tìm thấy {len(classes)} loại quả:\n")
    co_loi = False

    for cls in classes:
        train_cls_dir = TRAIN_DIR / cls
        val_cls_dir   = VAL_DIR / cls

        # Đếm ảnh train
        anh_train = list(train_cls_dir.glob("*.jpg")) + \
                    list(train_cls_dir.glob("*.jpeg")) + \
                    list(train_cls_dir.glob("*.png")) + \
                    list(train_cls_dir.glob("*.bmp"))

        # Đếm ảnh validation
        anh_val = []
        if val_cls_dir.exists():
            anh_val = list(val_cls_dir.glob("*.jpg")) + \
                      list(val_cls_dir.glob("*.jpeg")) + \
                      list(val_cls_dir.glob("*.png")) + \
                      list(val_cls_dir.glob("*.bmp"))

        trang_thai = "✅" if len(anh_train) >= 20 else "⚠️ "
        print(f"  {trang_thai} {cls:20s} → Train: {len(anh_train):4d} ảnh | Validation: {len(anh_val):4d} ảnh")

        if len(anh_train) < 10:
            print(f"       ⚠️  Cảnh báo: '{cls}' chỉ có {len(anh_train)} ảnh train (khuyến nghị ≥ 50 ảnh)")
            co_loi = True

    if co_loi:
        print("\n⚠️  Một số class có quá ít ảnh, độ chính xác có thể thấp.")
        print("   Nhưng vẫn có thể tiếp tục huấn luyện.\n")

    print()
    return True


def tao_data_generators():
    """
    Tạo ImageDataGenerator để load và augment ảnh tự động.
    Data augmentation giúp model tổng quát hóa tốt hơn.
    """
    # Augmentation cho tập train (tăng cường dữ liệu)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # Chuẩn hóa pixel về [0, 1]
        rotation_range=20,           # Xoay ảnh ngẫu nhiên ±20 độ
        width_shift_range=0.2,       # Dịch ngang ngẫu nhiên 20%
        height_shift_range=0.2,      # Dịch dọc ngẫu nhiên 20%
        shear_range=0.15,            # Biến dạng cắt
        zoom_range=0.2,              # Phóng to/thu nhỏ ngẫu nhiên 20%
        horizontal_flip=True,        # Lật ngang ngẫu nhiên
        fill_mode="nearest"          # Điền pixel khi biến đổi
    )

    # Chỉ chuẩn hóa cho tập validation (không augment)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Tạo generator từ thư mục
    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),   # Resize về kích thước cố định
        batch_size=BATCH_SIZE,
        class_mode="categorical",            # Phân loại nhiều class
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        str(VAL_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen


def xay_dung_mo_hinh(so_class):
    """
    Xây dựng mô hình CNN (Convolutional Neural Network).
    
    Kiến trúc:
    - 3 khối Conv2D + MaxPooling để trích xuất đặc trưng ảnh
    - Dropout để chống overfitting
    - Dense layer để phân loại
    
    Args:
        so_class: Số lượng loại quả cần phân loại
    
    Returns:
        Mô hình Keras đã được compile
    """
    model = keras.Sequential([
        # ===== Khối 1: Phát hiện các đặc trưng cơ bản (cạnh, góc) =====
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),    # Chuẩn hóa batch để huấn luyện ổn định hơn
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),      # Giảm kích thước xuống 1/2
        layers.Dropout(0.25),           # Tắt ngẫu nhiên 25% neuron để chống overfitting

        # ===== Khối 2: Phát hiện đặc trưng phức tạp hơn =====
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ===== Khối 3: Phát hiện đặc trưng cao cấp (hình dạng, màu sắc) =====
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ===== Flatten và Fully Connected Layers =====
        layers.GlobalAveragePooling2D(),  # Thay Flatten để giảm overfitting
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),              # Dropout mạnh hơn ở lớp cuối
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        # ===== Lớp đầu ra: softmax cho phân loại nhiều class =====
        layers.Dense(so_class, activation="softmax")
    ])

    # Compile model với optimizer Adam và loss categorical_crossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def ve_bieu_do(history, save_path=None):
    """
    Vẽ biểu đồ Accuracy và Loss trong quá trình huấn luyện.
    
    Args:
        history: Kết quả huấn luyện từ model.fit()
        save_path: Đường dẫn để lưu biểu đồ (tùy chọn)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Kết quả huấn luyện mô hình AI Nhận diện Trái cây",
                 fontsize=14, fontweight="bold")

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # --- Biểu đồ Accuracy ---
    ax1.plot(epochs_range, history.history["accuracy"],
             "b-o", markersize=4, label="Train Accuracy")
    ax1.plot(epochs_range, history.history["val_accuracy"],
             "r-o", markersize=4, label="Validation Accuracy")
    ax1.set_title("Độ chính xác (Accuracy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # --- Biểu đồ Loss ---
    ax2.plot(epochs_range, history.history["loss"],
             "b-o", markersize=4, label="Train Loss")
    ax2.plot(epochs_range, history.history["val_loss"],
             "r-o", markersize=4, label="Validation Loss")
    ax2.set_title("Hàm mất mát (Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Đã lưu biểu đồ: {save_path}")

    plt.show()


def luu_danh_sach_class(class_indices):
    """
    Lưu danh sách tên class vào file JSON.
    File này sẽ được app.py và predict.py đọc để hiển thị đúng tên.
    
    Args:
        class_indices: Dict ánh xạ tên class -> chỉ số (từ ImageDataGenerator)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Đảo ngược dict: chỉ số -> tên class
    # Vì model trả về chỉ số, ta cần map ngược lại để lấy tên
    index_to_class = {v: k for k, v in class_indices.items()}

    # Sắp xếp theo thứ tự index
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu danh sách class: {CLASS_FILE}")
    print(f"   Classes: {class_names}")


def main():
    print("=" * 60)
    print("   🍎 AI NHẬN DIỆN TRÁI CÂY - HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)
    print(f"   TensorFlow version: {tf.__version__}")
    print()

    # Tạo thư mục models nếu chưa có
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Bước 1: Kiểm tra dataset ----
    if not kiem_tra_dataset():
        print("\n❌ Dừng huấn luyện do lỗi dataset.")
        print("   Xem README.md để biết cách chuẩn bị dataset.")
        return

    # ---- Bước 2: Tạo data generators ----
    print("📂 Đang load dataset...")
    train_gen, val_gen = tao_data_generators()

    so_class   = len(train_gen.class_indices)
    so_mau_train = train_gen.samples
    so_mau_val   = val_gen.samples

    print(f"\n✅ Dataset đã load:")
    print(f"   - Số class: {so_class}")
    print(f"   - Ảnh train: {so_mau_train}")
    print(f"   - Ảnh validation: {so_mau_val}")
    print()

    # ---- Bước 3: Lưu danh sách class ----
    luu_danh_sach_class(train_gen.class_indices)

    # ---- Bước 4: Xây dựng mô hình ----
    print("\n🏗️  Đang xây dựng mô hình CNN...")
    model = xay_dung_mo_hinh(so_class)
    model.summary()
    print()

    # ---- Bước 5: Chuẩn bị callbacks ----
    callbacks = [
        # Dừng sớm nếu validation loss không cải thiện sau 7 epoch
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Lưu model tốt nhất trong quá trình train
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Giảm learning rate khi validation loss không cải thiện
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ---- Bước 6: Huấn luyện mô hình ----
    print("🚀 Bắt đầu huấn luyện...\n")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ---- Bước 7: Hiển thị kết quả cuối cùng ----
    val_acc  = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])

    print("\n" + "=" * 60)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)
    print(f"   📈 Validation Accuracy tốt nhất: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   📉 Validation Loss tốt nhất:     {val_loss:.4f}")
    print(f"   💾 Model đã lưu tại: {MODEL_PATH}")
    print()

    # ---- Bước 8: Vẽ biểu đồ ----
    bieu_do_path = MODEL_DIR / "training_history.png"
    ve_bieu_do(history, save_path=bieu_do_path)

    print("\n🎉 Hoàn thành! Bạn có thể chạy app.py để nhận diện trái cây.")


if __name__ == "__main__":
    main()
