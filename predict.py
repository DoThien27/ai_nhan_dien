"""
predict.py - Dự đoán loại trái cây từ một ảnh
================================================
Sử dụng model đã huấn luyện để nhận diện trái cây.

Cách dùng:
    python predict.py
    Sau đó nhập đường dẫn đến ảnh cần nhận diện.
"""

import json
import numpy as np
from pathlib import Path

# ============================================================
# Kiểm tra thư viện trước khi import
# ============================================================
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("❌ Lỗi: Chưa cài TensorFlow!")
    print("   Chạy lệnh: pip install tensorflow")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Lỗi: Chưa cài Pillow!")
    print("   Chạy lệnh: pip install Pillow")
    exit(1)

# ============================================================
# Cấu hình (phải khớp với train.py)
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fruit_model.keras"
CLASS_FILE = BASE_DIR / "models" / "class_names.json"

IMG_SIZE             = 128   # Phải bằng IMG_SIZE trong train.py
CONFIDENCE_THRESHOLD = 0.60  # Ngưỡng tin cậy tối thiểu (60%)


def tai_model_va_classes():
    """
    Load model đã huấn luyện và danh sách class.
    
    Returns:
        (model, class_names) hoặc (None, None) nếu lỗi
    """
    # Kiểm tra file model tồn tại
    if not MODEL_PATH.exists():
        print(f"❌ Lỗi: Không tìm thấy model tại: {MODEL_PATH}")
        print("   Hãy chạy train.py trước để huấn luyện mô hình.")
        return None, None

    # Kiểm tra file class tồn tại
    if not CLASS_FILE.exists():
        print(f"❌ Lỗi: Không tìm thấy file class tại: {CLASS_FILE}")
        print("   Hãy chạy lại train.py.")
        return None, None

    # Load danh sách class
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    print(f"✅ Đã load model: {MODEL_PATH.name}")
    print(f"✅ Các loại quả nhận diện được: {', '.join(class_names)}\n")

    # Load model (có thể mất vài giây)
    try:
        model = keras.models.load_model(str(MODEL_PATH))
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None, None

    return model, class_names


def xu_ly_anh(duong_dan_anh):
    """
    Đọc và tiền xử lý ảnh để đưa vào model dự đoán.
    Quy trình giống hệt lúc huấn luyện.
    
    Args:
        duong_dan_anh: Đường dẫn đến file ảnh
    
    Returns:
        numpy array shape (1, IMG_SIZE, IMG_SIZE, 3) hoặc None nếu lỗi
    """
    anh_path = Path(duong_dan_anh)

    # Kiểm tra file tồn tại
    if not anh_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file ảnh: {anh_path}")
        return None

    # Kiểm tra định dạng file
    dinh_dang_ho_tro = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    if anh_path.suffix.lower() not in dinh_dang_ho_tro:
        print(f"❌ Lỗi: Định dạng không hỗ trợ: {anh_path.suffix}")
        print(f"   Các định dạng hỗ trợ: {', '.join(dinh_dang_ho_tro)}")
        return None

    try:
        # Mở ảnh bằng Pillow và chuyển sang RGB
        anh = Image.open(anh_path).convert("RGB")

        # Resize về kích thước cố định (giống lúc train)
        anh = anh.resize((IMG_SIZE, IMG_SIZE))

        # Chuyển thành numpy array
        anh_array = np.array(anh, dtype=np.float32)

        # Chuẩn hóa pixel về [0, 1] (giống rescale=1/255 lúc train)
        anh_array = anh_array / 255.0

        # Thêm chiều batch: (H, W, C) -> (1, H, W, C)
        anh_array = np.expand_dims(anh_array, axis=0)

        return anh_array

    except Exception as e:
        print(f"❌ Lỗi khi đọc ảnh: {e}")
        return None


def du_doan(model, class_names, duong_dan_anh):
    """
    Thực hiện dự đoán loại quả từ ảnh.
    
    Args:
        model: Model Keras đã load
        class_names: Danh sách tên class
        duong_dan_anh: Đường dẫn ảnh cần dự đoán
    
    Returns:
        (ten_qua, do_tin_cay) hoặc (None, None) nếu lỗi
    """
    # Tiền xử lý ảnh
    anh_array = xu_ly_anh(duong_dan_anh)
    if anh_array is None:
        return None, None

    # Dự đoán - model trả về xác suất cho mỗi class
    ket_qua = model.predict(anh_array, verbose=0)

    # Lấy index của class có xác suất cao nhất
    index_cao_nhat = np.argmax(ket_qua[0])
    do_tin_cay = float(ket_qua[0][index_cao_nhat])
    ten_qua = class_names[index_cao_nhat]

    return ten_qua, do_tin_cay


def hien_thi_ket_qua(ten_qua, do_tin_cay, class_names, ket_qua_day_du=None):
    """
    Hiển thị kết quả dự đoán đẹp và rõ ràng.
    
    Args:
        ten_qua: Tên loại quả được dự đoán
        do_tin_cay: Độ tin cậy (0.0 - 1.0)
        class_names: Danh sách tất cả class
        ket_qua_day_du: Array xác suất của tất cả class (tùy chọn)
    """
    print("\n" + "=" * 50)
    print("         KẾT QUẢ NHẬN DIỆN")
    print("=" * 50)

    if do_tin_cay < CONFIDENCE_THRESHOLD:
        print(f"  ⚠️  AI không chắc chắn về kết quả!")
        print(f"  Dự đoán:     {ten_qua.upper()}")
        print(f"  Độ tin cậy:  {do_tin_cay*100:.1f}% (thấp hơn ngưỡng {CONFIDENCE_THRESHOLD*100:.0f}%)")
        print(f"\n  💡 Gợi ý: Thử ảnh rõ hơn hoặc cần thêm dữ liệu huấn luyện.")
    else:
        print(f"  🍎 Loại quả:   {ten_qua.upper()}")
        print(f"  📊 Độ tin cậy: {do_tin_cay*100:.1f}%")

    # Hiển thị thanh tiến độ trực quan
    print()
    bar_filled = int(do_tin_cay * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"  [{bar}] {do_tin_cay*100:.1f}%")

    # Hiển thị top-3 kết quả nếu có
    if ket_qua_day_du is not None and len(class_names) > 1:
        print("\n  📋 Top kết quả:")
        sorted_indices = np.argsort(ket_qua_day_du)[::-1]
        for i, idx in enumerate(sorted_indices[:3]):
            pct = ket_qua_day_du[idx] * 100
            marker = "→" if i == 0 else "  "
            print(f"  {marker} {i+1}. {class_names[idx]:15s} {pct:.1f}%")

    print("=" * 50)


def main():
    print("=" * 50)
    print("   🍎 AI NHẬN DIỆN TRÁI CÂY - DỰ ĐOÁN")
    print("=" * 50)
    print()

    # ---- Load model và class ----
    model, class_names = tai_model_va_classes()
    if model is None:
        return

    # ---- Vòng lặp nhận input từ người dùng ----
    while True:
        print("Nhập đường dẫn ảnh (hoặc gõ 'exit' để thoát):")
        duong_dan = input(">>> ").strip()

        # Xóa dấu nháy nếu người dùng copy path có dấu nháy
        duong_dan = duong_dan.strip('"').strip("'")

        if duong_dan.lower() in ("exit", "quit", "thoat", "q"):
            print("👋 Tạm biệt!")
            break

        if not duong_dan:
            print("⚠️  Vui lòng nhập đường dẫn ảnh.\n")
            continue

        # ---- Thực hiện dự đoán ----
        print(f"\n🔍 Đang phân tích ảnh: {duong_dan}")

        anh_array = xu_ly_anh(duong_dan)
        if anh_array is None:
            print()
            continue

        # Dự đoán và lấy toàn bộ xác suất
        ket_qua_day_du_raw = model.predict(anh_array, verbose=0)[0]
        index_cao_nhat = np.argmax(ket_qua_day_du_raw)
        do_tin_cay = float(ket_qua_day_du_raw[index_cao_nhat])
        ten_qua = class_names[index_cao_nhat]

        # Hiển thị kết quả
        hien_thi_ket_qua(ten_qua, do_tin_cay, class_names, ket_qua_day_du_raw)
        print()


if __name__ == "__main__":
    main()
