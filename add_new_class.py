import os
import sys
from pathlib import Path

# ============================================================
# Cấu hình đường dẫn
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"

def create_new_class(class_name):
    """
    Tạo thư mục cho loại quả mới trong tập train và validation.
    """
    class_name = class_name.strip().lower()
    
    if not class_name:
        print("❌ Tên loại quả không được để trống!")
        return False
        
    # Đường dẫn tới thư mục mới
    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name
    
    # Kiểm tra xem class đã tồn tại chưa
    if train_class_dir.exists() or val_class_dir.exists():
        print(f"⚠️  Loại quả '{class_name}' đã tồn tại trong dataset!")
        return False
        
    try:
        # Tạo thư mục
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*50)
        print(f"✅ Đã tạo thành công thư mục cho quả: {class_name.upper()}")
        print("="*50)
        print("👉 HƯỚNG DẪN BƯỚC TIẾP THEO:")
        print(f"1. Hãy copy khoảng 80% số ảnh của {class_name} vào thư mục:\n   📁 {train_class_dir}")
        print(f"2. Copy 20% số ảnh còn lại vào thư mục:\n   📁 {val_class_dir}")
        print("3. Sau khi copy xong ảnh, hãy chạy lại lệnh: python train.py")
        print("   (Mô hình sẽ tự động cấu hình lại lớp cuối để học thêm quả này!)")
        print("="*50 + "\n")
        
        return True
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi tạo thư mục: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("   ➕ THÊM LOẠI QUẢ MỚI VÀO DATASET")
    print("=" * 50)
    
    # Lấy tên quả từ tham số dòng lệnh nếu có, ngược lại thì hỏi người dùng
    if len(sys.argv) > 1:
        new_class = sys.argv[1]
    else:
        new_class = input("Nhập tên loại quả muốn thêm (vd: strawberry, watermelon): ")
        
    create_new_class(new_class)
