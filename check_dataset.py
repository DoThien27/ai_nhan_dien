"""
check_dataset.py - Kiểm tra dataset trước khi huấn luyện
==========================================================
Script này giúp bạn kiểm tra:
- Số lượng ảnh trong từng class
- Tỷ lệ phân bổ train/validation
- Cảnh báo nếu dataset không cân bằng

Cách chạy:
    python check_dataset.py
"""

from pathlib import Path
from collections import defaultdict

# ============================================================
# Cấu hình
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR   = BASE_DIR / "dataset" / "validation"

# Định dạng ảnh hỗ trợ
DINH_DANG_ANH = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

# Số ảnh tối thiểu khuyến nghị
MIN_ANH_TRAIN  = 50
MIN_ANH_VAL    = 10


def dem_anh_trong_thu_muc(thu_muc: Path) -> int:
    """
    Đếm số lượng file ảnh trong một thư mục.
    
    Args:
        thu_muc: Đường dẫn thư mục cần đếm
    
    Returns:
        Số lượng ảnh tìm thấy
    """
    if not thu_muc.exists():
        return 0
    so_anh = sum(
        1 for f in thu_muc.iterdir()
        if f.is_file() and f.suffix.lower() in DINH_DANG_ANH
    )
    return so_anh


def lay_danh_sach_class(thu_muc: Path):
    """
    Lấy danh sách class (tên thư mục con) trong dataset.
    
    Args:
        thu_muc: Thư mục dataset (train hoặc validation)
    
    Returns:
        List tên class đã sắp xếp
    """
    if not thu_muc.exists():
        return []
    return sorted([
        d.name for d in thu_muc.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def in_bang_thong_ke(du_lieu: dict):
    """
    In bảng thống kê đẹp ra màn hình.
    
    Args:
        du_lieu: Dict chứa thông tin từng class
    """
    if not du_lieu:
        print("   (Không có dữ liệu)")
        return

    # Tính độ rộng cột
    do_rong_ten = max(len(k) for k in du_lieu.keys()) + 2
    do_rong_ten = max(do_rong_ten, 15)

    # Header
    print(f"\n  {'Loại quả':<{do_rong_ten}} {'Train':>8} {'Validation':>12} {'Tổng':>8}  Trạng thái")
    print("  " + "-" * (do_rong_ten + 40))

    tong_train = 0
    tong_val   = 0
    co_canh_bao = False

    for cls, info in sorted(du_lieu.items()):
        n_train = info["train"]
        n_val   = info["val"]
        tong    = n_train + n_val
        tong_train += n_train
        tong_val   += n_val

        # Xác định trạng thái
        if n_train == 0:
            trang_thai = "❌ Không có ảnh train!"
            co_canh_bao = True
        elif n_train < MIN_ANH_TRAIN:
            trang_thai = f"⚠️  Train < {MIN_ANH_TRAIN} ảnh"
            co_canh_bao = True
        elif n_val == 0:
            trang_thai = "⚠️  Không có ảnh validation"
            co_canh_bao = True
        elif n_val < MIN_ANH_VAL:
            trang_thai = f"⚠️  Val < {MIN_ANH_VAL} ảnh"
            co_canh_bao = True
        else:
            trang_thai = "✅ Đủ"

        print(f"  {cls:<{do_rong_ten}} {n_train:>8,} {n_val:>12,} {tong:>8,}  {trang_thai}")

    # Tổng cộng
    print("  " + "-" * (do_rong_ten + 40))
    print(f"  {'TỔNG CỘNG':<{do_rong_ten}} {tong_train:>8,} {tong_val:>12,} {tong_train+tong_val:>8,}")

    return co_canh_bao, tong_train, tong_val


def kiem_tra_can_bang(du_lieu: dict):
    """
    Kiểm tra xem dataset có bị mất cân bằng không.
    
    Dataset mất cân bằng khi một class có quá nhiều ảnh so với class khác,
    dẫn đến model thiên vị class đó.
    
    Args:
        du_lieu: Dict chứa thông tin từng class
    """
    train_counts = [info["train"] for info in du_lieu.values() if info["train"] > 0]
    if len(train_counts) < 2:
        return

    max_count = max(train_counts)
    min_count = min(train_counts)

    if max_count > 0 and min_count > 0:
        ratio = max_count / min_count
        print(f"\n  📐 Tỷ lệ max/min (train): {ratio:.1f}x")

        if ratio > 5:
            print("  ⚠️  Dataset rất mất cân bằng (ratio > 5)!")
            print("     Nên bổ sung thêm ảnh cho các class ít hơn.")
        elif ratio > 2:
            print("  ⚠️  Dataset có phần mất cân bằng (ratio > 2).")
            print("     Cân nhắc thêm ảnh để cân bằng hơn.")
        else:
            print("  ✅ Dataset tương đối cân bằng.")


def main():
    print("=" * 65)
    print("   📊 KIỂM TRA DATASET - AI NHẬN DIỆN TRÁI CÂY")
    print("=" * 65)
    print(f"\n  📁 Thư mục train: {TRAIN_DIR}")
    print(f"  📁 Thư mục val:   {VAL_DIR}")

    # Lấy danh sách class từ cả 2 thư mục
    classes_train = set(lay_danh_sach_class(TRAIN_DIR))
    classes_val   = set(lay_danh_sach_class(VAL_DIR))
    tat_ca_class  = classes_train | classes_val

    if not tat_ca_class:
        print("\n  ❌ Không tìm thấy dữ liệu!")
        print(f"     Hãy tạo thư mục con trong: {TRAIN_DIR}")
        print("     Ví dụ: apple, banana, orange, ...")
        print("\n  📖 Xem README.md để biết cách chuẩn bị dataset.")
        return

    # Tổng hợp thông tin
    du_lieu = {}
    for cls in tat_ca_class:
        n_train = dem_anh_trong_thu_muc(TRAIN_DIR / cls)
        n_val   = dem_anh_trong_thu_muc(VAL_DIR / cls)
        du_lieu[cls] = {"train": n_train, "val": n_val}

    # In bảng thống kê
    print(f"\n  🍎 Tìm thấy {len(tat_ca_class)} loại quả:\n")
    ket_qua = in_bang_thong_ke(du_lieu)

    if ket_qua is None:
        return

    co_canh_bao, tong_train, tong_val = ket_qua

    # Kiểm tra cân bằng
    print()
    kiem_tra_can_bang(du_lieu)

    # Phân tích và đề xuất
    print("\n" + "=" * 65)
    print("  📋 PHÂN TÍCH:")
    print("=" * 65)

    class_thieu_train = [
        cls for cls, info in du_lieu.items()
        if info["train"] < MIN_ANH_TRAIN
    ]
    class_thieu_val = [
        cls for cls, info in du_lieu.items()
        if info["val"] < MIN_ANH_VAL
    ]
    class_khong_co_train = [
        cls for cls, info in du_lieu.items()
        if info["train"] == 0
    ]

    if class_khong_co_train:
        print(f"\n  ❌ Class không có ảnh train: {', '.join(class_khong_co_train)}")
        print("     Phải thêm ảnh mới có thể huấn luyện!")

    if class_thieu_train:
        print(f"\n  ⚠️  Class cần thêm ảnh train (< {MIN_ANH_TRAIN}):")
        for cls in class_thieu_train:
            can_them = MIN_ANH_TRAIN - du_lieu[cls]["train"]
            print(f"     - {cls}: {du_lieu[cls]['train']} ảnh (cần thêm ~{can_them} ảnh)")

    if class_thieu_val:
        print(f"\n  ⚠️  Class cần thêm ảnh validation (< {MIN_ANH_VAL}):")
        for cls in class_thieu_val:
            print(f"     - {cls}: {du_lieu[cls]['val']} ảnh")

    # Kết luận
    print()
    if not class_khong_co_train and len(tat_ca_class) >= 2:
        if co_canh_bao:
            print("  ⚡ Dataset CÓ THỂ huấn luyện, nhưng độ chính xác chưa cao.")
            print("     Nên thêm ảnh trước khi train để kết quả tốt hơn.")
        else:
            print("  ✅ Dataset ĐÃ SẴN SÀNG để huấn luyện!")
            print("     Chạy: python train.py")
    elif len(tat_ca_class) < 2:
        print("  ❌ Cần ít nhất 2 loại quả để huấn luyện!")
    else:
        print("  ❌ Dataset chưa đủ để huấn luyện. Hãy thêm ảnh trước!")

    print("\n  💡 Gợi ý: Tìm ảnh trên Google Images hoặc dùng dataset Fruits-360")
    print("     https://www.kaggle.com/datasets/moltean/fruits")
    print("=" * 65)


if __name__ == "__main__":
    main()
