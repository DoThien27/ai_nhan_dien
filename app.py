"""
app.py - Giao diện desktop Tkinter để nhận diện trái cây
=========================================================
Giao diện đồ họa giúp người dùng chọn ảnh và xem kết quả nhận diện.

Cách chạy:
    python app.py
"""

import json
import threading
import numpy as np
from pathlib import Path

# ============================================================
# Kiểm tra thư viện Tkinter (có sẵn trong Python)
# ============================================================
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
except ImportError:
    print("❌ Tkinter không khả dụng. Hãy cài Python đầy đủ.")
    exit(1)

# ============================================================
# Kiểm tra Pillow
# ============================================================
try:
    from PIL import Image, ImageTk
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
SUMMARY_FILE = BASE_DIR / "models" / "training_summary.json"

IMG_SIZE             = 128   # Phải bằng IMG_SIZE trong train.py
CONFIDENCE_THRESHOLD = 0.60  # Ngưỡng tin cậy tối thiểu

# ============================================================
# Bảng màu và thiết kế giao diện
# ============================================================
COLORS = {
    "bg_dark":      "#1a1a2e",   # Nền tối xanh đậm
    "bg_panel":     "#16213e",   # Nền panel
    "bg_card":      "#0f3460",   # Nền card
    "accent":       "#e94560",   # Màu nhấn đỏ hồng
    "accent2":      "#f5a623",   # Màu nhấn vàng cam
    "text_white":   "#ffffff",   # Chữ trắng
    "text_gray":    "#a0a0b0",   # Chữ xám
    "text_light":   "#c8d6e5",   # Chữ xanh nhạt
    "success":      "#2ecc71",   # Màu thành công
    "warning":      "#f39c12",   # Màu cảnh báo
    "error":        "#e74c3c",   # Màu lỗi
    "btn_primary":  "#e94560",   # Nút chính
    "btn_hover":    "#c0392b",   # Nút hover
    "btn_secondary":"#0f3460",   # Nút phụ
}

# Emoji cho từng loại quả (có thể thêm vào)
FRUIT_EMOJI = {
    "apple":      "🍎",
    "banana":     "🍌",
    "orange":     "🍊",
    "grape":      "🍇",
    "mango":      "🥭",
    "strawberry": "🍓",
    "watermelon": "🍉",
    "pineapple":  "🍍",
    "kiwi":       "🥝",
    "pear":       "🍐",
    "cherry":     "🍒",
    "peach":      "🍑",
    "lemon":      "🍋",
    "coconut":    "🥥",
    "default":    "🍈",
}

# Tên tiếng Việt cho từng loại quả
FRUIT_NAMES_VI = {
    "apple":       "Táo",
    "banana":      "Chuối",
    "orange":      "Cam",
    "grape":       "Nho",
    "mango":       "Xoài",
    "strawberry":  "Dâu tây",
    "watermelon":  "Dưa hấu",
    "pineapple":   "Dứa",
    "kiwi":        "Kiwi",
    "pear":        "Lê",
    "cherry":      "Cherry",
    "peach":       "Đào",
    "lemon":       "Chanh vàng",
    "coconut":     "Dừa",
    "tomato":      "Cà chua",
    "avocado":     "Bơ",
    "papaya":      "Đu đủ",
    "guava":       "Ổi",
    "dragon_fruit":"Thanh long",
    "durian":      "Sầu riêng",
    "lychee":      "Vải",
    "rambutan":    "Chôm chôm",
    "jackfruit":   "Mít",
}


class FruitRecognitionApp:
    """
    Lớp chính của ứng dụng nhận diện trái cây.
    Sử dụng Tkinter để tạo giao diện đồ họa.
    """

    def __init__(self, root):
        self.root = root
        self.model = None          # Model AI
        self.class_names = []      # Danh sách tên class
        self.anh_hien_tai = None   # Đường dẫn ảnh đang chọn
        self.photo_tk = None       # Ảnh Tkinter để hiển thị

        self._cau_hinh_cua_so()
        self._tao_giao_dien()
        self._tai_model_nen()     # Load model trong background

    def _cau_hinh_cua_so(self):
        """Cấu hình cửa sổ chính"""
        self.root.title("AI Nhận Diện Trái Cây")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(True, True)

        # Đặt icon nếu có
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        # Căn giữa màn hình
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def _tao_giao_dien(self):
        """Tạo toàn bộ giao diện"""
        # ===== TIÊU ĐỀ =====
        frame_title = tk.Frame(self.root, bg=COLORS["bg_dark"], pady=15)
        frame_title.pack(fill="x")

        tk.Label(
            frame_title,
            text="🍎  AI NHẬN DIỆN TRÁI CÂY",
            font=("Segoe UI", 22, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_dark"]
        ).pack()

        tk.Label(
            frame_title,
            text="Powered by TensorFlow / Keras CNN",
            font=("Segoe UI", 10),
            fg=COLORS["text_gray"],
            bg=COLORS["bg_dark"]
        ).pack()

        # Hiện thông tin file summary nếu có
        if SUMMARY_FILE.exists():
            try:
                with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                info_text = f"Model trained: {summary['trained_at'][:10]} | Acc: {summary['final_metrics']['val_accuracy']*100:.1f}% | Classes: {summary['total_classes']}"
                tk.Label(
                    frame_title,
                    text=info_text,
                    font=("Segoe UI", 9, "italic"),
                    fg=COLORS["success"],
                    bg=COLORS["bg_dark"]
                ).pack(pady=(5, 0))
            except Exception:
                pass

        # ===== THANH TRẠNG THÁI MODEL =====
        self.frame_status = tk.Frame(self.root, bg=COLORS["bg_panel"], pady=8)
        self.frame_status.pack(fill="x", padx=20, pady=(0, 10))

        self.lbl_status = tk.Label(
            self.frame_status,
            text="⏳ Đang tải model...",
            font=("Segoe UI", 10),
            fg=COLORS["warning"],
            bg=COLORS["bg_panel"]
        )
        self.lbl_status.pack()

        # ===== NỘI DUNG CHÍNH (2 cột) =====
        frame_main = tk.Frame(self.root, bg=COLORS["bg_dark"])
        frame_main.pack(fill="both", expand=True, padx=20, pady=5)
        frame_main.columnconfigure(0, weight=3)
        frame_main.columnconfigure(1, weight=2)
        frame_main.rowconfigure(0, weight=1)

        # --- Cột trái: hiển thị ảnh ---
        self._tao_panel_anh(frame_main)

        # --- Cột phải: kết quả và nút ---
        self._tao_panel_ket_qua(frame_main)

        # ===== THANH DƯỚI: Nút chọn và nhận diện =====
        self._tao_thanh_nut()

    def _tao_panel_anh(self, parent):
        """Tạo panel hiển thị ảnh (cột trái)"""
        frame_anh = tk.Frame(
            parent,
            bg=COLORS["bg_panel"],
            relief="flat",
            bd=0
        )
        frame_anh.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(
            frame_anh,
            text="🖼  ẢNH NHẬN DIỆN",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_light"],
            bg=COLORS["bg_panel"],
            pady=10
        ).pack()

        # Khung hiển thị ảnh
        self.frame_canvas = tk.Frame(
            frame_anh,
            bg=COLORS["bg_card"],
            width=380,
            height=380
        )
        self.frame_canvas.pack(padx=15, pady=(0, 15))
        self.frame_canvas.pack_propagate(False)

        # Label placeholder khi chưa chọn ảnh
        self.lbl_anh = tk.Label(
            self.frame_canvas,
            text="📁\n\nChưa có ảnh\n\nBấm 'Chọn Ảnh' để bắt đầu",
            font=("Segoe UI", 12),
            fg=COLORS["text_gray"],
            bg=COLORS["bg_card"],
            justify="center"
        )
        self.lbl_anh.pack(fill="both", expand=True)

        # Tên file ảnh
        self.lbl_ten_file = tk.Label(
            frame_anh,
            text="",
            font=("Segoe UI", 9),
            fg=COLORS["text_gray"],
            bg=COLORS["bg_panel"],
            wraplength=350
        )
        self.lbl_ten_file.pack(pady=(0, 10))

    def _tao_panel_ket_qua(self, parent):
        """Tạo panel kết quả (cột phải)"""
        frame_kq = tk.Frame(
            parent,
            bg=COLORS["bg_panel"],
            relief="flat"
        )
        frame_kq.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            frame_kq,
            text="📊  KẾT QUẢ",
            font=("Segoe UI", 11, "bold"),
            fg=COLORS["text_light"],
            bg=COLORS["bg_panel"],
            pady=10
        ).pack()

        # ---- Emoji loại quả ----
        self.lbl_emoji = tk.Label(
            frame_kq,
            text="🍈",
            font=("Segoe UI", 60),
            bg=COLORS["bg_panel"]
        )
        self.lbl_emoji.pack(pady=10)

        # ---- Tên loại quả ----
        self.lbl_ten_qua = tk.Label(
            frame_kq,
            text="---",
            font=("Segoe UI", 20, "bold"),
            fg=COLORS["text_white"],
            bg=COLORS["bg_panel"]
        )
        self.lbl_ten_qua.pack()

        # ---- Tên tiếng Việt ----
        self.lbl_ten_vi = tk.Label(
            frame_kq,
            text="",
            font=("Segoe UI", 13),
            fg=COLORS["accent2"],
            bg=COLORS["bg_panel"]
        )
        self.lbl_ten_vi.pack(pady=2)

        # ---- Độ tin cậy ----
        tk.Label(
            frame_kq,
            text="Độ tin cậy:",
            font=("Segoe UI", 10),
            fg=COLORS["text_gray"],
            bg=COLORS["bg_panel"]
        ).pack(pady=(15, 0))

        self.lbl_do_tin_cay = tk.Label(
            frame_kq,
            text="---",
            font=("Segoe UI", 28, "bold"),
            fg=COLORS["success"],
            bg=COLORS["bg_panel"]
        )
        self.lbl_do_tin_cay.pack()

        # ---- Thanh tiến độ ----
        frame_bar = tk.Frame(frame_kq, bg=COLORS["bg_panel"], padx=20)
        frame_bar.pack(fill="x", pady=5)

        # Nền thanh
        self.canvas_bar = tk.Canvas(
            frame_bar,
            height=20,
            bg=COLORS["bg_card"],
            highlightthickness=0
        )
        self.canvas_bar.pack(fill="x")
        self.thanh_progress = self.canvas_bar.create_rectangle(
            0, 0, 0, 20,
            fill=COLORS["success"],
            outline=""
        )

        # ---- Nhận xét ----
        self.lbl_nhan_xet = tk.Label(
            frame_kq,
            text="",
            font=("Segoe UI", 10),
            fg=COLORS["text_gray"],
            bg=COLORS["bg_panel"],
            wraplength=220,
            justify="center"
        )
        self.lbl_nhan_xet.pack(pady=10)

        # ---- Top-3 kết quả ----
        tk.Label(
            frame_kq,
            text="Top kết quả:",
            font=("Segoe UI", 9, "bold"),
            fg=COLORS["text_gray"],
            bg=COLORS["bg_panel"]
        ).pack(anchor="w", padx=20)

        self.frame_top3 = tk.Frame(frame_kq, bg=COLORS["bg_panel"])
        self.frame_top3.pack(fill="x", padx=20, pady=5)

        self.lbl_top3 = []
        for i in range(3):
            lbl = tk.Label(
                self.frame_top3,
                text="",
                font=("Segoe UI", 9),
                fg=COLORS["text_gray"],
                bg=COLORS["bg_panel"],
                anchor="w"
            )
            lbl.pack(fill="x")
            self.lbl_top3.append(lbl)

    def _tao_thanh_nut(self):
        """Tạo thanh nút bên dưới"""
        frame_nut = tk.Frame(self.root, bg=COLORS["bg_panel"], pady=15)
        frame_nut.pack(fill="x", padx=20, pady=10)

        # Nút Chọn Ảnh
        self.btn_chon_anh = tk.Button(
            frame_nut,
            text="📁  Chọn Ảnh",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["text_white"],
            activebackground=COLORS["accent"],
            activeforeground=COLORS["text_white"],
            relief="flat",
            padx=25,
            pady=10,
            cursor="hand2",
            command=self._chon_anh
        )
        self.btn_chon_anh.pack(side="left", padx=(20, 10))

        # Nút Nhận Diện
        self.btn_nhan_dien = tk.Button(
            frame_nut,
            text="🔍  Nhận Diện",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["btn_primary"],
            fg=COLORS["text_white"],
            activebackground=COLORS["btn_hover"],
            activeforeground=COLORS["text_white"],
            relief="flat",
            padx=25,
            pady=10,
            cursor="hand2",
            command=self._nhan_dien,
            state="disabled"
        )
        self.btn_nhan_dien.pack(side="left", padx=10)

        # Nút Xóa kết quả
        btn_xoa = tk.Button(
            frame_nut,
            text="🗑  Xóa",
            font=("Segoe UI", 12),
            bg=COLORS["bg_panel"],
            fg=COLORS["text_gray"],
            activebackground=COLORS["bg_card"],
            activeforeground=COLORS["text_white"],
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._xoa_ket_qua
        )
        btn_xoa.pack(side="left", padx=10)
        
        # Nút Thêm quả mới
        btn_them_qua = tk.Button(
            frame_nut,
            text="➕ Thêm Loại Quả",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["success"],
            fg=COLORS["text_white"],
            activebackground="#27ae60",
            activeforeground=COLORS["text_white"],
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._them_loai_qua
        )
        btn_them_qua.pack(side="right", padx=10)

        # Thêm hiệu ứng hover cho nút
        self._them_hover(self.btn_chon_anh, COLORS["bg_card"], COLORS["accent"])
        self._them_hover(btn_xoa, COLORS["bg_panel"], COLORS["bg_card"])
        self._them_hover(btn_them_qua, COLORS["success"], "#27ae60")

    def _them_hover(self, btn, color_normal, color_hover):
        """Thêm hiệu ứng đổi màu khi hover chuột lên nút"""
        btn.bind("<Enter>", lambda e: btn.configure(bg=color_hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=color_normal))

    def _tai_model_nen(self):
        """Load model trong luồng nền để không làm đóng băng giao diện"""
        def _load():
            # Kiểm tra file model
            if not MODEL_PATH.exists():
                self.root.after(0, lambda: self._cap_nhat_status(
                    "⚠️ Chưa có model. Hãy chạy train.py để huấn luyện!",
                    COLORS["warning"]
                ))
                return

            if not CLASS_FILE.exists():
                self.root.after(0, lambda: self._cap_nhat_status(
                    "⚠️ Thiếu file class. Hãy chạy lại train.py!",
                    COLORS["warning"]
                ))
                return

            # Đọc danh sách class
            with open(CLASS_FILE, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

            # Load model TensorFlow (import ở đây để tránh chậm khởi động)
            try:
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(str(MODEL_PATH))

                classes_str = ", ".join(self.class_names)
                self.root.after(0, lambda: self._cap_nhat_status(
                    f"✅ Model đã sẵn sàng! Nhận diện: {classes_str}",
                    COLORS["success"]
                ))
                # Kích hoạt nút nhận diện
                self.root.after(0, lambda: self.btn_nhan_dien.configure(state="normal"))

            except Exception as e:
                self.root.after(0, lambda: self._cap_nhat_status(
                    f"❌ Lỗi load model: {str(e)[:80]}",
                    COLORS["error"]
                ))

        # Chạy trong luồng riêng
        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _cap_nhat_status(self, text, color):
        """Cập nhật thanh trạng thái"""
        self.lbl_status.configure(text=text, fg=color)

    def _chon_anh(self):
        """Mở hộp thoại chọn file ảnh"""
        duong_dan = filedialog.askopenfilename(
            title="Chọn ảnh trái cây",
            filetypes=[
                ("Ảnh", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("PNG", "*.png"),
                ("Tất cả", "*.*")
            ]
        )

        if duong_dan:
            self.anh_hien_tai = duong_dan
            self._hien_thi_anh(duong_dan)
            # Xóa kết quả cũ khi chọn ảnh mới
            self._xoa_ket_qua(xoa_anh=False)

    def _hien_thi_anh(self, duong_dan):
        """Hiển thị ảnh được chọn lên giao diện"""
        try:
            anh = Image.open(duong_dan).convert("RGB")

            # Resize để vừa khung hiển thị (tối đa 360x360)
            anh.thumbnail((360, 360), Image.LANCZOS)

            self.photo_tk = ImageTk.PhotoImage(anh)
            self.lbl_anh.configure(image=self.photo_tk, text="")

            # Hiển thị tên file
            ten_file = Path(duong_dan).name
            self.lbl_ten_file.configure(text=f"📄 {ten_file}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh:\n{e}")

    def _nhan_dien(self):
        """Thực hiện nhận diện ảnh"""
        # Kiểm tra đã chọn ảnh chưa
        if not self.anh_hien_tai:
            messagebox.showwarning(
                "Chưa chọn ảnh",
                "Vui lòng bấm 'Chọn Ảnh' trước khi nhận diện!"
            )
            return

        # Kiểm tra model đã load chưa
        if self.model is None:
            messagebox.showerror(
                "Model chưa sẵn sàng",
                "Model chưa được tải hoặc chưa huấn luyện.\n\n"
                "Hãy chạy train.py để huấn luyện model trước!"
            )
            return

        # Vô hiệu hóa nút trong lúc xử lý
        self.btn_nhan_dien.configure(state="disabled", text="⏳ Đang xử lý...")
        self.root.update()

        # Chạy dự đoán trong luồng nền
        def _predict():
            try:
                import numpy as np
                from PIL import Image as PILImage

                # Tiền xử lý ảnh
                anh = PILImage.open(self.anh_hien_tai).convert("RGB")
                anh = anh.resize((IMG_SIZE, IMG_SIZE))
                anh_array = np.array(anh, dtype=np.float32) / 255.0
                anh_array = np.expand_dims(anh_array, axis=0)

                # Dự đoán
                ket_qua = self.model.predict(anh_array, verbose=0)[0]
                idx = np.argmax(ket_qua)
                do_tin_cay = float(ket_qua[idx])
                ten_qua = self.class_names[idx]

                # Cập nhật giao diện từ luồng chính
                self.root.after(0, lambda: self._hien_thi_ket_qua(
                    ten_qua, do_tin_cay, ket_qua
                ))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Lỗi", f"Lỗi khi nhận diện:\n{e}"
                ))
            finally:
                self.root.after(0, lambda: self.btn_nhan_dien.configure(
                    state="normal", text="🔍  Nhận Diện"
                ))

        t = threading.Thread(target=_predict, daemon=True)
        t.start()

    def _hien_thi_ket_qua(self, ten_qua, do_tin_cay, ket_qua_day_du):
        """Cập nhật giao diện với kết quả dự đoán"""
        pct = do_tin_cay * 100

        # Lấy emoji và tên tiếng Việt
        emoji = FRUIT_EMOJI.get(ten_qua.lower(), FRUIT_EMOJI["default"])
        ten_vi = FRUIT_NAMES_VI.get(ten_qua.lower(), ten_qua.capitalize())

        # Cập nhật emoji
        self.lbl_emoji.configure(text=emoji)

        # Cập nhật tên quả
        self.lbl_ten_qua.configure(text=ten_qua.upper())
        self.lbl_ten_vi.configure(text=f"({ten_vi})")

        # Cập nhật độ tin cậy và màu sắc
        if do_tin_cay >= 0.85:
            mau = COLORS["success"]
            nhan_xet = "✅ AI rất tự tin về kết quả này!"
        elif do_tin_cay >= CONFIDENCE_THRESHOLD:
            mau = COLORS["accent2"]
            nhan_xet = "⚡ AI khá tự tin về kết quả này."
        else:
            mau = COLORS["error"]
            nhan_xet = "⚠️ AI không chắc chắn. Thử ảnh rõ hơn!"

        self.lbl_do_tin_cay.configure(text=f"{pct:.1f}%", fg=mau)
        self.lbl_nhan_xet.configure(text=nhan_xet, fg=mau)

        # Cập nhật thanh tiến độ
        self.canvas_bar.update_idletasks()
        w = self.canvas_bar.winfo_width()
        self.canvas_bar.coords(
            self.thanh_progress,
            0, 0, int(w * do_tin_cay), 20
        )
        self.canvas_bar.itemconfig(self.thanh_progress, fill=mau)

        # Hiển thị top-3 kết quả
        sorted_idx = np.argsort(ket_qua_day_du)[::-1]
        for i, lbl in enumerate(self.lbl_top3):
            if i < len(sorted_idx):
                idx = sorted_idx[i]
                cls = self.class_names[idx]
                pct_i = ket_qua_day_du[idx] * 100
                emoji_i = FRUIT_EMOJI.get(cls.lower(), "🍈")
                marker = "▶ " if i == 0 else "   "
                mau_i = COLORS["text_white"] if i == 0 else COLORS["text_gray"]
                lbl.configure(
                    text=f"{marker}{emoji_i} {cls.capitalize():15s}  {pct_i:.1f}%",
                    fg=mau_i
                )
            else:
                lbl.configure(text="")

    def _xoa_ket_qua(self, xoa_anh=True):
        """Xóa kết quả và reset giao diện"""
        self.lbl_ten_qua.configure(text="---")
        self.lbl_ten_vi.configure(text="")
        self.lbl_emoji.configure(text="🍈")
        self.lbl_do_tin_cay.configure(text="---", fg=COLORS["success"])
        self.lbl_nhan_xet.configure(text="")
        self.canvas_bar.coords(self.thanh_progress, 0, 0, 0, 20)

        for lbl in self.lbl_top3:
            lbl.configure(text="")

        if xoa_anh:
            self.anh_hien_tai = None
            self.photo_tk = None
            self.lbl_anh.configure(
                image="",
                text="📁\n\nChưa có ảnh\n\nBấm 'Chọn Ảnh' để bắt đầu"
            )
            self.lbl_ten_file.configure(text="")
            
    def _them_loai_qua(self):
        """Mở hộp thoại tạo thư mục cho quả mới"""
        class_name = simpledialog.askstring(
            "Thêm Loại Quả Mới", 
            "Nhập tên loại quả muốn thêm bằng tiếng Anh (ví dụ: strawberry, watermelon):",
            parent=self.root
        )
        
        if class_name:
            class_name = class_name.strip().lower()
            if not class_name:
                messagebox.showerror("Lỗi", "Tên loại quả không hợp lệ!")
                return
                
            train_dir = BASE_DIR / "dataset" / "train" / class_name
            val_dir = BASE_DIR / "dataset" / "validation" / class_name
            
            if train_dir.exists() or val_dir.exists():
                messagebox.showwarning("Cảnh báo", f"Loại quả '{class_name}' đã tồn tại trong dataset!")
                return
                
            try:
                train_dir.mkdir(parents=True, exist_ok=True)
                val_dir.mkdir(parents=True, exist_ok=True)
                
                msg = (f"Đã tạo thư mục thành công!\n\n"
                       f"Bạn hãy làm theo các bước sau:\n"
                       f"1. Copy 80% số ảnh vào: dataset/train/{class_name}\n"
                       f"2. Copy 20% số ảnh vào: dataset/validation/{class_name}\n"
                       f"3. Đóng ứng dụng này và chạy lại file 'train.py' để AI học thêm quả mới.")
                messagebox.showinfo(f"Thành công tạo '{class_name}'", msg)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tạo thư mục:\n{e}")


def main():
    root = tk.Tk()
    app = FruitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
