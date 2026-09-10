"""
gui_dashboard.py - Giao diện trung tâm cho AI Nhận Diện Trái Cây
================================================================
Bao gồm cả 2 chức năng: Nhận diện ảnh (Inference) và Huấn luyện mô hình (Training).
"""

import sys
import os
import re
import json
import threading
import traceback
import numpy as np
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
except ImportError:
    print("❌ Lỗi: Thư viện Tkinter không khả dụng!")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("❌ Lỗi: Chưa cài Pillow! Hãy chạy: pip install Pillow")
    sys.exit(1)

# Import các script có sẵn trong dự án
try:
    import train
    import check_dataset
    import add_new_class
except ImportError as e:
    print(f"❌ Lỗi: Không thể import script gốc: {e}")
    sys.exit(1)


# ============================================================
# Cấu hình chung
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fruit_model.keras"
CLASS_FILE = BASE_DIR / "models" / "class_names.json"
SUMMARY_FILE = BASE_DIR / "models" / "training_summary.json"
HISTORY_IMG = BASE_DIR / "models" / "training_history.png"

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.60

COLORS = {
    "bg_dark":      "#1a1a2e",
    "bg_panel":     "#16213e",
    "bg_card":      "#0f3460",
    "accent":       "#e94560",
    "accent2":      "#f5a623",
    "text_white":   "#ffffff",
    "text_gray":    "#a0a0b0",
    "text_light":   "#c8d6e5",
    "success":      "#2ecc71",
    "warning":      "#f39c12",
    "error":        "#e74c3c",
    "btn_primary":  "#e94560",
    "btn_hover":    "#c0392b",
}

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
}

class RedirectText:
    """Class giúp chuyển hướng sys.stdout/sys.stderr vào Text widget của Tkinter an toàn"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')

    def write(self, string):
        # Xóa các mã escape màu ANSI bị rác
        clean_string = self.ansi_escape.sub('', string)
        if clean_string:
            # Gọi an toàn qua luồng chính của Tkinter
            self.text_widget.after(0, self._insert_text, clean_string)

    def _insert_text(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass


class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Dashboard - Trái Cây")
        self.root.geometry("1000x750")
        self.root.minsize(900, 700)
        
        # Biến cho Tab Nhận Diện
        self.model = None
        self.class_names = []
        self.anh_hien_tai = None
        self.photo_tk = None
        
        # Biến cho Tab Huấn Luyện
        self.is_training = False
        self.gui_callback = None
        self.last_val_acc = 0
        self.start_time = 0
        
        self._cau_hinh_style()
        self._init_ui()
        self._tai_model_nen()

    def _cau_hinh_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        # Cấu hình Notebook
        style.configure('TNotebook', background=COLORS['bg_dark'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['bg_panel'], foreground=COLORS['text_light'], 
                        padding=[20, 5], font=('Segoe UI', 11, 'bold'), borderwidth=0)
        style.map('TNotebook.Tab', 
                  background=[('selected', COLORS['accent'])],
                  foreground=[('selected', COLORS['text_white'])])
        style.configure('TFrame', background=COLORS['bg_dark'])
        self.root.configure(bg=COLORS["bg_dark"])

    def _init_ui(self):
        # Frame Tiêu đề chung
        frame_title = tk.Frame(self.root, bg=COLORS["bg_dark"], pady=10)
        frame_title.pack(fill="x")
        
        tk.Label(frame_title, text="🍍 AI DASHBOARD TRÁI CÂY 🍎", font=("Segoe UI", 24, "bold"), 
                 fg=COLORS["accent2"], bg=COLORS["bg_dark"]).pack()
                 
        self.lbl_global_status = tk.Label(frame_title, text="Đang khởi động...", font=("Segoe UI", 10), 
                                          fg=COLORS["text_gray"], bg=COLORS["bg_dark"])
        self.lbl_global_status.pack()

        # Tạo Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Tab 1: Inference
        self.tab_inference = tk.Frame(self.notebook, bg=COLORS["bg_dark"])
        self.notebook.add(self.tab_inference, text="👁️ NHẬN DIỆN TRÁI CÂY")
        self._init_inference_tab(self.tab_inference)
        
        # Tab 2: Training
        self.tab_training = tk.Frame(self.notebook, bg=COLORS["bg_dark"])
        self.notebook.add(self.tab_training, text="🧠 HUẤN LUYỆN AI")
        self._init_training_tab(self.tab_training)

    # =========================================================================
    # TAB 1: NHẬN DIỆN TRÁI CÂY (INFERENCE)
    # =========================================================================
    def _init_inference_tab(self, parent):
        frame_main = tk.Frame(parent, bg=COLORS["bg_dark"])
        frame_main.pack(fill="both", expand=True, pady=10)
        frame_main.columnconfigure(0, weight=3)
        frame_main.columnconfigure(1, weight=2)
        frame_main.rowconfigure(0, weight=1)

        # ---- Cột trái: Hiển thị ảnh ----
        frame_anh = tk.Frame(frame_main, bg=COLORS["bg_panel"])
        frame_anh.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.frame_canvas = tk.Frame(frame_anh, bg=COLORS["bg_card"], width=400, height=400)
        self.frame_canvas.pack(padx=20, pady=30)
        self.frame_canvas.pack_propagate(False)
        
        self.lbl_anh = tk.Label(self.frame_canvas, text="📁\n\nChưa có ảnh\n\nBấm 'Chọn Ảnh' để bắt đầu",
                                font=("Segoe UI", 12), fg=COLORS["text_gray"], bg=COLORS["bg_card"], justify="center")
        self.lbl_anh.pack(fill="both", expand=True)

        # Nút chức năng
        frame_nut = tk.Frame(frame_anh, bg=COLORS["bg_panel"])
        frame_nut.pack(fill="x", pady=20)
        
        btn_chon = tk.Button(frame_nut, text="📁 Chọn Ảnh", font=("Segoe UI", 12, "bold"), bg=COLORS["bg_card"], 
                             fg=COLORS["text_white"], activebackground=COLORS["accent"], relief="flat", padx=20, pady=8,
                             command=self._chon_anh)
        btn_chon.pack(side="left", padx=20)
        
        self.btn_nhan_dien = tk.Button(frame_nut, text="🔍 Nhận Diện", font=("Segoe UI", 12, "bold"), bg=COLORS["btn_primary"], 
                                       fg=COLORS["text_white"], activebackground=COLORS["btn_hover"], relief="flat", padx=20, pady=8,
                                       command=self._nhan_dien, state="disabled")
        self.btn_nhan_dien.pack(side="left")

        # ---- Cột phải: Kết quả ----
        frame_kq = tk.Frame(frame_main, bg=COLORS["bg_panel"])
        frame_kq.grid(row=0, column=1, sticky="nsew")

        tk.Label(frame_kq, text="📊 KẾT QUẢ", font=("Segoe UI", 13, "bold"), fg=COLORS["text_light"], bg=COLORS["bg_panel"]).pack(pady=20)
        
        self.lbl_emoji = tk.Label(frame_kq, text="🍈", font=("Segoe UI", 60), bg=COLORS["bg_panel"])
        self.lbl_emoji.pack(pady=5)
        
        self.lbl_ten_qua = tk.Label(frame_kq, text="---", font=("Segoe UI", 24, "bold"), fg=COLORS["text_white"], bg=COLORS["bg_panel"])
        self.lbl_ten_qua.pack()
        
        self.lbl_ten_vi = tk.Label(frame_kq, text="", font=("Segoe UI", 14), fg=COLORS["accent2"], bg=COLORS["bg_panel"])
        self.lbl_ten_vi.pack()
        
        self.lbl_do_tin_cay = tk.Label(frame_kq, text="---", font=("Segoe UI", 30, "bold"), fg=COLORS["success"], bg=COLORS["bg_panel"])
        self.lbl_do_tin_cay.pack(pady=15)
        
        # Top 3
        tk.Label(frame_kq, text="Top kết quả:", font=("Segoe UI", 10, "bold"), fg=COLORS["text_gray"], bg=COLORS["bg_panel"]).pack(anchor="w", padx=20)
        self.frame_top3 = tk.Frame(frame_kq, bg=COLORS["bg_panel"])
        self.frame_top3.pack(fill="x", padx=20, pady=5)
        self.lbl_top3 = []
        for i in range(3):
            lbl = tk.Label(self.frame_top3, text="", font=("Segoe UI", 10), fg=COLORS["text_gray"], bg=COLORS["bg_panel"], anchor="w")
            lbl.pack(fill="x")
            self.lbl_top3.append(lbl)

    def _chon_anh(self):
        duong_dan = filedialog.askopenfilename(title="Chọn ảnh trái cây", 
            filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png"), ("Tất cả", "*.*")])
        if duong_dan:
            self.anh_hien_tai = duong_dan
            anh = Image.open(duong_dan).convert("RGB")
            anh.thumbnail((400, 400), Image.LANCZOS)
            self.photo_tk = ImageTk.PhotoImage(anh)
            self.lbl_anh.configure(image=self.photo_tk, text="")
            self.lbl_ten_qua.configure(text="---")
            self.lbl_ten_vi.configure(text="")
            self.lbl_do_tin_cay.configure(text="---", fg=COLORS["success"])
            self.lbl_emoji.configure(text="🍈")
            for lbl in self.lbl_top3: lbl.configure(text="")

    def _nhan_dien(self):
        if not self.anh_hien_tai:
            messagebox.showwarning("Cảnh báo", "Hãy chọn ảnh trước!")
            return
        if self.model is None:
            messagebox.showerror("Lỗi", "Model chưa được load. Vui lòng Train mô hình ở Tab Huấn Luyện trước!")
            return
            
        self.btn_nhan_dien.configure(state="disabled", text="⏳ Đang xử lý...")
        
        def _predict():
            try:
                from PIL import Image as PILImage
                anh = PILImage.open(self.anh_hien_tai).convert("RGB")
                anh = anh.resize((IMG_SIZE, IMG_SIZE))
                anh_array = np.array(anh, dtype=np.float32) / 255.0
                anh_array = np.expand_dims(anh_array, axis=0)
                
                ket_qua = self.model.predict(anh_array, verbose=0)[0]
                self.root.after(0, lambda: self._hien_thi_ket_qua(ket_qua))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi dự đoán: {e}"))
            finally:
                self.root.after(0, lambda: self.btn_nhan_dien.configure(state="normal", text="🔍 Nhận Diện"))
                
        threading.Thread(target=_predict, daemon=True).start()

    def _hien_thi_ket_qua(self, ket_qua_day_du):
        idx = np.argmax(ket_qua_day_du)
        do_tin_cay = ket_qua_day_du[idx]
        ten_qua = self.class_names[idx]
        
        emoji = FRUIT_EMOJI.get(ten_qua.lower(), "🍈")
        ten_vi = FRUIT_NAMES_VI.get(ten_qua.lower(), ten_qua.capitalize())
        
        self.lbl_emoji.configure(text=emoji)
        self.lbl_ten_qua.configure(text=ten_qua.upper())
        self.lbl_ten_vi.configure(text=f"({ten_vi})")
        
        mau = COLORS["success"] if do_tin_cay >= CONFIDENCE_THRESHOLD else COLORS["error"]
        self.lbl_do_tin_cay.configure(text=f"{do_tin_cay*100:.1f}%", fg=mau)
        
        sorted_idx = np.argsort(ket_qua_day_du)[::-1]
        for i, lbl in enumerate(self.lbl_top3):
            if i < len(sorted_idx):
                c_idx = sorted_idx[i]
                c_name = self.class_names[c_idx]
                c_pct = ket_qua_day_du[c_idx] * 100
                lbl.configure(text=f"{i+1}. {c_name.capitalize():15s} : {c_pct:.1f}%")

    # =========================================================================
    # TAB 2: HUẤN LUYỆN AI (TRAINING HUB)
    # =========================================================================
    def _init_training_tab(self, parent):
        frame_main = tk.Frame(parent, bg=COLORS["bg_dark"])
        frame_main.pack(fill="both", expand=True, pady=10)
        
        # Khung cài đặt và thông tin ở trên
        frame_top = tk.Frame(frame_main, bg=COLORS["bg_dark"])
        frame_top.pack(fill="x")
        
        # Info Dataset
        frame_info = tk.LabelFrame(frame_top, text="Thông Tin Dataset", bg=COLORS["bg_panel"], fg=COLORS["text_light"], font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        frame_info.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.lbl_dataset_info = tk.Label(frame_info, text="Đang phân tích dữ liệu...", justify="left", font=("Segoe UI", 9), bg=COLORS["bg_panel"], fg=COLORS["text_white"])
        self.lbl_dataset_info.pack(anchor="nw")
        
        btn_them_qua = tk.Button(frame_info, text="➕ Thêm Quả Mới", bg=COLORS["success"], fg=COLORS["text_white"], 
                                 relief="flat", font=("Segoe UI", 9, "bold"), command=self._them_qua_moi)
        btn_them_qua.pack(anchor="nw", pady=(10, 0))

        # Cấu hình Train
        frame_config = tk.LabelFrame(frame_top, text="Cấu Hình Huấn Luyện", bg=COLORS["bg_panel"], fg=COLORS["text_light"], font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        frame_config.pack(side="right", fill="both", expand=True)
        
        tk.Label(frame_config, text="Số vòng lặp (Epochs):", bg=COLORS["bg_panel"], fg=COLORS["text_gray"]).grid(row=0, column=0, sticky="w", pady=5)
        self.ent_epochs = tk.Entry(frame_config, width=10)
        self.ent_epochs.insert(0, str(train.EPOCHS))
        self.ent_epochs.grid(row=0, column=1, pady=5)
        
        tk.Label(frame_config, text="Kích thước lô (Batch):", bg=COLORS["bg_panel"], fg=COLORS["text_gray"]).grid(row=1, column=0, sticky="w", pady=5)
        self.ent_batch = tk.Entry(frame_config, width=10)
        self.ent_batch.insert(0, str(train.BATCH_SIZE))
        self.ent_batch.grid(row=1, column=1, pady=5)

        tk.Label(frame_config, text="Tốc độ học (LR):", bg=COLORS["bg_panel"], fg=COLORS["text_gray"]).grid(row=2, column=0, sticky="w", pady=5)
        self.ent_lr = tk.Entry(frame_config, width=10)
        self.ent_lr.insert(0, str(train.LEARNING_RATE))
        self.ent_lr.grid(row=2, column=1, pady=5)

        # Các nút bấm Train / Stop
        frame_btn_train = tk.Frame(frame_config, bg=COLORS["bg_panel"])
        frame_btn_train.grid(row=3, column=0, columnspan=2, pady=(15, 5))
        
        self.btn_train = tk.Button(frame_btn_train, text="🚀 BẮT ĐẦU HUẤN LUYỆN", bg=COLORS["accent"], fg=COLORS["text_white"], 
                                   relief="flat", font=("Segoe UI", 11, "bold"), padx=10, pady=5, command=self._bat_dau_train)
        self.btn_train.pack(side="left", padx=5)

        self.btn_stop = tk.Button(frame_btn_train, text="⏹ DỪNG", bg=COLORS["warning"], fg=COLORS["bg_dark"], 
                                  relief="flat", font=("Segoe UI", 11, "bold"), padx=10, pady=5, command=self._dung_train, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        # Thanh tiến trình Epoch
        self.lbl_progress = tk.Label(frame_config, text="Tiến độ: Chưa bắt đầu", bg=COLORS["bg_panel"], fg=COLORS["text_light"], font=("Segoe UI", 9, "italic"))
        self.lbl_progress.grid(row=4, column=0, columnspan=2, pady=(10, 2))
        
        self.progress_bar = ttk.Progressbar(frame_config, orient="horizontal", mode="determinate", length=220)
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=(0, 5))

        # Thanh tiến trình Batch
        self.lbl_batch_progress = tk.Label(frame_config, text="Batch: Chưa bắt đầu", bg=COLORS["bg_panel"], fg=COLORS["text_light"], font=("Segoe UI", 9, "italic"))
        self.lbl_batch_progress.grid(row=6, column=0, columnspan=2, pady=(0, 2))
        
        self.batch_progress_bar = ttk.Progressbar(frame_config, orient="horizontal", mode="determinate", length=220)
        self.batch_progress_bar.grid(row=7, column=0, columnspan=2, pady=(0, 10))

        # Nút xem biểu đồ (sẽ hiện khi train xong)
        self.btn_view_history = tk.Button(frame_config, text="📈 Xem Biểu Đồ Huấn Luyện", bg=COLORS["success"], fg=COLORS["text_white"], 
                                          relief="flat", font=("Segoe UI", 10, "bold"), padx=10, pady=5, command=self._xem_bieu_do)
        self.btn_view_history.grid_remove()  # Ẩn mặc định

        # Hộp hiển thị log (Terminal ảo)
        frame_log = tk.LabelFrame(frame_main, text="Tiến Trình (Log Console)", bg=COLORS["bg_panel"], fg=COLORS["text_light"], font=("Segoe UI", 10, "bold"))
        frame_log.pack(fill="both", expand=True, pady=(15, 0))
        
        # Scrollbar cho Text
        scrollbar = tk.Scrollbar(frame_log)
        scrollbar.pack(side="right", fill="y")
        
        self.txt_log = tk.Text(frame_log, bg="#000000", fg="#00ff00", font=("Consolas", 10), state="disabled", yscrollcommand=scrollbar.set)
        self.txt_log.pack(fill="both", expand=True, padx=5, pady=5)
        scrollbar.config(command=self.txt_log.yview)

        # Cập nhật thông tin dataset
        self._cap_nhat_dataset_info()

    def _cap_nhat_dataset_info(self):
        """Đọc thư mục dataset và cập nhật nhãn hiển thị"""
        classes = check_dataset.lay_danh_sach_class(check_dataset.TRAIN_DIR)
        
        if not classes:
            self.lbl_dataset_info.configure(text="❌ Chưa có dữ liệu. Vui lòng thêm quả mới và nạp ảnh!", fg=COLORS["error"])
            return
            
        total_train = 0
        total_val = 0
        lines = [f"🍓 Đã nạp {len(classes)} loại quả:"]
        
        for cls in classes:
            n_train = check_dataset.dem_anh_trong_thu_muc(check_dataset.TRAIN_DIR / cls)
            n_val = check_dataset.dem_anh_trong_thu_muc(check_dataset.VAL_DIR / cls)
            total_train += n_train
            total_val += n_val
            lines.append(f"  - {cls.capitalize()}: {n_train} ảnh train")
            
        lines.append("-" * 30)
        lines.append(f"Tổng số ảnh Train: {total_train}")
        lines.append(f"Tổng số ảnh Val: {total_val}")
        
        # Lấy số lần đã huấn luyện
        if SUMMARY_FILE.exists():
            try:
                with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    if "training_iterations" in summary:
                        lines.append(f"Tổng số lần đã huấn luyện: {summary['training_iterations']}")
            except Exception:
                pass
        
        self.lbl_dataset_info.configure(text="\n".join(lines), fg=COLORS["text_white"])

    def _them_qua_moi(self):
        new_class = simpledialog.askstring("Thêm Quả", "Nhập tên loại quả tiếng Anh (vd: strawberry):", parent=self.root)
        if new_class:
            success = add_new_class.create_new_class(new_class)
            if success:
                messagebox.showinfo("Thành công", f"Đã tạo thư mục cho quả '{new_class}'!\nHãy bỏ ảnh vào folder dataset/train/{new_class.lower()} và bấm cập nhật.")
                self._cap_nhat_dataset_info()

    def _bat_dau_train(self):
        if self.is_training:
            return
            
        # Lưu các tham số vào train module
        try:
            train.EPOCHS = int(self.ent_epochs.get())
            train.BATCH_SIZE = int(self.ent_batch.get())
            train.LEARNING_RATE = float(self.ent_lr.get())
        except ValueError:
            messagebox.showerror("Lỗi Cấu Hình", "Epochs, Batch Size và LR phải là số hợp lệ!")
            return

        # Khóa nút
        self.is_training = True
        self.btn_train.configure(state="disabled", text="⏳ ĐANG HUẤN LUYỆN...", bg=COLORS["text_gray"])
        self.btn_stop.configure(state="normal")
        self.btn_view_history.grid_remove()
        self.btn_nhan_dien.configure(state="disabled")
        
        # Đặt lại tiến trình
        self.progress_bar["maximum"] = train.EPOCHS
        self.progress_bar["value"] = 0
        self.lbl_progress.configure(text=f"Tiến độ: Epoch 0 / {train.EPOCHS}")
        
        self.batch_progress_bar["value"] = 0
        self.lbl_batch_progress.configure(text="Batch: Đang chuẩn bị...")
        
        self.last_val_acc = 0
        import time
        self.start_time = time.time()
        
        self.txt_log.configure(state='normal')
        self.txt_log.delete(1.0, tk.END)
        self.txt_log.configure(state='disabled')

        # Chuyển hướng console output
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = RedirectText(self.txt_log)
        sys.stderr = RedirectText(self.txt_log)

        # Chạy thread
        threading.Thread(target=self._run_training_thread, daemon=True).start()

    def _run_training_thread(self):
        try:
            from tensorflow import keras
            
            # Callback nội bộ để gửi tiến độ lên GUI
            class GUICallback(keras.callbacks.Callback):
                def __init__(self, app):
                    super().__init__()
                    self.app = app
                    self.total_batches = 0
                    
                def on_epoch_begin(self, epoch, logs=None):
                    # Khởi tạo số lượng batch trong epoch
                    self.total_batches = self.params.get('steps', 0)
                    if self.total_batches > 0:
                        self.app.root.after(0, self.app._reset_batch_progress, self.total_batches)

                def on_train_batch_end(self, batch, logs=None):
                    # Gửi tiến độ batch hiện tại
                    if self.total_batches > 0:
                        self.app.root.after(0, self.app._cap_nhat_batch_tien_do, batch + 1, self.total_batches)

                def on_epoch_end(self, epoch, logs=None):
                    self.app.root.after(0, self.app._cap_nhat_tien_do, epoch + 1, train.EPOCHS, logs)
                    
            train.GUI_CALLBACKS = [GUICallback(self)]
            self.gui_callback = train.GUI_CALLBACKS[0]
            
            print(f"👉 Khởi chạy quá trình huấn luyện với {train.EPOCHS} Epochs...")
            train.main()
        except Exception as e:
            print("\n❌ LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH TRAIN:")
            traceback.print_exc()
        finally:
            self.root.after(0, self._hoan_thanh_train)

    def _dung_train(self):
        if self.is_training and self.gui_callback and self.gui_callback.model:
            self.btn_stop.configure(state="disabled", text="⏳ Đang Dừng...")
            print("\n⚠️ NHẬN ĐƯỢC LỆNH DỪNG! Sẽ dừng sau khi epoch hiện tại kết thúc...")
            self.gui_callback.model.stop_training = True

    def _reset_batch_progress(self, total_batches):
        self.batch_progress_bar["maximum"] = total_batches
        self.batch_progress_bar["value"] = 0
        self.lbl_batch_progress.configure(text=f"Batch: 0 / {total_batches} (0%)")

    def _cap_nhat_batch_tien_do(self, current_batch, total_batches):
        self.batch_progress_bar["value"] = current_batch
        pct = int((current_batch / total_batches) * 100) if total_batches > 0 else 0
        self.lbl_batch_progress.configure(text=f"Batch: {current_batch} / {total_batches} ({pct}%)")

    def _cap_nhat_tien_do(self, current_epoch, total_epochs, logs=None):
        self.progress_bar["value"] = current_epoch
        self.lbl_progress.configure(text=f"Tiến độ: Epoch {current_epoch} / {total_epochs}")
        if logs and "val_accuracy" in logs:
            self.last_val_acc = logs["val_accuracy"]

    def _xem_bieu_do(self):
        if HISTORY_IMG.exists():
            try:
                os.startfile(HISTORY_IMG)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể mở ảnh: {e}")
        else:
            messagebox.showwarning("Chưa có biểu đồ", "Không tìm thấy file biểu đồ huấn luyện.")

    def _hoan_thanh_train(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        import time
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)} phút {int(secs)} giây"
        
        self.is_training = False
        self.btn_train.configure(state="normal", text="🚀 BẮT ĐẦU HUẤN LUYỆN", bg=COLORS["accent"])
        self.btn_stop.configure(state="disabled", text="⏹ DỪNG")
        self.btn_view_history.grid(row=8, column=0, columnspan=2, pady=10)
        
        # Đặt thanh tiến trình về trạng thái hoàn tất
        self.lbl_progress.configure(text="Tiến độ: Đã hoàn tất!")
        self.lbl_batch_progress.configure(text="Batch: Hoàn tất 100%")
        
        self._cap_nhat_dataset_info()
        
        msg = f"Đã huấn luyện xong!\n\nVal Accuracy tốt nhất: {self.last_val_acc*100:.2f}%\nThời gian chạy: {time_str}\n\nModel mới đã được nạp sang Tab Nhận Diện."
        messagebox.showinfo("Hoàn Thành", msg)
        
        # Reload model cho inference tab
        self._tai_model_nen()

    # =========================================================================
    # CORE: LOAD MODEL
    # =========================================================================
    def _tai_model_nen(self):
        self.lbl_global_status.configure(text="⏳ Đang tải model...", fg=COLORS["warning"])
        
        def _load():
            if not MODEL_PATH.exists() or not CLASS_FILE.exists():
                self.root.after(0, lambda: self.lbl_global_status.configure(
                    text="⚠️ Chưa có model. Hãy chuyển sang tab HUẤN LUYỆN AI để tạo mô hình!", fg=COLORS["warning"]))
                return

            with open(CLASS_FILE, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

            try:
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(str(MODEL_PATH))
                
                self.root.after(0, lambda: self.lbl_global_status.configure(
                    text=f"✅ Model đã nạp thành công! Sẵn sàng nhận diện {len(self.class_names)} loại quả.", fg=COLORS["success"]))
                self.root.after(0, lambda: self.btn_nhan_dien.configure(state="normal"))
            except Exception as e:
                self.root.after(0, lambda: self.lbl_global_status.configure(
                    text=f"❌ Lỗi load model: {e}", fg=COLORS["error"]))

        threading.Thread(target=_load, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()
