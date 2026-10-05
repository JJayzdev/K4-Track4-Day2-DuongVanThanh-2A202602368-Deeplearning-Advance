# Báo Cáo Thực Nghiệm Deep Learning Nâng Cao - Lab Day 2: Phân Loại Cỏ Dại DeepWeeds

- **Học viên:** Dương Văn Thành
- **Mã học viên:** 2A202602368
- **Lớp / Track:** Track 4 - Advanced Deep Learning (VinAI & Đại học Công nghệ)
- **Dataset:** DeepWeeds (Olsen et al., 2019) - 17.509 ảnh 9 lớp (Fold 0)

---

## 1. Hướng Dẫn Tái Lập Thực Nghiệm (Reproducibility Guide)

### 1.1 Yêu Cầu Môi Trường
- **Hệ điều hành:** Windows 10/11 hoặc Linux
- **Python:** >= 3.9
- **Phần cứng:** GPU NVIDIA (Khuyến nghị VRAM >= 6GB, thực nghiệm chạy trên NVIDIA GeForce RTX 3070 Ti Laptop GPU 8GB)
- **Thư viện chính:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install timm pandas openpyxl matplotlib scipy
  ```

### 1.2 Cấu Trúc Thư Mục Nộp Bài
```text
submissions/2A202602368_duong_van_thanh/
├── README.md               # Hướng dẫn tái lập thực nghiệm và sitemap sản phẩm
├── report.md               # Báo cáo khoa học toàn diện
├── results.xlsx            # Bảng tính số liệu 7 sheets đúng quy chuẩn GUIDE.md
├── grade_result.txt        # Kết quả tự chấm từ công cụ eval.py grade
├── code/                   # Toàn bộ mã nguồn thực thi
│   ├── dataset.py          # Data pipeline, augmentations, check split
│   ├── model.py            # Model factory, timm backbones, linear probe
│   ├── losses.py           # CE, Focal, Label Smoothing, CutMix/Mixup
│   ├── benchmark.py        # Đo Params, GMACs, độ trễ CUDA synchronize
│   ├── inference.py        # TTA, Temperature Scaling, Ensemble, FixRes
│   ├── train.py            # Training engine chuẩn (AdamW, AMP, Cosine, EMA)
│   ├── excel_tracker.py    # Bộ công cụ ghi tự động vào results.xlsx
│   ├── run_checkpoint1.py  # So sánh 5 backbone (B01..B05)
│   ├── run_checkpoint2.py  # Ablation >= 3 trục trên ConvNeXt-Tiny (T00..T07)
│   ├── run_checkpoint3.py  # So sánh phương pháp suy luận (I00..I05)
│   ├── run_checkpoint4.py  # Vòng chung kết 3 seeds, đánh giá test (F01)
│   └── run_all_checkpoints.py # Master runner thực thi tuần tự từ A-Z
├── curves/                 # Đồ thị learning curves & visualization
│   ├── eda_class_distribution.png
│   ├── B01_resnet50.png
│   ├── B02_convnext_tiny.png
│   ├── B03_resnext50_32x4d.png
│   ├── B04_swin_tiny_patch4_window7_224.png
│   ├── B05_mobilenetv3_large_100.png
│   ├── T0x_*.png
│   ├── inference_tradeoff.png
│   └── confusion_matrix_test.png
└── predictions/            # File dự đoán định dạng eval.py cho Val & Test
    ├── B01..B05_seed0_val.csv
    ├── T00_seed0..2_test.csv
    ├── F01_seed0..2_test.csv
    ├── F01_seed0..2_val.csv
    └── F01uncal_seed0..2_test.csv
```

### 1.3 Lệnh Chạy Lại (One-Click Reproduction)
Để tái lập toàn bộ kết quả từ đầu:
```bash
# Thiết lập biến môi trường UTF-8 (đặc biệt quan trọng trên Windows)
$env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"

# 1. Chạy kiểm tra bộ unit test (38 tests)
pytest tests/

# 2. Chạy Checkpoint 1 (So sánh 5 backbone)
python submissions/2A202602368_duong_van_thanh/code/run_checkpoint1.py

# 3. Chạy Checkpoint 2 (Ablation công thức huấn luyện)
python submissions/2A202602368_duong_van_thanh/code/run_checkpoint2.py

# 4. Chạy Checkpoint 3 (Phương pháp suy luận & benchmark độ trễ)
python submissions/2A202602368_duong_van_thanh/code/run_checkpoint3.py

# 5. Chạy Checkpoint 4 (Vòng chung kết 3 seeds & chấm điểm tự động)
python submissions/2A202602368_duong_van_thanh/code/run_checkpoint4.py

# Hoặc chạy toàn bộ tuần tự tự động:
python submissions/2A202602368_duong_van_thanh/code/run_all_checkpoints.py
```
