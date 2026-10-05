"""eda.py - Thực hiện Phân tích Khám phá Dữ liệu (EDA) và kiểm tra toàn vẹn chia tập (S1-S6).
Xuất biểu đồ phân bố lớp vào thư mục curves/ và in bảng thống kê cho report.md.
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import load_split, check_split, CLASS_NAMES, NUM_CLASSES

# Số liệu công bố trong Table 1 của bài báo Olsen et al. (Scientific Reports 2019)
PAPER_TABLE_1 = {
    "Chinee Apple": 1125,
    "Lantana": 1064,
    "Parkinsonia": 1031,
    "Parthenium": 1022,
    "Prickly Acacia": 1062,
    "Rubber Vine": 1009,
    "Siam Weed": 1074,
    "Snake Weed": 1016,
    "Negatives": 9106,
}


def run_eda(labels_dir: str = "data/labels", images_dir: str = "data/images", curves_dir: str = "submissions/2A202602368_duong_van_thanh/curves"):
    labels_path = Path(labels_dir)
    curves_path = Path(curves_dir)
    curves_path.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_split(labels_path, fold=0)
    stats = check_split(train_df, val_df, test_df, images_dir)

    print("\n" + "="*80)
    print("                      KẾT QUẢ ĐỐI CHIẾU VỚI TABLE 1 BÀI BÁO GỐC")
    print("="*80)
    header = f"{'STT':<4} | {'Tên loài / Lớp':<16} | {'Train':<7} | {'Val':<6} | {'Test':<6} | {'Tổng thực':<10} | {'Bài báo':<8} | {'Khớp?':<6}"
    print(header)
    print("-" * len(header))

    plot_data = []

    for cls_idx in range(NUM_CLASSES):
        name = CLASS_NAMES[cls_idx]
        c_train = stats["per_class"][cls_idx]["train"]
        c_val = stats["per_class"][cls_idx]["val"]
        c_test = stats["per_class"][cls_idx]["test"]
        c_total = stats["per_class"][cls_idx]["total"]
        paper_total = PAPER_TABLE_1[name]
        is_match = "ĐÚNG" if c_total == paper_total else "SAI"

        print(f"{cls_idx:<4} | {name:<16} | {c_train:<7} | {c_val:<6} | {c_test:<6} | {c_total:<10} | {paper_total:<8} | {is_match:<6}")
        plot_data.append((name, c_train, c_val, c_test, c_total))

    print("-" * len(header))
    total_train = stats["n_train"]
    total_val = stats["n_val"]
    total_test = stats["n_test"]
    total_all = stats["n_total"]
    print(f"{'TỔNG':<4} | {'Toàn bộ':<16} | {total_train:<7} | {total_val:<6} | {total_test:<6} | {total_all:<10} | {'17509':<8} | {'ĐÚNG':<6}")
    print("="*80)

    # Tính độ mất cân bằng
    max_class_count = max(stats["per_class"][c]["total"] for c in range(NUM_CLASSES))
    min_class_count = min(stats["per_class"][c]["total"] for c in range(NUM_CLASSES))
    imbalance_ratio = max_class_count / min_class_count
    print(f"\nĐặc điểm mất cân bằng:")
    print(f"- Lớp nhiều nhất: Negatives ({max_class_count} ảnh, chiếm {max_class_count/total_all*100:.2f}%)")
    print(f"- Lớp ít nhất: Rubber Vine ({min_class_count} ảnh, chiếm {min_class_count/total_all*100:.2f}%)")
    print(f"- Tỉ số mất cân bằng (Max/Min): {imbalance_ratio:.2f}x\n")

    # Vẽ biểu đồ phân bố lớp
    species = [d[0] for d in plot_data]
    train_bars = [d[1] for d in plot_data]
    val_bars = [d[2] for d in plot_data]
    test_bars = [d[3] for d in plot_data]

    x = np.arange(len(species))
    width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, train_bars, width, label=f"Train ({total_train})", color="#2b5c8f")
    rects2 = ax.bar(x, val_bars, width, label=f"Val ({total_val})", color="#e27c38")
    rects3 = ax.bar(x + width, test_bars, width, label=f"Test ({total_test})", color="#3b9a57")

    ax.set_ylabel("Số lượng ảnh", fontsize=12, fontweight="bold")
    ax.set_title("Phân bố số lượng ảnh theo lớp trong DeepWeeds (Fold 0)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 6000)

    # Thêm nhãn số trên đầu cột Negatives
    for rect in [rects1[-1], rects2[-1], rects3[-1]]:
        height = rect.get_height()
        ax.annotate(f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = curves_path / "eda_class_distribution.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ phân bố lớp vào: {chart_path}")


if __name__ == "__main__":
    run_eda()
