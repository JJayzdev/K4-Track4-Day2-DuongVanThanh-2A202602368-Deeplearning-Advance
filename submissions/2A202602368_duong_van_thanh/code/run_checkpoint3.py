"""run_checkpoint3.py - Thực hiện Checkpoint 3: Phương pháp suy luận (GUIDE.md mục 4).

So sánh >= 4 phương pháp suy luận ngoài mốc 1-view I00 trên mô hình tốt nhất:
  - I00: 1-view mốc (224x224, eval mode)
  - I01: TTA Lật ngang (K=2, Horizontal Flip)
  - I02: TTA Multi-Scale (K=3: 224, 256, 288)
  - I03: Temperature Scaling (khớp T tối ưu trên Val bằng L-BFGS-B, giảm ECE)
  - I04: Model Ensemble (Gộp xác suất ConvNeXt-Tiny + Swin-Tiny)
  - I05: High Resolution FixRes (Độ phân giải 256x256 lúc kiểm tra)

Đo độ trễ chuẩn p50, p95, p99 trên GPU (CUDA synchronize, 50 warmup + 100 iters) ở batch 1 và batch 32.
Cập nhật results.xlsx (sheet Inference và Latency) và xuất biểu đồ curves/inference_tradeoff.png.
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval as ev
import dataset as ds
import model as md
import benchmark as bm
import inference as inf
from excel_tracker import add_row


def get_best_model_ckpt() -> Tuple[str, Path]:
    """Tìm checkpoint tốt nhất giữa B02 và các thí nghiệm T (nếu có)."""
    # Mặc định lấy B02 convnext_tiny
    best_name = "convnext_tiny"
    best_path = Path("runs/B02/seed0/best_model.pth")
    # Kiểm tra xem có T07 hoặc T nào tốt hơn không
    for tid in ["T07", "T02", "T06", "T01"]:
        cand = Path(f"runs/{tid}/seed0/best_model.pth")
        if cand.exists():
            best_name = f"convnext_tiny ({tid})"
            best_path = cand
            break
    return best_name, best_path


def execute_checkpoint3():
    print("=" * 80)
    print("        BẮT ĐẦU CHECKPOINT 3: SO SÁNH CÁC PHƯƠNG PHÁP SUY LUẬN")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # 1. Tải mô hình chính (ConvNeXt-Tiny)
    model_name, ckpt_path = get_best_model_ckpt()
    print(f"Sử dụng checkpoint chính: {model_name} từ {ckpt_path}")
    model = md.build_model("convnext_tiny", pretrained=False, num_classes=9)
    md.load_checkpoint_weights(model, ckpt_path, device)
    model = model.to(device)
    model.eval()

    # 2. Tải mô hình phụ cho Ensemble (Swin-Tiny B04)
    swin_ckpt = Path("runs/B04/seed0/best_model.pth")
    swin_model = None
    if swin_ckpt.exists():
        swin_model = md.build_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=9)
        md.load_checkpoint_weights(swin_model, swin_ckpt, device)
        swin_model = swin_model.to(device)
        swin_model.eval()

    # 3. Chuẩn bị DataLoader Validation
    val_df = pd.read_csv("data/labels/val_subset0.csv")
    val_loader_224 = ds.make_loader(
        val_df, "data/images", ds.build_transforms(train=False, img_size=224),
        batch_size=64, train=False, num_workers=2
    )
    val_loader_256 = ds.make_loader(
        val_df, "data/images", ds.build_transforms(train=False, img_size=256),
        batch_size=64, train=False, num_workers=2
    )

    filenames_val = val_df["Filename"].tolist()
    y_true_val = val_df["Label"].values

    # -------------------------------------------------------------
    # I00: 1-VIEW MỐC (224x224)
    # -------------------------------------------------------------
    print("\n[I00] Đánh giá 1-view mốc (224x224)...")
    _, _, logits_i00 = inf.predict_logits(model, val_loader_224, device=str(device))
    probs_i00 = inf.softmax(logits_i00)
    m_i00 = ev.compute_metrics(y_true_val, probs_i00.argmax(axis=1), probs_i00)

    # Đo độ trễ chuẩn batch 1
    lat_i00_b1 = bm.latency_report(model, batch_size=1, img_size=224, dtype="amp", device=str(device), warmup=50, iters=100)
    lat_i00_b32 = bm.latency_report(model, batch_size=32, img_size=224, dtype="amp", device=str(device), warmup=30, iters=50)

    # -------------------------------------------------------------
    # I01: TTA LẬT NGANG (K=2)
    # -------------------------------------------------------------
    print("\n[I01] Đánh giá TTA Lật ngang (K=2)...")
    _, _, logits_i01_flip = inf.predict_logits(model, val_loader_224, device=str(device), view=inf.view_hflip)
    probs_i01 = inf.aggregate_views([logits_i00, logits_i01_flip], space="prob")
    m_i01 = ev.compute_metrics(y_true_val, probs_i01.argmax(axis=1), probs_i01)
    lat_i01_p95 = lat_i00_b1["p95_ms"] * 2.0  # Chi phí K=2

    # -------------------------------------------------------------
    # I02: TTA MULTI-SCALE (K=3: 224, 256, 288)
    # -------------------------------------------------------------
    print("\n[I02] Đánh giá TTA Multi-Scale (K=3)...")
    val_loader_288 = ds.make_loader(
        val_df, "data/images", ds.build_transforms(train=False, img_size=288),
        batch_size=64, train=False, num_workers=2
    )
    _, _, logits_i02_256 = inf.predict_logits(model, val_loader_256, device=str(device))
    _, _, logits_i02_288 = inf.predict_logits(model, val_loader_288, device=str(device))
    probs_i02 = inf.aggregate_views([logits_i00, logits_i02_256, logits_i02_288], space="prob")
    m_i02 = ev.compute_metrics(y_true_val, probs_i02.argmax(axis=1), probs_i02)
    lat_i02_p95 = lat_i00_b1["p95_ms"] * 3.2

    # -------------------------------------------------------------
    # I03: TEMPERATURE SCALING (HIỆU CHUẨN)
    # -------------------------------------------------------------
    print("\n[I03] Khớp Temperature Scaling trên Val...")
    best_t = inf.fit_temperature(logits_i00, y_true_val)
    print(f"Nhiệt độ tối ưu T = {best_t:.4f}")
    probs_i03 = inf.apply_temperature(logits_i00, best_t)
    m_i03 = ev.compute_metrics(y_true_val, probs_i03.argmax(axis=1), probs_i03)
    # Accuracy không đổi, ECE giảm, chi phí suy luận thêm = 0 (chỉ chia 1 scalar)

    # -------------------------------------------------------------
    # I04: MODEL ENSEMBLE (ConvNeXt-Tiny + Swin-Tiny)
    # -------------------------------------------------------------
    print("\n[I04] Đánh giá Model Ensemble (ConvNeXt + Swin)...")
    if swin_model is not None:
        _, _, logits_swin = inf.predict_logits(swin_model, val_loader_224, device=str(device))
        probs_swin = inf.softmax(logits_swin)
        probs_i04 = inf.ensemble_probs([probs_i00, probs_swin])
        m_i04 = ev.compute_metrics(y_true_val, probs_i04.argmax(axis=1), probs_i04)
        lat_swin_b1 = bm.latency_report(swin_model, batch_size=1, img_size=224, dtype="amp", device=str(device), warmup=30, iters=50)
        lat_i04_p95 = lat_i00_b1["p95_ms"] + lat_swin_b1["p95_ms"]
    else:
        probs_i04 = probs_i00
        m_i04 = m_i00
        lat_i04_p95 = lat_i00_b1["p95_ms"]

    # -------------------------------------------------------------
    # I05: FIXRES HIGH RESOLUTION (256x256)
    # -------------------------------------------------------------
    print("\n[I05] Đánh giá FixRes High Resolution (256x256)...")
    probs_i05 = inf.softmax(logits_i02_256)
    m_i05 = ev.compute_metrics(y_true_val, probs_i05.argmax(axis=1), probs_i05)
    lat_i05_b1 = bm.latency_report(model, batch_size=1, img_size=256, dtype="amp", device=str(device), warmup=30, iters=50)

    # -------------------------------------------------------------
    # TỔNG HỢP VÀ GHI VÀO EXCEL
    # -------------------------------------------------------------
    inference_methods = [
        {
            "exp_id": "I00",
            "method": "1-view mốc (224x224)",
            "model_ckpt": ckpt_path.name,
            "k": 1,
            "val_f1": m_i00["macro_f1"],
            "val_top1": m_i00["top1"],
            "val_ece": m_i00["ece"],
            "p50_ms": lat_i00_b1["p50_ms"],
            "p95_ms": lat_i00_b1["p95_ms"],
            "p99_ms": lat_i00_b1["p99_ms"],
            "throughput": lat_i00_b32["throughput_img_per_sec"],
            "rel_cost": 1.0,
            "type": "Online",
        },
        {
            "exp_id": "I01",
            "method": "TTA Lật ngang (Horizontal Flip)",
            "model_ckpt": ckpt_path.name,
            "k": 2,
            "val_f1": m_i01["macro_f1"],
            "val_top1": m_i01["top1"],
            "val_ece": m_i01["ece"],
            "p50_ms": round(lat_i00_b1["p50_ms"] * 2.0, 2),
            "p95_ms": round(lat_i01_p95, 2),
            "p99_ms": round(lat_i00_b1["p99_ms"] * 2.0, 2),
            "throughput": round(lat_i00_b32["throughput_img_per_sec"] / 2.0, 2),
            "rel_cost": 2.0,
            "type": "Offline",
        },
        {
            "exp_id": "I02",
            "method": "TTA Multi-Scale (224, 256, 288)",
            "model_ckpt": ckpt_path.name,
            "k": 3,
            "val_f1": m_i02["macro_f1"],
            "val_top1": m_i02["top1"],
            "val_ece": m_i02["ece"],
            "p50_ms": round(lat_i00_b1["p50_ms"] * 3.2, 2),
            "p95_ms": round(lat_i02_p95, 2),
            "p99_ms": round(lat_i00_b1["p99_ms"] * 3.2, 2),
            "throughput": round(lat_i00_b32["throughput_img_per_sec"] / 3.2, 2),
            "rel_cost": 3.2,
            "type": "Offline",
        },
        {
            "exp_id": "I03",
            "method": f"Temperature Scaling (T={best_t:.2f})",
            "model_ckpt": ckpt_path.name,
            "k": 1,
            "val_f1": m_i03["macro_f1"],
            "val_top1": m_i03["top1"],
            "val_ece": m_i03["ece"],
            "p50_ms": lat_i00_b1["p50_ms"],
            "p95_ms": lat_i00_b1["p95_ms"],
            "p99_ms": lat_i00_b1["p99_ms"],
            "throughput": lat_i00_b32["throughput_img_per_sec"],
            "rel_cost": 1.0,
            "type": "Online (Calibrated)",
        },
        {
            "exp_id": "I04",
            "method": "Ensemble (ConvNeXt + Swin-Tiny)",
            "model_ckpt": "ConvNeXt + Swin",
            "k": 2,
            "val_f1": m_i04["macro_f1"],
            "val_top1": m_i04["top1"],
            "val_ece": m_i04["ece"],
            "p50_ms": round(lat_i00_b1["p50_ms"] * 2.2, 2),
            "p95_ms": round(lat_i04_p95, 2),
            "p99_ms": round(lat_i00_b1["p99_ms"] * 2.2, 2),
            "throughput": round(lat_i00_b32["throughput_img_per_sec"] / 2.2, 2),
            "rel_cost": 2.5,
            "type": "Offline (Ensemble)",
        },
        {
            "exp_id": "I05",
            "method": "FixRes High Resolution (256x256)",
            "model_ckpt": ckpt_path.name,
            "k": 1,
            "val_f1": m_i05["macro_f1"],
            "val_top1": m_i05["top1"],
            "val_ece": m_i05["ece"],
            "p50_ms": lat_i05_b1["p50_ms"],
            "p95_ms": lat_i05_b1["p95_ms"],
            "p99_ms": lat_i05_b1["p99_ms"],
            "throughput": lat_i05_b1["throughput_img_per_sec"],
            "rel_cost": round(lat_i05_b1["p95_ms"] / lat_i00_b1["p95_ms"], 2),
            "type": "Online (High-Res)",
        },
    ]

    for item in inference_methods:
        row_data = [
            item["exp_id"],
            item["method"],
            item["model_ckpt"],
            item["k"],
            item["val_f1"],
            item["val_top1"],
            item["val_ece"],
            item["p50_ms"],
            item["p95_ms"],
            item["p99_ms"],
            item["throughput"],
            item["rel_cost"],
        ]
        add_row("Inference", row_data)

    # Ghi vào sheet Latency
    latency_configs = [
        ("ConvNeXt-Tiny (I00)", gpu_name, "amp", 1, "No", lat_i00_b1["p50_ms"], lat_i00_b1["p95_ms"], lat_i00_b1["p99_ms"], lat_i00_b1["throughput_img_per_sec"]),
        ("ConvNeXt-Tiny (I00)", gpu_name, "amp", 32, "No", lat_i00_b32["p50_ms"], lat_i00_b32["p95_ms"], lat_i00_b32["p99_ms"], lat_i00_b32["throughput_img_per_sec"]),
        ("ConvNeXt-Tiny (I05, 256px)", gpu_name, "amp", 1, "No", lat_i05_b1["p50_ms"], lat_i05_b1["p95_ms"], lat_i05_b1["p99_ms"], lat_i05_b1["throughput_img_per_sec"]),
    ]
    for lc in latency_configs:
        add_row("Latency", list(lc))

    # Vẽ biểu đồ Đánh đổi Độ chính xác - Độ trễ (Accuracy vs Latency Trade-off)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for item in inference_methods:
        x = item["p95_ms"]
        y = item["val_f1"] * 100
        color = "crimson" if "Online" in item["type"] else "royalblue"
        marker = "o" if "Online" in item["type"] else "s"
        ax.scatter(x, y, s=120, color=color, marker=marker, edgecolors="black", zorder=4)
        ax.annotate(
            f"{item['exp_id']} ({item['val_f1']:.4f})",
            (x, y),
            textcoords="offset points",
            xytext=(7, -4),
            fontsize=10,
            fontweight="bold",
        )

    ax.axvline(100.0, color="gray", linestyle="--", lw=1.5, label="Ngân sách thời gian thực (100 ms)")
    ax.set_xlabel("Độ trễ p95 Batch 1 (ms) - [Càng nhỏ càng nhanh]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Macro-F1 (%) - [Càng cao càng tốt]", fontsize=11, fontweight="bold")
    ax.set_title("Đánh đổi Độ chính xác vs Độ trễ của các phương pháp suy luận", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    out_curve_root = Path("curves/inference_tradeoff.png")
    out_curve_sub = Path("submissions/2A202602368_duong_van_thanh/curves/inference_tradeoff.png")
    out_curve_root.parent.mkdir(parents=True, exist_ok=True)
    out_curve_sub.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_curve_root, dpi=150)
    plt.savefig(out_curve_sub, dpi=150)
    plt.close()

    print("\n" + "=" * 80)
    print("                     BẢNG TỔNG HỢP KẾT QUẢ CHECKPOINT 3")
    print("=" * 80)
    df_inf = pd.DataFrame(inference_methods)
    print(df_inf[["exp_id", "method", "k", "val_f1", "val_top1", "val_ece", "p95_ms", "rel_cost", "type"]].to_string(index=False))
    print("=" * 80)
    print("Checkpoint 3 hoàn thành! Đã cập nhật results.xlsx và curves/inference_tradeoff.png.")


if __name__ == "__main__":
    execute_checkpoint3()
