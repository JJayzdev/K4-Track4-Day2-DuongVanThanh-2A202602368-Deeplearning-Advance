"""run_checkpoint1.py - Thực hiện Checkpoint 1: So sánh 5 backbone (GUIDE.md mục 2).

Chạy 5 backbone đại diện cho các họ kiến trúc với cùng baseline recipe T00, seed 0:
  1. resnet50 (Họ ResNet tiêu chuẩn - mốc)
  2. convnext_tiny (Họ ConvNeXt hiện đại hoá)
  3. resnext50_32x4d (Họ ResNeXt - đa nhánh/cardinality)
  4. swin_tiny_patch4_window7_224 (Họ Vision Transformer - Hierarchical/Window Attention)
  5. mobilenetv3_large_100 (Họ mạng nhẹ - Edge/Mobile)

Tự động ghi kết quả vào:
  - curves/B0x_<backbone>.png
  - results.xlsx (sheet Backbones và Summary)
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import time
import pandas as pd

from train import Config, run
from excel_tracker import add_row, init_excel

BACKBONES = [
    {
        "exp_id": "B01",
        "backbone": "resnet50",
        "tag": "a1_in1k",
        "notes": "Họ ResNet tiêu chuẩn (Mốc baseline)",
    },
    {
        "exp_id": "B02",
        "backbone": "convnext_tiny",
        "tag": "in12k_ft_in1k",
        "notes": "Họ ConvNeXt hiện đại hoá (7x7 depthwise)",
    },
    {
        "exp_id": "B03",
        "backbone": "resnext50_32x4d",
        "tag": "a1h_in1k",
        "notes": "Họ ResNeXt (cardinality 32x4d)",
    },
    {
        "exp_id": "B04",
        "backbone": "swin_tiny_patch4_window7_224",
        "tag": "ms_in1k",
        "notes": "Họ Vision Transformer (Swin shifted window)",
    },
    {
        "exp_id": "B05",
        "backbone": "mobilenetv3_large_100",
        "tag": "ra_in1k",
        "notes": "Họ mạng nhẹ (MobileNetV3-Large)",
    },
]


def execute_checkpoint1():
    print("=" * 80)
    print("           BẮT ĐẦU CHECKPOINT 1: SO SÁNH 5 BACKBONE")
    print("=" * 80)

    results = []

    for b in BACKBONES:
        exp_id = b["exp_id"]
        backbone_name = b["backbone"]
        tag = b["tag"]
        notes = b["notes"]

        print(f"\n[{exp_id}] Đang huấn luyện backbone: {backbone_name} (tag: {tag})...")

        # Cấu hình chuẩn T00 (GUIDE mục 1.4)
        cfg = Config(
            exp_id=exp_id,
            backbone=backbone_name,
            seed=0,
            epochs=12,
            batch_size=64,
            lr_backbone=1e-4,
            lr_head=1e-3,
            weight_decay=0.05,
            warmup_epochs=1.0,
            loss="ce",
            aug="basic",
            amp=True,
            images_dir="data/images",
            labels_dir="data/labels",
            out_dir="runs",
            pred_dir="predictions",
            save_test_predictions=False, # Chỉ val ở Bước 1 (Quy tắc S2, S4)
        )

        t0 = time.time()
        summary = run(cfg)
        elapsed = time.time() - t0

        row_data = [
            exp_id,
            backbone_name,
            tag,
            summary["params_m"],
            summary["gmacs"],
            224,
            cfg.epochs,
            cfg.seed,
            summary["val_macro_f1"],
            summary["val_top1"],
            summary["avg_epoch_sec"],
            summary["latency_p50_ms"],
            summary["latency_p95_ms"],
            notes,
        ]

        add_row("Backbones", row_data)

        # Thêm vào Summary
        summary_row = [
            len(results) + 1,
            exp_id,
            "Backbone",
            f"{backbone_name} ({notes})",
            summary["val_macro_f1"],
            summary["val_top1"],
            summary["latency_p95_ms"],
            f"Params: {summary['params_m']}M, GMACs: {summary['gmacs']}",
        ]
        add_row("Summary", summary_row)

        results.append({
            "exp_id": exp_id,
            "backbone": backbone_name,
            "params_m": summary["params_m"],
            "gmacs": summary["gmacs"],
            "val_f1": summary["val_macro_f1"],
            "val_acc": summary["val_top1"],
            "train_sec": summary["avg_epoch_sec"],
            "p95_ms": summary["latency_p95_ms"],
            "notes": notes,
        })

    print("\n" + "=" * 80)
    print("                     BẢNG TỔNG HỢP KẾT QUẢ CHECKPOINT 1")
    print("=" * 80)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("=" * 80)
    print("Checkpoint 1 đã hoàn thành! Dữ liệu đã được ghi vào results.xlsx và thư mục curves/.")


if __name__ == "__main__":
    execute_checkpoint1()
