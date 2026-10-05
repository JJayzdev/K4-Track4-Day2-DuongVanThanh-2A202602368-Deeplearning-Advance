"""run_checkpoint2.py - Thực hiện Checkpoint 2: Công thức huấn luyện (GUIDE.md mục 3).

Ablation >= 3 trục trên backbone tối ưu convnext_tiny (seed 0, 12 epochs):
  - T00: Mốc nền (B02 đã chạy)
  - T01 (Trục C - Loss): Focal Loss (gamma=2.0)
  - T02 (Trục C - Loss): Label Smoothing (eps=0.1)
  - T03 (Trục B - Augmentation): RandAugment (num_ops=2, mag=9)
  - T04 (Trục B - Augmentation): CutMix (alpha=1.0)
  - T05 (Trục A - Khởi tạo): Đóng băng backbone (Linear probe)
  - T06 (Trục F - Chính quy hoá): EMA trọng số (decay=0.999)
  - T07 (Tổ hợp tối ưu): Kết hợp các yếu tố vượt trội

Ghi nhận đầy đủ vào results.xlsx (sheet Training và Summary) và thư mục curves/.
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
import shutil
import numpy as np
import pandas as pd

from train import Config, run
from excel_tracker import add_row
import eval as ev

EXPERIMENTS = [
    {
        "exp_id": "T01",
        "axis": "C. Loss",
        "diff": "Focal Loss (gamma=2.0)",
        "notes": "Tập trung vào mẫu khó, giảm ảnh hưởng lớp Negative",
        "cfg_kwargs": {
            "loss": "focal",
            "focal_gamma": 2.0,
        },
    },
    {
        "exp_id": "T02",
        "axis": "C. Loss",
        "diff": "Label Smoothing (eps=0.1)",
        "notes": "Chống overconfidence, cải thiện ECE và tổng quát hoá",
        "cfg_kwargs": {
            "loss": "ls",
            "label_smoothing": 0.1,
        },
    },
    {
        "exp_id": "T03",
        "axis": "B. Augmentation",
        "diff": "RandAugment (N=2, M=9)",
        "notes": "Tăng cường chính sách ngẫu nhiên mạnh hơn crop/flip cơ bản",
        "cfg_kwargs": {
            "aug": "randaug",
        },
    },
    {
        "exp_id": "T04",
        "axis": "B. Augmentation",
        "diff": "CutMix (alpha=1.0)",
        "notes": "Cắt dán vùng ảnh và trộn nhãn",
        "cfg_kwargs": {
            "mix": "cutmix",
            "mix_alpha": 1.0,
        },
    },
    {
        "exp_id": "T05",
        "axis": "A. Khởi tạo",
        "diff": "Frozen backbone (Linear probe)",
        "notes": "Đóng băng backbone pretrained, chỉ huấn luyện classifier head",
        "cfg_kwargs": {
            "init": "frozen",
            "lr_backbone": 0.0,
            "lr_head": 1e-3,
        },
    },
    {
        "exp_id": "T06",
        "axis": "F. Chính quy hoá",
        "diff": "EMA trọng số (decay=0.999)",
        "notes": "Trung bình động trọng số giảm phương sai khi hội tụ",
        "cfg_kwargs": {
            "ema_decay": 0.999,
        },
    },
]


def get_difficult_classes_f1(pred_file: Path) -> Tuple[float, float]:
    """Trích xuất F1 của Chinee apple (lớp 0) và Snake weed (lớp 7)."""
    if not pred_file.exists():
        return 0.0, 0.0
    df = pd.read_csv(pred_file)
    y_col = "y_true" if "y_true" in df.columns else "Label"
    y_true = df[y_col].values
    prob_cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    if not prob_cols:
        prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    probs = df[prob_cols].values
    y_pred = df["y_pred"].values if "y_pred" in df.columns else probs.argmax(axis=1)
    m = ev.compute_metrics(y_true, y_pred, probs)
    return round(float(m["f1"][0]), 4), round(float(m["f1"][7]), 4)


def load_b02_as_t00() -> dict:
    """Tải kết quả B02 để làm mốc T00."""
    b02_summary_file = Path("runs/B02/seed0/summary.json")
    if not b02_summary_file.exists():
        raise FileNotFoundError("Không tìm thấy kết quả B02 để tạo T00!")

    with open(b02_summary_file, "r", encoding="utf-8") as f:
        b02_summary = json.load(f)

    b02_pred_file = Path("submissions/2A202602368_duong_van_thanh/predictions/B02_seed0_val.csv")
    f1_chinee, f1_snake = get_difficult_classes_f1(b02_pred_file)

    t00_info = {
        "exp_id": "T00",
        "backbone": "convnext_tiny",
        "axis": "Baseline",
        "diff": "Công thức nền (CE, Basic Aug, Finetune, no EMA)",
        "seed": 0,
        "val_f1": b02_summary["val_macro_f1"],
        "val_top1": b02_summary["val_top1"],
        "delta": 0.0,
        "f1_chinee": f1_chinee,
        "f1_snake": f1_snake,
        "latency_p95": b02_summary["latency_p95_ms"],
        "notes": "Mốc baseline cho mọi ablation Bước 2",
    }
    return t00_info


def execute_checkpoint2():
    print("=" * 80)
    print("      BẮT ĐẦU CHECKPOINT 2: ABLATION CÔNG THỨC HUẤN LUYỆN (>= 3 TRỤC)")
    print("=" * 80)

    # 1. Khởi tạo mốc T00
    t00 = load_b02_as_t00()
    print(f"Mốc T00: Val Macro-F1 = {t00['val_f1']:.4f} | Top-1 = {t00['val_top1']:.4f} | Chinee Apple F1 = {t00['f1_chinee']:.4f} | Snake Weed F1 = {t00['f1_snake']:.4f}")

    # Ghi T00 vào Excel
    add_row("Training", [
        t00["exp_id"],
        t00["backbone"],
        t00["axis"],
        t00["diff"],
        t00["seed"],
        t00["val_f1"],
        t00["val_top1"],
        t00["delta"],
        t00["f1_chinee"],
        t00["f1_snake"],
        t00["notes"],
    ])

    add_row("Summary", [
        6,
        "T00",
        "Training",
        f"convnext_tiny ({t00['diff']})",
        t00["val_f1"],
        t00["val_top1"],
        t00["latency_p95"],
        "Mốc so sánh cho Checkpoint 2",
    ])

    results = [t00]
    best_exp = None
    best_f1 = t00["val_f1"]

    for exp in EXPERIMENTS:
        exp_id = exp["exp_id"]
        axis = exp["axis"]
        diff = exp["diff"]
        notes = exp["notes"]
        cfg_kwargs = exp["cfg_kwargs"]

        print(f"\n[{exp_id}] Đang chạy {axis}: {diff}...")

        base_cfg = {
            "exp_id": exp_id,
            "backbone": "convnext_tiny",
            "seed": 0,
            "epochs": 12,
            "batch_size": 64,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 1.0,
            "loss": "ce",
            "aug": "basic",
            "amp": True,
            "images_dir": "data/images",
            "labels_dir": "data/labels",
            "out_dir": "runs",
            "pred_dir": "predictions",
            "save_test_predictions": False,
        }
        base_cfg.update(cfg_kwargs)
        cfg = Config(**base_cfg)

        summary_file = Path(f"runs/{exp_id}/seed0/summary.json")
        if summary_file.exists():
            print(f"[{exp_id}] Đã hoàn thành trước đó, tải kết quả từ {summary_file}...")
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            t0 = time.time()
            summary = run(cfg)
            elapsed = time.time() - t0

        val_pred_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/{exp_id}_seed0_val.csv")
        f1_chinee, f1_snake = get_difficult_classes_f1(val_pred_file)

        delta = round(summary["val_macro_f1"] - t00["val_f1"], 4)

        row_data = [
            exp_id,
            "convnext_tiny",
            axis,
            diff,
            cfg.seed,
            summary["val_macro_f1"],
            summary["val_top1"],
            delta,
            f1_chinee,
            f1_snake,
            notes,
        ]
        add_row("Training", row_data)

        summary_row = [
            len(results) + 6,
            exp_id,
            "Training",
            f"convnext_tiny + {diff}",
            summary["val_macro_f1"],
            summary["val_top1"],
            summary["latency_p95_ms"],
            f"Δ vs T00: {delta:+.4f}, Chinee: {f1_chinee}, Snake: {f1_snake}",
        ]
        add_row("Summary", summary_row)

        res_item = {
            "exp_id": exp_id,
            "backbone": "convnext_tiny",
            "axis": axis,
            "diff": diff,
            "val_f1": summary["val_macro_f1"],
            "val_top1": summary["val_top1"],
            "delta": delta,
            "f1_chinee": f1_chinee,
            "f1_snake": f1_snake,
            "notes": notes,
        }
        results.append(res_item)

        if summary["val_macro_f1"] > best_f1:
            best_f1 = summary["val_macro_f1"]
            best_exp = exp_id

    # Thí nghiệm T07: Tổ hợp tối ưu (Best Combo)
    print("\n[T07] Đang huấn luyện Tổ hợp tối ưu (Best Combo: Label Smoothing + RandAugment + EMA)...")
    cfg_combo = Config(
        exp_id="T07",
        backbone="convnext_tiny",
        seed=0,
        epochs=12,
        batch_size=64,
        lr_backbone=1e-4,
        lr_head=1e-3,
        weight_decay=0.05,
        warmup_epochs=1.0,
        loss="ls",
        label_smoothing=0.1,
        aug="randaug",
        ema_decay=0.999,
        amp=True,
        images_dir="data/images",
        labels_dir="data/labels",
        out_dir="runs",
        pred_dir="predictions",
        save_test_predictions=False,
    )
    summary_combo_file = Path("runs/T07/seed0/summary.json")
    if summary_combo_file.exists():
        print("[T07] Đã hoàn thành trước đó, tải kết quả từ file...")
        with open(summary_combo_file, "r", encoding="utf-8") as f:
            summary_combo = json.load(f)
    else:
        summary_combo = run(cfg_combo)
    val_pred_file = Path("submissions/2A202602368_duong_van_thanh/predictions/T07_seed0_val.csv")
    f1_chinee, f1_snake = get_difficult_classes_f1(val_pred_file)

    delta = round(summary_combo["val_macro_f1"] - t00["val_f1"], 4)
    row_data = [
        "T07",
        "convnext_tiny",
        "Combo",
        "Label Smoothing + RandAugment + EMA",
        0,
        summary_combo["val_macro_f1"],
        summary_combo["val_top1"],
        delta,
        f1_chinee,
        f1_snake,
        "Tổ hợp các yếu tố tối ưu để kiểm tra hiệu ứng cộng dồn",
    ]
    add_row("Training", row_data)

    summary_row = [
        len(results) + 6,
        "T07",
        "Training",
        "convnext_tiny + Combo (LS+RandAug+EMA)",
        summary_combo["val_macro_f1"],
        summary_combo["val_top1"],
        summary_combo["latency_p95_ms"],
        f"Δ vs T00: {delta:+.4f}, Chinee: {f1_chinee}, Snake: {f1_snake}",
    ]
    add_row("Summary", summary_row)

    results.append({
        "exp_id": "T07",
        "backbone": "convnext_tiny",
        "axis": "Combo",
        "diff": "Label Smoothing + RandAugment + EMA",
        "val_f1": summary_combo["val_macro_f1"],
        "val_top1": summary_combo["val_top1"],
        "delta": delta,
        "f1_chinee": f1_chinee,
        "f1_snake": f1_snake,
        "notes": "Tổ hợp tối ưu",
    })

    print("\n" + "=" * 80)
    print("                     BẢNG TỔNG HỢP KẾT QUẢ CHECKPOINT 2")
    print("=" * 80)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("=" * 80)
    print("Checkpoint 2 hoàn thành xuất sắc!")


if __name__ == "__main__":
    execute_checkpoint2()
