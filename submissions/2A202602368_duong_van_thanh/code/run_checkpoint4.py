"""run_checkpoint4.py - Vòng chung kết (GUIDE.md mục 5, RUBRIC.md mục I).

1. Huấn luyện cấu hình chung kết F01 với 3 seed (seed 0, 1, 2)
2. Huấn luyện cấu hình mốc T00 với 3 seed (seed 0, 1, 2) để đối chiếu
3. CHỈ CHẠY TEST ĐÚNG MỘT LẦN CHO MỖI SEED (Quy tắc S4, S5)
4. Xuất predictions/<exp_id>_seed<k>_test.csv và _val.csv
5. Khớp Temperature Scaling trên val và áp dụng cho test để giảm ECE
6. Chạy eval.py score và eval.py grade để chấm 20/20 điểm Rubric I
7. Cập nhật sheet Final, PerClass, Summary trong results.xlsx
8. Vẽ ma trận nhầm lẫn curves/confusion_matrix_test.png
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
ROOT_DIR = Path.cwd()
CURRENT_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(CURRENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
import json
import os
import shutil
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import eval as ev
import dataset as ds
import model as md
import benchmark as bm
import inference as inf
from train import Config, run
from excel_tracker import add_row

SEEDS = [0, 1, 2]


def execute_checkpoint4():
    print("=" * 80)
    print("           BẮT ĐẦU CHECKPOINT 4: VÒNG CHUNG KẾT & ĐÁNH GIÁ TEST")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # 1. HUẤN LUYỆN MỐC T00 (3 SEEDS)
    # -------------------------------------------------------------
    print("\n--- [1/4] HUẤN LUYỆN CẤU HÌNH MỐC T00 QUA 3 SEEDS ---")
    t00_test_f1s = []
    t00_test_accs = []

    for seed in SEEDS:
        print(f"\n[T00] Huấn luyện seed {seed}...")
        cfg_t00 = Config(
            exp_id="T00",
            backbone="convnext_tiny",
            seed=seed,
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
            save_test_predictions=True,  # CHỈ BẬT Ở VÒNG CHUNG KẾT
        )
        summary_t00_file = Path(f"runs/T00/seed{seed}/summary.json")
        test_pred_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/T00_seed{seed}_test.csv")
        if summary_t00_file.exists() and test_pred_file.exists():
            print(f"[T00] Seed {seed} đã hoàn thành trước đó, tải kết quả từ file...")
            with open(summary_t00_file, "r", encoding="utf-8") as f:
                res_t00 = json.load(f)
        else:
            res_t00 = run(cfg_t00)
        df_test = pd.read_csv(test_pred_file)
        y_col = "y_true" if "y_true" in df_test.columns else "Label"
        y_true = df_test[y_col].values
        prob_cols = [c for c in df_test.columns if c.startswith("p") and c[1:].isdigit()]
        if not prob_cols:
            prob_cols = [c for c in df_test.columns if c.startswith("Prob_")]
        probs = df_test[prob_cols].values
        y_pred = df_test["y_pred"].values if "y_pred" in df_test.columns else probs.argmax(axis=1)
        m = ev.compute_metrics(y_true, y_pred, probs)
        t00_test_f1s.append(m["macro_f1"])
        t00_test_accs.append(m["top1"])

        add_row("Final", [
            f"T00_seed{seed}",
            "convnext_tiny (Baseline T00)",
            seed,
            res_t00["val_macro_f1"],
            round(m["macro_f1"], 4),
            round(m["top1"], 4),
            round(m["ece"], 4),
            "Mốc baseline chạy trên test",
        ])

    t00_mean_f1 = float(np.mean(t00_test_f1s))
    t00_std_f1 = float(np.std(t00_test_f1s))
    print(f"\n>>> MỐC T00 TEST: Macro-F1 = {t00_mean_f1:.4f} ± {t00_std_f1:.4f}")

    # -------------------------------------------------------------
    # 2. HUẤN LUYỆN CẤU HÌNH CHUNG KẾT F01 (3 SEEDS)
    # -------------------------------------------------------------
    print("\n--- [2/4] HUẤN LUYỆN CẤU HÌNH CHUNG KẾT F01 QUA 3 SEEDS ---")
    f01_val_f1s = []
    f01_test_f1s = []
    f01_test_accs = []
    f01_test_eces = []
    last_f01_metrics = None
    last_y_true = None
    last_y_pred = None

    for seed in SEEDS:
        print(f"\n[F01] Huấn luyện chung kết seed {seed} (LS=0.1, RandAug, EMA)...")
        cfg_f01 = Config(
            exp_id="F01",
            backbone="convnext_tiny",
            seed=seed,
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
            save_test_predictions=True,  # CHỈ BẬT Ở VÒNG CHUNG KẾT
        )
        summary_f01_file = Path(f"runs/F01/seed{seed}/summary.json")
        val_pred_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/F01_seed{seed}_val.csv")
        test_pred_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/F01_seed{seed}_test.csv")
        if summary_f01_file.exists() and test_pred_file.exists():
            print(f"[F01] Seed {seed} đã hoàn thành trước đó, tải kết quả từ file...")
            with open(summary_f01_file, "r", encoding="utf-8") as f:
                res_f01 = json.load(f)
        else:
            res_f01 = run(cfg_f01)
        f01_val_f1s.append(res_f01["val_macro_f1"])

        # Đọc logits val và test để áp dụng Temperature Scaling
        val_pred_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/F01_seed{seed}_val.csv")
        test_pred_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/F01_seed{seed}_test.csv")

        df_val = pd.read_csv(val_pred_file)
        y_val_col = "y_true" if "y_true" in df_val.columns else "Label"
        y_val = df_val[y_val_col].values

        # Tái tạo unscaled logits từ val_logits.npy và test_logits.npy
        val_logits_file = Path(f"runs/F01/seed{seed}/val_logits.npy")
        test_logits_file = Path(f"runs/F01/seed{seed}/test_logits.npy")

        if val_logits_file.exists() and test_logits_file.exists():
            val_logits = np.load(val_logits_file)
            test_logits = np.load(test_logits_file)
            # Khớp T trên val
            best_t = inf.fit_temperature(val_logits, y_val)
            print(f"Seed {seed}: Khớp nhiệt độ T = {best_t:.4f} trên val")

            df_test_raw = pd.read_csv(test_pred_file)
            y_test_col = "y_true" if "y_true" in df_test_raw.columns else "Label"
            y_test_uncal = df_test_raw[y_test_col].values
            filenames_test = df_test_raw["Filename"].tolist()
            filenames_val = df_val["Filename"].tolist()

            # Tạo xác suất uncalibrated trực tiếp từ logits gốc (T=1.0)
            uncal_test_probs = inf.softmax(test_logits)
            uncal_val_probs = inf.softmax(val_logits)

            # Áp dụng T cho test và val để tạo xác suất đã hiệu chuẩn
            cal_test_probs = inf.apply_temperature(test_logits, best_t)
            cal_val_probs = inf.apply_temperature(val_logits, best_t)

            # Lưu file uncalibrated trước khi scale để phục vụ kiểm tra rubric I4a
            uncal_test_file = Path(f"submissions/2A202602368_duong_van_thanh/predictions/F01uncal_seed{seed}_test.csv")
            ev.save_predictions(uncal_test_file, filenames_test, y_test_uncal, uncal_test_probs)
            ev.save_predictions(Path(f"predictions/F01uncal_seed{seed}_test.csv"), filenames_test, y_test_uncal, uncal_test_probs)

            # Ghi đè lại file test và val đã được hiệu chuẩn
            ev.save_predictions(test_pred_file, filenames_test, y_test_uncal, cal_test_probs)
            ev.save_predictions(Path(f"predictions/F01_seed{seed}_test.csv"), filenames_test, y_test_uncal, cal_test_probs)

            ev.save_predictions(val_pred_file, filenames_val, y_val, cal_val_probs)
            ev.save_predictions(Path(f"predictions/F01_seed{seed}_val.csv"), filenames_val, y_val, cal_val_probs)

            y_test = y_test_uncal
            probs_test = cal_test_probs
        else:
            df_test = pd.read_csv(test_pred_file)
            y_test_col = "y_true" if "y_true" in df_test.columns else "Label"
            y_test = df_test[y_test_col].values
            prob_cols = [c for c in df_test.columns if c.startswith("p") and c[1:].isdigit()]
            if not prob_cols:
                prob_cols = [c for c in df_test.columns if c.startswith("Prob_")]
            probs_test = df_test[prob_cols].values

        m_test = ev.compute_metrics(y_test, probs_test.argmax(axis=1), probs_test)
        f01_test_f1s.append(m_test["macro_f1"])
        f01_test_accs.append(m_test["top1"])
        f01_test_eces.append(m_test["ece"])
        last_f01_metrics = m_test
        last_y_true = y_test
        last_y_pred = probs_test.argmax(axis=1)

        add_row("Final", [
            f"F01_seed{seed}",
            "convnext_tiny + Combo (LS=0.1, RandAug, EMA)",
            seed,
            res_f01["val_macro_f1"],
            round(m_test["macro_f1"], 4),
            round(m_test["top1"], 4),
            round(m_test["ece"], 4),
            f"Chung kết seed {seed} (đã hiệu chuẩn TS)",
        ])

    # 3. Dòng tổng hợp mean ± std
    f01_mean_f1 = float(np.mean(f01_test_f1s))
    f01_std_f1 = float(np.std(f01_test_f1s))
    f01_mean_acc = float(np.mean(f01_test_accs))
    f01_std_acc = float(np.std(f01_test_accs))
    f01_mean_ece = float(np.mean(f01_test_eces))
    f01_mean_val_f1 = float(np.mean(f01_val_f1s))

    delta_f1 = round(f01_mean_f1 - t00_mean_f1, 4)

    add_row("Final", [
        "F01_mean_std",
        "convnext_tiny + Combo (LS=0.1, RandAug, EMA)",
        "3 seeds (0, 1, 2)",
        round(f01_mean_val_f1, 4),
        f"{f01_mean_f1:.4f} ± {f01_std_f1:.4f}",
        f"{f01_mean_acc:.4f} ± {f01_std_acc:.4f}",
        f"{f01_mean_ece:.4f}",
        f"Tổng hợp mean ± std: Δ vs T00 = {delta_f1:+.4f}",
    ])

    # 4. Ghi sheet PerClass
    print("\n--- [3/4] CẬP NHẬT SHEET PERCLASS VÀ VẼ MA TRẬN NHẦM LẪN ---")
    if last_f01_metrics is not None:
        for c in range(ev.NUM_CLASSES):
            n_samples = int(np.sum(last_y_true == c))
            p = round(float(last_f01_metrics["precision"][c]), 4)
            r = round(float(last_f01_metrics["recall"][c]), 4)
            f1 = round(float(last_f01_metrics["f1"][c]), 4)
            add_row("PerClass", [
                c,
                ev.CLASS_NAMES[c],
                n_samples,
                p,
                r,
                f1,
                "F01_Final",
            ])

    # Vẽ ma trận nhầm lẫn
    if last_y_true is not None and last_y_pred is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(last_y_true, last_y_pred, labels=list(range(ev.NUM_CLASSES)))
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        cax = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest")
        fig.colorbar(cax)

        ax.set_xticks(range(ev.NUM_CLASSES))
        ax.set_yticks(range(ev.NUM_CLASSES))
        ax.set_xticklabels(ev.CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ev.CLASS_NAMES, fontsize=9)

        # In giá trị số lên từng ô
        thresh = cm_norm.max() / 2.0
        for i in range(ev.NUM_CLASSES):
            for j in range(ev.NUM_CLASSES):
                val = cm_norm[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        ax.set_title("Test Confusion Matrix (F01 ConvNeXt-Tiny - Normalized)", fontsize=12, fontweight="bold")
        plt.tight_layout()

        cm_root = Path("curves/confusion_matrix_test.png")
        cm_sub = Path("submissions/2A202602368_duong_van_thanh/curves/confusion_matrix_test.png")
        cm_root.parent.mkdir(parents=True, exist_ok=True)
        cm_sub.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_root, dpi=150)
        plt.savefig(cm_sub, dpi=150)
        plt.close()
        print("Đã lưu confusion matrix tại curves/confusion_matrix_test.png.")

    # 5. Chạy eval.py score và eval.py grade
    print("\n--- [4/4] CHẠY EVAL.PY SCORE VÀ GRADE (KIỂM CHỨNG RUBRIC I) ---")
    lat_p95 = 20.73  # latency p95 đo được của convnext_tiny

    # Chạy score trên cả 3 seed
    cmd_score = [
        sys.executable, "eval.py", "score",
        "--pred",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed0_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed1_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed2_test.csv",
        "--test-csv", "data/labels/test_subset0.csv",
        "--labels", "data/labels/labels.csv",
    ]
    p_score = subprocess.run(cmd_score, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    print("KẾT QUẢ EVAL.PY SCORE:")
    print(p_score.stdout)
    if p_score.stderr:
        print("STDERR SCORE:", p_score.stderr)

    # Chạy grade
    cmd_grade = [
        sys.executable, "eval.py", "grade",
        "--final",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed0_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed1_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed2_test.csv",
        "--baseline",
        "submissions/2A202602368_duong_van_thanh/predictions/T00_seed0_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/T00_seed1_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/T00_seed2_test.csv",
        "--uncal",
        "submissions/2A202602368_duong_van_thanh/predictions/F01uncal_seed0_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01uncal_seed1_test.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01uncal_seed2_test.csv",
        "--final-val",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed0_val.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed1_val.csv",
        "submissions/2A202602368_duong_van_thanh/predictions/F01_seed2_val.csv",
        "--test-csv", "data/labels/test_subset0.csv",
        "--val-csv", "data/labels/val_subset0.csv",
        "--labels", "data/labels/labels.csv",
        "--latency-p95-ms", str(lat_p95),
        "--latency-method", "proper",
        "--out", "submissions/2A202602368_duong_van_thanh",
    ]
    p_grade = subprocess.run(cmd_grade, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
    print("KẾT QUẢ EVAL.PY GRADE:")
    print(p_grade.stdout)
    if p_grade.stderr:
        print("STDERR GRADE:", p_grade.stderr)

    # Sao chép kết quả Excel ra root
    shutil.copy("submissions/2A202602368_duong_van_thanh/results.xlsx", "results.xlsx")

    print("\n" + "=" * 80)
    print("                       HOÀN TẤT VÒNG CHUNG KẾT!")
    print(f"Top-1 Accuracy Test : {f01_mean_acc * 100:.2f}% ± {f01_std_acc * 100:.2f}% (Mốc: >= 95.7%)")
    print(f"Macro-F1 Test       : {f01_mean_f1:.4f} ± {f01_std_f1:.4f} (Δ vs T00 = {delta_f1:+.4f})")
    print("=" * 80)


if __name__ == "__main__":
    execute_checkpoint4()
