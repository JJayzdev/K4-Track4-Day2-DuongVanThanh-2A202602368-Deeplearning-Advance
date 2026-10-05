"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import random
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Đảm bảo import được eval.py ở repo gốc và các module nội bộ
CURRENT_DIR = Path(__file__).resolve().parent
p = CURRENT_DIR
while p != p.parent:
    if (p / "eval.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break
    p = p.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import eval as ev
import dataset as ds
import model as md
import losses as ls
import benchmark as bm


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên cho tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (model.param_groups)."""
    groups = md.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0."""
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler: GradScaler,
                    cfg: Config, device: torch.device, ema: Optional[EMA] = None) -> Dict[str, float]:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone bị đóng băng, giữ nguyên BatchNorm ở chế độ eval
    if cfg.init == "frozen":
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.size(0)

        mixed_targets = None
        if cfg.mix and cfg.mix.lower() != "none":
            x, mixed_targets = ls.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix.lower())

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
            if mixed_targets is not None:
                loss = ls.mixed_loss(criterion, logits, mixed_targets)
            else:
                loss = criterion(logits, y)

        if cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_y = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            x, y, filenames = batch
            x = x.to(device, non_blocking=True)
            y_cuda = y.to(device, non_blocking=True)
            batch_size = x.size(0)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                if criterion is not None:
                    loss = criterion(logits, y_cuda)
                    total_loss += loss.item() * batch_size

            all_filenames.extend(filenames)
            all_y.extend(y.numpy())
            all_logits.append(logits.float().cpu().numpy())
            total_samples += batch_size

    filenames = all_filenames
    y_true = np.array(all_y)
    logits_arr = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, total_samples) if criterion is not None else 0.0

    return filenames, y_true, logits_arr, avg_loss


def plot_curves(history: List[Dict[str, Any]], path: str | Path, title: str) -> None:
    """Vẽ đường cong training và lưu vào file ảnh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    val_acc = [h["val_top1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Subplot 1: Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="crimson", lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Metrics
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", color="forestgreen", lw=2)
    axes[1].plot(epochs, val_acc, label="Val Top-1 Acc", color="darkorange", lw=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> Dict[str, Any]:
    """Huấn luyện một cấu hình và lưu kết quả, trả về dict tóm tắt."""
    start_total_time = time.time()
    set_seed(cfg.seed)

    out_folder = run_dir(cfg)
    out_folder.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    # 1. Lưu config.json
    with open(out_folder / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 2. Đọc split & kiểm tra
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, fold=cfg.fold)
    split_stats = ds.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Tạo transform & DataLoader
    train_transform = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = ds.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds.make_loader(
        train_df, cfg.images_dir, train_transform,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = ds.make_loader(
        val_df, cfg.images_dir, val_transform,
        batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
    )

    # 4. Tạo Model, Criterion, Optimizer, Scheduler, EMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = md.build_model(
        name=cfg.backbone,
        pretrained=(cfg.init != "scratch"),
        num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)

    # Đếm params & GMACs
    param_count_m = md.count_params(model)
    gmacs = md.count_gmacs(model, img_size=cfg.img_size)

    # Criterion
    loss_kwargs = {}
    if cfg.loss == "ls":
        loss_kwargs["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        loss_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().values
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        w = ls.class_weights(counts, beta=beta).to(device)
        loss_kwargs["weight"] = w

    criterion = ls.build_criterion(kind=cfg.loss, **loss_kwargs)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Vòng lặp huấn luyện theo Epoch
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_weights_path = out_folder / "best_model.pth"
    epoch_times = []

    print(f"\n>>> BẮT ĐẦU HUẤN LUYỆN: Exp {cfg.exp_id} | Backbone: {cfg.backbone} | Seed: {cfg.seed} | Epochs: {cfg.epochs}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            device=device,
            ema=ema,
        )
        epoch_sec = time.time() - t0
        epoch_times.append(epoch_sec)

        # Đánh giá trên VAL
        if ema is not None:
            backup = ema.apply_shadow(model)

        val_filenames, val_y, val_logits, val_loss = evaluate(model, val_loader, eval_criterion, device)
        val_probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        val_probs = val_probs / np.sum(val_probs, axis=-1, keepdims=True)
        val_pred = val_probs.argmax(axis=1)

        val_metrics = ev.compute_metrics(val_y, val_pred, val_probs)
        macro_f1 = val_metrics["macro_f1"]
        top1_acc = val_metrics["top1"]

        if ema is not None:
            ema.restore(model, backup)

        log_item = {
            "epoch": epoch,
            "train_loss": round(train_res["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_top1": round(top1_acc, 4),
            "lr": train_res["lr"],
            "epoch_sec": round(epoch_sec, 2),
        }
        history.append(log_item)

        print(f"Epoch {epoch:2d}/{cfg.epochs} | Train Loss: {log_item['train_loss']:.4f} | Val Loss: {log_item['val_loss']:.4f} | Val F1: {macro_f1:.4f} | Val Top-1: {top1_acc:.4f} | Time: {epoch_sec:.1f}s")

        # Lưu checkpoint theo macro-F1 val (hòa thì giữ epoch sớm hơn)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)

    # 6. Tải checkpoint tốt nhất để xuất logit val và tính chỉ số chuẩn
    if best_weights_path.exists():
        md.load_checkpoint_weights(model, best_weights_path, device)

    val_filenames, val_y, val_logits, _ = evaluate(model, val_loader, None, device)
    val_probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
    val_probs = val_probs / np.sum(val_probs, axis=-1, keepdims=True)
    val_pred = val_probs.argmax(axis=1)

    # Lưu file dự đoán VAL
    val_pred_file = pred_path(cfg, "val")
    ev.save_predictions(val_pred_file, val_filenames, val_y, val_probs)
    sub_val_pred = Path("submissions/2A202602368_duong_van_thanh/predictions") / val_pred_file.name
    sub_val_pred.parent.mkdir(parents=True, exist_ok=True)
    if sub_val_pred != val_pred_file:
        ev.save_predictions(sub_val_pred, val_filenames, val_y, val_probs)
    np.save(out_folder / "val_logits.npy", val_logits)

    final_val_metrics = ev.compute_metrics(val_y, val_pred, val_probs)

    # 7. Đánh giá TEST (CHỈ BẬT KHI save_test_predictions = True Ở BƯỚC 4)
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(
            test_df, cfg.images_dir, val_transform,
            batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
        )
        test_filenames, test_y, test_logits, _ = evaluate(model, test_loader, None, device)
        test_probs = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
        test_probs = test_probs / np.sum(test_probs, axis=-1, keepdims=True)
        test_pred = test_probs.argmax(axis=1)

        test_pred_file = pred_path(cfg, "test")
        ev.save_predictions(test_pred_file, test_filenames, test_y, test_probs)
        sub_test_pred = Path("submissions/2A202602368_duong_van_thanh/predictions") / test_pred_file.name
        sub_test_pred.parent.mkdir(parents=True, exist_ok=True)
        if sub_test_pred != test_pred_file:
            ev.save_predictions(sub_test_pred, test_filenames, test_y, test_probs)
        np.save(out_folder / "test_logits.npy", test_logits)
        test_metrics = ev.compute_metrics(test_y, test_pred, test_probs)

    # 8. Lưu history.csv và vẽ curves
    pd.DataFrame(history).to_csv(out_folder / "history.csv", index=False)
    curve_filename = f"{cfg.exp_id}_{cfg.backbone}.png"
    curve_path = Path("curves") / curve_filename
    sub_curve_path = Path("submissions/2A202602368_duong_van_thanh/curves") / curve_filename
    plot_curves(history, curve_path, title=f"{cfg.exp_id} - {cfg.backbone} (Val Macro-F1: {best_macro_f1:.4f})")
    plot_curves(history, sub_curve_path, title=f"{cfg.exp_id} - {cfg.backbone} (Val Macro-F1: {best_macro_f1:.4f})")
    plot_curves(history, out_folder / "curves.png", title=f"{cfg.exp_id} - {cfg.backbone}")

    # Đo độ trễ sơ bộ batch 1
    latency_stat = bm.latency_report(model, batch_size=1, img_size=cfg.img_size, dtype="amp" if cfg.amp else "fp32", device=str(device), warmup=5, iters=30)

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "params_m": round(param_count_m, 2),
        "gmacs": round(gmacs, 2),
        "best_epoch": best_epoch,
        "val_macro_f1": round(final_val_metrics["macro_f1"], 4),
        "val_top1": round(final_val_metrics["top1"], 4),
        "val_ece": round(final_val_metrics["ece"], 4),
        "avg_epoch_sec": round(float(np.mean(epoch_times)), 2),
        "latency_p50_ms": latency_stat["p50"],
        "latency_p95_ms": latency_stat["p95"],
        "total_time_sec": round(time.time() - start_total_time, 2),
    }

    if test_metrics is not None:
        summary["test_macro_f1"] = round(test_metrics["macro_f1"], 4)
        summary["test_top1"] = round(test_metrics["top1"], 4)
        summary["test_ece"] = round(test_metrics["ece"], 4)

    with open(out_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>>> HOÀN TẤT THÍ NGHIỆM {cfg.exp_id}: Best Val Macro-F1 = {summary['val_macro_f1']:.4f} (Epoch {best_epoch}) | Params = {summary['params_m']}M | GMACs = {summary['gmacs']}")
    return summary


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Biến ['key=val', ...] thành dict ép kiểu chuẩn theo dataclass Config."""
    fields = Config.__dataclass_fields__
    overrides = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Tham số không hợp lệ: {p}, kỳ vọng định dạng key=value")
        key, val = p.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in fields:
            raise ValueError(f"Trường {key} không tồn tại trong Config. Các trường có sẵn: {list(fields.keys())}")

        field_type = fields[key].type
        # Ép kiểu cơ bản
        if val.lower() == "none":
            overrides[key] = None
        elif "bool" in str(field_type).lower():
            overrides[key] = val.lower() in ("true", "1", "yes")
        elif "int" in str(field_type).lower():
            overrides[key] = int(val)
        elif "float" in str(field_type).lower():
            overrides[key] = float(val)
        else:
            overrides[key] = val
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè siêu tham số dạng key=value")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    run(cfg)


if __name__ == "__main__":
    main()
