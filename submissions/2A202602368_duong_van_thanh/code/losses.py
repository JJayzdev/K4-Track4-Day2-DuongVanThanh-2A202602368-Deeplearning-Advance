"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).
Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

from typing import Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", kw.get("weight", None))
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loss: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target)
        
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).
    Khi gamma = 0 và alpha = None, bằng đúng Cross-Entropy tiêu chuẩn.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Lấy log_prob và prob của đúng lớp ground-truth
        target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * target_log_probs
        return loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.
    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al.): w_c = (1 - beta) / (1 - beta^n_c)
    """
    counts_np = np.array(counts, dtype=np.float64)
    if beta <= 0.0:
        w = 1.0 / counts_np
        w = w / w.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts_np)
        w = (1.0 - beta) / effective_num
        w = w / w.sum() * len(counts_np)

    return torch.tensor(w, dtype=torch.float32)


def rand_bbox(size, lam: float):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn batch ảnh và nhãn theo mixup hoặc cutmix."""
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[perm, :, bbx1:bbx2, bby1:bby2]
        # Điều chỉnh lại lam theo diện tích thực tế
        actual_lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1)) / (x.size(-1) * x.size(-2))
        return x_mixed, (y_a, y_b, actual_lam)
    else:
        raise ValueError(f"Không hỗ trợ mix mode: {mode}")


def mixed_loss(criterion, logits: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss cho batch đã trộn: lam * L(logits, y_a) + (1 - lam) * L(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
