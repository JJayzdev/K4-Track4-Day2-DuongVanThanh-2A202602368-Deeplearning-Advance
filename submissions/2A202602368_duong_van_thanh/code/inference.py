"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Callable
import copy
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax chuẩn hóa theo hàng với số mũ ổn định."""
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device: str = "cuda", view: Optional[Callable] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    target_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(target_device)
    model.eval()

    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            x, y, filenames = batch
            x = x.to(target_device)
            if view is not None:
                x = view(x)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)

            all_filenames.extend(filenames)
            all_y_true.extend(y.numpy())
            all_logits.append(logits.float().cpu().numpy())

    return all_filenames, np.array(all_y_true), np.concatenate(all_logits, axis=0)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) trên chiều rộng."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> List[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    _, _, H, W = x.shape
    crops = [
        x[:, :, :crop, :crop],             # top-left
        x[:, :, :crop, W - crop:],         # top-right
        x[:, :, H - crop:, :crop],         # bottom-left
        x[:, :, H - crop:, W - crop:],     # bottom-right
        x[:, :, (H - crop) // 2:(H + crop) // 2, (W - crop) // 2:(W + crop) // 2],  # center
    ]
    return crops


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    scaled = []
    for s in sizes:
        scaled.append(F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False))
    return scaled


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62)."""
    if space == "prob":
        probs = [softmax(l) for l in logits_per_view]
        return np.mean(probs, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return softmax(avg_logits)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69)."""
    def nll_objective(T: float) -> float:
        scaled_logits = val_logits / T
        scaled_logits -= np.max(scaled_logits, axis=-1, keepdims=True)
        exp_z = np.exp(scaled_logits)
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        # NLL = - 1/N sum log(p[i, y_i])
        n = len(val_labels)
        correct_probs = np.clip(probs[np.arange(n), val_labels], 1e-12, 1.0)
        return -np.mean(np.log(correct_probs))

    res = scipy.optimize.minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
    best_t = float(res.x)
    return best_t


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    return softmax(logits / T)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75)."""
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    try:
        # Sử dụng tiện ích tích hợp sẵn của PyTorch nếu có
        import torch.nn.utils.fusion as fusion
        for name, m in fused_model.named_children():
            if isinstance(m, nn.Sequential):
                for i in range(len(m) - 1):
                    if isinstance(m[i], nn.Conv2d) and isinstance(m[i+1], nn.BatchNorm2d):
                        m[i] = fusion.fuse_conv_bn_eval(m[i], m[i+1])
                        m[i+1] = nn.Identity()
    except Exception:
        pass

    return fused_model
