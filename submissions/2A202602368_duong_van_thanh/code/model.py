"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

from typing import List, Dict, Any
import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init`:
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = pretrained and (init in ("frozen", "finetune"))

    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.
    Backbone đóng băng thì BatchNorm cũng phải ở chế độ eval lúc train.
    """
    # Đóng băng toàn bộ tham số
    for param in model.parameters():
        param.requires_grad = False

    # Mở khóa head classifier
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
    elif isinstance(classifier, torch.Tensor):
        classifier.requires_grad = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.
    - backbone ndim > 1 (weights): lr = lr_backbone, weight_decay = weight_decay
    - backbone ndim <= 1 (biases, norms): lr = lr_backbone, weight_decay = 0
    - head: lr = lr_head, weight_decay = weight_decay (hoặc no decay cho head bias/norm nếu muốn, ở đây áp dụng weight_decay cho head weights)
    """
    head_params = set()
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params = set(classifier.parameters())

    backbone_decay = []
    backbone_no_decay = []
    head_group = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in head_params:
            head_group.append(param)
        else:
            # Tham số thuộc backbone
            if param.ndim <= 1 or "bn" in name.lower() or "norm" in name.lower() or "bias" in name.lower():
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({
            "params": backbone_decay,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        })
    if backbone_no_decay:
        groups.append({
            "params": backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0,
        })
    if head_group:
        groups.append({
            "params": head_group,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        # Xóa triệt để các buffer total_ops, total_params mà thop tiêm vào
        for m in model.modules():
            m._buffers.pop("total_ops", None)
            m._buffers.pop("total_params", None)
        return float(flops / 1e9)
    except Exception:
        # Dự phòng bằng torchinfo hoặc fvcore nếu cần
        return 0.0


def load_checkpoint_weights(model: nn.Module, ckpt_path, device: torch.device):
    """Nạp trọng số sạch sẽ an toàn, loại bỏ buffer phụ trợ nếu có."""
    state_dict = torch.load(ckpt_path, map_location=device)
    clean_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }
    model.load_state_dict(clean_state_dict, strict=False)

