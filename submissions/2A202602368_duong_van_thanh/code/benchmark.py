"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100, sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây (ms)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo đạc
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    target_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(target_device)
    model.eval()

    x = torch.randn(batch_size, 3, img_size, img_size, device=target_device)
    sync = torch.cuda.synchronize if target_device.type == "cuda" else None

    if dtype == "fp16":
        model = model.half()
        x = x.half()
        
        @torch.inference_mode()
        def fn():
            return model(x)
            
    elif dtype == "amp":
        @torch.inference_mode()
        def fn():
            with torch.cuda.amp.autocast():
                return model(x)
                
    elif dtype == "fp32":
        model = model.float()
        x = x.float()
        
        @torch.inference_mode()
        def fn():
            return model(x)
    else:
        raise ValueError(f"Không hỗ trợ dtype: {dtype}")

    res = bench(fn, warmup=warmup, iters=iters, sync=sync)
    p50 = res["p50"]
    images_per_s = batch_size / (p50 / 1000.0) if p50 > 0 else 0.0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(p50, 2),
        "p95": round(res["p95"], 2),
        "p99": round(res["p99"], 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(res["p95"], 2),
        "p99_ms": round(res["p99"], 2),
        "mean": round(res["mean"], 2),
        "images_per_s": round(images_per_s, 1),
        "throughput_img_per_sec": round(images_per_s, 1),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ thực tế khi chạy TTA với K views."""
    target_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(target_device)
    model.eval()

    views = [torch.randn(batch_size, 3, img_size, img_size, device=target_device) for _ in range(k_views)]
    sync = torch.cuda.synchronize if target_device.type == "cuda" else None

    if dtype == "fp16":
        model = model.half()
        views = [v.half() for v in views]
        
        @torch.inference_mode()
        def fn():
            outs = [model(v) for v in views]
            return torch.stack(outs).mean(0)
    elif dtype == "amp":
        @torch.inference_mode()
        def fn():
            with torch.cuda.amp.autocast():
                outs = [model(v) for v in views]
                return torch.stack(outs).mean(0)
    else:
        model = model.float()
        views = [v.float() for v in views]
        
        @torch.inference_mode()
        def fn():
            outs = [model(v) for v in views]
            return torch.stack(outs).mean(0)

    res = bench(fn, warmup=warmup, iters=iters, sync=sync)
    p50 = res["p50"]
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "k_views": k_views,
        "img_size": img_size,
        "p50": round(p50, 2),
        "p95": round(res["p95"], 2),
        "p99": round(res["p99"], 2),
        "images_per_s": round(batch_size / (p50 / 1000.0), 1),
    }
