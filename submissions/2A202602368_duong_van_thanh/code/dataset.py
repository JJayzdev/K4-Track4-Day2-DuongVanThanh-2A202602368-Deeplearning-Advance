"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> Dict[str, Any]:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir` (nếu thư mục ảnh đã sẵn sàng)
    """
    images_dir = Path(images_dir)
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    # 1. Giao từng cặp tập
    train_val_intersect = train_files.intersection(val_files)
    train_test_intersect = train_files.intersection(test_files)
    val_test_intersect = val_files.intersection(test_files)

    assert len(train_val_intersect) == 0, f"Giao train ∩ val không rỗng: {len(train_val_intersect)} file"
    assert len(train_test_intersect) == 0, f"Giao train ∩ test không rỗng: {len(train_test_intersect)} file"
    assert len(val_test_intersect) == 0, f"Giao val ∩ test không rỗng: {len(val_test_intersect)} file"

    # 2. Hợp ba tập
    union_files = train_files | val_files | test_files
    assert len(union_files) == 17509, f"Hợp ba tập không bằng 17.509 ảnh: {len(union_files)}"

    # 3. Phân bố lớp
    train_counts = train_df["Label"].value_counts().sort_index().to_dict()
    val_counts = val_df["Label"].value_counts().sort_index().to_dict()
    test_counts = test_df["Label"].value_counts().sort_index().to_dict()
    total_counts = (train_df["Label"].value_counts() + val_df["Label"].value_counts() + test_df["Label"].value_counts()).sort_index().to_dict()

    # 4. Kiểm tra file trên đĩa nếu thư mục ảnh tồn tại
    missing_files = []
    if images_dir.exists() and any(images_dir.iterdir()):
        for f in union_files:
            if not (images_dir / f).exists():
                missing_files.append(f)
        assert len(missing_files) == 0, f"Có {len(missing_files)} file ảnh trong CSV không tồn tại trong {images_dir}"

    stats = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total": n_total,
        "train_pct": n_train / n_total * 100,
        "val_pct": n_val / n_total * 100,
        "test_pct": n_test / n_total * 100,
        "per_class": {
            cls_idx: {
                "name": CLASS_NAMES[cls_idx],
                "train": train_counts.get(cls_idx, 0),
                "val": val_counts.get(cls_idx, 0),
                "test": test_counts.get(cls_idx, 0),
                "total": total_counts.get(cls_idx, 0),
            }
            for cls_idx in range(NUM_CLASSES)
        },
        "overlap": {
            "train_val": len(train_val_intersect),
            "train_test": len(train_test_intersect),
            "val_test": len(val_test_intersect),
        },
        "missing_files": len(missing_files),
    }

    print(f"=== CHECK SPLIT HOÀN TẤT ===")
    print(f"Tổng số ảnh: {n_total} (Train: {n_train} [{stats['train_pct']:.2f}%], Val: {n_val} [{stats['val_pct']:.2f}%], Test: {n_test} [{stats['test_pct']:.2f}%])")
    print(f"Giao các tập: Train ∩ Val: {len(train_val_intersect)}, Train ∩ Test: {len(train_test_intersect)}, Val ∩ Test: {len(val_test_intersect)}")
    print(f"Hợp 3 tập: {len(union_files)} ảnh (kỳ vọng 17.509)")
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> T.Compose:
    """Tạo torchvision transform cho train hoặc val/test.
    `aug`:
      - "basic": RandomResizedCrop + RandomHorizontalFlip
      - "color": basic + ColorJitter
      - "randaug": basic + RandAugment
      - "trivial": basic + TrivialAugmentWide
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        t_list = []
        t_list.append(T.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC))
        t_list.append(T.RandomHorizontalFlip(p=0.5))

        if aug == "color":
            t_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        elif aug == "randaug":
            t_list.append(T.RandAugment(num_ops=2, magnitude=9))
        elif aug == "trivial":
            t_list.append(T.TrivialAugmentWide())
        elif aug == "basic":
            pass
        else:
            raise ValueError(f"Không hỗ trợ loại aug: {aug}")

        t_list.append(T.ToTensor())
        t_list.append(normalize)
        return T.Compose(t_list)
    else:
        # Val / Test: Resize 256 -> CenterCrop(img_size) -> ToTensor -> Normalize
        # Không có augmentation ngẫu nhiên
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize,
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh DeepWeeds từ thư mục ảnh và nhãn theo DataFrame.
    Trả về (image_tensor, int(label), filename)
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        filename = self.filenames[idx]
        label = self.labels[idx]
        img_path = self.images_dir / filename

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label, filename


def _worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: Optional[str] = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    data_sampler = None
    shuffle = False

    if train:
        if sampler == "balanced":
            counts = df["Label"].value_counts().to_dict()
            class_weights = {cls: 1.0 / count for cls, count in counts.items()}
            sample_weights = [class_weights[y] for y in df["Label"]]
            data_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            shuffle = True
    else:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=data_sampler,
        drop_last=(train and not data_sampler),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn if train else None,
    )
