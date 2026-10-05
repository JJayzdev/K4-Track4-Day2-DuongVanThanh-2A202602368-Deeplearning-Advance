"""extract_and_verify.py - Giám sát, xác minh MD5 và giải nén images.zip vào data/images.
"""
import hashlib
import os
from pathlib import Path
import sys
import time
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "submissions/2A202602368_duong_van_thanh/code")
from dataset import load_split, check_split

EXPECTED_MD5 = "b7b30f96d466fba86016aa5a26606e0f"
ZIP_PATH = Path("data/images.zip")
IMAGES_DIR = Path("data/images")


def extract_and_verify():
    print(f"Bắt đầu kiểm tra file {ZIP_PATH}...")
    if not ZIP_PATH.exists():
        print(f"Lỗi: Không tìm thấy {ZIP_PATH}")
        return False

    print("Đang tính MD5 checksum...")
    h = hashlib.md5()
    with open(ZIP_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_md5 = h.hexdigest()
    print(f"MD5 thực tế: {actual_md5}")
    print(f"MD5 kỳ vọng: {EXPECTED_MD5}")

    if actual_md5 != EXPECTED_MD5:
        print(f"CẢNH BÁO: MD5 không trùng khớp! ({actual_md5} != {EXPECTED_MD5})")
    else:
        print("Xác minh MD5 HOÀN HẢO!")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Đang giải nén {ZIP_PATH} vào {IMAGES_DIR}...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(IMAGES_DIR)
    print(f"Giải nén hoàn tất trong {time.time() - t0:.2f}s!")

    # Kiểm tra số lượng ảnh
    # Một số file zip chứa thư mục images/ bên trong hoặc các file .jpg trực tiếp
    jpg_files = list(IMAGES_DIR.glob("*.jpg"))
    if len(jpg_files) == 0:
        # Kiểm tra nếu giải nén thành data/images/images/*.jpg
        nested = list(IMAGES_DIR.glob("*/*.jpg"))
        if len(nested) > 0:
            print(f"Phát hiện ảnh nằm trong thư mục con, đang di chuyển {len(nested)} ảnh ra {IMAGES_DIR}...")
            for f in nested:
                target = IMAGES_DIR / f.name
                if not target.exists():
                    f.rename(target)
            jpg_files = list(IMAGES_DIR.glob("*.jpg"))

    print(f"Số lượng file .jpg tìm thấy trong {IMAGES_DIR}: {len(jpg_files)} (kỳ vọng 17.509)")

    # Chạy check_split toàn diện
    train_df, val_df, test_df = load_split("data/labels", fold=0)
    stats = check_split(train_df, val_df, test_df, IMAGES_DIR)
    print("Mọi kiểm tra Checkpoint 0 đã ĐẠT CHUẨN!")
    return True


if __name__ == "__main__":
    extract_and_verify()
