"""run_all_checkpoints.py - Master runner thực hiện tuần tự Checkpoint 2, Checkpoint 3, Checkpoint 4.

Đảm bảo pipeline chạy mượt mà, đầy đủ các bước từ:
  - Checkpoint 2: Huấn luyện ablation >= 3 trục trên convnext_tiny
  - Checkpoint 3: Đánh giá >= 4 phương pháp suy luận và đo độ trễ chuẩn GPU
  - Checkpoint 4: Vòng chung kết 3 seeds, đánh giá test chuẩn, chạy eval.py score và grade
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
from run_checkpoint2 import execute_checkpoint2
from run_checkpoint3 import execute_checkpoint3
from run_checkpoint4 import execute_checkpoint4


def main():
    t_start = time.time()
    print("=" * 80)
    print("         BẮT ĐẦU CHUỖI THỰC NGHIỆM TỔNG LỰC: CHECKPOINT 2 -> 3 -> 4")
    print("=" * 80)

    # 1. Chạy Checkpoint 2
    print("\n>>> BẮT ĐẦU CHECKPOINT 2: ABLATION CÔNG THỨC HUẤN LUYỆN")
    t0 = time.time()
    execute_checkpoint2()
    print(f">>> HOÀN TẤT CHECKPOINT 2 trong {(time.time() - t0)/60:.1f} phút!")

    # 2. Chạy Checkpoint 3
    print("\n>>> BẮT ĐẦU CHECKPOINT 3: PHƯƠNG PHÁP SUY LUẬN & ĐO ĐỘ TRỄ")
    t0 = time.time()
    execute_checkpoint3()
    print(f">>> HOÀN TẤT CHECKPOINT 3 trong {(time.time() - t0)/60:.1f} phút!")

    # 3. Chạy Checkpoint 4
    print("\n>>> BẮT ĐẦU CHECKPOINT 4: VÒNG CHUNG KẾT & CHẤM ĐIỂM TEST")
    t0 = time.time()
    execute_checkpoint4()
    print(f">>> HOÀN TẤT CHECKPOINT 4 trong {(time.time() - t0)/60:.1f} phút!")

    total_time = (time.time() - t_start) / 60
    print("\n" + "=" * 80)
    print(f"   TẤT CẢ CÁC CHECKPOINT ĐÃ HOÀN TẤT TRỌN VẸN TRONG {total_time:.1f} PHÚT!")
    print("=" * 80)


if __name__ == "__main__":
    main()
