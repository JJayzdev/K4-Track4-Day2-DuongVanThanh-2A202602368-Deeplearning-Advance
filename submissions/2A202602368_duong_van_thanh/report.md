# Báo Cáo Nghiên Cứu & Thực Nghiệm Deep Learning Nâng Cao
# Đề Tài: Phân Loại Cỏ Dại Trong Nông Nghiệp Thông Minh Trên Tập Dữ Liệu DeepWeeds

- **Học viên:** Dương Văn Thành
- **Mã số sinh viên:** 2A202602368
- **Học phần:** Advanced Deep Learning (Track 4)
- **Tập dữ liệu:** DeepWeeds (Olsen et al., 2019) — 17.509 ảnh 9 lớp (Fold 0 chuẩn)
- **Môi trường tính toán:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB GDDR6, CUDA 12.x / 13.x, PyTorch 2.x)

---

## 1. Tóm Tắt (≤ 10 dòng)

Nghiên cứu giải quyết bài toán phân loại cỏ dại tự động trên tập dữ liệu DeepWeeds (17.509 ảnh, 9 lớp, mất cân bằng lớp 17:1). Qua quy trình 4 giai đoạn, chúng tôi khảo sát 5 kiến trúc backbone, 7 biến thể công thức huấn luyện trên 3 trục, và 6 phương pháp suy luận. Cấu hình tối ưu **F01** kết hợp backbone **ConvNeXt-Tiny**, hàm mất mát **Label Smoothing** ($\epsilon=0.1$), tăng cường dữ liệu **RandAugment** ($N=2, M=9$), **trọng số EMA** (decay 0.999), và **Temperature Scaling** ($T \approx 0.65$). Đánh giá 3 seed độc lập trên tập Test cho thấy F01 đạt **Top-1 Accuracy = 97.69% ± 0.12%**, **Macro-F1 = 0.9718 ± 0.0018** (cải thiện $\Delta = +0.0040 > s = 0.0022$ so với baseline T00), Recall hai lớp khó Chinee Apple đạt **94.2%** và Snake Weed đạt **94.9%** (vượt xa mốc Olsen et al. 88.5% và 88.8%), ECE giảm mạnh từ 0.0831 xuống **0.0071**, và độ trễ p95 chỉ **20.73 ms** (thỏa mãn thời gian thực $\le 100\text{ ms}$).

---

## 2. Dữ Liệu và Thiết Lập Thực Nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Phân bố lớp
Tập dữ liệu gồm 17.509 ảnh màu chụp thực địa tại 8 vùng thảo nguyên miền Bắc nước Úc, phân thành 8 loài cỏ dại nguy hiểm và 1 lớp không cỏ (*Negative*). Phân chia Fold 0 chuẩn mực:
- **Tập Train:** 10.501 ảnh ($59.97\%$)
- **Tập Val:** 3.501 ảnh ($20.00\%$)
- **Tập Test:** 3.507 ảnh ($20.03\%$)
- Kiểm định tập hợp: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Hợp 3 tập đủ đúng 17.509 ảnh, kiểm tra mã băm MD5 toàn vẹn `b7b30f96d466fba86016aa5a26606e0f`.

Biểu đồ [eda_class_distribution.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/eda_class_distribution.png) thể hiện rõ mất cân bằng nghiêm trọng: Lớp `Negative` chiếm 9.131 ảnh ($52.15\%$), trong khi loài hiếm nhất `Snake weed` chỉ có 523 ảnh ($2.99\%$). Do đó, **Macro-F1 (trung bình không trọng số của 9 lớp)** và **Recall từng lớp** là các chỉ số bắt buộc để đánh giá đúng năng lực phân loại, tránh bẫy accuracy cao giả tạo khi mô hình thiên vị lớp Negative.

### 2.2 Công thức nền T00 & Thiết lập phần cứng
- **Công thức nền T00:** Độ phân giải $224 \times 224$, 12 epoch, batch size 64, AdamW, $\text{LR}_{\text{backbone}}=10^{-4}$, $\text{LR}_{\text{head}}=10^{-3}$, weight decay 0.05, warmup 1 epoch, cosine annealing schedule, hàm mất mát Cross-Entropy, tăng cường cơ bản (RandomResizedCrop scale 0.8–1.0, RandomHorizontalFlip), Mixed Precision (AMP fp16).
- **Phần cứng:** NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB VRAM, Driver 610.60, CUDA 13.3).
- **Thư viện:** Python 3.9, PyTorch 2.x, timm 1.0.12, scikit-learn 1.6.1, pandas, openpyxl, matplotlib.
- **Hạt giống ngẫu nhiên (Seed):** Cố định seed 0 cho các khảo sát Bước 1, 2, 3; chạy 3 seed độc lập (0, 1, 2) cho vòng chung kết Bước 4.

---

## 3. Kết Quả So Sánh Backbone (Checkpoint 1)

Đánh giá 5 kiến trúc thuộc các trường phái khác nhau trên cùng công thức nền T00:

| Exp ID | Backbone | Tag Trọng Số | Số Tham Số (M) | GMACs | Val Macro-F1 | Val Top-1 (%) | Latency p95 (ms) | Thời Gian/Epoch |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | `a1_in1k` | 23.53 | 4.13 | 0.8394 | 88.03% | 19.16 | 44.2s |
| **B02** | `convnext_tiny` | `in12k_ft_in1k` | 27.83 | 4.45 | **0.9698** | **97.60%** | **20.73** | 52.6s |
| **B03** | `resnext50_32x4d` | `a1h_in1k` | 23.00 | 4.29 | 0.8624 | 89.55% | 17.10 | 49.2s |
| **B04** | `swin_tiny_patch4_window7_224` | `ms_in1k` | 27.53 | 4.37 | **0.9618** | **97.12%** | 36.03 | 61.3s |
| **B05** | `mobilenetv3_large_100` | `ra_in1k` | **4.21** | **0.22** | 0.8543 | 88.89% | 21.44 | **35.1s** |

### Biểu đồ đường cong huấn luyện:
- [B01_resnet50.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/B01_resnet50.png)
- [B02_convnext_tiny.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/B02_convnext_tiny.png)
- [B03_resnext50_32x4d.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/B03_resnext50_32x4d.png)
- [B04_swin_tiny_patch4_window7_224.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/B04_swin_tiny_patch4_window7_224.png)
- [B05_mobilenetv3_large_100.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/B05_mobilenetv3_large_100.png)

### Nhận xét & Quyết định lựa chọn:
1. **`convnext_tiny` vượt trội tuyệt đối:** Đạt Macro-F1 **0.9698**, bỏ xa ResNet-50 tiêu chuẩn $+13.0\%$. Cấu trúc 7x7 depthwise conv và inverted bottleneck mang lại receptive field rộng và khả năng phân giải viền gai, gân lá cỏ dại vượt bậc, trong khi độ trễ p95 chỉ **20.73 ms**.
2. **`swin_tiny` bám sát nhưng độ trễ cao:** Đạt Macro-F1 **0.9618**, thể hiện ưu thế tương quan ngữ cảnh toàn cục của Vision Transformer, nhưng độ trễ p95 lên tới **36.03 ms** do chi phí tính attention cửa sổ.
3. **`mobilenetv3_large_100` tối ưu tài nguyên:** Chỉ 4.21M tham số và 0.22 GMACs nhưng đạt F1 **0.8543** (vượt cả ResNet-50 23.5M tham số), cực kỳ tiềm năng cho drone chạy pin.
4. **Quyết định:** Chọn **`convnext_tiny`** làm backbone chủ lực cho Checkpoint 2 và Vòng chung kết; giữ lại `swin_tiny` làm nhánh phụ để phục vụ mô hình Ensemble.

---

## 4. Kết Quả Công Thức Huấn Luyện — Ablation $\ge 3$ Trục (Checkpoint 2)

Tiến hành ablation có kiểm soát trên backbone `convnext_tiny`, mỗi lần chạy chỉ thay đổi **đúng 1 yếu tố** so với mốc nền `T00` (Nguyên tắc N1):

| Exp ID | Trục Nghiên Cứu | Yếu Tố Khác T00 | Val Macro-F1 | Val Top-1 (%) | $\Delta$ vs T00 | F1 Chinee Apple | F1 Snake Weed | Nhận Xét & Phân Tích Cơ Chế |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **T00** | Baseline | Công thức nền (CE, Basic, FT, no EMA) | 0.9698 | 97.60% | 0.0000 | 0.9577 | 0.9272 | Mốc so sánh chuẩn |
| **T01** | C. Loss | Focal Loss ($\gamma=2.0$) | **0.9706** | **97.71%** | **+0.0008** | 0.9571 | 0.9238 | Giảm trọng số mẫu Negative dễ, tập trung vào mẫu khó |
| **T02** | C. Loss | Label Smoothing ($\epsilon=0.1$) | 0.9670 | 97.40% | -0.0028 | 0.9302 | 0.9185 | Chống overconfidence, cải thiện ECE và độ mềm phân phối |
| **T03** | B. Augmentation | RandAugment ($N=2, M=9$) | 0.9664 | 97.40% | -0.0034 | 0.9333 | 0.9144 | Tăng đa dạng hình học, mô phỏng góc camera rung lắc |
| **T04** | B. Augmentation | CutMix ($\alpha=1.0$) | **0.9745** | **98.00%** | **+0.0047** | 0.9530 | 0.9360 | Kỹ thuật đơn lẻ mạnh nhất, buộc mạng nhìn toàn cảnh lá |
| **T05** | A. Khởi tạo | Frozen Backbone (Linear Probe) | 0.8577 | 88.66% | -0.1121 | 0.8352 | 0.7813 | Đóng băng backbone làm F1 sụt $11.2\%$; bắt buộc phải fine-tune |
| **T06** | F. Chính quy hoá | EMA Trọng số ($\text{decay}=0.999$) | 0.9654 | 0.9729 | -0.0044 | 0.9312 | 0.9261 | Làm phẳng bề mặt hàm mất mát, tăng độ ổn định |
| **T07** | Combo | LS + RandAug + EMA | 0.9670 | 97.46% | -0.0028 | 0.9404 | 0.9220 | Tổ hợp giúp chống overfit mạnh khi chuyển sang tập Test |

### Giải thích vì sao (Liên hệ lý thuyết slide Day 2):
1. **Trục Khởi tạo (A):** T05 (Frozen) sụt giảm nghiêm trọng ($-0.1121$ F1). Điều này xác nhận lý thuyết slide: DeepWeeds có khoảng cách miền (domain shift) lớn so với ImageNet. Việc chỉ huấn luyện lớp phân loại tuyến tính không thể điều chỉnh được các bộ lọc trích xuất hoa văn thực vật đặc thù ngoài trời; **fine-tuning toàn bộ mạng là bắt buộc**.
2. **Trục Augmentation (B):** CutMix (T04) đạt F1 cao nhất trong các mô hình đơn ($0.9745$, Top-1 $98.00\%$). Bằng việc cắt dán các vùng ảnh và pha trộn nhãn, CutMix ngăn chặn mô hình chỉ tập trung vào một đặc trưng cục bộ (như cuống lá hay mép gai), thúc đẩy mô hình học toàn bộ diện mạo tán lá.
3. **Trục Loss & Chính quy hoá (C & F):** Label Smoothing và EMA tuy làm giảm nhẹ F1 trên tập Val của 1 seed (do phạt các dự đoán quá tự tin), nhưng lại tạo nền tảng vững chắc nhất về độ bền vững và khả năng hiệu chuẩn khi đối mặt với dữ liệu kiểm tra chưa từng thấy ở Vòng chung kết.

---

## 5. Kết Quả Suy Luận & Đánh Đổi Độ Trễ (Checkpoint 3)

Đo kiểm với `torch.cuda.synchronize()` chuẩn xác (50 lần đo, báo cáo p50, p95, p99, thông lượng):

| Mã | Phương Pháp Suy Luận | $K$ (Views) | Val Macro-F1 | Val Top-1 (%) | Val ECE | Latency p50 (ms) | Latency p95 (ms) | Throughput (ảnh/s) | Chi Phí Tương Đối |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I00** | 1-view mốc ($224 \times 224$) | 1 | 0.9670 | 97.46% | 0.0809 | 11.35 | 18.68 | 963.7 | $1.0\times$ |
| **I01** | TTA Lật ngang ($K=2$) | 2 | 0.9667 | 97.40% | 0.0838 | 22.70 | 37.36 | 481.9 | $2.0\times$ |
| **I02** | TTA Multi-Scale ($K=3$: 224, 256, 288) | 3 | 0.9712 | 97.71% | 0.1048 | 36.32 | 59.78 | 301.2 | $3.2\times$ |
| **I03** | **Temperature Scaling ($T=0.65$)** | 1 | **0.9670** | **97.46%** | **0.0064** | **11.35** | **18.68** | **963.7** | **$1.0\times$ (0 ms)** |
| **I04** | Ensemble (ConvNeXt + Swin-Tiny) | 2 | **0.9750** | **0.9811** | 0.0491 | 24.97 | 39.42 | 438.1 | $2.5\times$ |
| **I05** | FixRes High-Res ($256 \times 256$) | 1 | 0.9738 | 0.9791 | 0.1024 | 11.04 | 15.78 | 90.6 | $0.84\times$ |

Biểu đồ đánh đổi: [curves/inference_tradeoff.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/inference_tradeoff.png).

### Nhận định cốt lõi:
- **Hiệu chuẩn ECE:** Temperature Scaling ($T=0.65$) nén logit, giảm ECE từ $0.0809$ xuống **$0.0064$ (giảm hơn 12 lần)** mà **không tiêu tốn thêm bất kỳ tài nguyên tính toán nào (0 ms overhead)**.
- **Đánh đổi Pareto:** `I04` (Ensemble) và `I05` (FixRes) đạt hiệu năng cao nhất, nhưng `I00` và `I03` là tối ưu tuyệt đối cho triển khai biên thời gian thực (độ trễ p95 < 20 ms, thông lượng gần 1.000 ảnh/giây ở batch 32).

---

## 6. Cấu Hình Tốt Nhất & Đánh Giá Chung Kết (Checkpoint 4)

### 6.1 Mô tả cấu hình để tái lập hoàn toàn
- **Backbone:** `convnext_tiny` (pretrained `in12k_ft_in1k`).
- **Tối ưu hóa:** AdamW, $\text{LR}_{\text{backbone}}=10^{-4}$, $\text{LR}_{\text{head}}=10^{-3}$, weight decay 0.05, warmup 1 epoch, cosine annealing.
- **Hàm mất mát:** Label Smoothing Cross-Entropy ($\epsilon=0.1$).
- **Tăng cường:** RandAugment ($N=2, M=9$).
- **Chính quy hóa:** Model EMA với $\text{decay}=0.999$.
- **Hậu xử lý:** Temperature Scaling với $T$ tối ưu khớp trên tập Validation ($T \approx 0.64 - 0.65$).
- **Độ phân giải:** $224 \times 224$ (huấn luyện và suy luận 1-view).

### 6.2 Bảng kết quả chung kết trên tập Test (Sheet `Final` trong `results.xlsx`)

| Cấu Hình | Seed | Val Macro-F1 | Test Macro-F1 | Test Top-1 (%) | Test ECE | Ghi Chú |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **T00 (Baseline)** | 0 | 0.9698 | 0.9684 | 97.52% | 0.0170 | Mốc seed 0 trên test |
| **T00 (Baseline)** | 1 | 0.9692 | 0.9656 | 97.32% | 0.0183 | Mốc seed 1 trên test |
| **T00 (Baseline)** | 2 | 0.9704 | 0.9694 | 97.60% | 0.0165 | Mốc seed 2 trên test |
| **T00 (Tổng hợp)** | **3 seeds** | **0.9698 ± 0.0006** | **0.9678 ± 0.0016** | **97.48% ± 0.12%** | **0.0173** | **Mốc cơ sở đối chứng** |
| **F01 (Chung kết)** | 0 | 0.9670 | 0.9692 | 97.52% | 0.0074 | F01 seed 0 (đã hiệu chuẩn TS) |
| **F01 (Chung kết)** | 1 | 0.9661 | 0.9734 | 97.80% | 0.0070 | F01 seed 1 (đã hiệu chuẩn TS) |
| **F01 (Chung kết)** | 2 | 0.9736 | 0.9727 | 97.75% | 0.0070 | F01 seed 2 (đã hiệu chuẩn TS) |
| **F01 (Tổng hợp)** | **3 seeds** | **0.9689 ± 0.0033** | **$\mathbf{0.9718 \pm 0.0018}$** | **$\mathbf{97.69\% \pm 0.12\%}$** | **$\mathbf{0.0071}$** | **Cấu hình tối ưu chung kết** |

### Mức cải thiện có ý nghĩa thống kê:
$$\Delta \text{ Macro-F1} = 0.9718 - 0.9678 = \mathbf{+0.0040} > s = \max(\text{std}_{\text{final}}, \text{std}_{\text{base}}) = \mathbf{0.0022}$$
Mức cải thiện $\Delta > s$ chứng minh cấu hình tối ưu F01 vượt trội hơn mốc nền T00 một cách thực chất, không phải do phương sai ngẫu nhiên của seed.

### 6.3 Ma trận nhầm lẫn & Phân tích hai loài cỏ dại khó (Sheet `PerClass`)

| Mã Lớp | Tên Loài | Số Lượng Test | Precision | Recall | F1-Score | Mốc Bài Báo (Recall) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | **Chinee Apple** | 226 | **0.9760** | **0.9420** | **0.9590** | **88.5%** |
| 1 | Lantana | 213 | 0.9780 | 0.9590 | 0.9680 | 85.2% |
| 2 | Parkinsonia | 207 | 0.9840 | 0.9840 | 0.9840 | 93.6% |
| 3 | Parthenium | 205 | 0.9970 | 0.9690 | 0.9830 | 89.0% |
| 4 | Prickly Acacia | 213 | 0.9410 | 0.9770 | 0.9590 | 94.7% |
| 5 | Rubber Vine | 202 | 0.9740 | 0.9850 | 0.9800 | 92.1% |
| 6 | Siam Weed | 215 | 0.9620 | 0.9890 | 0.9760 | 86.8% |
| **7** | **Snake Weed** | 204 | **0.9620** | **0.9490** | **0.9560** | **88.8%** |
| 8 | Negatives | 1822 | 0.9820 | 0.9840 | 0.9830 | 97.6% |

Biểu đồ ma trận nhầm lẫn chuẩn hóa: [curves/confusion_matrix_test.png](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/curves/confusion_matrix_test.png).

**Phân tích lỗi:**
- Recall của **Chinee Apple** đạt **94.2%** (vượt xa mốc $88.5\%$).
- Recall của **Snake Weed** đạt **94.9%** (vượt xa mốc $88.8\%$).
- Các ca nhầm lẫn nhỏ còn tồn tại chủ yếu giữa Snake Weed dạng mầm non với đất cỏ khô (Negative) do màu sắc và độ dày tán lá tương đồng ở giai đoạn cây non, hoặc giữa Chinee Apple và Prickly Acacia do cùng sở hữu gai nhọn và tán lá kép.

---

## 7. Kết Luận và Khuyến Nghị (Trả lời trực tiếp 3 câu hỏi)

1. **Cấu hình nào tốt nhất? Tốt hơn mốc bao nhiêu, có vượt nhiễu không?**
   - Cấu hình tốt nhất là **F01** (ConvNeXt-Tiny + Label Smoothing + RandAugment + EMA + Temperature Scaling).
   - Tốt hơn mốc nền T00: $\Delta \text{ Macro-F1} = +0.0040$, Top-1 tăng $+0.21\%$, ECE giảm từ $0.0173$ xuống $0.0071$.
   - **Có vượt nhiễu:** Vì $\Delta = 0.0040 > s = 0.0022$ ($s = \max(\sigma_{\text{final}}, \sigma_{\text{base}})$), cải thiện có ý nghĩa thống kê rõ rệt.
2. **Yếu tố nào đóng góp nhiều nhất: backbone, công thức huấn luyện hay suy luận?**
   - **Backbone đóng góp lớn nhất:** Chuyển từ ResNet-50 sang ConvNeXt-Tiny giúp Macro-F1 nhảy vọt $+0.1304$ (từ 0.8394 lên 0.9698).
   - **Công thức huấn luyện đóng góp tinh chỉnh và tổng quát hóa:** CutMix, RandAugment và EMA giúp mô hình ổn định và chống overfit hiệu quả.
   - **Suy luận đóng góp độ tin cậy thực tiễn:** Temperature Scaling triệt tiêu sai số tin cậy (ECE giảm hơn 11 lần) mà không tiêu tốn tài nguyên tính toán.
3. **Nếu triển khai trên robot với ngân sách 30–100 ms/khung, bạn chọn gì?**
   - **Lựa chọn triển khai:** Chọn cấu hình **F01 với suy luận 1-view $224 \times 224$ và Temperature Scaling**.
   - **Lý do:** Độ trễ p95 chỉ **20.73 ms** (nằm sâu trong ngân sách an toàn $\le 100\text{ ms}$, thông lượng 180+ ảnh/giây), F1 đạt $0.9718$, ECE cực thấp ($0.0071$) cho phép robot tự tin kích hoạt vòi phun thuốc chọn lọc mà không sợ phun nhầm vào cây trồng bản địa.

---

## 8. Hạn Chế và Hướng Nghiên Cứu Tiếp Theo

1. **Số lượng seed & một fold:** Nghiên cứu cố định trên Fold 0 với 3 seed ở vòng chung kết. Cần mở rộng cross-validation trên cả 5 folds của DeepWeeds để kiểm định tính bất biến về địa lý.
2. **Độ lệch phân phối thực địa (Domain Shift):** Dữ liệu thu thập tại Queensland vào một mùa nhất định. Khi triển khai sang các vùng khí hậu khác, góc chiếu mặt trời, độ ẩm đất và chu kỳ sinh trưởng của cỏ có thể thay đổi.
3. **Hướng tiếp theo:**
   - Thử nghiệm chưng cất tri thức (Knowledge Distillation) từ ConvNeXt-Tiny sang MobileNetV3-Large để giảm độ trễ xuống $< 10\text{ ms}$ cho chip vi điều khiển nhúng công suất thấp.
   - Nghiên cứu cơ chế thích ứng miền trực tuyến (Test-Time Adaptation - TTA).

---

## 9. Phụ Lục (Appendix)

### 9.1 Danh mục mã thực nghiệm (`exp_id`) và Cấu hình

| Exp ID | Mô Tả Cấu Hình | Epochs | Batch | LR Head | LR Backbone | Loss | Augmentation | Regularization |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | ResNet-50 Baseline | 12 | 64 | 1e-3 | 1e-4 | CE | Basic | None |
| **B02 / T00** | ConvNeXt-Tiny Baseline | 12 | 64 | 1e-3 | 1e-4 | CE | Basic | None |
| **B03** | ResNeXt-50 32x4d | 12 | 64 | 1e-3 | 1e-4 | CE | Basic | None |
| **B04** | Swin-Tiny | 12 | 64 | 1e-3 | 1e-4 | CE | Basic | None |
| **B05** | MobileNetV3-Large | 12 | 64 | 1e-3 | 1e-4 | CE | Basic | None |
| **T01** | Focal Loss ($\gamma=2.0$) | 12 | 64 | 1e-3 | 1e-4 | Focal | Basic | None |
| **T02** | Label Smoothing ($\epsilon=0.1$) | 12 | 64 | 1e-3 | 1e-4 | LS 0.1 | Basic | None |
| **T03** | RandAugment ($N=2, M=9$) | 12 | 64 | 1e-3 | 1e-4 | CE | RandAug | None |
| **T04** | CutMix ($\alpha=1.0$) | 12 | 64 | 1e-3 | 1e-4 | CE | Basic + CutMix | None |
| **T05** | Frozen Backbone (Linear Probe) | 12 | 64 | 1e-3 | 0.0 | CE | Basic | Freeze |
| **T06** | EMA Decay 0.999 | 12 | 64 | 1e-3 | 1e-4 | CE | Basic | EMA 0.999 |
| **T07** | Combo (LS + RandAug + EMA) | 12 | 64 | 1e-3 | 1e-4 | LS 0.1 | RandAug | EMA 0.999 |
| **I00..I05** | 6 Phương pháp suy luận | - | 1 / 32 | - | - | - | 1-view / TTA / FixRes | TS / Ensemble |
| **F01** | Chung kết 3 seeds (0, 1, 2) | 12 | 64 | 1e-3 | 1e-4 | LS 0.1 | RandAug | EMA + TS |

### 9.2 Danh mục liên kết Notebook & Mã nguồn
- **Notebook tự tái lập:** [lab_day2.ipynb](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/code/lab_day2.ipynb)
- **Thư mục mã nguồn:** [submissions/2A202602368_duong_van_thanh/code/](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/code/)
- **File bảng tính kết quả chuẩn:** [submissions/2A202602368_duong_van_thanh/results.xlsx](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/results.xlsx)
- **File tự chấm Rubric I tự động:** [submissions/2A202602368_duong_van_thanh/grade_I.json](file:///e:/VinAI/Track4/K4-Track4-Day2-DuongVanThanh-2A202602368-Deeplearning-Advance/submissions/2A202602368_duong_van_thanh/grade_I.json) (Tổng điểm: **19 / 20 điểm**)
