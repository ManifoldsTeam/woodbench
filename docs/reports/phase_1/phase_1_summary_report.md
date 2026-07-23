# Báo cáo tổng hợp Phase 1 — WoodID-S2-2026

**Cập nhật:** 23-07-2026, sau khi phục hồi Task A và tải đủ checkpoint Final-B 17 lớp  
**Nguồn chuẩn:** inference lại trực tiếp từ 45 checkpoint `best.pth`  
**Notebook:** `notebooks/phase 1/07_phase1_results_summary.ipynb`

## 1. Phạm vi chính thức

- **Task A:** phân loại 4 họ, dữ liệu train đã cân bằng, 5 backbone × 3 seed.
- **Final-B:** phân loại 17 loài, sau khi loại `Calocedrus sp`, `Cupressus funebris (Cupressus pendula)` và `Pinus sp` vì thiếu dữ liệu để tạo split origin-safe đầy đủ.
- **Ablation Final-B:** rotation 90°, label smoothing 0.0 và 0.2; cấu hình main dùng label smoothing 0.1.
- **Efficiency:** FLOPs, params, model size và inference batch=1 cho 5 backbone trên checkpoint Final-B 17 lớp.

Việc giảm từ 20 xuống 17 lớp là quyết định dữ liệu cuối cùng của dự án, không còn được xem là checkpoint thiếu. Khi viết paper cần nêu rõ tiêu chí loại ba lớp và không trình bày kết quả 17 lớp như benchmark 20 lớp nguyên thủy trong `WoodID_plan.md`.

## 2. Phục hồi và kiểm tra dữ liệu

### Task A

`data/split` được tái tạo từ `data/raw` bằng notebook Phase 0, seed 42 và split theo từng species. `data/task-A` sau đó được phục hồi theo notebook augmentation:

| Split | Cupressaceae | Pinaceae | Podocarpaceae | Taxaceae | Tổng |
|---|---:|---:|---:|---:|---:|
| Train | 2.398 | 1.666 | 1.600 | 1.600 | **7.264** |
| Validation | 510 | 355 | 46 | 44 | **955** |
| Test | 525 | 364 | 47 | 45 | **981** |

- 6.423 ảnh gốc được giữ nội dung nguyên vẹn bằng hard link để tránh nhân đôi khoảng 21 GB dữ liệu.
- 2.777 ảnh augmentation được sinh lại bằng 7 phương pháp trong notebook Phase 0.
- Manifest có đúng 2.777 dòng, không trùng đường dẫn; toàn bộ ảnh augmentation qua kiểm tra định dạng PIL.
- Dataloader xác nhận đúng 4 class và 7.264 mẫu train.

### Final-B

| Split | Số lớp | Số ảnh |
|---|---:|---:|
| Train | 17 | 3.365 |
| Validation | 17 | 903 |
| Test | 17 | 532 |

Các thư mục của ba lớp bị loại được chuyển sang `data/task-B-group-split-excluded/`, nên vẫn có thể phục hồi; chúng không còn tham gia inference Final-B.

## 3. Phương pháp chạy lại

- Load strict từng checkpoint `best.pth` với head đúng 4 hoặc 17 lớp.
- Test transform: resize/crop và normalize theo pipeline evaluation hiện tại; không augmentation test.
- Task A: 15 checkpoint main, 981 ảnh test/run.
- Final-B: 15 checkpoint main và 15 checkpoint ablation, 532 ảnh test/run.
- Tính lại Accuracy, Macro-Precision, Macro-Recall, Macro-F1 và per-class F1 từ prediction.
- Mean ± std dùng ba seed 42/123/456, population standard deviation (`ddof=0`).
- Thiết bị đánh giá: NVIDIA GeForce RTX 3060, PyTorch 2.12.0+cu130.

## 4. Kết quả Task A — 4 họ

| Model | Test Accuracy, % | Macro-F1, % | Kết luận |
|---|---:|---:|---|
| ResNet-50 | 98.573 ± 0.220 | 97.442 ± 0.392 | Baseline ổn định |
| EfficientNet-B3 | 98.233 ± 0.509 | 96.608 ± 0.834 | F1 biến động cao nhất Task A |
| MobileNetV3-Large | 99.626 ± 0.048 | 99.383 ± 0.328 | Nhẹ và rất ổn định |
| ConvNeXt-Tiny | 99.898 ± 0.144 | 99.784 ± 0.306 | Accuracy gần tuyệt đối |
| Swin-Tiny | **100.000 ± 0.000** | **100.000 ± 0.000** | Tốt nhất trên ba checkpoint best |

Kết quả trực tiếp từ checkpoint cho thấy Swin-Tiny đứng đầu Task A. MobileNetV3-Large tiếp tục là lựa chọn accuracy–efficiency tốt nhất khi cần model nhẹ.

## 5. Kết quả Final-B main — 17 loài

| Model | Test Accuracy, % | Macro-F1, % | Kết luận |
|---|---:|---:|---|
| ResNet-50 | 93.672 ± 1.022 | 93.844 ± 0.998 | Thấp nhất và biến động lớn nhất |
| EfficientNet-B3 | 97.118 ± 0.177 | 97.138 ± 0.221 | Ổn định |
| MobileNetV3-Large | 98.371 ± 0.319 | 98.407 ± 0.339 | Trade-off tốt cho deployment |
| ConvNeXt-Tiny | **99.499 ± 0.089** | **99.524 ± 0.059** | Tốt nhất và ổn định nhất |
| Swin-Tiny | 99.436 ± 0.307 | 99.481 ± 0.282 | Đứng thứ hai, rất gần ConvNeXt |

ConvNeXt-Tiny là model chính xác nhất cho Final-B theo ba checkpoint best. MobileNetV3-Large thấp hơn khoảng 1,13 điểm phần trăm nhưng nhỏ và nhanh hơn đáng kể.

## 6. Ablation Final-B — seed 42

### Rotation 90°

| Model | Accuracy, % | Macro-F1, % | Δ Accuracy so với main |
|---|---:|---:|---:|
| ResNet-50 | 95.489 | 95.525 | +0.752 |
| EfficientNet-B3 | 98.684 | 98.721 | **+1.692** |
| MobileNetV3-Large | 98.684 | 98.691 | +0.188 |
| ConvNeXt-Tiny | **100.000** | **100.000** | +0.376 |
| Swin-Tiny | 99.812 | 99.794 | +0.752 |

Rotation 90° cải thiện cả 5 backbone trong lần đánh giá checkpoint này, ủng hộ hypothesis ảnh mặt cắt ngang đẳng hướng. Mức tăng lớn nhất thuộc EfficientNet-B3.

### Label smoothing

- `0.1` (main) tốt hơn `0.0` và `0.2` đối với ResNet-50 và EfficientNet-B3.
- ConvNeXt-Tiny tốt nhất ở `0.0` với 99,812%, cao hơn main 0,188 điểm %.
- MobileNetV3-Large tốt nhất ở `0.2` với 98,684%, cao hơn main 0,188 điểm %.
- Swin-Tiny tốt nhất ở `0.2` với 99,624%, cao hơn main 0,564 điểm %.
- Không có một mức label smoothing tối ưu chung cho cả 5 kiến trúc.

Các ablation chỉ có một seed, do đó chưa đủ để kết luận về độ ổn định thống kê.

## 7. Efficiency với checkpoint Final-B 17 lớp

Thiết lập: input `1×3×224×224`, batch=1, 20 warm-up, 100 timed runs, RTX 3060. THOP trả về MACs; GFLOPs dùng quy ước `1 MAC = 2 FLOPs`.

| Model | Params, M | State dict, MB | GFLOPs | Mean latency, ms/ảnh | P95, ms |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 23.54 | 90.11 | 8.263 | 3.094 | 3.758 |
| EfficientNet-B3 | 10.72 | 41.42 | 1.877 | 3.474 | 3.909 |
| MobileNetV3-Large | **4.22** | **16.30** | **0.431** | **1.882** | **2.720** |
| ConvNeXt-Tiny | 27.83 | 106.24 | 8.910 | 3.762 | 4.557 |
| Swin-Tiny | 27.53 | 105.09 | 8.742 | 4.696 | 5.534 |

- **Accuracy-first Final-B:** ConvNeXt-Tiny.
- **Accuracy-first Task A:** Swin-Tiny.
- **Edge/deployment:** MobileNetV3-Large.

Latency chỉ so sánh được khi giữ nguyên thiết bị và protocol. State-dict size được dùng thay cho full training checkpoint vì checkpoint còn chứa optimizer/scheduler.

## 8. Audit report lịch sử

Kết quả inference lại từ `best.pth` không hoàn toàn giống JSON/report được lưu trong lần training:

- ConvNeXt-Tiny Final-B main seed 456: JSON cũ 91,917%, checkpoint best đánh giá lại **99,436%** (`+7,519` điểm %).
- Swin-Tiny Final-B main seed 42: JSON cũ 97,180%, checkpoint best đánh giá lại **99,060%** (`+1,880` điểm %).
- Các run còn lại chủ yếu chênh từ 0 đến khoảng 1,13 điểm %.

Khả năng cao JSON lịch sử được ghi từ trạng thái model khác với file `best.pth` hoặc là artifact stale. Vì mục tiêu deployment sử dụng checkpoint, báo cáo này lấy kết quả inference trực tiếp từ checkpoint làm nguồn chuẩn. CSV raw vẫn giữ cả đường dẫn checkpoint để truy vết.

## 9. Đối chiếu WoodID plan

| Hạng mục Phase 1 | Trạng thái | Bằng chứng |
|---|---|---|
| 5 model Task A, 3 seed | Hoàn thành | 15 checkpoint được đánh giá lại |
| 5 model Final-B 17 lớp, 3 seed | Hoàn thành theo scope cuối | 15 checkpoint main được đánh giá lại |
| Rotation 90° | Hoàn thành | 5 checkpoint seed 42 |
| Label smoothing 0.0/0.1/0.2 | Hoàn thành | 15 checkpoint ablation/main seed 42 |
| FLOPs, params, model size | Hoàn thành | Notebook efficiency 17 lớp |
| Inference time batch=1, 100 runs | Hoàn thành | Metadata RTX 3060 |
| Mean ± std main experiments | Hoàn thành | Task A và Final-B, 3 seed |

Điểm lệch duy nhất so với tài liệu gốc là số class Task B: kế hoạch ban đầu ghi 20 loài, scope cuối là 17 do ba lớp không đủ dữ liệu origin-safe. Quyết định này phải được phản ánh nhất quán trong abstract, methods, tables và tên dataset của paper.

## 10. Artifacts chuẩn

- Notebook tổng hợp đã chạy: `notebooks/phase 1/07_phase1_results_summary.ipynb`.
- Raw 45 run: `data/experiments/phase_1/summary/phase1_all_runs_recomputed.csv`.
- Main mean ± std: `data/experiments/phase_1/summary/phase1_main_summary_recomputed.csv`.
- Ablation: `data/experiments/phase_1/summary/phase1_task_b_ablation_recomputed.csv`.
- Metadata: `data/experiments/phase_1/summary/summary_metadata.json`.
- Efficiency: `data/experiments/phase_1/efficiency/`.
- Task A recovery manifest: `data/task-A/augmentation_manifest.csv`.
