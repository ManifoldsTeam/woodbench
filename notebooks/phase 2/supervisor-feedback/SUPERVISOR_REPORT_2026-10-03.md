# Báo cáo bổ sung theo góp ý của thầy 03/10/2026

## 1. Grouped 5-fold CV, so sánh 4 model với paper

| Model | Accuracy CV (%) | Macro-F1 CV (%) | Accuracy trong paper (%) | Macro-F1 trong paper (%) | Run nguồn |
|---|---:|---:|---:|---:|---|
| Swin-Tiny | 99,73 ± 0,20 | 99,72 ± 0,24 | 99,12 ± 0,58 | 99,20 ± 0,53 | [IDs](../../../data/experiments/Task-B-origin-cv/reports/paper-comparison-20261003T161037Z-972cf8ab/comparison.csv "source_run_ids=swin_tiny-s42-f1-6ec4bf52b1724dd5b5bf92374fe719e8, swin_tiny-s42-f2-75a48a2a3da64c1d8456c83de249c26c, swin_tiny-s42-f3-c0ffdeace9f64b33aa94024288b79dbd, swin_tiny-s42-f4-51aba68e99c848a0a1848779b8a431a2, swin_tiny-s42-f5-53f480dd0cca4936a59aec6d9e349678; baseline_source_run_ids=taskB17__main__seed_123__swin_tiny, taskB17__main__seed_42__swin_tiny, taskB17__main__seed_456__swin_tiny") |
| ConvNeXt-Tiny | 99,33 ± 0,43 | 99,38 ± 0,35 | 99,56 ± 0,35 | 99,60 ± 0,33 | [IDs](../../../data/experiments/Task-B-origin-cv/reports/paper-comparison-20261003T161037Z-972cf8ab/comparison.csv "source_run_ids=convnext_tiny-s42-f1-69e7bdc8fc504ea08671fd9b0831a1e6, convnext_tiny-s42-f2-3ef5f80da2474768acc9a94a6fa40b50, convnext_tiny-s42-f3-9a6b46dfc9b449a8aeeb1284a68037da, convnext_tiny-s42-f4-af98025426d142839970bb494f06871e, convnext_tiny-s42-f5-93ded80e8b414d6287e74784165b098e; baseline_source_run_ids=taskB17__main__seed_123__convnext_tiny, taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_456__convnext_tiny") |
| EfficientNet-B3 | 95,96 ± 1,12 | 95,72 ± 1,10 | 97,81 ± 0,32 | 97,84 ± 0,32 | [IDs](../../../data/experiments/Task-B-origin-cv/reports/paper-comparison-20261003T161037Z-972cf8ab/comparison.csv "source_run_ids=efficientnet_b3-s42-f1-36d5d27057b74d82a4980c789a7d397b, efficientnet_b3-s42-f2-1d6e5b932dd84bff82c7527af467cc68, efficientnet_b3-s42-f3-194ae84bdddd4808abc9d92038ba3fdd, efficientnet_b3-s42-f4-eb79eaf935a8444a97d8b6fc549bbccd, efficientnet_b3-s42-f5-d6fe6beeca6842cb82e1207f88ccefe2; baseline_source_run_ids=taskB17__main__seed_123__efficientnet_b3, taskB17__main__seed_42__efficientnet_b3, taskB17__main__seed_456__efficientnet_b3") |
| MobileNetV3-Large | 94,51 ± 1,39 | 94,26 ± 1,56 | 98,37 ± 0,32 | 98,45 ± 0,35 | [IDs](../../../data/experiments/Task-B-origin-cv/reports/paper-comparison-20261003T161037Z-972cf8ab/comparison.csv "source_run_ids=mobilenetv3_large-s42-f1-9921147063f4472d9749eb40de6c2f37, mobilenetv3_large-s42-f2-9eaada9dc1024a37a06f257d9d814025, mobilenetv3_large-s42-f3-3c356412002345368e138bf5ec2be2a4, mobilenetv3_large-s42-f4-96f9df8248d448a39bf78fd46ccf7dfe, mobilenetv3_large-s42-f5-46741ac0c3534186941fc7db2a2ee42e; baseline_source_run_ids=taskB17__main__seed_123__mobilenetv3_large, taskB17__main__seed_42__mobilenetv3_large, taskB17__main__seed_456__mobilenetv3_large") |

CV: trung bình ± độ lệch chuẩn mẫu của 5 fold, checkpoint cuối; paper: giữ nguyên số liệu của [bảng Task B](../../../../WoodID/results/tab_taskb.tex), gộp 3 seed và dùng best checkpoint. Hai cột khác protocol, không phải so sánh trên cùng một tập test. [Registry 32 run](../../../data/experiments/Task-B-origin-cv/reports/paper-comparison-20261003T161037Z-972cf8ab/origin-cv-paper-20261003T161037Z-registry.csv), [metadata](../../../data/experiments/Task-B-origin-cv/reports/paper-comparison-20261003T161037Z-972cf8ab/metadata.json).

## 2. Kiểm tra ảnh hưởng của viền và màu

Đã chạy inference trên **532 ảnh test/17 loài**, dùng **best checkpoint Phase 1 trong paper, seed 42**, không retrain. Hai model dùng chung pipeline `Resize(224×224) → phép kiểm tra → normalize`; kết quả là **accuracy / macro-F1 (%)**. Mức giảm tính theo baseline riêng của từng model, bằng **điểm phần trăm**.

| Phép kiểm tra | ConvNeXt Acc / F1 (%) | Giảm Acc | MobileNet Acc / F1 (%) | Giảm Acc | Run nguồn |
|---|---:|---:|---:|---:|---|
| Ảnh gốc | 99,81 / 99,83 | 0,00 | 98,68 / 98,79 | 0,00 | [IDs](task_02_robustness_tests/runs/inference-20261003T163224Z-1fde2140/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |
| Center-crop 70% | 96,62 / 96,69 | 3,20 | 91,54 / 91,71 | 7,14 | [IDs](task_02_robustness_tests/runs/inference-20261003T163224Z-1fde2140/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |
| Che viền 15% | 99,44 / 99,46 | 0,38 | 88,16 / 87,88 | 10,53 | [IDs](task_02_robustness_tests/runs/inference-20261003T163224Z-1fde2140/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |
| Grayscale | 73,68 / 70,33 | 26,13 | 47,18 / 44,25 | 51,50 | [IDs](task_02_robustness_tests/runs/inference-20261003T163224Z-1fde2140/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |

**Nhận xét:** Che viền chỉ làm ConvNeXt giảm **0,38 điểm**, nhưng MobileNet giảm **10,53 điểm**. Grayscale làm cả hai giảm mạnh, lần lượt **26,13 / 51,50 điểm**; MobileNet nhạy hơn với các can thiệp đã thử. Kết quả cho thấy độ nhạy với màu/viền, chưa đủ để khẳng định model chỉ dựa vào các yếu tố này.

Nguồn: [kết quả mới](task_02_robustness_tests/runs/inference-20261003T163224Z-1fde2140/results.csv), [metadata](task_02_robustness_tests/runs/inference-20261003T163224Z-1fde2140/metadata.json).

## 3. Dark artifact ở 5 loài

Đã quét toàn bộ **4.800 ảnh/17 loài**: **0 ảnh** có hơn 1% pixel với cường độ dưới 0,01 hoặc hơn 5% pixel dưới 0,05, tính trên cường độ RGB trung bình chuẩn hóa về [0,1]. Không phát hiện ảnh vượt hai ngưỡng này. [Nguồn](task_03_dark_artifact_check/image_brightness_analysis.csv).

## 4. Degradation: 16 px và blur

Cùng checkpoint, 532 ảnh test và pipeline như Task 2; đã kiểm tra baseline và dự đoán ảnh gốc khớp hoàn toàn giữa hai notebook. Kết quả là **accuracy / macro-F1 (%)**; mức giảm là **điểm phần trăm**.

| Phép kiểm tra | ConvNeXt Acc / F1 (%) | Giảm Acc | MobileNet Acc / F1 (%) | Giảm Acc | Run nguồn |
|---|---:|---:|---:|---:|---|
| Ảnh gốc | 99,81 / 99,83 | 0,00 | 98,68 / 98,79 | 0,00 | [IDs](task_04_degradation_rerun/runs/inference-20261003T163731Z-63675241/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |
| 16×16 → 224×224 | 27,07 / 15,10 | 72,74 | 4,70 / 1,27 | 93,98 | [IDs](task_04_degradation_rerun/runs/inference-20261003T163731Z-63675241/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |
| Blur σ=5 (kernel 31) | 37,59 / 26,88 | 62,22 | 5,08 / 1,41 | 93,61 | [IDs](task_04_degradation_rerun/runs/inference-20261003T163731Z-63675241/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |
| Blur σ=10 (kernel 61) | 23,68 / 12,00 | 76,13 | 3,20 / 0,56 | 95,49 | [IDs](task_04_degradation_rerun/runs/inference-20261003T163731Z-63675241/results.csv "source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large; baseline_source_run_ids=taskB17__main__seed_42__convnext_tiny, taskB17__main__seed_42__mobilenetv3_large") |

**Nhận xét:** Khi hạ độ phân giải hoặc làm mờ, accuracy ConvNeXt còn **23,68–37,59%**, MobileNet còn **3,20–5,08%**. Cả hai giảm mạnh khi mất chi tiết, MobileNet kém bền hơn trong các phép thử này. Chưa thể dùng riêng kết quả này để xác nhận model học đúng đặc điểm giải phẫu.

Nguồn: [kết quả mới](task_04_degradation_rerun/runs/inference-20261003T163731Z-63675241/results.csv), [metadata](task_04_degradation_rerun/runs/inference-20261003T163731Z-63675241/metadata.json).

## 5. ResNet-50 với backbone LR = 1e-4, seed 42

Đã có run tuning: **test accuracy 96,05% (511/532 ảnh)**, best validation **96,12%**. Notebook resume sau epoch 15, chạy tiếp đến epoch 50; history lưu 35 epoch và test dùng model cuối. [Nguồn](../../../data/experiments/Task-B-new/lr_tuning/test_metrics.json), [notebook](task_05_resnet_lr_tuning/resnet_lr_tuning.ipynb).

Mốc ResNet main seed 42 trong registry là **93,80%**, chênh lệch mô tả **+2,26 điểm**, khác số +2,38 trong notebook dùng baseline gán tay. Cần thống nhất best/final và kiểm tra resume trước khi quy toàn bộ cải thiện cho LR. Run tuning này **khác** ResNet CV còn thiếu ở mục 1.

## 6. Giải thích 7 lỗi P. krempfii → P. dalatensis

**7 lỗi trong paper là 7 lần dự đoán sai khi gộp ConvNeXt main qua 3 seed, tương ứng 5 ảnh khác nhau.** Seed 42 sai 1 lần, seed 123 sai 1 lần, seed 456 sai 5 lần.

Ví dụ, ảnh **`(130).png` bị nhận nhầm ở cả 3 seed**, nên được đếm thành **3 lượt lỗi**. Bốn ảnh `(129)`, `(131)`, `(133)`, `(134)` mỗi ảnh sai một lần ở seed 456. Tổng: **3 + 4 = 7 lượt lỗi trên 5 ảnh**.

Cả 5 ảnh đều có origin **`3384. Pinus krempfii.4`**: chúng thuộc cùng nhóm nguồn theo quy tắc tách origin từ tên file. Như vậy lỗi tập trung ở một origin; cần xem trực tiếp các ảnh và xác nhận nhãn mẫu để biết nguyên nhân, chưa thể kết luận do màu, viền hay đặc điểm giống nhau giữa hai loài.

Con số **125** trong [file lỗi tổng](task_06_error_analysis/pk_all_errors.csv) là lượt lỗi của **30 run gồm nhiều model, seed và ablation**, trên **9 ảnh**. Khi báo cáo 7 lỗi của ConvNeXt trong paper, chỉ xét nhóm **ConvNeXt main / 3 seed** ở trên.


Truy vết Task 2/4: [registry tổng hợp 2 training run](reports/inference-summary-20261003T164818Z-8d57aa9f/supervisor-inference-20261003T164818Z-registry.csv), [16 kết quả inference](reports/inference-summary-20261003T164818Z-8d57aa9f/results.csv), [metadata tổng hợp](reports/inference-summary-20261003T164818Z-8d57aa9f/metadata.json).
