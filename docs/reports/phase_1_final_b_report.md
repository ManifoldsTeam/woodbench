# Báo cáo lần chạy cuối — Final-B

- Bài toán: phân loại 17 loài cây.
- Dữ liệu: train 3.365 ảnh, validation 903 ảnh, test 532 ảnh.
- Metric: Accuracy và macro-averaged F1/Precision/Recall.
- `90rotation` là cấu hình đạt kết quả cao nhất hoặc đồng hạng cao nhất trong phần lớn model.

## 1. ResNet-50

Nguồn: [01_train_resnet50.ipynb](../../notebooks/phase%201/final-B/01_train_resnet50.ipynb)

| Experiment | Seed | Val Acc | Test Acc | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| main | 42 | 90.59% | 94.55% | 94.69% | 95.16% | 94.65% |
| main | 123 | 90.37% | 92.29% | 92.34% | 93.40% | 92.43% |
| main | 456 | 90.59% | 93.61% | 93.75% | 94.55% | 93.74% |
| **taskB_90rotation** | **42** | **92.58%** | **95.68%** | **95.73%** | **96.23%** | **95.64%** |
| label_smoothing_0.0 | 42 | 90.59% | 93.80% | 93.95% | 94.55% | 93.92% |
| label_smoothing_0.2 | 42 | 90.70% | 94.17% | 94.29% | 94.88% | 94.22% |

## 2. EfficientNet-B3

Nguồn: [02_train_efficientnet_b3.ipynb](../../notebooks/phase%201/final-B/02_train_efficientnet_b3.ipynb)

| Experiment | Seed | Val Acc | Test Acc | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| main | 42 | 95.79% | 97.37% | 97.42% | 97.51% | 97.44% |
| main | 123 | 96.57% | 97.37% | 97.42% | 97.63% | 97.40% |
| main | 456 | 96.23% | 97.37% | 97.42% | 97.63% | 97.40% |
| **taskB_90rotation** | **42** | **97.67%** | **98.87%** | **98.93%** | **99.03%** | **98.96%** |
| label_smoothing_0.0 | 42 | 96.68% | 97.18% | 97.18% | 97.30% | 97.20% |
| label_smoothing_0.2 | 42 | 95.68% | 96.05% | 96.06% | 96.19% | 96.15% |

## 3. MobileNetV3-Large

Nguồn: [03_train_mobilenetv3_large.ipynb](../../notebooks/phase%201/final-B/03_train_mobilenetv3_large.ipynb)

| Experiment | Seed | Val Acc | Test Acc | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| main | 42 | 95.35% | 98.68% | 98.73% | 98.82% | 98.69% |
| main | 123 | 95.57% | 98.50% | 98.55% | 98.65% | 98.53% |
| main | 456 | 95.57% | 98.68% | 98.75% | 98.83% | 98.75% |
| taskB_90rotation | 42 | 97.45% | 98.50% | 98.51% | 98.53% | 98.56% |
| label_smoothing_0.0 | 42 | 95.35% | 97.93% | 97.99% | 98.13% | 97.95% |
| label_smoothing_0.2 | 42 | 95.13% | 98.50% | 98.55% | 98.67% | 98.53% |

## 4. ConvNeXt-Tiny

Nguồn: [04_train_convnext_tiny.ipynb](../../notebooks/phase%201/final-B/04_train_convnext_tiny.ipynb)

| Experiment | Seed | Val Acc | Test Acc | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| main | 42 | 99.89% | 99.62% | 99.65% | 99.65% | 99.67% |
| main | 123 | 99.89% | 99.62% | 99.66% | 99.65% | 99.67% |
| main | 456 | 99.89% | 91.92% | 91.73% | 94.57% | 92.43% |
| **taskB_90rotation** | **42** | **99.67%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| label_smoothing_0.0 | 42 | 99.67% | 99.62% | 99.66% | 99.65% | 99.67% |
| label_smoothing_0.2 | 42 | 99.45% | 99.06% | 99.10% | 99.11% | 99.12% |

## 5. Swin-Tiny

Nguồn: [05_train_swin_tiny.ipynb](../../notebooks/phase%201/final-B/05_train_swin_tiny.ipynb)

| Experiment | Seed | Val Acc | Test Acc | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| main | 42 | 99.67% | 97.18% | 97.39% | 97.99% | 97.31% |
| main | 123 | 99.78% | 100.00% | 100.00% | 100.00% | 100.00% |
| main | 456 | 99.67% | 99.25% | 99.28% | 99.34% | 99.28% |
| **taskB_90rotation** | **42** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| label_smoothing_0.0 | 42 | 99.56% | 99.62% | 99.65% | 99.65% | 99.67% |
| label_smoothing_0.2 | 42 | 100.00% | 99.44% | 99.49% | 99.49% | 99.51% |

## Tóm tắt

- Kết quả test cao nhất: **100.00%**, đạt bởi ConvNeXt-Tiny (`taskB_90rotation`) và Swin-Tiny (`main`, seed 123; `taskB_90rotation`, seed 42).
- Trong ba seed của cấu hình `main`, MobileNetV3-Large ổn định nhất về Test Accuracy (98.50–98.68%); ConvNeXt-Tiny có một lần giảm mạnh ở seed 456 (91.92%).
