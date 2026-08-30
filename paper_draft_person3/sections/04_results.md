# 4. Experimental Results

> **Nguồn số liệu.** Mục 4.1 đến 4.4 lấy từ `docs/reports/phase_1/phase_1_summary_report.md`, tính lại bằng inference trực tiếp từ 45 checkpoint `best.pth` trên NVIDIA GeForce RTX 3060 (PyTorch 2.12.0+cu130). Mục 4.5 lấy từ bộ dự đoán từng ảnh của **lượt huấn luyện lại** trong `docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z/`, vì đây là nguồn dự đoán cấp ảnh duy nhất hiện có. Hai nguồn này **không trùng khớp** về accuracy, khác biệt được nêu ở cuối mục.

Task A được đánh giá trên 981 ảnh test của 4 họ, Task B trên 532 ảnh test của 17 loài. Giá trị trung bình và độ lệch chuẩn tính trên ba seed 42, 123 và 456 với độ lệch chuẩn tổng thể.

## 4.1. Task A: Family-Level Classification Results

Bảng LaTeX: `paper_draft/tables/table1_task_a.tex`

| Kiến trúc | Top-1 Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| ResNet-50 | 98,573 ± 0,220 | 97,442 ± 0,392 |
| EfficientNet-B3 | 98,233 ± 0,509 | 96,608 ± 0,834 |
| MobileNetV3-Large | 99,626 ± 0,048 | 99,383 ± 0,328 |
| ConvNeXt-Tiny | 99,898 ± 0,144 | 99,784 ± 0,306 |
| **Swin-Tiny** | **100,000 ± 0,000** | **100,000 ± 0,000** |

Cả năm kiến trúc đạt accuracy trên 98% và Macro-F1 trên 96%. Swin-Tiny đạt accuracy và Macro-F1 bằng 100,000% trên cả ba seed với độ lệch chuẩn bằng 0. ConvNeXt-Tiny đứng thứ hai với 99,898%, sai một ảnh ở một seed. MobileNetV3-Large đạt 99,626% và có độ lệch chuẩn accuracy nhỏ nhất trong nhóm không đạt kết quả tuyệt đối.

Khoảng cách giữa các kiến trúc thể hiện rõ hơn ở Macro-F1 so với ở accuracy. EfficientNet-B3 có Macro-F1 thấp nhất (96,608%) và độ lệch chuẩn Macro-F1 lớn nhất (0,834). Phân bố tập test không đồng đều giữa bốn họ, gồm Cupressaceae 525 ảnh, Pinaceae 364 ảnh, Podocarpaceae 47 ảnh và Taxaceae 45 ảnh.

Tập huấn luyện Task A đã cân bằng bằng tăng cường ngoại tuyến và sử dụng phân chia phân tầng theo loài, không phải phân chia theo mẫu gốc (Mục 3.2).

## 4.2. Task B: Species-Level Benchmark Results

Bảng LaTeX: `paper_draft/tables/table2_task_b.tex`

| Kiến trúc | Top-1 Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| ResNet-50 | 93,672 ± 1,022 | 93,844 ± 0,998 |
| EfficientNet-B3 | 97,118 ± 0,177 | 97,138 ± 0,221 |
| MobileNetV3-Large | 98,371 ± 0,319 | 98,407 ± 0,339 |
| **ConvNeXt-Tiny** | **99,499 ± 0,089** | **99,524 ± 0,059** |
| Swin-Tiny | 99,436 ± 0,307 | 99,481 ± 0,282 |

Task B phân tách các kiến trúc rõ hơn Task A. Biên độ accuracy giữa kiến trúc cao nhất và thấp nhất là 5,827 điểm phần trăm ở Task B, so với 1,427 điểm phần trăm ở Task A.

**Nhóm dẫn đầu.** ConvNeXt-Tiny và Swin-Tiny đạt accuracy lần lượt 99,499% và 99,436%. Khoảng cách giữa hai kiến trúc này là 0,063 điểm phần trăm, nhỏ hơn độ lệch chuẩn của Swin-Tiny (0,307). Kiểm định t ghép cặp trên ba seed cho $t(2)=0{,}229$ và $p=0{,}840$, không đủ bằng chứng bác bỏ giả thuyết không về khác biệt giữa hai kiến trúc. ConvNeXt-Tiny có độ lệch chuẩn nhỏ nhất ở cả hai chỉ số (0,089 và 0,059).

**So với ResNet-50 và MobileNetV3-Large.** ConvNeXt-Tiny cao hơn ResNet-50 5,827 điểm phần trăm accuracy ($t(2)=8{,}596$, $p=0{,}013$) và cao hơn MobileNetV3-Large 1,128 điểm phần trăm ($t(2)=5{,}196$, $p=0{,}035$). Swin-Tiny cao hơn ResNet-50 5,764 điểm phần trăm ($p=0{,}025$), nhưng khác biệt so với MobileNetV3-Large không đạt ngưỡng ý nghĩa ($t(2)=3{,}900$, $p=0{,}060$). ResNet-50 có độ lệch chuẩn accuracy 1,022, lớn hơn ConvNeXt-Tiny khoảng mười một lần.

MobileNetV3-Large đạt 98,371%, cao hơn EfficientNet-B3 1,253 điểm phần trăm ($t(2)=7{,}559$, $p=0{,}017$), mặc dù có ít tham số hơn.

Bảy trong mười phép so sánh cặp đạt ý nghĩa ở mức 0,05. Kiểm định thực hiện với $n=3$ nên độ mạnh thống kê thấp, và chưa hiệu chỉnh cho so sánh bội.

## 4.3. Computational Efficiency and Hardware Trade-offs

Bảng LaTeX: `paper_draft/tables/table3_efficiency.tex`

| Kiến trúc | Params (M) | GFLOPs | Size (MB) | Latency (ms/ảnh) | P95 (ms) | Task B Acc (%) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 23,54 | 8,263 | 90,11 | 3,094 | 3,758 | 93,672 |
| EfficientNet-B3 | 10,72 | 1,877 | 41,42 | 3,474 | 3,909 | 97,118 |
| **MobileNetV3-Large** | **4,22** | **0,431** | **16,30** | **1,882** | **2,720** | 98,371 |
| ConvNeXt-Tiny | 27,83 | 8,910 | 106,24 | 3,762 | 4,557 | **99,499** |
| Swin-Tiny | 27,53 | 8,742 | 105,09 | 4,696 | 5,534 | 99,436 |

MobileNetV3-Large có giá trị thấp nhất ở cả bốn chỉ số chi phí. So với ConvNeXt-Tiny, MobileNetV3-Large dùng ít hơn 6,59 lần số tham số, 20,67 lần số phép tính, 6,52 lần dung lượng, và có độ trễ thấp hơn 2,00 lần, đổi lại accuracy thấp hơn 1,128 điểm phần trăm.

Thứ tự xếp hạng theo GFLOPs không trùng với thứ tự xếp hạng theo độ trễ đo được. Swin-Tiny có độ trễ cao nhất (4,696 ms/ảnh) mặc dù GFLOPs thấp hơn ConvNeXt-Tiny. EfficientNet-B3 có GFLOPs thấp hơn ResNet-50 4,40 lần nhưng độ trễ cao hơn 0,380 ms/ảnh.

ConvNeXt-Tiny, kiến trúc đạt accuracy Task B cao nhất, có dung lượng và GFLOPs cao nhất trong năm kiến trúc. Độ trễ chỉ so sánh được khi giữ nguyên thiết bị và giao thức đo.

## 4.4. Ablation Studies

Bảng LaTeX: `paper_draft/tables/table4_ablation.tex`

> **Nguồn.** Cột xoay lấy từ bảng ở mục 6 của `phase_1_summary_report.md`. Mười lăm giá trị label smoothing lấy từ `src/phase-1/phase-1/summary/phase1_task_b_ablation_recomputed.csv`, vì báo cáo tổng hợp chỉ mô tả khảo sát này bằng văn xuôi, không có bảng đầy đủ.

| Kiến trúc | Xoay liên tục ±30° | Xoay rời rạc 90° | Δ | LS 0,0 | LS 0,1 | LS 0,2 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 94,737 | 95,489 | +0,752 | 93,797 | **94,737** | 93,985 |
| EfficientNet-B3 | 96,992 | 98,684 | **+1,692** | 96,429 | **96,992** | 95,677 |
| MobileNetV3-Large | 98,496 | 98,684 | +0,188 | 98,120 | 98,496 | **98,684** |
| ConvNeXt-Tiny | 99,624 | **100,000** | +0,376 | **99,812** | 99,624 | 97,932 |
| Swin-Tiny | 99,060 | 99,812 | +0,752 | 99,436 | 99,060 | **99,624** |

**Phép xoay đẳng hướng.** Phép xoay rời rạc bốn góc làm tăng accuracy ở cả năm kiến trúc, với mức tăng từ 0,188 đến 1,692 điểm phần trăm. Mức tăng lớn nhất ghi nhận ở **EfficientNet-B3**, từ 96,992% lên 98,684%, tương ứng 1,692 điểm phần trăm và giảm 56,2% số lỗi tương đối. ConvNeXt-Tiny đạt 100,000% ở cấu hình xoay rời rạc. Hai kiến trúc có mức tăng thấp nhất là MobileNetV3-Large (0,188) và ConvNeXt-Tiny (0,376), đều là các kiến trúc vốn đã có accuracy cao ở cấu hình mặc định.

Kết quả nhất quán trên toàn bộ năm kiến trúc phù hợp với giả thuyết đẳng hướng nêu ở Mục 3.3. Phép so sánh này là giữa hai kiểu xoay, không phải giữa có xoay và không xoay.

**Label smoothing.** Không có mức nào cho accuracy cao nhất ở tất cả năm kiến trúc. Mức $\epsilon=0{,}1$ tốt nhất cho ResNet-50 và EfficientNet-B3, mức $\epsilon=0{,}0$ tốt nhất cho ConvNeXt-Tiny (99,812%), và mức $\epsilon=0{,}2$ tốt nhất cho MobileNetV3-Large (98,684%) và Swin-Tiny (99,624%). Biên độ dao động theo $\epsilon$ lớn nhất ở ConvNeXt-Tiny, chênh 1,880 điểm phần trăm giữa mức 0,0 và mức 0,2.

Cả hai khảo sát chỉ chạy trên **một seed duy nhất (seed 42)**. Biên độ chênh lệch quan sát được nằm trong hoặc gần biên độ biến thiên theo seed của cấu hình chính, vốn đạt tới 1,022 điểm phần trăm ở ResNet-50 (Bảng 2). [CẦN DỮ LIỆU: kết quả ablation ở seed 123 và 456, gồm 10 lượt chạy cho khảo sát xoay và 20 lượt chạy cho khảo sát label smoothing]

## 4.5. Error Analysis and Interpretability

> **Lưu ý nguồn.** Mục này dùng bộ dự đoán từng ảnh của lượt huấn luyện lại (30 lượt chạy, đánh giá trên NVIDIA GeForce RTX 4090, PyTorch 2.11.0+cu128), là nguồn dự đoán cấp ảnh duy nhất hiện có. Accuracy của lượt chạy này khác với Bảng 2, chi tiết ở cuối mục.

### Ma trận nhầm lẫn

Trên tập test 17 lớp, ConvNeXt-Tiny sai **7 trên 1.596** dự đoán khi gộp ba seed. Toàn bộ 7 lỗi thuộc về **một cặp loài duy nhất là *Pinus dalatensis* và *Pinus krempfii***, cả hai cùng họ Pinaceae. Ma trận nhầm lẫn 17×17 của kiến trúc này do đó gần như chéo hoàn toàn, không có lỗi nào ở họ Cupressaceae, Podocarpaceae hay Taxaceae.

Khi gộp cả năm kiến trúc và ba seed, tổng số lỗi là 204. Năm cặp bị nhầm nhiều nhất trình bày ở Bảng 5.

**Bảng 5.** Năm cặp loài bị nhầm lẫn nhiều nhất trên tập test Task B, gộp năm kiến trúc và ba seed, tính cả hai chiều.

| # | Cặp loài | Số lần | Quan hệ phân loại |
|---:|---|---:|---|
| 1 | *Pinus dalatensis* ↔ *Pinus krempfii* | 62 | Cùng họ Pinaceae |
| 2 | *Chamaecyparis obtusa* ↔ *Podocarpus pilgeri* | 55 | Khác họ (Cupressaceae, Podocarpaceae) |
| 3 | *Podocarpus pilgeri* ↔ *Taxus wallichiana* | 20 | Khác họ (Podocarpaceae, Taxaceae) |
| 4 | *Taxus wallichiana* ↔ *Xanthocyparis vietnamensis* | 12 | Khác họ (Taxaceae, Cupressaceae) |
| 5 | *Callitris columellaris* ↔ *Cunninghamia lanceolata* | 11 | Cùng họ Cupressaceae |

Phân bố lỗi theo quan hệ phân loại: 94 trên 204 lỗi (46,1%) xảy ra giữa hai loài cùng họ, và 110 lỗi (53,9%) xảy ra giữa hai loài khác họ. Trong nhóm cùng họ, nhầm lẫn nội bộ họ Pinaceae chiếm 82 lỗi, trong khi nội bộ họ Cupressaceae chỉ chiếm 12 lỗi. Số lỗi theo kiến trúc giảm mạnh từ ResNet-50 (122 lỗi) xuống ConvNeXt-Tiny (7 lỗi).

Hai quan sát đi ngược kỳ vọng ban đầu. Thứ nhất, họ Cupressaceae không phải nguồn nhầm lẫn chính mặc dù có nhiều loài nhất trong tập (8 trên 17). Thứ hai, cặp bị nhầm nhiều thứ hai là một cặp **khác họ** giữa *Chamaecyparis obtusa* và *Podocarpus pilgeri*, chiếm hơn một phần tư tổng số lỗi. [CẦN DỮ LIỆU: kiểm tra thủ công một mẫu ảnh của hai loài này để xác định nguyên nhân, ví dụ tương đồng về màu sắc hoặc điều kiện chụp]

### Grad-CAM

[CẦN DỮ LIỆU: bản đồ nhiệt Grad-CAM của ConvNeXt-Tiny, Swin-Tiny và MobileNetV3-Large trên bốn ảnh đại diện cho bốn họ]

Mục này chưa hoàn thiện. Module sinh bản đồ kích hoạt đã được cài đặt cho cả năm kiến trúc, nhưng việc sinh ảnh đòi hỏi đồng thời ảnh gốc và checkpoint Task B, cả hai chưa có tại thời điểm viết. Mọi phát biểu về vùng tập trung của mạng, chẳng hạn kích thước lòng quản bào hay mật độ tia gỗ, chỉ được viết sau khi bản đồ nhiệt thực sự được sinh và kiểm tra.

### Khác biệt giữa hai lượt huấn luyện

Accuracy Task B của lượt huấn luyện lại khác với Bảng 2 ở mức không thể quy về làm tròn.

**Bảng 6.** Đối chiếu accuracy Task B giữa lượt chạy trong Bảng 2 và lượt huấn luyện lại.

| Kiến trúc | Bảng 2 (%) | Huấn luyện lại (%) | Δ (điểm phần trăm) |
|---|---:|---:|---:|
| ResNet-50 | 93,672 | 92,356 | −1,316 |
| EfficientNet-B3 | 97,118 | 97,807 | +0,690 |
| MobileNetV3-Large | 98,371 | 98,371 | 0,000 |
| ConvNeXt-Tiny | 99,499 | 99,561 | +0,062 |
| Swin-Tiny | 99,436 | 99,123 | −0,313 |

Thứ hạng của ConvNeXt-Tiny và của ResNet-50 không đổi, nhưng khoảng cách giữa ConvNeXt-Tiny và Swin-Tiny mở rộng từ 0,063 lên 0,438 điểm phần trăm. [CẦN DỮ LIỆU: xác định lượt chạy nào là kết quả chính thức của bài báo, và tính lại toàn bộ Mục 4 theo một nguồn duy nhất]
