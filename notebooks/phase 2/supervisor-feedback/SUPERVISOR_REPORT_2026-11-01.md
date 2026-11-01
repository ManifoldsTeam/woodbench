# Báo cáo bổ sung theo yêu cầu của thầy, 01/11/2026

Thầy yêu cầu 5 việc: (1) chọn LR cho từng model trên val, thử `1e-5` và `1e-4`; (2) chạy lại CV 5-fold cho đủ
5 model với LR đã chọn, lấy best-val checkpoint, val tách theo origin trong phần train của mỗi fold, có per-species
F1; (3) retrain ConvNeXt trên ảnh grayscale; (4) retrain ConvNeXt trên ảnh 16px; (5) đo lại latency/VRAM nếu
checkpoint đổi. Cả 5 việc đã làm xong ngày 01/11 trên một RTX 4090 (Vast). Toàn bộ log, cấu hình, dự đoán và metric
nằm trong `artifacts/` (mục 7).

**Tóm tắt**

- Cả 5 model đều chọn **backbone LR 1e-4**, tức backbone và head cùng một LR. Với ResNet, MobileNet và EfficientNet,
  1e-4 thắng rõ (+3 đến +9 điểm macro-F1 trên val). Với ConvNeXt và Swin, hai LR hòa nhau và 1e-4 có val loss thấp
  hơn.
- CV 5-fold theo origin (giao thức mới): ConvNeXt **99,13 ± 0,39%**, Swin 98,99 ± 1,01%, EfficientNet 97,73 ± 0,86%,
  ResNet 97,25 ± 1,02%, MobileNet 96,61 ± 0,88%. ResNet không còn tụt hẳn so với các model khác.
- ConvNeXt fold 1: ảnh màu 99,17%, **grayscale 94,68% (−4,5 điểm)**, **16px 92,81% (−6,4 điểm)**.
- Latency/VRAM: trên cùng một GPU, checkpoint mới và model trọng số ngẫu nhiên cho số đo như nhau (lệch dưới 1%), vì
  các chỉ số này chỉ phụ thuộc kiến trúc. Đổi LR chỉ làm đổi cột accuracy/F1 của bảng efficiency.

## 1. Chọn LR trên val (seed 42, split Final-B)

Mỗi model train 2 lần trên split Final-B (train 3.365 / val 903 / test 532 ảnh), chỉ khác backbone LR. Head LR cố
định `1e-4`. Các cài đặt còn lại giống Final-B main: AdamW, warmup 5 epoch rồi cosine, tối đa 50 epoch, early stopping
patience 10 theo val acc, label smoothing 0,1, AMP. Sau khi train, nạp `best.pth` rồi chấm val và test.

Quy tắc chọn được cố định trước khi xem kết quả: macro-F1 trên val cao hơn thì thắng. Nếu chênh dưới 0,5 điểm
(khoảng 4–5 ảnh val) thì coi là hòa và chọn LR có val loss thấp hơn. Số test chỉ ghi lại, không dùng để chọn.

| Model | Backbone LR | Val macro-F1 (%) | Val acc (%) | Val loss | Best epoch | Test acc (%) | Test macro-F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 1e-5 | 99,85 | 99,89 | 0,6052 | 10 | 98,68 | 98,79 |
| | **1e-4** | 99,68 | 99,67 | **0,5867** | 12 | 99,44 | 99,44 |
| Swin-Tiny | 1e-5 | 99,81 | 99,78 | 0,5989 | 18 | 99,25 | 99,27 |
| | **1e-4** | **100,00** | 100,00 | **0,5759** | 26 | 99,62 | 99,66 |
| EfficientNet-B3 | 1e-5 | 95,45 | 95,68 | 0,7065 | 31 | 95,68 | 95,58 |
| | **1e-4** | **98,44** | 98,45 | 0,6243 | 17 | 98,31 | 98,40 |
| MobileNetV3-Large | 1e-5 | 93,44 | 93,69 | 0,7768 | 37 | 97,74 | 97,74 |
| | **1e-4** | **98,66** | 98,67 | 0,6182 | 34 | 98,87 | 98,96 |
| ResNet-50 | 1e-5 | 89,27 | 90,37 | 0,9111 | 41 | 93,05 | 93,33 |
| | **1e-4** | **98,69** | 98,67 | 0,6229 | 31 | 98,12 | 98,21 |

**Lý do chọn:** ConvNeXt (chênh 0,17 điểm) và Swin (chênh 0,19 điểm) hòa về macro-F1, nên chọn theo val loss. Ba
model còn lại có macro-F1 cao hơn ở 1e-4 lần lượt: EfficientNet +2,99, MobileNet +5,23, ResNet +9,42 điểm.

Nguồn: [summary](../../../artifacts/phase_2/supervisor-feedback/task_07_lr_selection/results/reports/lr-select-20261101T075231Z/summary.md),
[decisions.csv](../../../artifacts/phase_2/supervisor-feedback/task_07_lr_selection/results/reports/lr-select-20261101T075231Z/decisions.csv),
[runs.csv](../../../artifacts/phase_2/supervisor-feedback/task_07_lr_selection/results/reports/lr-select-20261101T075231Z/runs.csv).

**Cần thầy lưu ý:** mô tả cũ "backbone 1e-5, head 1e-4" không đúng với các run Final-B cũ. Code cũ xếp một tham số
vào nhóm head nếu tên của nó chứa `head`, `fc` hoặc `classifier`. Trong timm, cách này khớp cả các block MLP của
ConvNeXt/Swin (`mlp.fc1`, `mlp.fc2`) và `conv_head` của EfficientNet/MobileNet. Hệ quả là 93% tham số ConvNeXt, 63%
của Swin và 30% của MobileNet đã chạy ở 1e-4. Lần này head chỉ gồm classifier thật (`model.get_classifier()`), nên
nhánh 1e-5 mới thật sự là 1e-5. Vì LR chọn được là 1e-4 cho mọi model, cấu hình mới chính là "AdamW lr = 1e-4" như
[kế hoạch ban đầu](../../../docs/references/WoodID_plan.md) (mục 3.4).

## 2. CV 5-fold theo origin, đủ 5 model

**Giao thức** (`best-val-origin-cv-v2`):

- 5 fold giữ nguyên như lần 03/10, chia theo origin trên toàn bộ 4.800 ảnh.
- Trong phần train của mỗi fold, mỗi loài lấy `round(0,2 × số origin)` origin làm val (ít nhất 1). Code kiểm tra
  không có origin hay md5 nào nằm ở hai phần.
- Backbone và head LR 1e-4. Train tối đa 50 epoch, early stopping patience 10 theo val acc.
- Fold held-out được chấm **một lần** bằng `best.pth`.

Mỗi fold có khoảng 2.900 ảnh train, 930 val và 960 held-out.

| Model | Accuracy (%) | Macro-F1 (%) | OOF accuracy (%) | Best epoch theo fold |
|---|---:|---:|---:|---|
| ConvNeXt-Tiny | **99,13 ± 0,39** | **99,16 ± 0,43** | 99,12 | 12, 11, 3, 21, 11 |
| Swin-Tiny | 98,99 ± 1,01 | 99,02 ± 0,91 | 99,00 | 13, 21, 11, 21, 5 |
| EfficientNet-B3 | 97,73 ± 0,86 | 97,71 ± 0,83 | 97,73 | 24, 29, 42, 33, 41 |
| ResNet-50 | 97,25 ± 1,02 | 97,19 ± 1,00 | 97,25 | 28, 35, 28, 31, 31 |
| MobileNetV3-Large | 96,61 ± 0,88 | 96,46 ± 0,94 | 96,60 | 48, 37, 35, 31, 41 |

Mean ± độ lệch chuẩn mẫu qua 5 fold. Đây là độ biến thiên giữa các fold, không phải khoảng tin cậy.
Nguồn: [summary.csv](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2/reports/cv-20261101T091132Z-2b618083/summary.csv),
[registry](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2/reports/cv-20261101T091132Z-2b618083/origin-cv-20261101T091132Z-registry.csv),
[notebook đã chạy](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/evaluate_and_register_cv_step2_executed.ipynb).

**So với CV ngày 03/10** (khác giao thức, chỉ để tham khảo): lần trước không có val, train cố định 50 epoch, lấy
checkpoint cuối và dùng LR theo quy tắc cũ. Kết quả lúc đó: Swin 99,73, ConvNeXt 99,33, EfficientNet 95,96,
MobileNet 94,51, ResNet chưa có. Lần này EfficientNet và MobileNet tăng 1,8–2,1 điểm. Swin giảm 0,7 điểm, chủ yếu do
fold 5 (xem mục 6.2). Mỗi fold lần này cũng train trên ít ảnh hơn khoảng 20%, vì phải tách val.

### 2.1 F1 từng loài, mean ± std qua 5 fold (%)

Xếp theo F1 thấp nhất giữa 5 model; *n* là tổng số ảnh held-out của loài qua 5 fold.

| Loài | n | ConvNeXt | Swin | EfficientNet | ResNet | MobileNet |
|---|---:|---:|---:|---:|---:|---:|
| *Taxus wallichiana* | 223 | 98,8 ± 1,3 | 98,6 ± 1,6 | 92,7 ± 5,9 | 90,0 ± 7,9 | 86,7 ± 13,7 |
| *Pinus dalatensis* | 334 | 95,9 ± 1,4 | 96,5 ± 4,5 | 92,8 ± 3,8 | 91,8 ± 4,2 | 92,4 ± 4,4 |
| *Podocarpus pilgeri* | 310 | 98,7 ± 1,7 | 99,2 ± 0,6 | 92,5 ± 5,3 | 94,8 ± 3,7 | 91,9 ± 4,2 |
| *Xanthocyparis vietnamensis* | 284 | 98,9 ± 1,3 | 98,7 ± 1,7 | 98,7 ± 1,9 | 95,4 ± 4,7 | 93,7 ± 3,2 |
| *Pinus krempfii* | 304 | 96,4 ± 1,9 | 96,6 ± 4,6 | 94,1 ± 4,1 | 93,9 ± 3,6 | 94,5 ± 3,0 |
| *Pinus cernua* | 287 | 99,4 ± 1,4 | 96,8 ± 7,2 | 97,9 ± 0,8 | 95,5 ± 3,7 | 94,3 ± 6,6 |
| *Pseudotsuga brevifolia* | 235 | 98,9 ± 2,4 | 99,3 ± 1,0 | 97,5 ± 2,3 | 95,5 ± 2,9 | 96,4 ± 2,6 |
| *Fokienia hodginsii* | 285 | 100,0 ± 0,0 | 100,0 ± 0,0 | 98,7 ± 2,5 | 99,8 ± 0,4 | 97,9 ± 3,3 |
| *Callitris columellaris* | 214 | 99,3 ± 1,1 | 99,3 ± 1,0 | 99,0 ± 1,3 | 98,6 ± 1,5 | 98,0 ± 2,0 |
| *Cunninghamia lanceolata* | 186 | 100,0 ± 0,0 | 99,7 ± 0,7 | 98,6 ± 1,3 | 99,0 ± 1,0 | 98,2 ± 1,8 |
| *Calocedrus rupestris* | 272 | 99,8 ± 0,4 | 99,5 ± 0,5 | 99,8 ± 0,5 | 99,6 ± 0,9 | 98,7 ± 1,4 |
| *Picea glauca* | 278 | 100,0 ± 0,0 | 99,7 ± 0,7 | 99,6 ± 0,8 | 99,3 ± 1,2 | 98,7 ± 0,5 |
| *Thujopsis dolabrata* | 327 | 99,8 ± 0,3 | 99,9 ± 0,3 | 99,7 ± 0,4 | 99,7 ± 0,4 | 99,2 ± 0,9 |
| *Araucaria araucana* | 293 | 100,0 ± 0,0 | 100,0 ± 0,0 | 100,0 ± 0,0 | 99,5 ± 0,7 | 99,7 ± 0,7 |
| *Cryptomeria japonica* | 320 | 99,9 ± 0,3 | 99,9 ± 0,3 | 99,7 ± 0,4 | 100,0 ± 0,0 | 99,9 ± 0,3 |
| *Pinus radiata* | 331 | 100,0 ± 0,0 | 100,0 ± 0,0 | 99,8 ± 0,3 | 99,9 ± 0,3 | 99,7 ± 0,4 |
| *Chamaecyparis obtusa* | 317 | 99,9 ± 0,3 | 100,0 ± 0,0 | 99,8 ± 0,4 | 100,0 ± 0,0 | 100,0 ± 0,0 |

**Nhận xét:**

- Ba loài thông Việt Nam (*P. dalatensis*, *P. krempfii*, *P. cernua*) là nhóm khó nhất, kể cả với ConvNeXt và Swin
  (95,9–96,6% ở *P. dalatensis* và *P. krempfii*).
- *Taxus* và *Podocarpus* chỉ khó với ba CNN nhỏ hơn; *Taxus* ở MobileNet dao động mạnh giữa các fold (± 13,7).
- 9 loài còn lại đạt F1 ≥ 97,9% với mọi model.

Nguồn: [per_species_summary.csv](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2/reports/cv-20261101T091132Z-2b618083/per_species_summary.csv)
(có thêm precision và recall), [per_class.csv](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2/reports/cv-20261101T091132Z-2b618083/per_class.csv)
(từng fold).

## 3–4. ConvNeXt retrain trên grayscale và 16px (fold 1)

Ba run dùng cùng giao thức, cùng fold 1, cùng 112 origin và 959 ảnh held-out; chỉ khác input. Biến đổi áp dụng cho
cả train, val và held-out:

- `grayscale`: `Grayscale(3)`, tức 3 kênh giống nhau.
- `px16`: resize 224 → 16 → 224 (bilinear), giống probe tháng 7.

| Input | Acc held-out (%) | Macro-F1 (%) | Giảm acc so với RGB | Best epoch / số epoch đã chạy |
|---|---:|---:|---:|---:|
| RGB (fold 1 của mục 2) | 99,17 | 99,27 | — | 12 / 22 |
| Grayscale | **94,68** | 94,87 | −4,49 điểm | 23 / 33 |
| 16px | **92,81** | 92,60 | −6,36 điểm | 37 / 47 |

F1 từng loài (%), chỉ liệt kê các loài bị ảnh hưởng; các loài khác ≥ 98,9 ở cả hai biến thể, trừ *Picea glauca*
(100 / 93,0) và *Pseudotsuga brevifolia* (96,5 / 94,1):

| Loài | RGB | Grayscale | 16px |
|---|---:|---:|---:|
| *Podocarpus pilgeri* | 100,0 | 90,6 | **69,9** |
| *Xanthocyparis vietnamensis* | 100,0 | 92,6 | **75,5** |
| *Taxus wallichiana* | 100,0 | 87,0 | 84,9 |
| *Pinus dalatensis* | 93,7 | **74,1** | 85,2 |
| *Pinus krempfii* | 93,9 | 81,9 | 91,0 |
| *Pinus cernua* | 100,0 | 99,0 | 85,7 |
| *Calocedrus rupestris* | 100,0 | 91,9 | 98,2 |

**Nhận xét:**

- **Bỏ màu chỉ làm giảm 4,5 điểm.** Phần lớn thông tin model dùng không nằm ở màu, nhưng màu giúp tách các loài thông
  Việt Nam (*P. dalatensis* giảm gần 20 điểm F1).
- **Ảnh 16px vẫn đạt 92,8%.** Ở độ phân giải này gần như không còn thấy chi tiết giải phẫu, nên model vẫn nhận đúng
  phần lớn ảnh dựa trên màu và cấu trúc thô (tỷ lệ gỗ sớm/gỗ muộn, vòng năm, tông màu).
- Hai loài giảm mạnh nhất ở 16px là *Podocarpus* và *Xanthocyparis*, cho thấy nhận dạng hai loài này cần chi tiết ở
  độ phân giải cao.
- Số 16px này **thay** số Swin 20 epoch tháng 7 (94,36%), không so trực tiếp với số đó vì khác model và khác giao
  thức.
- Số inference ảnh xám ngày 03/10 (ConvNeXt 73,68%) đo một câu hỏi khác: checkpoint đã train trên ảnh màu gặp ảnh xám
  lần đầu thì đạt bao nhiêu. Số mới (94,68%) trả lời câu hỏi "chỉ với thông tin không màu thì học được đến đâu".

Nguồn: [grayscale metrics.json](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2-grayscale/convnext_tiny/fold_1/metrics.json),
[16px metrics.json](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2-px16/convnext_tiny/fold_1/metrics.json),
[RGB metrics.json](../../../artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2/convnext_tiny/fold_1/metrics.json).

## 5. Latency/VRAM

Đo theo đúng giao thức của bảng efficiency hiện tại: batch 1, 3 × 224 × 224, FP32, 50 lần chạy khởi động rồi đo 200
lần bằng `torch.cuda.Event`, MACs đếm bằng THOP. Code đo dùng lại `measure_latency` của `phase2_benchmark.py`.
Checkpoint là `best.pth` LR 1e-4 của mục 1. Để kiểm tra xem đổi checkpoint có làm đổi số đo hay không, mỗi kiến trúc
được đo hai lần: với checkpoint mới và với trọng số khởi tạo ngẫu nhiên.

GPU: NVIDIA GeForce **RTX 4090**.

| Model | Params (M) | GMACs | Size (MiB) | Latency trung vị (ms) | p95 (ms) | VRAM tăng thêm (MiB) | Latency, trọng số ngẫu nhiên (ms) | Test acc mục 1 (%) | CV acc (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 27,83 | 4,45 | 106,2 | 4,19 | 4,39 | 15,2 | 4,21 | 99,44 | 99,13 |
| Swin-Tiny | 27,53 | 4,37 | 105,0 | 8,03 | 8,51 | 26,6 | 8,05 | 99,62 | 98,99 |
| ResNet-50 | 23,54 | 4,13 | 90,0 | 4,74 | 5,03 | 11,4 | 4,77 | 98,12 | 97,25 |
| MobileNetV3-Large | 4,22 | 0,22 | 16,2 | 4,79 | 5,11 | 7,7 | 4,81 | 98,87 | 96,61 |
| EfficientNet-B3 | 10,72 | 0,94 | 41,2 | 10,00 | 10,23 | 18,9 | 10,03 | 98,31 | 97,73 |

**Nhận xét:**

- Checkpoint mới và trọng số ngẫu nhiên cho latency lệch dưới 1% và VRAM tăng thêm giống hệt nhau. Latency, VRAM,
  params và FLOPs chỉ phụ thuộc kiến trúc, nên đổi LR **không** làm đổi các cột này trên cùng một GPU. Cột cần cập
  nhật trong bảng efficiency là accuracy/F1.
- Với batch 1 trên RTX 4090, thời gian chủ yếu đến từ việc khởi chạy kernel, nên MobileNet (0,22 GMACs) không nhanh
  hơn ResNet (4,13 GMACs). EfficientNet chậm nhất vì có nhiều lớp nhỏ.
- **Số trên 4090 không thay được cột RTX 3060 của paper** (`tab_env.tex` ghi 3060). Nếu thầy muốn bảng chính dùng
  4090 thì phải đổi môi trường đo trong paper. Nếu giữ 3060, các số latency/VRAM cũ vẫn đúng, chỉ cần cập nhật cột
  accuracy.

Nguồn: [efficiency_17class_rtx4090.csv](../../../artifacts/phase_2/supervisor-feedback/task_08_efficiency_rtx4090/profile-20261101T094606Z/efficiency_17class_rtx4090.csv),
[latency_samples.csv](../../../artifacts/phase_2/supervisor-feedback/task_08_efficiency_rtx4090/profile-20261101T094606Z/latency_samples.csv),
[metadata](../../../artifacts/phase_2/supervisor-feedback/task_08_efficiency_rtx4090/profile-20261101T094606Z/efficiency_metadata.json),
[script](task_08_efficiency_rtx4090/profile_efficiency.py).

## 6. Ghi chú phương pháp

### 6.1 Cache ảnh 224×224

Ảnh gốc là PNG 1280×1024 (khoảng 2 MB). Container chỉ được dùng khoảng 24 core CPU, không đủ để giải nén ảnh kịp
cho GPU: lúc đầu GPU chỉ dùng khoảng 6%, ước tính bước 1 mất 8–14 giờ.

Mọi pipeline đều bắt đầu bằng `Resize((224, 224))`. Vì vậy toàn bộ ảnh được resize trước **một lần** bằng đúng phép
đó và lưu PNG không mất dữ liệu (`scripts/build_resized_cache.py`). Kiểm tra trên 300 ảnh ngẫu nhiên: tensor đưa vào
model **giống hệt** khi đọc từ ảnh gốc. Sau thay đổi này, tốc độ tăng khoảng 20 lần.

- Mọi run báo cáo ở đây đều đọc từ cache; `run_config.json`/`run.json` ghi `data_root: data/task-B-group-split-224`.
- Với 16px, input giống hệt pipeline gốc, vì phép Resize đầu tiên không làm thay đổi ảnh cache.
- Với grayscale, pipeline chuyển xám *trước* khi resize, còn khi đọc cache thì resize đã xảy ra trước. Hai cách có
  thể lệch ±1 mức xám do làm tròn, không ảnh hưởng đến kết luận.

Ảnh gốc đã khớp checksum với bản local trước khi tạo cache. Checksum của cache nằm trong
[01-11-2026-task-B-group-split-224.sha256](../../../docs/timeline/01-11-2026-task-B-group-split-224.sha256).

### 6.2 Best-val checkpoint đôi khi rơi vào epoch rất sớm

Ở ConvNeXt fold 3 (best epoch 3) và Swin fold 5 (best epoch 5), val acc đạt đỉnh ngay sau warmup rồi dao động, trong
khi val loss vẫn tiếp tục giảm. Ví dụ ConvNeXt fold 3:

- Epoch 3: val acc 99,24%, val loss 0,645. Đây là epoch được chọn làm best.
- Epoch 12: val acc 98,71%, val loss **0,607**.

Early stopping vì vậy dừng ở epoch 13–15. Swin fold 5 đạt 97,36%, thấp nhất trong 5 fold của model này, và kéo std
của Swin lên 1,01. Quy tắc "best val acc, patience 10" giữ đúng như đã thống nhất (giống Final-B), không đổi sau khi
xem kết quả. Nếu thầy muốn, có thể chạy lại các fold đó với tiêu chí val loss để kiểm tra độ nhạy.

## 7. Minh chứng

| Nội dung | Vị trí trong repo |
|---|---|
| Bước 1: cấu hình, history, metric, dự đoán val/test từng run | `artifacts/phase_2/supervisor-feedback/task_07_lr_selection/results/` |
| Bước 1: log từng run, môi trường (`env.txt`, `pip_freeze.txt`, `conda_env.yaml`, `nvidia-smi.txt`), GPU monitor | `.../task_07_lr_selection/launch-20261101T071156Z/`, `.../logs/20261101T071205Z-sxcGt2/` |
| Bước 1: lần chạy bị hủy vì nghẽn đọc ảnh (giữ log, không dùng số) | `.../task_07_lr_selection/ABORTED-slow-io-*` |
| Bước 2: metric/run/history/predictions từng fold, registry, OOF, per-species | `artifacts/phase_2/supervisor-feedback/task_01_fivefold_cv_training/results-v2/` |
| Bước 3–4 | `.../task_01_fivefold_cv_training/results-v2-grayscale/`, `.../results-v2-px16/` |
| Bước 2–4: log, môi trường, GPU monitor, smoke test | `.../task_01_fivefold_cv_training/launch-step2-*`, `launch-step34-*`, `logs/20261101T080223Z-4riqUX/`, `smoke-*` |
| Bước 5 | `artifacts/phase_2/supervisor-feedback/task_08_efficiency_rtx4090/` |
| sha256 của 47 checkpoint (`.pth` không đưa vào git, khoảng 6,6 GB) | `artifacts/phase_2/supervisor-feedback/checkpoints_2026-11-01.sha256` |

Log dạng `*.log.gz` (đọc bằng `zcat`). `oof_predictions.csv` được nén thành `.csv.gz`.

## 8. Cần thầy quyết định

1. Bảng chính Task B của paper (Final-B, 3 seed × 5 model) đang dùng cấu hình LR cũ. Lấy CV mới làm kết quả chính,
   hay train lại 15 run Final-B với LR 1e-4 (ước tính khoảng 1–1,5 giờ trên 4090, dựa trên thời gian của bước 1)? Nếu train lại, checkpoint của demo cũng phải
   thay.
2. Sửa mô tả LR trong paper thành một LR chung 1e-4 (mục 1).
3. Bảng efficiency: giữ cột latency/VRAM RTX 3060 hiện có và chỉ cập nhật accuracy, hay chuyển sang số RTX 4090 (mục 5)?
4. Có cần kiểm tra độ nhạy tiêu chí chọn checkpoint (val loss thay cho val acc) ở mục 6.2 không?
