# Handoff 01-11-2026: chạy yêu cầu mới của thầy trên Vast

> **Trạng thái (01-11-2026, 10:00 UTC): cả 5 bước đã xong trên Vast (RTX 4090).** Báo cáo cho thầy:
> [SUPERVISOR_REPORT_2026-11-01.md](../../notebooks/phase%202/supervisor-feedback/SUPERVISOR_REPORT_2026-11-01.md).
> Kết quả tóm tắt và các điểm lệch so với runbook ở [mục 9](#9-kết-quả-thực-hiện-01-11-2026).

Tài liệu này đủ để tiếp tục trên một instance Vast **chỉ có repo woodbench** (không có WoodID, WoodID-demo).
Mọi số liệu dưới đây đã được kiểm lại trên máy local ngày 01-11-2026 (woodbench `7f004df`), trừ chỗ ghi
"suy luận".

## 1. Thầy yêu cầu gì

1. **Chọn LR cho từng model trên val.** Thử backbone LR `1e-5` và `1e-4`, seed 42 hoặc fold 1. Model nào tốt hơn
   ở LR nào thì dùng LR đó cho bước sau.
2. **Chạy lại CV 5-fold cho đủ 5 model** (thêm ResNet) với LR đã chọn. Thống nhất lấy best-val checkpoint; val
   tách theo origin bên trong phần train của mỗi fold. Ghi thêm per-species F1 trung bình qua các fold.
3. **Retrain ConvNeXt trên ảnh grayscale** (1 fold là đủ), xem còn bao nhiêu %.
4. **Retrain ConvNeXt trên ảnh 16px** bằng pipeline hiện tại, để thay số liệu tháng 7.
5. **Nếu checkpoint đổi do đổi LR thì đo lại latency/VRAM trên RTX 3060.**

Thứ tự bắt buộc: 1 → 2 → (3, 4) → 5. Bước 3 và 4 dùng cùng giao thức với bước 2. Bước 1–4 chạy trên Vast; bước 5
đo trên RTX 3060 ở máy local, như bảng efficiency hiện tại (xem mục 7).

## 2. Số liệu đã có và vì sao không dùng lại được

| Bước | Đã có | Vấn đề |
|---|---|---|
| 1 | Chỉ ResNet, backbone LR 1e-4, seed 42: `data/experiments/Task-B-new/lr_tuning/` ([notebook](../../notebooks/phase%202/supervisor-feedback/task_05_resnet_lr_tuning/resnet_lr_tuning.ipynb)) | Xem mục 3.3. Số 96,05% là model epoch 50; `best.pth` (epoch 35) chỉ đạt **95,68%** |
| 2 | `data/experiments/Task-B-origin-cv/` (03/10): 4 model × 5 fold; ResNet chỉ có `fold_1/run.json`, không có kết quả | Không có val (`n_val=0`), 50 epoch cố định, lấy `final.pth`, FP32, LR theo quy tắc chuỗi con (mục 3.1), không có F1 từng loài trong `metrics.json` |
| 3 | Chỉ inference ảnh xám trên checkpoint RGB (03/10): ConvNeXt 73,68%, MobileNet 47,18% | Không retrain |
| 4 | Tháng 7 (30/07, `artifacts/phase_1b/anatomy_destruction_probe/`): **Swin**, retrain 20 epoch trên ảnh 224→16→224, 94,36%, chọn theo best val macro-F1, baseline lấy từ checkpoint Task-B cũ (Swin 99,06%). Tháng 10: ConvNeXt inference 16px, không retrain, 27,07% | Khác model, khác pipeline |
| 5 | `WoodID/artifact/person2/efficiency/efficiency_17class.csv` (RTX 3060, có latency + VRAM) | Đo trên checkpoint seed 42 của registry hiện tại |

Số chuẩn hiện tại của Task B (để so): registry
`docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z/` (đánh giá lại từ `best.pth`).
Seed 42, best val acc / test acc:

| Model | Best val acc | Test acc | Best epoch |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 99,89 | 99,81 | 14 |
| Swin-Tiny | 99,56 | 99,62 | 10 |
| MobileNetV3-Large | 94,91 | 98,68 | 43 |
| EfficientNet-B3 | 96,12 | 97,37 | 43 |
| ResNet-50 | 90,48 | 93,80 | 47 |

## 3. Lỗi trong code cần biết trước khi chạy

### 3.1 Chia nhóm LR head/backbone theo chuỗi con

`src/train/optimizer.py` (`create_optimizer`, dòng 44) và
`notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training/train_kfold.py` (dòng 185): tham số nào có
tên chứa `"head"`, `"fc"` hoặc `"classifier"` thì nhận head LR (1e-4), còn lại nhận backbone LR. Trong timm,
block MLP của ConvNeXt/Swin tên `mlp.fc1`/`mlp.fc2`, EfficientNet/MobileNet có `conv_head`. Kết quả:

| Model | Đang nhận head LR theo quy tắc cũ | Classifier thật (`model.get_classifier()`) |
|---|---:|---:|
| ConvNeXt-Tiny | 93,1% | 0,05% |
| Swin-Tiny | 62,8% | 0,05% |
| MobileNetV3-Large | 29,6% | 0,52% |
| EfficientNet-B3 | 5,7% | 0,24% |
| ResNet-50 | 0,1% | 0,15% |

Hệ quả cho bước 1: nếu không sửa, nhánh "1e-5" của ConvNeXt/Swin thực chất vẫn chạy phần lớn ở 1e-4, phép so
sánh vô nghĩa. Khi `backbone_lr == head_lr` thì `create_optimizer` dùng một nhóm duy nhất, nên nhánh 1e-4 không
bị ảnh hưởng.

**Đã sửa (01-11):** `split_head_parameters(model, head_selector)` trong `src/train/optimizer.py`;
`create_optimizer(..., head_selector="classifier")` lấy đúng `model.get_classifier()`. Mặc định vẫn là
`"substring"` để run cũ tái lập được; script bước 1–4 truyền `"classifier"`.

### 3.2 Final-B chấm test trên model cuối, không nạp lại best

`notebooks/phase 1/task-B-17-new/final_b_17class.py`, hàm `train_one`: `engine.fit(save_best_only=True, ...)` có
lưu `*_best.pth`, nhưng dòng 580 gọi `evaluate_model(model, test_loader, device)` trên model đang nằm trong bộ
nhớ, tức epoch cuối. `test_metrics.json` lúc train vì thế là của model cuối. Số chuẩn hiện tại không bị ảnh hưởng
vì registry 20260819T043733Z đã chấm lại từ `best.pth`. Ngoài ra LR (dòng 522–524: `lr=1e-4`,
`backbone_lr=1e-5`) và thư mục output (dòng 75: `Task-B-new/final`) đang hard-code.

### 3.3 Run ResNet LR cũ (`lr_tuning`) không dùng làm bằng chứng

- Chấm test trên biến `model` sau vòng lặp (epoch 50), không nạp `best.pth`.
- Resume từ `best.pth` epoch 15: mất lịch sử 15 epoch đầu (`epochs_trained: 35` là sai, thực tế 50), scheduler
  tạo lại với `T_max=35` nên LR quay về 1e-4.
- Transform khác Final-B: `RandomResizedCrop` + `RandomRotation(90)` + jitter saturation/hue khi train;
  `Resize(256)` + `CenterCrop(224)` khi chấm. Final-B dùng `Resize(224)` thẳng.
- Baseline gõ tay `0.93672` (registry: 93,80%).
- Best và final khác nhau ở 6/532 ảnh (final đúng hơn 2 ảnh, McNemar p = 0,69): chỉ là nhiễu.

### 3.4 Ghi chú dữ liệu

Cột `md5` trong manifest là hash của ảnh nguồn `data/task-B-clean`. Khoảng 1/4 file trong
`data/task-B-group-split` khác byte so với nguồn, nên **không dùng md5 của manifest để kiểm dữ liệu**. Dùng
checksum của bản local (mục 4).

## 4. Chuẩn bị Vast

### 4.1 Code

```zsh
cd /workspace
git clone -b docs/handoff-2026-11-01 git@github.com:ManifoldsTeam/woodbench.git   # nhánh có code bước 1–4
cd woodbench
conda activate wood                       # tạo theo scripts/setup_ssh_server.sh nếu chưa có
pip install -r requirements.txt           # nếu env mới
PYTHONPATH="$PWD" pytest -q tests/test_lr_selection.py tests/test_optimizer_head_selector.py \
  tests/test_origin_cv.py tests/test_cv_training_runs.py
python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), timm.__version__)"
```

Code đã viết và test trên máy local (01-11): dry-run cả hai script trên dữ liệu thật và train thử 2 epoch
MobileNet trên RTX 3060. Trên Vast chỉ cần chạy; nếu phải sửa hàm nào, chạy `gitnexus_impact` theo
`CLAUDE.md` trước.

### 4.2 Dữ liệu (19 GB, 4.800 ảnh)

[`01-11-2026-task-B-group-split.sha256`](01-11-2026-task-B-group-split.sha256) là sha256 của từng ảnh trong
`data/task-B-group-split` trên máy local, tức bản đã sinh ra registry hiện tại. Lấy dữ liệu từ đâu cũng phải
qua bước kiểm này.

Cách 1, nhanh hơn: tải `data-B-new.tar.zst` (18,25 GB, 18/08) từ Google Drive `wood/` về Vast. Chưa kiểm được
nội dung tarball này từ máy local; nếu bung ra không có đúng thư mục `task-B-group-split/` hoặc checksum không
khớp thì dùng cách 2.

```zsh
cd /workspace/woodbench/data
tar -I zstd -xf data-B-new.tar.zst
ls    # cần có task-B-group-split/{train,val,test}; đổi tên nếu tarball dùng tên khác
```

Cách 2: chép từ máy local.

```zsh
# máy local
tar -cf task-B-group-split.tar -C ~/Code/ManifoldsTeam/woodbench/data task-B-group-split
rsync -avP task-B-group-split.tar <vast>:/workspace/woodbench/data/
# Vast
cd /workspace/woodbench/data && tar -xf task-B-group-split.tar
```

Kiểm (bắt buộc, trên Vast):

```zsh
cd /workspace/woodbench/data
sha256sum -c --quiet ../docs/timeline/01-11-2026-task-B-group-split.sha256 && echo DATA OK
for s in train val test; do echo "$s $(find task-B-group-split/$s -name '*.png' | wc -l)"; done
# phải ra: train 3365, val 903, test 532
```

Pretrained ImageNet được timm tải từ HuggingFace Hub lúc chạy lần đầu; cần mạng ra ngoài.

### 4.3 Không cần đẩy checkpoint cũ

Cả bước 1–4 đều train lại từ trọng số ImageNet, không đọc checkpoint cũ. Số so sánh của cấu hình cũ đã có trong
registry (đã nằm trong git) và trong bảng ở mục 2. Checkpoint cũ đã có bản trên Drive (`Task-B-new/`,
`Task-B-origin-cv/`); chỉ cần tải về nếu muốn chấm lại một checkpoint cũ trên Vast, việc này không thuộc 5 yêu
cầu.

## 5. Bước 1: chọn LR

### 5.1 Code (đã làm)

`notebooks/phase 2/supervisor-feedback/task_07_lr_selection/`, hướng dẫn chi tiết trong
[README](../../notebooks/phase%202/supervisor-feedback/task_07_lr_selection/README.md):

- `train_lr_select.py`: một run = một model × một backbone LR. Dùng lại cấu hình Final-B main
  (`final_b_17class.py` không bị sửa), head bằng classifier thật, sau khi train nạp `model_best.pth` rồi chấm
  val và test. `run_config.json` ghi LR, `head_selector`, số tham số mỗi nhóm, `git_commit`, phiên bản, GPU.
- `run_lr_select.sh`: chạy 10 run, giới hạn số run song song bằng `MAX_PARALLEL`.
- `summarize_lr_select.py`: bảng 10 run và LR được chọn cho từng model theo quy tắc mục 5.3.

### 5.2 Ma trận chạy (10 run)

| Cố định | Giá trị |
|---|---|
| Split | `data/task-B-group-split` (train 3.365 / val 903 / test 532), seed 42 |
| Giống Final-B main | AdamW, weight decay 1e-4, cosine + 5 epoch warmup, tối đa 50 epoch, early stopping patience 10 theo val acc, label smoothing 0,1, AMP, batch theo `BACKBONE_CONFIGS`, transform `src/data/transforms.py` |
| Head LR | 1e-4, `head_selector="classifier"` |
| Thay đổi | backbone LR ∈ {1e-5, 1e-4} × 5 model |
| Output | `data/experiments/Task-B-lr-select/bblr_<1e-5|1e-4>/seed_42/<backbone>/` |

Không resume run dùng để báo cáo: nếu gián đoạn, xóa thư mục run đó và chạy lại từ đầu (script từ chối ghi vào
thư mục đã có file).

```zsh
cd '/workspace/woodbench/notebooks/phase 2/supervisor-feedback/task_07_lr_selection'
python train_lr_select.py --backbone convnext_tiny --backbone_lr 1e-5 --dryrun
MAX_PARALLEL=5 nohup bash run_lr_select.sh > "launch-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
disown
# khi đủ 10 metrics.json:
python summarize_lr_select.py
```

### 5.3 Quy tắc chọn

- Dùng `best.pth` của mỗi run, tính trên **val**: macro-F1 (chính), acc, loss.
- Chênh macro-F1 < 0,5 điểm (khoảng 4–5 ảnh val) thì coi là hòa và xét val loss.
- Hòa cả hai thì chọn chung một LR cho mọi model (dễ bảo vệ hơn trong paper). Bước này `summarize_lr_select.py`
  không tự làm: nó chọn theo hai tiêu chí trên và ghi lý do vào `decisions.csv`.
- Ghi số test vào bảng nhưng **không dùng test để chọn**.

Suy luận (chưa kiểm): ResNet, EfficientNet, MobileNet có best epoch muộn (40–50) ở cấu hình cũ, nên nhiều khả
năng 1e-4 tốt hơn. ConvNeXt/Swin ở cấu hình cũ đã chạy phần lớn ở 1e-4 và hội tụ sớm.

### 5.4 Kết quả cần mang về

Bảng 5 model × 2 LR (val macro-F1/acc/loss, best epoch, epochs trained, test acc/F1), `run_config.json`,
`history`, `best.pth` của LR được chọn, commit hash.

## 6. Bước 2–4: CV mới, grayscale, 16px

### 6.1 Code (đã làm, giao thức `best-val-origin-cv-v2`)

Chi tiết trong [README Task 1](../../notebooks/phase%202/supervisor-feedback/task_01_fivefold_cv_training/README.md).

1. **Val theo origin trong phần train của mỗi fold** (`origin_cv.split_inner_validation`): mỗi loài lấy
   `round(0.2 × số origin)` origin làm val (ít nhất 1, chừa ít nhất 1 để train). Loài 10 origin: 2 test / 2 val
   / 6 train. Seed `[42, fold]`, nên 5 fold test giữ nguyên như v1. `metrics.json` ghi danh sách origin
   train/val/test; code kiểm không trùng origin hay md5 giữa 3 phần.
2. **Giao thức giống bước 1:** warmup 5 + cosine, AMP, early stopping patience 10 theo val acc, lưu `best.pth`,
   chấm fold held-out **một lần** bằng `best.pth`.
3. **LR:** `--backbone_lr` bắt buộc. Launcher nhận `BACKBONE_LR` (chung) và `BACKBONE_LRS="model:lr ..."`
   (riêng từng model), dừng nếu model nào thiếu LR. Head chọn bằng classifier thật.
4. **Per-species F1:** `metrics.json` mỗi fold có P/R/F1 từng loài. `evaluate_and_register_cv.ipynb` xuất
   thêm `per_species_summary.csv` (mean ± std qua fold).
5. **`--input_variant {rgb,grayscale,px16}`** áp dụng cho cả train, val và held-out. `grayscale` là
   `Grayscale(3)`; `px16` là resize 224→16→224 bilinear như probe tháng 7.
6. Output mặc định `data/experiments/Task-B-origin-cv-v2`; resume không nhận output v1.

### 6.2 Chạy

- Dry-run, rồi smoke test 2 epoch cho cả 5 model vào output riêng (README Task 1, mục 1).
- Bước 2: 25 run. Đặt LR theo `decisions.csv` của bước 1, ví dụ
  `BACKBONE_LR=1e-5 BACKBONE_LRS="resnet50:1e-4" bash run_all_models.sh` (lệnh `nohup` đầy đủ ở README mục 2).
- Bước 3, 4: ConvNeXt, `--fold 1`, `--input_variant grayscale` và `px16`, output riêng
  (`Task-B-origin-cv-v2-grayscale`, `Task-B-origin-cv-v2-px16`; lệnh ở README mục 5). So với ConvNeXt RGB fold
  1 của bước 2: cùng fold, cùng giao thức, nên phần chênh lệch là do input.
- Khi báo thầy: số 16px mới là ConvNeXt theo giao thức mới, **thay** chứ không so trực tiếp với số Swin tháng 7.

### 6.3 Kết quả cần mang về

Toàn bộ thư mục output (có `best.pth`, `predictions.csv`, `metrics.json`, `run.json`, `history.json`), log
session, registry/summary/`per_species_summary.csv` từ `evaluate_and_register_cv.ipynb`. Với grayscale/px16,
đổi `OUTPUT_ROOT` trong notebook sang thư mục tương ứng.
Backup trước khi Destroy instance.

## 7. Bước 5: latency/VRAM

Train luôn chạy trên RTX 4090 (Vast). Riêng bảng efficiency của paper được đo trên RTX 3060 ở máy local:
`WoodID/results/tab_env.tex` ghi "Efficiency profile & NVIDIA GeForce RTX 3060", `macros.tex` có
`\EfficiencyGPU{NVIDIA GeForce RTX 3060}`, và thầy yêu cầu đo lại "trên 3060". Script đo
(`notebooks/phase 2/phase2_benchmark.py`, `run_efficiency_profile`) vì thế tự dừng nếu GPU không phải RTX 3060,
và ghi output vào repo `WoodID` nằm cạnh woodbench.

Latency đo trên 4090 là một bộ số khác, không thay được cột 3060 trong bảng. Hai hướng:

- **Giữ 3060 (đề xuất):** mang 5 `best.pth` đã chọn về máy local (khoảng 0,4 GB), trỏ hàm đo sang checkpoint
  mới, chạy `01_learning_curves_and_efficiency.ipynb`. Việc này chỉ đọc checkpoint, giống các việc xử lý artifact
  khác ở local.
- **Đổi sang 4090:** phải đo lại cả 5 model, sửa `tab_env.tex`/`macros.tex`, và cần thầy đồng ý vì thầy ghi rõ
  3060.

Latency, VRAM và FLOPs chỉ phụ thuộc kiến trúc, nên trên cùng một GPU dự kiến không đổi ngoài nhiễu đo. Cột
acc/F1 mới là phần thay đổi.

## 8. Câu hỏi còn mở

1. Bảng chính của paper (Final-B, 3 seed × 5 model) đang dùng cấu hình LR cũ. Nếu CV chạy LR mới: lấy CV làm kết
   quả chính, hay train lại 15 run Final-B? Nếu train lại, checkpoint của demo cũng phải thay.
2. Mô tả "backbone 1e-5, head 1e-4" trong paper chưa đúng với ConvNeXt/Swin/MobileNet (mục 3.1). Cần báo thầy.
3. Grayscale/16px: tài liệu này chọn fold 1 của CV v2. Nếu thầy muốn so với split Final-B thì chạy trên split
   đó thay vì fold 1.

## 9. Kết quả thực hiện 01-11-2026

Chạy trên Vast, RTX 4090 24 GB, quota CPU khoảng 24 core, repo tại `/root/woodbench`, nhánh `docs/handoff-2026-11-01`
(commit `36dd63e`). Env conda `wood` (Python 3.12, torch 2.15.0+cu132, timm 1.0.30); danh sách gói đầy đủ ở
`pip_freeze.txt`/`conda_env.yaml` trong mỗi thư mục `launch-*`.

| Bước | Kết quả | Minh chứng |
|---|---|---|
| Dữ liệu | Tarball `data-B-new.tar.zst` đúng thư mục `task-B-group-split/`; 4.800/4.800 sha256 khớp; train 3365 / val 903 / test 532 | — |
| 1 | Cả 5 model chọn backbone LR **1e-4** | `artifacts/phase_2/supervisor-feedback/task_07_lr_selection/results/` |
| 2 | 25/25 fold; ConvNeXt 99,13 ± 0,39, Swin 98,99 ± 1,01, EfficientNet 97,73 ± 0,86, ResNet 97,25 ± 1,02, MobileNet 96,61 ± 0,88 | `.../task_01_fivefold_cv_training/results-v2/` |
| 3 | ConvNeXt grayscale fold 1: 94,68% (RGB 99,17%) | `.../results-v2-grayscale/` |
| 4 | ConvNeXt 16px fold 1: 92,81% | `.../results-v2-px16/` |
| 5 | Đo trên **RTX 4090** (theo yêu cầu, không đo trên 3060); latency/VRAM của checkpoint mới bằng trọng số ngẫu nhiên | `.../task_08_efficiency_rtx4090/` |

**Lệch so với runbook:**

1. **Cache ảnh 224×224.** Mọi pipeline bắt đầu bằng `Resize((224,224))`. Ảnh gốc là PNG 1280×1024; với quota khoảng
   24 core, GPU chỉ dùng khoảng 6% và bước 1 ước tính mất 8–14 giờ. `scripts/build_resized_cache.py` resize trước một
   lần bằng đúng phép đó, lưu PNG lossless vào `data/task-B-group-split-224`. Kiểm tra 300 ảnh ngẫu nhiên: tensor đưa
   vào model giống hệt. Tốc độ tăng khoảng 20 lần. Checksum của cache nằm ở
   [01-11-2026-task-B-group-split-224.sha256](01-11-2026-task-B-group-split-224.sha256). Với grayscale có thể lệch
   ±1 mức xám, vì pipeline chuyển xám trước khi resize.
2. **Lần chạy đầu của bước 1 đã bị hủy** sau 1–3 epoch (nghẽn đọc ảnh). Output dở đã xóa; log giữ lại ở
   `task_07_lr_selection/ABORTED-slow-io-*`.
3. **Bước 5 đo trên 4090.** `phase2_benchmark.run_efficiency_profile` chặn mọi GPU khác 3060, nên dùng script mới
   `task_08_efficiency_rtx4090/profile_efficiency.py`, tái sử dụng `measure_latency`/`live_environment`. Script giới hạn
   PyTorch còn 8 thread CPU, vì với 255 thread trên quota 24 core, chỉ riêng khởi tạo model đã mất khoảng 40 giây.
4. Smoke test (2 epoch × 5 model) ghi vào `data/experiments/Task-B-origin-cv-smoke-*`; đây không phải kết quả báo cáo.

**Checkpoint:** 47 file `.pth` (khoảng 6,6 GB) chỉ có trên instance, ở `data/experiments/Task-B-lr-select/` và
`Task-B-origin-cv-v2*/`; sha256 ở `artifacts/phase_2/supervisor-feedback/checkpoints_2026-11-01.sha256`. Instance không
có volume (`workspace_is_volume`), nên phải tải về trước khi destroy.

**Câu hỏi mở** (thay mục 8): xem mục 8 của báo cáo cho thầy. Ngoài ra, best-val theo val acc đôi khi rơi vào epoch rất
sớm (ConvNeXt fold 3 epoch 3, Swin fold 5 epoch 5) trong khi val loss vẫn giảm (báo cáo, mục 6.2).

