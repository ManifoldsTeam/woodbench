# Task 1 — Grouped 5-fold CV (giao thức v2)

Train 5 model độc lập theo origin của lần Phase 1 cuối. Mỗi model chạy lần lượt
5 fold; mỗi fold khởi tạo lại từ ImageNet. Trong phần train của fold, khoảng 20%
origin mỗi loài được tách làm val; train đến khi val acc không tăng 10 epoch, rồi
chấm fold held-out **một lần** bằng `best.pth`. Held-out không dùng để chọn gì.

v2 (`protocol: best-val-origin-cv-v2`, 01-11-2026) thay v1 (fixed 50 epoch,
`final.pth`, FP32, không val). Output v1 ở `data/experiments/Task-B-origin-cv/` giữ
nguyên; v2 ghi vào `Task-B-origin-cv-v2/` và resume không trộn hai bản. Lý do:
[handoff 01-11-2026](../../../../docs/timeline/01-11-2026-handoff-vast.md), mục 6.

## Cấu hình mặc định

| Tham số | Giá trị |
|---|---|
| Models | ConvNeXt-Tiny, MobileNetV3-Large, ResNet-50, EfficientNet-B3, Swin-Tiny |
| Epochs | Tối đa **50**; early stopping patience 10 theo inner-val acc (`EARLY_STOPPING_PATIENCE=0` tắt) |
| Inner val | `round(0.2 × số origin)` origin mỗi loài trong phần train của fold (ít nhất 1, chừa ít nhất 1 để train); loài 10 origin: 2 test / 2 val / 6 train |
| Batch | **64** cho ResNet/MobileNet; **32** cho ba model còn lại |
| Optimizer / LR | AdamW, weight decay `1e-4`; head `1e-4` trên `model.get_classifier()`; backbone LR **bắt buộc**, lấy từ [bước 1](../task_07_lr_selection/README.md) |
| Scheduler / loss | Warmup tuyến tính 5 epoch rồi cosine; label smoothing `0.1`; AMP trên CUDA |
| Checkpoint | `best.pth` theo inner-val acc; held-out chấm từ checkpoint này |
| Input | `INPUT_VARIANT=rgb` (mặc định), `grayscale` hoặc `px16` (resize 224→16→224), áp dụng cho mọi split |
| Seed | `42` để chia fold và inner val (`default_rng([seed, fold])`), khởi tạo lại mỗi model/fold |
| Augmentation | Resize 224×224, H/VFlip p=0.5, rotation ±30°, brightness/contrast jitter 0.2, ImageNet normalize |
| Held-out | Resize 224×224 và normalize; không augmentation/crop |

Engine dùng cùng transform với Phase 1. `USE_90ROTATION=0` là main;
`1` là ablation chọn góc 0/90/180/270°. Không đặt `BATCH_SIZE` để giữ batch theo model.
Giao thức v2 giống Final-B main (warmup + cosine, AMP, early stopping, best checkpoint);
khác ở chỗ head là classifier thật và val tách theo origin trong từng fold.

Origin được đọc từ manifest
`docs/reports/phase_1/phase_1_leakage_origin_investigation/task-B-clean_group_split_plan.csv`.
Pool 4.800 ảnh train/val/test, 17 lớp, 552 key; giữ collision rename của Phase 1,
kiểm tra thiếu ảnh/loài và overlap origin/MD5 theo manifest. Chỉ chấp nhận thiếu
ba lớp đã bị loại ở Phase 1 (`Calocedrus sp`, `Cupressus funebris`, `Pinus sp`).
Ba loài dùng filename key
vẫn chưa có specimen ID được xác minh thêm. Mean ± sample std mô tả biến thiên
qua fold, không phải confidence interval.

## 1. Chuẩn bị và smoke test trên Vast

Repo ở `/workspace/woodbench`, có dataset `data/task-B-group-split`, manifest,
môi trường `wood` và GPU CUDA. Mỗi shell zsh cần activate cùng môi trường.

```zsh
conda activate wood
cd '/workspace/woodbench/notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training'
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# Kiểm fold, inner val, nhóm tham số, JSON và checkpoint; chưa train.
python train_kfold.py --backbone convnext_tiny --backbone_lr 1e-5 --dryrun

# Hai epoch cho mỗi model, output riêng; không dùng kết quả smoke để báo cáo.
smoke_root="/workspace/woodbench/data/experiments/Task-B-origin-cv-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
for model in convnext_tiny mobilenetv3_large resnet50 efficientnet_b3 swin_tiny; do
  python train_kfold.py --backbone "$model" --backbone_lr 1e-5 --fold 1 \
    --epochs 2 --warmup_epochs 1 --output_root "$smoke_root"
done
```

## 2. Bước 2: chạy ngầm 5 model, giữ chạy khi mất SSH

Từ folder Task 1, sau khi smoke test thành công. Launcher dừng (mã 2) nếu model nào
chưa có backbone LR. `BACKBONE_LR` áp dụng cho mọi model; `BACKBONE_LRS` ghi đè cho
từng model theo `decisions.csv` của bước 1. Ví dụ (thay bằng LR đã chọn):

```zsh
unset MODELS BATCH_SIZE INPUT_VARIANT
BACKBONE_LR=1e-5 BACKBONE_LRS="resnet50:1e-4 efficientnet_b3:1e-4" \
N_FOLDS=5 MAX_PARALLEL=5 USE_90ROTATION=0 RESUME=0 \
nohup bash run_all_models.sh \
  > "launch-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
disown
```

5 model chạy song song trên GPU mà môi trường cho phép nhìn thấy. Đây không
phải lệnh phân mỗi model sang một GPU riêng. Nếu VRAM không đủ, giảm
`MAX_PARALLEL` trước khi đổi batch để giữ cấu hình đối chiếu Phase 1.
Launcher in đường dẫn log session, chờ mã thoát của từng model và trả lỗi nếu có
model thất bại. Mỗi lần launch giữ log riêng tại `logs/<session>/<model>.log`.

## 3. Monitor từ shell zsh khác hoặc sau khi kết nối SSH lại

```zsh
conda activate wood
cd '/workspace/woodbench/notebooks/phase 2/supervisor-feedback/task_01_fivefold_cv_training'
python check_progress.py --watch --verbose
```

Monitor cập nhật mỗi 30 giây: tổng fold đã hoàn thành, trạng thái từng fold,
`epoch=12/50`, held-out accuracy và best epoch của fold hoàn thành. `Ctrl-C` chỉ dừng monitor.
`in_progress` là process đang sống; `interrupted` là fold dở đã ngừng; `error`
là config/artifact không hợp lệ. Khi đủ kết quả: `25/25 folds completed`.

Để xem log và GPU, mở thêm shell trong cùng folder:

```zsh
session_dir=$(ls -td logs/*/ | head -n 1)
tail -f "$session_dir/convnext_tiny.log"
# Hoặc xem một lần:
nvidia-smi
```

## 4. Resume và các kịch bản

Nếu process đã ngừng, chạy lại với **cùng code, dữ liệu và cấu hình** (cùng
`BACKBONE_LR`/`BACKBONE_LRS`):

```zsh
unset MODELS BATCH_SIZE INPUT_VARIANT
BACKBONE_LR=1e-5 BACKBONE_LRS="resnet50:1e-4 efficientnet_b3:1e-4" \
N_FOLDS=5 MAX_PARALLEL=5 USE_90ROTATION=0 RESUME=1 \
nohup bash run_all_models.sh \
  > "launch-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
disown
```

Resume giữ fold hoàn thành có config/split/code/checksum khớp; **fold dở train
lại từ epoch 0**, không tiếp tục epoch cuối. Output cũ khác cấu hình sẽ bị từ chối.

| Kịch bản | Thao tác |
|---|---|
| Mất SSH nhưng training vẫn sống | Kết nối lại và monitor; không launch thêm |
| Process ngừng / instance restart | Dùng lệnh `RESUME=1` ở trên |
| Chạy lại riêng fold 3 | `python train_kfold.py --backbone convnext_tiny --backbone_lr <lr> --fold 3 --resume` |
| OOM khi chạy song song | Giảm `MAX_PARALLEL` rồi resume, giữ nguyên batch |
| Đổi batch, epoch, LR, seed, augmentation hoặc code | Chọn `OUTPUT_ROOT` mới; không trộn với run cũ |
| Output riêng | Launcher đặt `OUTPUT_ROOT=/đường/dẫn`; monitor dùng `--output /đường/dẫn` |
| Chuyển instance | Copy code, dataset, manifest và toàn bộ output; dùng lại cấu hình cũ |

Override tùy chọn: `MODELS`, `MAX_PARALLEL`, `N_FOLDS`, `EPOCHS`, `BATCH_SIZE`,
`BACKBONE_LR`, `BACKBONE_LRS`, `HEAD_LR`, `WEIGHT_DECAY`, `LABEL_SMOOTHING`, `SEED`,
`NUM_WORKERS`, `WARMUP_EPOCHS`, `EARLY_STOPPING_PATIENCE`, `INNER_VAL_FRACTION`,
`INPUT_VARIANT`, `DATA_ROOT`, `MANIFEST_PATH`, `OUTPUT_ROOT`, `LOG_ROOT`, `PYTHON`,
`WANDB_PROJECT`.
`BATCH_SIZE` nếu đặt sẽ áp dụng cho mọi model. Có thể đặt `WANDB_API_KEY` trước
launch nếu cần tracking. Giữ nguyên các override đó khi resume.

## 5. Bước 3–4: ConvNeXt grayscale và 16px, fold 1

Cùng giao thức và cùng fold 1 với ConvNeXt RGB của bước 2, nên chênh lệch là do input.
Mỗi biến thể một output riêng (`<lr>` là LR ConvNeXt đã chọn ở bước 1):

```zsh
for variant in grayscale px16; do
  nohup python train_kfold.py --backbone convnext_tiny --backbone_lr <lr> --fold 1 \
    --input_variant "$variant" \
    --output_root "/workspace/woodbench/data/experiments/Task-B-origin-cv-v2-$variant" \
    > "launch-$variant-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
done
python check_progress.py --model convnext_tiny --verbose \
  --output /workspace/woodbench/data/experiments/Task-B-origin-cv-v2-px16
```

Số 16px mới **thay** số Swin 20 epoch tháng 7, không so trực tiếp với số đó.

### Máy ít CPU: dùng cache 224×224

Nếu GPU dùng ít vì CPU không giải nén kịp PNG 1280×1024, tạo cache một lần rồi trỏ `DATA_ROOT`
(launcher) hoặc `--data_root` (script) vào đó. Input của model không đổi, vì pipeline bắt đầu bằng
`Resize((224,224))`. Lần chạy 01-11-2026 dùng cách này; chi tiết ở
[báo cáo 01-11](../SUPERVISOR_REPORT_2026-11-01.md), mục 6.1.

```zsh
python scripts/build_resized_cache.py data/task-B-group-split data/task-B-group-split-224   # từ repo root
```

## 6. Xuất kết quả và backup

Output: `data/experiments/Task-B-origin-cv-v2/<model>/fold_<n>/` chứa `best.pth`,
`predictions.csv`, `metrics.json` (best epoch, origin train/val/test, per-class
P/R/F1), `run.json`, `history.json`.
Sau khi đủ fold, mở [evaluate_and_register_cv.ipynb](evaluate_and_register_cv.ipynb)
và chạy các cell: mỗi lần xuất tạo registry CSV riêng, một dòng/run, cùng
per-class metrics, OOF, summary và `per_species_summary.csv` (F1/precision/recall
từng loài, mean ± std qua fold) trong `reports/<session>/`; giữ
`run_id`/`source_run_ids` và đường dẫn tương đối theo repo ROOT.
Backup toàn bộ output, code và logs trước khi Destroy instance.

Notebook [fivefold_cv_training.ipynb](fivefold_cv_training.ipynb) gọi cùng engine
nhưng chạy các model tuần tự; dùng launcher ở trên để chạy song song.
Từ repo root, kiểm tra bằng:
`PYTHONPATH="$PWD" pytest -q tests/test_origin_cv.py tests/test_cv_training_runs.py tests/test_optimizer_head_selector.py`.
