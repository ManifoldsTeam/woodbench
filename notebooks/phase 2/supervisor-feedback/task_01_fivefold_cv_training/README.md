# Task 1 — Grouped 5-fold CV

Train 5 model độc lập theo origin của lần Phase 1 cuối. Mỗi model chạy lần lượt
5 fold; mỗi fold khởi tạo lại từ ImageNet, train đủ số epoch đã chốt, rồi đánh giá
held-out đúng một lần. Không chọn checkpoint bằng kết quả held-out.

## Cấu hình mặc định

| Tham số | Giá trị |
|---|---|
| Models | ConvNeXt-Tiny, MobileNetV3-Large, ResNet-50, EfficientNet-B3, Swin-Tiny |
| Epochs | **50 mỗi fold** |
| Batch | **64** cho ResNet/MobileNet; **32** cho ba model còn lại |
| Optimizer / LR | AdamW; backbone `1e-5`, head `1e-4`; weight decay `1e-4` |
| Scheduler / loss | Cosine, `T_max=50`; label smoothing `0.1` |
| Seed | `42` để chia 5 fold theo origin từng loài và khởi tạo lại mỗi model/fold |
| Augmentation | Resize 224×224, H/VFlip p=0.5, rotation ±30°, brightness/contrast jitter 0.2, ImageNet normalize |
| Held-out | Resize 224×224 và normalize; không augmentation/crop |

Engine dùng cùng transform với Phase 1. `USE_90ROTATION=0` là main;
`1` là ablation chọn góc 0/90/180/270°. Không đặt `BATCH_SIZE` để giữ batch theo model.
Training hiện dùng FP32; Phase 1 dùng AMP và chọn best checkpoint/early stopping.
CV dùng fixed epochs và `final.pth`, nên ghi rõ khác biệt protocol khi đối chiếu.

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

# Kiểm tra cấu trúc data/model/checkpoint, chưa chạy forward/backward.
python train_kfold.py --backbone convnext_tiny --dryrun

# Một epoch cho mỗi model, output riêng; không dùng kết quả smoke để báo cáo.
smoke_root="/workspace/woodbench/data/experiments/Task-B-origin-cv-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
for model in convnext_tiny mobilenetv3_large resnet50 efficientnet_b3 swin_tiny; do
  python train_kfold.py --backbone "$model" --fold 1 --epochs 1 --output_root "$smoke_root"
done
```

## 2. Chạy ngầm 5 model, giữ chạy khi mất SSH

Từ folder Task 1, sau khi smoke test thành công:

```zsh
unset MODELS BATCH_SIZE
N_FOLDS=5 EPOCHS=50 MAX_PARALLEL=5 USE_90ROTATION=0 RESUME=0 \
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
`epoch=12/50` và held-out accuracy của fold hoàn thành. `Ctrl-C` chỉ dừng monitor.
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

Nếu process đã ngừng, chạy lại với **cùng code, dữ liệu và cấu hình**:

```zsh
unset MODELS BATCH_SIZE
N_FOLDS=5 EPOCHS=50 MAX_PARALLEL=5 USE_90ROTATION=0 RESUME=1 \
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
| Chạy lại riêng fold 3 | `python train_kfold.py --backbone convnext_tiny --fold 3 --resume` |
| OOM khi chạy song song | Giảm `MAX_PARALLEL` rồi resume, giữ nguyên batch |
| Đổi batch, epoch, LR, seed, augmentation hoặc code | Chọn `OUTPUT_ROOT` mới; không trộn với run cũ |
| Output riêng | Launcher đặt `OUTPUT_ROOT=/đường/dẫn`; monitor dùng `--output /đường/dẫn` |
| Chuyển instance | Copy code, dataset, manifest và toàn bộ output; dùng lại cấu hình cũ |

Override tùy chọn: `MODELS`, `MAX_PARALLEL`, `N_FOLDS`, `EPOCHS`, `BATCH_SIZE`,
`BACKBONE_LR`, `HEAD_LR`, `WEIGHT_DECAY`, `LABEL_SMOOTHING`, `SEED`, `NUM_WORKERS`,
`DATA_ROOT`, `MANIFEST_PATH`, `OUTPUT_ROOT`, `LOG_ROOT`, `PYTHON`, `WANDB_PROJECT`.
`BATCH_SIZE` nếu đặt sẽ áp dụng cho mọi model. Có thể đặt `WANDB_API_KEY` trước
launch nếu cần tracking. Giữ nguyên các override đó khi resume.

## 5. Xuất kết quả và backup

Output: `data/experiments/Task-B-origin-cv/<model>/fold_<n>/` chứa `final.pth`,
`predictions.csv`, `metrics.json`, `run.json`, `history.json`.
Sau khi đủ fold, mở [evaluate_and_register_cv.ipynb](evaluate_and_register_cv.ipynb)
và chạy các cell: mỗi lần xuất tạo registry CSV riêng, một dòng/run, cùng
per-class metrics, OOF và summary trong `reports/<session>/`; giữ
`run_id`/`source_run_ids` và đường dẫn tương đối theo repo ROOT.
Backup toàn bộ output, code và logs trước khi Destroy instance.

Notebook [fivefold_cv_training.ipynb](fivefold_cv_training.ipynb) gọi cùng engine
nhưng chạy các model tuần tự; dùng launcher ở trên để chạy song song.
Từ repo root, kiểm tra bằng:
`PYTHONPATH="$PWD" pytest -q tests/test_origin_cv.py tests/test_cv_training_runs.py`.
