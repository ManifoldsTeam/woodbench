# Task 7 — Bước 1: chọn backbone LR cho từng model

Mỗi model train hai lần trên split Final-B (`data/task-B-group-split`, train 3.365 / val 903 / test 532),
chỉ khác backbone LR: `1e-5` và `1e-4`. Chọn LR theo **val** của `model_best.pth`; test chỉ ghi lại.
Lý do và bối cảnh: [handoff 01-11-2026](../../../../docs/timeline/01-11-2026-handoff-vast.md), mục 3 và 5.

## Cấu hình

| Tham số | Giá trị |
|---|---|
| Giống Final-B main (`final_b_17class.train_one`) | AdamW wd 1e-4, warmup 5 + cosine, tối đa 50 epoch, early stopping patience 10 theo val acc, label smoothing 0,1, AMP, batch theo `BACKBONE_CONFIGS`, transform `src/data/transforms.py` |
| Head | `model.get_classifier()` nhận head LR `1e-4` (`head_selector="classifier"`) |
| Thay đổi | backbone LR ∈ {`1e-5`, `1e-4`} × 5 model, seed 42 |
| Chấm | Sau khi train, nạp `model_best.pth`, chấm val và test ở FP32 |
| Output | `data/experiments/Task-B-lr-select/bblr_<lr>/seed_42/<backbone>/` |

Khác Final-B cũ ở hai điểm, cả hai có chủ đích: head là classifier thật thay vì quy tắc chuỗi con
`head/fc/classifier`, và val/test chấm từ best checkpoint. `run_config.json` ghi số tham số mỗi nhóm.

Run **không resume**: thư mục run đã có file thì script dừng. Nếu gián đoạn, xóa thư mục run đó rồi chạy lại.
`metrics.json` được ghi cuối cùng, nên run không có `metrics.json` là run dở.

## Chạy trên Vast

```zsh
conda activate wood
cd '/workspace/woodbench/notebooks/phase 2/supervisor-feedback/task_07_lr_selection'

# Kiểm data, model, nhóm tham số và một forward; không ghi gì.
for model in convnext_tiny swin_tiny mobilenetv3_large efficientnet_b3 resnet50; do
  python train_lr_select.py --backbone "$model" --backbone_lr 1e-5 --dryrun
done

# 10 run, chạy ngầm. MAX_PARALLEL mặc định 2; CV v1 từng chạy 5 model song song trên 4090.
MAX_PARALLEL=5 nohup bash run_lr_select.sh \
  > "launch-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
disown
```

Launcher chạy mọi cặp `MODELS × LRS`, mỗi run một log và một file `.exit` trong
`artifacts/phase_2/supervisor-feedback/task_07_lr_selection/logs/<session>/`, trả mã khác 0 nếu có run lỗi.
Override: `MODELS`, `LRS`, `MAX_PARALLEL`, `SEED`, `EPOCHS`, `NUM_WORKERS`, `OUTPUT_ROOT`, `DATA_ROOT`,
`LOG_ROOT`, `PYTHON`. Đặt `WANDB_API_KEY` nếu cần tracking.

Theo dõi:

```zsh
session_dir=$(ls -td ../../../../artifacts/phase_2/supervisor-feedback/task_07_lr_selection/logs/*/ | head -n 1)
tail -f "$session_dir/convnext_tiny_bblr1e-5.log"
ls ../../../../data/experiments/Task-B-lr-select/bblr_*/seed_42/*/metrics.json | wc -l   # đủ: 10
```

### Máy ít CPU: dùng cache 224×224

Nếu GPU dùng ít vì CPU không giải nén kịp PNG 1280×1024, tạo cache một lần rồi trỏ `DATA_ROOT`
(launcher) hoặc `--data_root` (script) vào đó. Input của model không đổi, vì pipeline bắt đầu bằng
`Resize((224,224))`. Lần chạy 01-11-2026 dùng cách này; chi tiết ở
[báo cáo 01-11](../SUPERVISOR_REPORT_2026-11-01.md), mục 6.1.

```zsh
python scripts/build_resized_cache.py data/task-B-group-split data/task-B-group-split-224   # từ repo root
```

## Chọn LR

```zsh
python summarize_lr_select.py
```

Ghi `runs.csv`, `decisions.csv`, `summary.md` vào `data/experiments/Task-B-lr-select/reports/lr-select-<UTC>/`.
Quy tắc (cố định trước khi xem kết quả): val macro-F1 cao hơn thắng; chênh dưới 0,5 điểm (khoảng 4–5 ảnh val)
là hòa và val loss thấp hơn thắng. Model thiếu một trong hai LR hiện `incomplete`.

LR được chọn đưa vào bước 2–4 qua `BACKBONE_LR` (chung) hoặc `BACKBONE_LRS="resnet50:1e-4 ..."` (riêng từng
model) của [Task 1](../task_01_fivefold_cv_training/README.md).

## Mang về

Thư mục `reports/lr-select-<UTC>/`, mỗi run: `run_config.json`, `history.json`, `metrics.json`,
`predictions_{val,test}.csv`; `model_best.pth` của LR được chọn; log session. `model_last.pth` không cần.

Test, từ repo root: `PYTHONPATH="$PWD" pytest -q tests/test_lr_selection.py tests/test_optimizer_head_selector.py`.
