---
phase: planning
title: Task B: Species Classification — Project Planning
description: Task breakdown and timeline for species-level classification of 20 conifer species
---

# Project Planning — Task B: Species Classification (20-class)

## Current Status (2026-05-22)

- **Task A:** Complete (5 models × 3 seeds, acc >97%)
- **Task B data:** `data/task-B/` (3973 / 933 / 924, 20 classes)
- **Infrastructure:** Hydra `train.py` + `scripts/run_experiments.py` + configs + wandb (`WandbTracker`)
- **Smoke test:** 1 epoch ResNet-50 OK → `experiments/taskB/resnet50/seed_42/`
- **Hardware:** 2× RTX 4090 24GB — sẵn sàng chạy main training
- **Next:** 15 main runs (5 models × 3 seeds), sau đó ablation

---

## Milestones

- [x] **M0:** Task B infrastructure (Hydra, configs, launcher, wandb, output layout)
- [ ] **M1:** Task B main experiments (5 models × 3 seeds = 15 runs)
- [ ] **M2:** Ablation experiments (90° rotation, label smoothing, resolution)
- [ ] **M3:** Efficiency profiling (inference time, FLOPs, model size)
- [ ] **M4:** Visualization (confusion matrix, per-class F1, Grad-CAM)

---

## Task Checklist

### Done

| ID | Task | Notes |
|----|------|-------|
| 1.1 | Đọc và dùng `src/` modules | `train.py` gọi TrainingEngine, get_dataloaders |
| 1.2 | `configs/experiment/taskB.yaml` | `data_root`, 20 classes, wandb project |
| 1.3 | Ablation configs | `taskB_90rotation`, `label_smooth_*`, `resolution_384` |
| 1.4 | Entry point thống nhất | Root `train.py` (Hydra); đã xóa `scripts/train.py` trùng |
| 1.5 | wandb logging | `src/train/wandb_tracker.py`, bật qua `train.wandb.enabled` |
| 1.6 | `scripts/run_experiments.py` | Gọi Hydra, ablation presets, `--max_workers` |
| 1.7 | Output layout ổn định | `experiments/{taskB}/{backbone}/seed_{seed}/` |

### In Progress

| ID | Task | Notes |
|----|------|-------|
| 2.0 | Main training 15 runs | Chờ lên lịch 2×4090 (xem mục Parallel Schedule) |

### Not Started

| ID | Task | Notes |
|----|------|-------|
| 2.1 | Đánh giá + metrics sau M1 | `scripts/evaluate.py` |
| 3.1 | Ablation 90° rotation | `experiment=taskB_90rotation`, seed 42 |
| 3.2 | Ablation label smoothing | 0.0 / 0.1 / 0.2 |
| 3.3 | Ablation resolution 384 | VRAM cao hơn — chạy 1 job/GPU |
| 4.1–4.5 | Efficiency + plots | Sau M1 |

### Blocked / Risks

| Risk | Mitigation |
|------|------------|
| `run_experiments.py` chưa gán `CUDA_VISIBLE_DEVICES` | Chạy 2 terminal (mỗi GPU một `CUDA_VISIBLE_DEVICES`) hoặc bổ sung gán GPU trong launcher |
| Swin + ConvNeXt cùng lúc trên 1×4090 | Giữ `batch_size` 32–48 cho Swin/EfficientNet |
| Pretrained download lần đầu | Chạy 1 model trước để cache timm weights |

---

## Main Experiments — 5 models × 3 seeds = 15 runs

| Model | Seeds | Est. / run (50 ep, 4090) | Batch size gợi ý |
|-------|-------|---------------------------|------------------|
| ResNet-50 | 42, 123, 456 | ~35 min | 64 |
| MobileNetV3-Large | 42, 123, 456 | ~30 min | 64 |
| EfficientNet-B3 | 42, 123, 456 | ~45 min | 48 |
| ConvNeXt-Tiny | 42, 123, 456 | ~50 min | 48 |
| Swin-Tiny | 42, 123, 456 | ~60 min | 32 |

**Tổng thời gian tuần tự:** ~11.0 h  
**Với 2×4090 song song (lý tưởng):** ~5.5–6.5 h wall-clock  
**Thực tế (cân bằng job + overhead):** **~6–8 h**

---

## Parallel Schedule — 2× RTX 4090

Giả định: mỗi process chỉ thấy **một GPU** (`CUDA_VISIBLE_DEVICES=0` hoặc `1`), `train.epochs=50`, config `experiment=taskB`.

### Chiến lược

1. **max_workers = 2** — mỗi wave chạy 2 job (1 job / GPU).
2. **Ghép cặp** model nhẹ + model nặng vừa phải để giảm thời gian chờ GPU.
3. **15 runs → 8 waves** → thời gian wall ≈ `8 × T_wave` với `T_wave` ≈ max(job_a, job_b) trong wave.

### Lịch gợi ý (8 waves)

| Wave | GPU 0 | GPU 1 | ~Thời gian wave |
|------|-------|-------|-----------------|
| 1 | resnet50 / seed 42 | mobilenetv3_large / seed 42 | 35 min |
| 2 | resnet50 / seed 123 | mobilenetv3_large / seed 123 | 35 min |
| 3 | resnet50 / seed 456 | mobilenetv3_large / seed 456 | 35 min |
| 4 | efficientnet_b3 / seed 42 | convnext_tiny / seed 42 | 50 min |
| 5 | efficientnet_b3 / seed 123 | convnext_tiny / seed 123 | 50 min |
| 6 | efficientnet_b3 / seed 456 | convnext_tiny / seed 456 | 50 min |
| 7 | swin_tiny / seed 42 | convnext_tiny / seed 42 (nếu wave 4–6 xong) hoặc idle | 60 min |
| 8 | swin_tiny / seed 123 | swin_tiny / seed 456 | 60 min |

**Lưu ý wave 7–8:** Swin chạy lâu nhất — có thể chạy 3 seed Swin tuần tự trên 2 GPU (GPU0: s42, GPU1: s123, rồi s456 trên GPU rảnh) để không lãng phí.

### Lịch tối ưu hơn (Swin song song 2 seed)

| Wave | GPU 0 | GPU 1 |
|------|-------|-------|
| 1 | resnet50_s42 | mobilenetv3_large_s42 |
| 2 | resnet50_s123 | mobilenetv3_large_s123 |
| 3 | resnet50_s456 | mobilenetv3_large_s456 |
| 4 | efficientnet_b3_s42 | convnext_tiny_s42 |
| 5 | efficientnet_b3_s123 | convnext_tiny_s123 |
| 6 | efficientnet_b3_s456 | convnext_tiny_s456 |
| 7 | swin_tiny_s42 | swin_tiny_s123 |
| 8 | swin_tiny_s456 | *(idle hoặc bắt đầu ablation nhẹ)* |

**Wall-clock ước tính:** 35×3 + 50×3 + 60×2 ≈ **375 min (~6.25 h)**

### Lệnh mẫu (manual 2 GPU — khuyến nghị trước khi sửa launcher)

```bash
conda activate wood
cd /path/to/woodbench

# Terminal 1 — GPU 0
export CUDA_VISIBLE_DEVICES=0
python train.py experiment=taskB model.backbone=resnet50 train.seed=42 \
  data.dataloader.batch_size=64

# Terminal 2 — GPU 1
export CUDA_VISIBLE_DEVICES=1
python train.py experiment=taskB model.backbone=mobilenetv3_large train.seed=42 \
  data.dataloader.batch_size=64
```

### Batch launcher (cần gán GPU thủ công hoặc 2 terminal)

```bash
# Chạy tuần tự từng cặp wave — an toàn nếu chưa patch CUDA_VISIBLE_DEVICES
python scripts/run_experiments.py --experiment taskB \
  --backbone resnet50 --seeds 42 123 456 --max_workers 1

# Hoặc sau khi patch GPU rotation:
python scripts/run_experiments.py --experiment taskB --all_models \
  --seeds 42 123 456 --max_workers 2
```

Output: `experiments/taskB/{backbone}/seed_{seed}/` (sync lên Drive bằng rclone).

---

## Ablation (sau M1, ~3–4 h thêm với 2 GPU)

| Ablation | Hydra experiment | Runs | Ghi chú |
|----------|------------------|------|---------|
| 90° rotation | `taskB_90rotation` | 1× best model, seed 42 | So với baseline |
| Label smooth 0.0 / 0.2 | `taskB_label_smoothing_*` | 2 | Baseline 0.1 đã có trong M1 |
| Resolution 384 | `taskB_resolution_384` | 1 | `batch_size=24–32`, 1 job/GPU |

Chạy trên **ConvNeXt-Tiny** (best Task A) trước, seed 42; multi-seed ablation tùy thời gian.

---

## Machine Assignment — 2 vast.ai instances

Mỗi instance = 1 GPU độc lập, không shared VRAM. Chạy notebook trên mỗi instance.

### Machine Assignment

| Máy | Notebooks | Batch | ~Thời gian |
|-----|-----------|-------|-------------|
| Machine A | ConvNeXt (3 seeds) | 32 | 50×3 = 150 min |
| Machine A | MobileNet (3 seeds) | 64 | 30×3 = 90 min |
| Machine B | Swin (3 seeds) | 32 | 60×3 = 180 min |
| Machine B | ResNet (3 seeds) | 64 | 35×3 = 105 min |
| Machine A (sau cùng) | EfficientNet (3 seeds) | 32 | 45×3 = 135 min |

**Machine A total:** 150+90+135 = 375 min (~6.25h)  
**Machine B total:** 180+105 = 285 min (~4.75h)  
**Makespan:** max(375, 285) = **~6.25h** wall-clock (2 GPU rời)


### Notebooks

```
notebooks/phase 1/task-B/
├── TaskB_convnext_tiny.ipynb
├── TaskB_swin_tiny.ipynb
├── TaskB_efficientnet_b3.ipynb
├── TaskB_mobilenetv3_large.ipynb
├── TaskB_resnet50.ipynb
└── TaskB_Results_Driver.ipynb  # tổng hợp wandb sau khi xong
```

**Workflow:**
1. Machine A: `TaskB_convnext_tiny.ipynb`
2. Machine B: `TaskB_swin_tiny.ipynb` (song song với A)
3. Machine A: `TaskB_mobilenetv3_large.ipynb`
4. Machine B: `TaskB_resnet50.ipynb`
5. Machine A (sau): `TaskB_efficientnet_b3.ipynb`
6. Tổng hợp: `TaskB_Results_Driver.ipynb`

**Total makespan:** max(Machine A ~5h, Machine B ~4.75h) ≈ **5h**

---

## Resources

| Resource | Path / value |
|----------|----------------|
| Dataset | HuggingFace `ny0g507h3p/wood-task-B` hoặc `data/task-B/` |
| Checkpoints | wandb `woodbench-taskB` |
| GPU | 1× RTX 4090 24GB mỗi instance |
| Conda env | `wood` |

---

## Next Steps (ưu tiên)

1. **Khởi tạo 2 instance vast.ai** (1 GPU mỗi instance)
2. **Machine A**: `TaskB_convnext_tiny.ipynb` + `TaskB_mobilenetv3_large.ipynb` + `TaskB_efficientnet_b3.ipynb`
3. **Machine B**: `TaskB_swin_tiny.ipynb` + `TaskB_resnet50.ipynb` (song song với A)
4. **Tổng hợp**: `TaskB_Results_Driver.ipynb` → `evaluate.py` → báo cáo
5. **Ablation**: sau M1, chạy trên ConvNeXt seed 42

---

## Summary (cho timeline)

5 notebook (1 model × 3 seeds), 2 GPU rời. Makespan ~5h wall-clock.
